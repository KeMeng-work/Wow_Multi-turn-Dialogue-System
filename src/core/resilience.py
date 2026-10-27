"""
过载保护与运行时指标（压测第 2 步 P2，2026-10-07）

组件：
  - AdmissionController：全局在途请求上限（准入控制）。满额时最多排队 queue_timeout 秒，仍无名额则拒绝（503）
  - AsyncBulkhead / ThreadBulkhead：依赖隔离（舱壁）。协程侧用 asyncio 信号量，线程侧（检索工具在线程池中）用线程信号量
  - RetryBudget：重试预算。滑动窗口内重试次数不超过 max(min_retries, ratio × 请求数)，防止过载时重试放大流量
  - LoopLagMonitor：事件循环延迟。后台协程每 interval 秒醒来一次，实际醒来时间与预期之差即事件循环被阻塞的时长
  - snapshot()：汇总以上指标与线程池排队数，供 /metrics 输出

设计约束：
  - 名额 / 舱壁一律用 async with / with 获取，异常、取消、客户端断开时都会在 finally 中归还，不会泄漏
  - max ≤ 0 表示不限（压测并发阶梯时关闭准入，测系统自然拐点）
"""

import asyncio
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from config.config_path import OVERLOAD_CONFIG


class AdmissionRejected(Exception):
    """准入控制拒绝：在途请求已满且排队超时"""


class BulkheadFull(Exception):
    """舱壁已满且等待超时，由调用方降级"""


# ═══════════════════════════ 准入控制 / 协程舱壁 ═══════════════════════════

class AsyncBulkhead:
    """协程侧并发上限。limit ≤ 0 时不限，仅计数。"""

    def __init__(self, name: str, limit: int, timeout_s: float, exc: type = BulkheadFull):
        self.name = name
        self.limit = limit
        self.timeout_s = timeout_s
        self._exc = exc
        self._sem = asyncio.Semaphore(limit) if limit > 0 else None
        self.in_use = 0
        self.waiting = 0
        self.peak_in_use = 0
        self.acquired_total = 0
        self.rejected_total = 0

    async def __aenter__(self):
        if self._sem is not None:
            self.waiting += 1
            try:
                async with asyncio.timeout(self.timeout_s):
                    await self._sem.acquire()
            except TimeoutError:
                self.rejected_total += 1
                raise self._exc(f"{self.name} 已满（上限 {self.limit}），等待 {self.timeout_s}s 超时")
            finally:
                self.waiting -= 1
        self.in_use += 1
        self.acquired_total += 1
        self.peak_in_use = max(self.peak_in_use, self.in_use)
        return self

    async def __aexit__(self, *exc):
        self.in_use -= 1
        if self._sem is not None:
            self._sem.release()
        return False

    def stats(self, reset_peak: bool = False) -> dict:
        s = {"limit": self.limit, "in_use": self.in_use, "waiting": self.waiting,
             "peak_in_use": self.peak_in_use, "acquired_total": self.acquired_total,
             "rejected_total": self.rejected_total}
        if reset_peak:
            self.peak_in_use = self.in_use
        return s


class AdmissionController(AsyncBulkhead):
    """全局准入：/chat 与 /chat/stream 入口处获取，请求结束时归还"""

    def __init__(self, max_inflight: int, queue_timeout_s: float):
        super().__init__("admission", max_inflight, queue_timeout_s, exc=AdmissionRejected)


# ═══════════════════════════ 线程侧舱壁 ═══════════════════════════

class ThreadBulkhead:
    """线程侧并发上限（检索工具在线程池中执行）。limit ≤ 0 时不限，仅计数。"""

    def __init__(self, name: str, limit: int, timeout_s: float):
        self.name = name
        self.limit = limit
        self.timeout_s = timeout_s
        self._sem = threading.BoundedSemaphore(limit) if limit > 0 else None
        self._lock = threading.Lock()
        self.in_use = 0
        self.peak_in_use = 0
        self.acquired_total = 0
        self.rejected_total = 0

    def __enter__(self):
        if self._sem is not None and not self._sem.acquire(timeout=self.timeout_s):
            with self._lock:
                self.rejected_total += 1
            raise BulkheadFull(f"{self.name} 已满（上限 {self.limit}），等待 {self.timeout_s}s 超时")
        with self._lock:
            self.in_use += 1
            self.acquired_total += 1
            self.peak_in_use = max(self.peak_in_use, self.in_use)
        return self

    def __exit__(self, *exc):
        with self._lock:
            self.in_use -= 1
        if self._sem is not None:
            self._sem.release()
        return False

    def stats(self, reset_peak: bool = False) -> dict:
        with self._lock:
            s = {"limit": self.limit, "in_use": self.in_use, "peak_in_use": self.peak_in_use,
                 "acquired_total": self.acquired_total, "rejected_total": self.rejected_total}
            if reset_peak:
                self.peak_in_use = self.in_use
        return s


# ═══════════════════════════ 重试预算 ═══════════════════════════

class RetryBudget:
    """
    滑动窗口重试预算（线程安全，改写在线程池中调用）。
    每次首发请求调用 record_request()；需要重试时调用 try_acquire()，预算不足返回 False → 放弃重试直接降级。
    """

    def __init__(self, ratio: float, min_retries: int, window_s: float):
        self.ratio = ratio
        self.min_retries = min_retries
        self.window_s = window_s
        self._requests: deque = deque()
        self._retries: deque = deque()
        self._lock = threading.Lock()
        self.denied_total = 0
        self.retries_total = 0

    def _trim(self, now: float) -> None:
        for q in (self._requests, self._retries):
            while q and now - q[0] > self.window_s:
                q.popleft()

    def record_request(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            self._requests.append(now)

    def try_acquire(self) -> bool:
        now = time.monotonic()
        with self._lock:
            self._trim(now)
            allowed = max(self.min_retries, int(self.ratio * len(self._requests)))
            if len(self._retries) >= allowed:
                self.denied_total += 1
                return False
            self._retries.append(now)
            self.retries_total += 1
            return True

    def stats(self) -> dict:
        with self._lock:
            self._trim(time.monotonic())
            return {"ratio": self.ratio, "window_requests": len(self._requests),
                    "window_retries": len(self._retries), "retries_total": self.retries_total,
                    "denied_total": self.denied_total}


# ═══════════════════════════ 事件循环延迟 ═══════════════════════════

class LoopLagMonitor:
    """后台协程每 interval_s 醒来一次；实际间隔 − 预期间隔 = 事件循环被同步代码占住的时长"""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.last_ms = 0.0
        self.max_ms = 0.0          # 自上次 /metrics 读取以来的最大值（读取后重置）
        self.ewma_ms = 0.0
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            t0 = loop.time()
            await asyncio.sleep(self.interval_s)
            lag = max(0.0, (loop.time() - t0 - self.interval_s) * 1000)
            self.last_ms = lag
            self.max_ms = max(self.max_ms, lag)
            self.ewma_ms = 0.8 * self.ewma_ms + 0.2 * lag

    def stats(self, reset_peak: bool = False) -> dict:
        s = {"last_ms": round(self.last_ms, 1), "max_ms": round(self.max_ms, 1), "ewma_ms": round(self.ewma_ms, 1)}
        if reset_peak:
            self.max_ms = self.last_ms
        return s


# ═══════════════════════════ 全局实例 ═══════════════════════════

ADMISSION = AdmissionController(OVERLOAD_CONFIG['admission_max_inflight'],
                                OVERLOAD_CONFIG['admission_queue_timeout_s'])
# 4B 前台：改写（协程侧）+ 充分性判定（线程侧）；两者之和 + 后台 LLM 并发 ≤ 4B 的 max-num-seqs，
# 使请求不在 vLLM 内部排队，前台优先由舱壁保证
REWRITE_BULKHEAD = AsyncBulkhead("rewrite_4b", OVERLOAD_CONFIG['bulkhead_rewrite'],
                                 OVERLOAD_CONFIG['bulkhead_rewrite_timeout_s'])
EVALUATOR_BULKHEAD = ThreadBulkhead("evaluator_4b", OVERLOAD_CONFIG['bulkhead_evaluator'],
                                    OVERLOAD_CONFIG['bulkhead_evaluator_timeout_s'])
RERANK_BULKHEAD = ThreadBulkhead("rerank_tei", OVERLOAD_CONFIG['bulkhead_rerank'],
                                 OVERLOAD_CONFIG['bulkhead_rerank_timeout_s'])
RETRY_BUDGET = RetryBudget(OVERLOAD_CONFIG['retry_budget_ratio'],
                           OVERLOAD_CONFIG['retry_budget_min'],
                           OVERLOAD_CONFIG['retry_budget_window_s'])
LOOP_LAG = LoopLagMonitor()

_executor: Optional[ThreadPoolExecutor] = None


def register_executor(executor: ThreadPoolExecutor) -> None:
    """api.py lifespan 创建默认线程池后登记，供 snapshot 读取排队数"""
    global _executor
    _executor = executor


def snapshot(reset_peaks: bool = True) -> dict:
    """/metrics 输出；峰值类指标读取后重置为当前值，使每次采样反映"上次采样以来"的峰值"""
    pool = None
    if _executor is not None:
        pool = {"max_workers": _executor._max_workers,
                "threads": len(_executor._threads),
                "queue": _executor._work_queue.qsize()}
    return {
        "admission": ADMISSION.stats(reset_peaks),
        "bulkheads": {
            "rewrite_4b": REWRITE_BULKHEAD.stats(reset_peaks),
            "evaluator_4b": EVALUATOR_BULKHEAD.stats(reset_peaks),
            "rerank_tei": RERANK_BULKHEAD.stats(reset_peaks),
        },
        "retry_budget": RETRY_BUDGET.stats(),
        "loop_lag": LOOP_LAG.stats(reset_peaks),
        "thread_pool": pool,
    }
