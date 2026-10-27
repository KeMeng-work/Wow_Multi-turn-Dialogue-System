"""
熔断器模块 (CircuitBreaker)

三状态机：CLOSED（闭合）→ OPEN（断开）→ HALF_OPEN（半开）→ CLOSED
1、使用滑动窗口（Sorted Set）统计 60 秒内的请求数和失败数。
2、当样本数 >5 and 失败率 > 50% ，状态机从 闭合 转为 断开，断开 持续30秒转为 半开。
3、半开阶段最多允许 2 个探测请求，if成功恢复闭合，else 重新断开。
4、状态全部持久化到 Redis，服务重启后自动恢复上次状态。

失败定义（计入统计）：未捕获的 Exception、连接异常、TimeoutError、HTTP 5XX
不计入失败：限流、HTTP 4xx（客户端错误）

Author: Ke Meng
Created: 2026-04-10
Version: 1.0.0
"""

import time
import logging
import uuid
from enum import Enum
from typing import Optional

import redis

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"       # 正常，放行所有请求
    OPEN = "OPEN"         # 断开，快速失败
    HALF_OPEN = "HALF_OPEN"    # 半开，放行少量探测请求


class CircuitBreakerOpen(Exception):
    """熔断器断开时抛出，由调用方捕获并返回降级响应"""
    pass


class CircuitBreaker:
    """
    基于 Redis 的持久化熔断器

    使用方式：
        cb = CircuitBreaker(redis_client=redis_client, name="agent")

        # 请求前
        cb.before_call()               # 断开时抛出 CircuitBreakerOpen

        # 请求后
        cb.on_success()                # 成功时调用
        cb.on_failure(exc)             # 失败时调用，传入异常实例
    """

    def __init__(
        self,
        redis_client:      redis.Redis,
        name:              str = "agent",
        failure_threshold: float = 0.5,    # 失败率：50%
        min_samples:       int = 5,      # 最小请求数
        window_seconds:    int = 60,     # 滑动窗口长度（秒）
        open_duration:     int = 30,     # 断开持续时间（秒）
        half_open_max:     int = 2,      # 半开阶段最大探测请求数
    ):
        self.redis = redis_client
        self.name = name
        self.failure_threshold = failure_threshold
        self.min_samples = min_samples
        self.window = window_seconds
        self.open_duration = open_duration
        self.half_open_max = half_open_max

        # ── Redis Key 定义 ────────────────────────────────────
        self._state_key = f"cb:{name}:state"        # 当前状态字符串
        self._open_time_key = f"cb:{name}:open_time"    # 进入 OPEN 的时间戳
        self._half_count_key = f"cb:{name}:half_count"   # 半开阶段已放行探测数
        self._failures_key = f"cb:{name}:failures"     # Sorted Set：失败时间戳
        self._requests_key = f"cb:{name}:requests"     # Sorted Set：请求时间戳

    # ══════════════════════════════════════════════════════════
    # 公共接口
    # ══════════════════════════════════════════════════════════

    def get_state(self) -> CircuitState:
        """
        读取当前熔断状态。
        自动处理 OPEN → HALF_OPEN 的超时转换，无需外部轮询。
        """
        # Redis读取状态字符串
        raw = self.redis.get(self._state_key)
        # 若无状态，默认为闭合（CLOSED）
        state = CircuitState(raw) if raw else CircuitState.CLOSED

        # 如果当前是断开状态，检查是否已超过恢复窗口时间
        if state == CircuitState.OPEN:
            open_time = self.redis.get(self._open_time_key)
            # 如果记录存在且当前时间已超过断开持续时间
            if open_time and (time.time() - float(open_time)) >= self.open_duration:
                # 自动转为半开状态，允许最多2个探测请求
                self._transition_to(CircuitState.HALF_OPEN)
                return CircuitState.HALF_OPEN

        return state

    def before_call(self) -> None:
        """
        请求前调用，检查是否允许放行。
        :raises CircuitBreakerOpen: 熔断断开或半开名额用尽时拒绝请求
        """
        state = self.get_state()
        # ── 情况1：熔断器处于断开状态 ────────────────────────────
        # 直接拒绝所有请求，快速失败，并记录监控指标
        if state == CircuitState.OPEN:
            logger.warning("熔断器 [%s] 断开，拒绝请求", self.name)
            self._incr_metric("cb:metrics:rejected")
            raise CircuitBreakerOpen("服务熔断中，请稍后再试")

        # ── 情况2：熔断器处于半开状态 ────────────────────────────
        # 只允许有限数量的探测请求通过，超过则拒绝
        if state == CircuitState.HALF_OPEN:
            # 原子递增探测计数，并设置过期时间
            count = self.redis.incr(self._half_count_key)
            self.redis.expire(self._half_count_key, self.open_duration * 2)
            # 如果探测请求数超过允许的最大值，则拒绝
            if count > self.half_open_max:
                raise CircuitBreakerOpen("服务恢复探测中，请稍后再试")

        # 记录本次请求到滑动窗口（闭合和半开均记录）
        self._record_event(self._requests_key)

    def on_success(self) -> None:
        """请求成功后调用"""
        state = self.get_state()
        if state == CircuitState.HALF_OPEN:
            # 半开探测成功，恢复闭合
            self._transition_to(CircuitState.CLOSED)
            logger.info("熔断器 [%s] 恢复闭合", self.name)

    def on_failure(self, exc: Exception) -> None:
        """
        请求失败后调用。
        会根据异常类型判断是否计入失败统计，不是所有错误都触发熔断。
        """
        # ── 不计入失败的情况 ──────────────────────────────────
        # 1. 限流触发（客户端行为，不代表下游故障）
        try:
            from src.core.high_frequency_query_cache.rate_limiter import RateLimitExceeded
            if isinstance(exc, RateLimitExceeded):
                return
        except ImportError:
            pass

        # 2. HTTP 4xx（客户端错误，不代表下游故障）
        if hasattr(exc, 'status_code') and 400 <= getattr(exc, 'status_code', 0) < 500:
            return

        # ── 计入失败统计 ──────────────────────────────────────
        self._record_event(self._failures_key)

        # 获取当前熔断器状态
        state = self.get_state()

        # ── 情况1：当前处于半开状态 ──────────────────────────────────
        # 半开状态下的探测请求失败，说明下游服务仍未恢复
        # 需要重新断开熔断器，进入 OPEN 状态，等待下一个恢复窗口
        if state == CircuitState.HALF_OPEN:
            # 半开探测失败，重新断开
            self._transition_to(CircuitState.OPEN)
            logger.warning("熔断器 [%s] 半开探测失败，重新断开", self.name)
            return

        # ── 情况2：当前处于闭合状态 ──────────────────────────────────
        # 计算滑动窗口内的失败率，若达到阈值则触发熔断，转为 OPEN 状态
        if state == CircuitState.CLOSED:
            rate = self._failure_rate()  # 计算失败率（样本数不足时返回 None）
            if rate is not None and rate >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                logger.error(
                    "熔断器 [%s] 触发断开，当前失败率=%.1f%%",
                    self.name, rate * 100,
                )
                self._incr_metric("cb:metrics:triggered")   # 记录触发次数，用于监控

    def get_fallback(self) -> str:
        """熔断时的降级响应，返回固定提示语。"""
        return "系统暂时繁忙，请稍后再试。"

    # ══════════════════════════════════════════════════════════
    # 内部方法
    # ══════════════════════════════════════════════════════════

    def _transition_to(self, state: CircuitState) -> None:
        """状态转换，同时清理相关 Redis key"""
        pipe = self.redis.pipeline()
        # 持久化当前状态，过期时间 3600 秒
        pipe.set(self._state_key, state.value, ex=3600)

        if state == CircuitState.OPEN:
            # 进入断开状态：记录断开时间戳（用于超时恢复），清除半开计数
            pipe.set(self._open_time_key, time.time(), ex=3600)
            pipe.delete(self._half_count_key)

        elif state == CircuitState.CLOSED:
            # 恢复闭合：清除所有熔断相关数据（断开时间、半开计数、滑动窗口统计）
            pipe.delete(self._open_time_key)
            pipe.delete(self._half_count_key)
            pipe.delete(self._failures_key)
            pipe.delete(self._requests_key)

        elif state == CircuitState.HALF_OPEN:
            # 进入半开状态：重置探测计数为 0，并设置过期时间（避免长期占用）
            pipe.set(self._half_count_key, 0, ex=self.open_duration * 2)

        pipe.execute()

    def _record_event(self, key: str) -> None:
        """
        向滑动窗口 Sorted Set 写入当前时间戳，并清理窗口外的旧数据。
        Score 和 Member 均使用时间戳（加微小随机偏移避免 member 重复）。
        """
        now = time.time()
        # 使用 `时间戳:随机后缀` 作为 member，确保唯一性（高并发下不会覆盖）
        member = f"{now:.6f}:{uuid.uuid4().hex[:8]}"
        pipe = self.redis.pipeline()
        pipe.zadd(key, {member: now})  # 添加当前事件
        pipe.zremrangebyscore(key, 0, now - self.window)  # 删除窗口外的旧记录
        pipe.expire(key, self.window * 2)  # 整体过期时间，避免残留
        pipe.execute()

    def _failure_rate(self) -> Optional[float]:
        """
        计算滑动窗口内的失败率。
        样本数不足 min_samples 时返回 None，不触发熔断。
        """
        now    = time.time()
        cutoff = now - self.window

        pipe = self.redis.pipeline()
        pipe.zcount(self._requests_key, cutoff, now)  # 窗口内总请求数
        pipe.zcount(self._failures_key, cutoff, now)  # 窗口内失败数
        total, failures = pipe.execute()

        if total < self.min_samples:    # 样本太少，不触发熔断
            return None
        return failures / total

    def _incr_metric(self, key: str) -> None:
        """写入监控计数，保留 24 小时"""
        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, 86400)
            pipe.execute()
        except Exception:
            pass