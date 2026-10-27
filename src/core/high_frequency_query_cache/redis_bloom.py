"""
高频查询缓存模块（High Frequency Query Cache）
- ReAct Agent前置模块，不属于ReAct 工具集

该模块负责对改写后的用户查询进行高级归一化处理，避免只有完全一致的字符串才能命中缓存。
归一化处理后通过 Redis 布隆过滤器 + 计数器实现高频查询缓存，减少重复计算。
如缓存命中，则直接返回最终答案。
如缓存未命中，则进入后续ReAct Agent模块。

主要功能：
    - 高级查询归一化（全角半角符号、货币符号、中英文数字类型等）
    - Redis布隆过滤器快速判断查询是否曾出现
    - 统计查询频率，超过阈值后直接返回缓存，避免调用下游Agent
    - 分层 TTL：普通高频 30分钟，超高频（≥10次）2小时
    - TTL 随机偏移 ±60 秒，防止缓存雪崩
    - 互斥锁接口：防止高频 query 缓存过期时被同时大量击穿
    - 提供 get/update/clear/is_high_freq/try_lock/release_lock 接口
    - 监控指标写入 Redis（命中率、击穿次数等）

依赖：
    - redis: 历史存储
    - cn2an: 中英文数字转换
    - bloom: 布隆过滤器，redis-stack自带

Author: Ke Meng
Created: 2026-01-22
Version: 1.1.0
Last Modified: 2026-04-10

变更记录：
    - 1.1.0 (2026-04-10):
                        新增 1、分层 TTL：普通高频 30 分钟，超高频（≥10次）2 小时
                        新增 2、TTL 随机偏移 ±60 秒，防止缓存雪崩（大量 key 同时过期冲击下游）
                        新增 3、互斥锁接口（try_lock / release_lock）：
                                防止缓存击穿——高频 query 缓存过期时多个请求同时穿透到 Agent
                                锁超时 5 秒，防止持有锁的请求异常后锁永不释放
                        新增 4、is_high_freq()：判断 query 是否已达高频阈值（供 main.py 决策是否加锁）
                        新增 5、监控指标写入 Redis：缓存命中数、未命中数、击穿数

    - 1.0.1 (2026-03-18):
                        改动1、引入cn2an库解决中英文数字类型转换，去除自定义词典（过于繁琐且效果不佳）

    - 1.0.0 (2026-01-22): 初始版本
"""

import logging
import random
import re
import redis
from typing import Optional, Callable

try:
    import cn2an
    CN2AN_AVAILABLE = True
except ImportError:
    CN2AN_AVAILABLE = False
    print("Warning: cn2an library not installed. Chinese number normalization will be disabled.")

logger = logging.getLogger(__name__)

# ── TTL 配置 ─────────────────────────────────────────────────
_TTL_NORMAL = 1800     # 普通高频（5~9次）：30 分钟
_TTL_HOT = 7200     # 超高频（≥10次） ：2 小时
_TTL_JITTER = 60       # 随机偏移上限：±60 秒
_HOT_THRESHOLD = 10      # 超高频判定次数

# ── 互斥锁配置 ────────────────────────────────────────────────
_LOCK_TTL = 30       # 锁超时秒数，防止持有锁的请求崩溃后锁永不释放，当前 30s 足够宽松，系统上线后监控 LLM 微调


def _build_full_to_half_map():
    """构建全角字符到半角字符的映射字典"""
    return {
        0xff01: 0x21, 0xff02: 0x22, 0xff03: 0x23, 0xff04: 0x24,
        0xff05: 0x25, 0xff06: 0x26, 0xff07: 0x27, 0xff08: 0x28,
        0xff09: 0x29, 0xff0a: 0x2a, 0xff0b: 0x2b, 0xff0c: 0x2c,
        0xff0d: 0x2d, 0xff0e: 0x2e, 0xff0f: 0x2f, 0xff10: 0x30,
        0xff11: 0x31, 0xff12: 0x32, 0xff13: 0x33, 0xff14: 0x34,
        0xff15: 0x35, 0xff16: 0x36, 0xff17: 0x37, 0xff18: 0x38,
        0xff19: 0x39, 0xff1a: 0x3a, 0xff1b: 0x3b, 0xff1c: 0x3c,
        0xff1d: 0x3d, 0xff1e: 0x3e, 0xff1f: 0x3f, 0xff20: 0x40,
        0xff21: 0x61, 0xff22: 0x62, 0xff23: 0x63, 0xff24: 0x64,
        0xff25: 0x65, 0xff26: 0x66, 0xff27: 0x67, 0xff28: 0x68,
        0xff29: 0x69, 0xff2a: 0x6a, 0xff2b: 0x6b, 0xff2c: 0x6c,
        0xff2d: 0x6d, 0xff2e: 0x6e, 0xff2f: 0x6f, 0xff30: 0x70,
        0xff31: 0x71, 0xff32: 0x72, 0xff33: 0x73, 0xff34: 0x74,
        0xff35: 0x75, 0xff36: 0x76, 0xff37: 0x77, 0xff38: 0x78,
        0xff39: 0x79, 0xff3a: 0x7a, 0xff3b: 0x5b, 0xff3c: 0x5c,
        0xff3d: 0x5d, 0xff3e: 0x5e, 0xff3f: 0x5f, 0xff40: 0x60,
        0xff41: 0x61, 0xff42: 0x62, 0xff43: 0x63, 0xff44: 0x64,
        0xff45: 0x65, 0xff46: 0x66, 0xff47: 0x67, 0xff48: 0x68,
        0xff49: 0x69, 0xff4a: 0x6a, 0xff4b: 0x6b, 0xff4c: 0x6c,
        0xff4d: 0x6d, 0xff4e: 0x6e, 0xff4f: 0x6f, 0xff50: 0x70,
        0xff51: 0x71, 0xff52: 0x72, 0xff53: 0x73, 0xff54: 0x74,
        0xff55: 0x75, 0xff56: 0x76, 0xff57: 0x77, 0xff58: 0x78,
        0xff59: 0x79, 0xff5a: 0x7a, 0xff5b: 0x7b, 0xff5c: 0x7c,
        0xff5d: 0x7d, 0xff5e: 0x7e
    }


def _build_currency_map():
    """构建货币符号统一映射字典（键为正则表达式，值为标准符号）"""
    return {
        r'\$': '$', r'usd\b': '$', r'dollar\b': '$', r'dollars\b': '$',
        r'€': '€', r'eur\b': '€', r'euro\b': '€', r'euros\b': '€',
        r'£': '£', r'gbp\b': '£', r'pound\b': '£', r'pounds\b': '£',
        r'¥': '¥', r'cny\b': '¥', r'jpy\b': '¥', r'yuan\b': '¥', r'yen\b': '¥',
    }


_full_to_half = _build_full_to_half_map()
CURRENCY_MAP = _build_currency_map()


def advanced_normalize(text: str) -> str:
    """
    对查询文本进行高级归一化处理。
    """
    if not text:
        return ""
    # 1. 去除首尾空格，全部转为小写
    s = text.strip().lower()
    # 2. 全角字符转半角
    s = s.translate(_full_to_half)
    # 3. 货币符号统一
    for pattern, replacement in CURRENCY_MAP.items():
        s = re.sub(pattern, replacement, s, flags=re.IGNORECASE)
    # 4. 中文数字与阿拉伯数字互转
    if CN2AN_AVAILABLE:
        try:
            s = cn2an.transform(s, "cn2an")
        except Exception as e:
            logger.debug(f"cn2an transform failed: {e}")
    else:
        logger.warning("cn2an not installed, Chinese numbers may not be normalized.")
    # 5. 去除千位分隔符（逗号或空格）
    s = re.sub(r'(?<=\d)[, ](?=\d)', '', s)

    def _sci_to_float(match):
        # 6. 科学计数法转普通浮点数，如 1.23e+4 → 12300
        try:
            f = float(match.group(0))
            if f.is_integer():
                return str(int(f))
            else:
                return f"{f:.10f}".rstrip('0').rstrip('.')
        except:
            return match.group(0)
    s = re.sub(r'\d+\.?\d*[eE][+-]?\d+', _sci_to_float, s)
    # 7. 规范化括号周围空格：左括号前加空格，右括号后加空格（若与字母数字相邻）
    s = re.sub(r'([a-z0-9])\(', r'\1 (', s)
    s = re.sub(r'\)([a-z0-9])', r') \1', s)
    # 8. 合并多个空格为一个空格，并去除首尾空格
    s = re.sub(r'\s+', ' ', s).strip()

    return s


def _jitter_ttl(base_ttl: int) -> int:
    """
    为 TTL 添加 ±_TTL_JITTER 秒随机偏移，防止大量 key 同时过期引发缓存雪崩。
    最小值不低于 base_ttl一半。
    """
    offset = random.randint(-_TTL_JITTER, _TTL_JITTER)
    return max(base_ttl // 2, base_ttl + offset)


class HighFreqCache:
    """
    高频查询缓存器，基于 Redis 布隆过滤器和计数器实现。

    核心逻辑：
        - 对查询进行归一化（可通过 normalize_func 自定义）
        - 使用布隆过滤器快速判断查询是否曾出现（减少 Redis 读压力）
        - 通过计数器统计查询频率，窗口内超过阈值则缓存响应
        - 分层 TTL：普通高频 30 分钟，超高频 2 小时，均加随机偏移
        - 互斥锁防止高频 query 缓存过期时被同时大量击穿
        - 后续相同查询直接返回缓存，避免重复计算
    """

    def __init__(
        self,
        redis_client:    redis.Redis,
        bloom_key:       str = "bf:queries",
        counter_prefix:  str = "cnt:",
        cache_prefix:    str = "cache:",
        lock_prefix:     str = "lock:cache:",    # 互斥锁前缀
        threshold:       int = 5,
        window_seconds:  int = 1800,
        normalize_func:  Optional[Callable[[str], str]] = None,
        bloom_error_rate: float = 0.01,
        bloom_capacity:  int = 1000000,
        namespace:       str = "",
    ):
        self.redis = redis_client
        # 版本命名空间（知识库:提示词:主模型）：任一变化时换一套键与布隆过滤器，旧缓存自然失效
        ns = f"{namespace}:" if namespace else ""
        self.namespace = namespace
        self.bloom_key = f"{bloom_key}:{namespace}" if namespace else bloom_key
        self.counter_prefix = f"{counter_prefix}{ns}"
        self.cache_prefix = f"{cache_prefix}{ns}"
        self.lock_prefix = f"{lock_prefix}{ns}"
        self.threshold = threshold
        self.window = window_seconds
        self.normalize = normalize_func if normalize_func else advanced_normalize

        self._ensure_bloom_filter(bloom_error_rate, bloom_capacity)

    # ══════════════════════════════════════════════════════════
    # 原有接口（TTL 升级为分层+jitter）
    # ══════════════════════════════════════════════════════════

    def _ensure_bloom_filter(self, error_rate: float, capacity: int):
        """确保布隆过滤器存在，若不存在则自动创建。"""
        try:
            self.redis.execute_command('BF.INFO', self.bloom_key)
        except redis.exceptions.ResponseError:
            self.redis.execute_command('BF.RESERVE', self.bloom_key, error_rate, capacity)
            logger.info(f"布隆过滤器已创建: {self.bloom_key}")

    def _get_counter_key(self, norm_query: str) -> str:
        """生成查询计数的 Redis 键名。"""
        return f"{self.counter_prefix}{norm_query}"

    def _get_cache_key(self, norm_query: str) -> str:
        """生成缓存响应的 Redis 键名。"""
        return f"{self.cache_prefix}{norm_query}"

    def _get_lock_key(self, norm_query: str) -> str:
        """生成互斥锁的 Redis 键名。"""
        return f"{self.lock_prefix}{norm_query}"

    def _norm(self, query: str, variant: str = "") -> str:
        """
        归一化查询；variant 为影响答案的会话约束指纹（短期记忆 constraint_hash）：
        同一问题在不同约束下（如"只看 2020 年以后" vs "只看 2015 年以后"）是不同的缓存项。
        """
        norm = self.normalize(query)
        return f"{norm}#{variant}" if variant else norm

    def get(self, query: str, variant: str = "") -> Optional[str]:
        """获取查询的缓存响应"""
        # 1. 记录原始查询并归一化
        logger.info(f"[Cache] 接收到的查询改写模块输出: {query}")
        norm = self._norm(query, variant)
        logger.info(f"[Cache] Normalized query: {norm}")

        # 2. 布隆过滤器快速判断：从未见过的查询直接 miss
        exists = self.redis.execute_command('BF.EXISTS', self.bloom_key, norm)
        if not exists:
            logger.info(f"[Cache] Query '{norm}' not in bloom filter, cache miss")
            self._incr_metric("cache:metrics:miss")
            return None

        # 3. 获取计数器中的频率计数，未达到高频阈值则 miss
        counter_key = self._get_counter_key(norm)
        count = self.redis.get(counter_key)
        if count is None or int(count) < self.threshold:
            logger.info(f"[Cache] Query '{norm}' count {count} below threshold {self.threshold}, cache miss")
            self._incr_metric("cache:metrics:miss")
            return None

        # 4. 读取缓存响应
        cache_key = self._get_cache_key(norm)
        cached = self.redis.get(cache_key)

        if cached:
            # 缓存命中
            logger.info(f"[Cache] HIT! Returning cached response for '{norm}'")
            self._incr_metric("cache:metrics:hit")
        else:
            # 理论上已达到高频但缓存缺失（可能过期），记录击穿
            logger.warning(f"[Cache] Query '{norm}' passed threshold but cache missing (expired?)")
            self._incr_metric("cache:metrics:stampede")
        return cached

    def update(self, query: str, response: str, variant: str = ""):
        """
        更新查询的计数和缓存。
        升级：使用分层 TTL + 随机 jitter，替换原有固定 TTL。
        """
        norm = self._norm(query, variant)
        counter_key = self._get_counter_key(norm)
        cache_key = self._get_cache_key(norm)

        pipe = self.redis.pipeline()
        pipe.incr(counter_key)  # 计数 +1
        pipe.expire(counter_key, self.window)   # 重置计数窗口过期时间
        pipe.execute_command('BF.ADD', self.bloom_key, norm)    # 记录到布隆过滤器
        results = pipe.execute()
        new_count = results[0]  # 获取增加后的计数值
        logger.info(f"[Cache] Query '{norm}' new count: {new_count}")

        if new_count >= self.threshold:
            # ── 分层 TTL ─────────────────────────────────────
            # 根据计数选择基础 TTL：超高频（≥10次）用2小时，否则30分钟
            base_ttl = _TTL_HOT if new_count >= _HOT_THRESHOLD else _TTL_NORMAL
            # 添加随机偏移，确保不低于 base_ttl 的一半
            ttl = _jitter_ttl(base_ttl)
            # 延长已有缓存的过期时间
            self.redis.expire(cache_key, ttl)
            logger.info("查询 '%s' 超高频续期 TTL=%ds", norm, ttl)

            # 分支1：首次达到阈值（计数等于阈值）
            if new_count == self.threshold:
                # 写入新缓存
                self.redis.setex(cache_key, ttl, response)
                logger.info(
                    "查询 '%s' 首次达到阈值 %d，已缓存（TTL=%ds）",
                    norm, self.threshold, ttl,
                )
            # 分支2：超高频（计数超过阈值）且缓存不存在
            elif not self.redis.exists(cache_key):
                # 超高频但缓存已过期：续写缓存
                self.redis.setex(cache_key, ttl, response)
                logger.info(
                    "查询 '%s' 超高频续写缓存（count=%d, TTL=%ds）",
                    norm, new_count, ttl,
                )
            # 分支3：超高频且缓存已存在（正常情况，expire 已续期，无需再写）
            else:
                logger.debug(f"[Cache] Query '{norm}' already cached")

    def clear(self, query: str, variant: str = ""):
        """清除特定查询的计数和缓存（管理用）"""
        norm = self._norm(query, variant)
        pipe = self.redis.pipeline()
        pipe.delete(self._get_counter_key(norm))
        pipe.delete(self._get_cache_key(norm))
        pipe.execute()

    # ══════════════════════════════════════════════════════════
    # 新增接口：互斥锁 + 高频判断
    # ══════════════════════════════════════════════════════════

    def is_high_freq(self, query: str, variant: str = "") -> bool:
        """
        判断 query 是否已达到高频阈值（count >= threshold）。
        供 main.py 决策是否需要加互斥锁。
        """
        norm = self._norm(query, variant)
        # 1、布隆过滤器快速判断：从未见过该查询则直接返回 False
        exists = self.redis.execute_command('BF.EXISTS', self.bloom_key, norm)
        if not exists:
            return False
        # 2、获取计数，若存在且 ≥ 阈值则返回 True
        count = self.redis.get(self._get_counter_key(norm))
        return count is not None and int(count) >= self.threshold

    def try_lock(self, query: str, variant: str = "") -> bool:
        """
        尝试获取互斥锁（SETNX）。
        锁超时 _LOCK_TTL 秒，防止持有锁的请求异常后锁永不释放。

        :return: True = 成功获取锁，调用下游 Agent
                 False = 锁已被其他请求持有，等待并重试
        """
        lock_key = self._get_lock_key(self._norm(query, variant))
        acquired = self.redis.set(lock_key, "1", nx=True, ex=_LOCK_TTL)
        if not acquired:
            self._incr_metric("cache:metrics:lock_wait")
        return bool(acquired)

    def release_lock(self, query: str, variant: str = "") -> None:
        """
        释放互斥锁。
        在 update() 写入缓存之后调用，让等待的请求可以读到新缓存。
        """
        lock_key = self._get_lock_key(self._norm(query, variant))
        self.redis.delete(lock_key)

    # ══════════════════════════════════════════════════════════
    # 内部监控
    # ══════════════════════════════════════════════════════════

    def _incr_metric(self, key: str) -> None:
        """写入监控计数，保留 24 小时，供监控脚本读取"""
        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, 86400)
            pipe.execute()
        except Exception:
            pass