"""
限流模块 (RateLimiter)

三级令牌桶限流，全部通过 Lua 脚本保证 Redis 原子操作：
    级别一：全局限流   — 系统每秒最多 5 个请求
    级别二：用户限流   — 单个 user_id 每分钟最多 20 个请求
    级别三：热点Query  — 同一归一化 query 每秒最多 3 次

触发限流时抛出 RateLimitExceeded，由调用方 api.py 转换为 HTTP 429 + Retry-After 头。
限流触发计数写入 Redis，供监控读取。

Author: Ke Meng
Created: 2026-04-10
Version: 1.0.0
"""

import time
import logging
import redis

logger = logging.getLogger(__name__)

# Retry-After 建议等待秒数（写入响应头）
RETRY_AFTER_SECONDS = 30

# ── Lua 脚本：令牌桶原子操作 ─────────────────────────────────
# 返回 1 = 允许通过，0 = 拒绝
# KEYS[1]  = bucket key
# ARGV[1]  = capacity（桶容量）
# ARGV[2]  = refill_rate（每秒补充令牌数）
# ARGV[3]  = now（当前时间戳，浮点）
# ARGV[4]  = ttl（key 过期时间，整数秒）
_TOKEN_BUCKET_LUA = """
local key        = KEYS[1]
local capacity   = tonumber(ARGV[1])
local rate       = tonumber(ARGV[2])
local now        = tonumber(ARGV[3])
local ttl        = tonumber(ARGV[4])

local data       = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens     = tonumber(data[1])
local last       = tonumber(data[2])

if tokens == nil then
    tokens = capacity
    last   = now
end

local elapsed    = math.max(0, now - last)
local new_tokens = math.min(capacity, tokens + elapsed * rate)

if new_tokens >= 1 then
    redis.call('HMSET', key, 'tokens', new_tokens - 1, 'last_refill', now)
    redis.call('EXPIRE', key, ttl)
    return 1
else
    redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
    redis.call('EXPIRE', key, ttl)
    return 0
end
"""


class RateLimitExceeded(Exception):
    """
    限流触发时抛出。
    携带限流原因和 Retry-After 秒数，由 api.py 转换为 HTTP 429。
    熔断器的 on_failure 会忽略此异常，不计入失败统计。
    """
    def __init__(self, reason: str, retry_after: int = RETRY_AFTER_SECONDS):
        self.reason = reason
        self.retry_after = retry_after
        super().__init__(reason)


class RateLimiter:
    """
    三级令牌桶限流器

    使用方式：
        limiter = RateLimiter(redis_client=redis_client)
        limiter.check(user_id="u123", norm_query="归一化后的查询")
        # 触发限流时抛出 RateLimitExceeded，正常则无返回值
    """

    def __init__(
        self,
        redis_client:  redis.Redis,
        global_rate:   int = 5,          # 全局：每秒请求上限
        user_rate:     int = 20,          # 用户：每分钟请求上限
        query_rate:    int = 3,           # 热点 query：每秒请求上限
        global_key:    str = "rl:global",
        user_prefix:   str = "rl:user:",
        query_prefix:  str = "rl:query:",
    ):
        self.redis = redis_client
        self.global_rate = global_rate
        self.user_rate = user_rate
        self.query_rate = query_rate
        self.global_key = global_key
        self.user_prefix = user_prefix
        self.query_prefix = query_prefix

        # 注册 Lua 脚本（只编译一次，复用 SHA）
        self._lua = self.redis.register_script(_TOKEN_BUCKET_LUA)

    # ── 公共接口 ───────────────────────────────────────────────

    def check(self, user_id: str, norm_query: str) -> None:
        """
        执行三级限流检查。任意一级触发则抛出 RateLimitExceeded。

        :param user_id:    用户 ID
        :param norm_query: 归一化后的 query（由 HighFreqCache.normalize 处理）
        :raises RateLimitExceeded: 任意一级限流触发时
        """
        # 如果 user_id 为空，统一作为匿名用户走全局限流
        if not user_id:
            user_id = "anonymous"

        # ── 第一级：全局限流（5次/秒）────────────────────────────
        if not self._acquire(self.global_key, self.global_rate, self.global_rate, ttl=3600):
            logger.warning("全局限流触发")
            self._incr_metric("rl:metrics:global_hits")
            raise RateLimitExceeded("系统请求过于繁忙，请稍后再试")

        # ── 第二级：用户限流（20次/分钟）──────────────────────────
        user_key = f"{self.user_prefix}{user_id}"
        if not self._acquire(user_key, self.user_rate, self.user_rate / 60, ttl=120):
            logger.warning("用户限流触发: user_id=%s", user_id)
            self._incr_metric("rl:metrics:user_hits")
            raise RateLimitExceeded("您的请求过于频繁，请稍后再试")

        # ── 第三级：热点 Query 限流（3次/秒）──────────────────────
        query_key = f"{self.query_prefix}{norm_query}"
        if not self._acquire(query_key, self.query_rate, self.query_rate, ttl=10):
            logger.warning("热点 Query 限流触发: query=%s", norm_query[:50])
            self._incr_metric("rl:metrics:query_hits")
            raise RateLimitExceeded("该查询请求过于频繁，请稍后再试")

    # ── 内部方法 ───────────────────────────────────────────────

    def _acquire(self, key: str, capacity: int, refill_rate: float, ttl: int) -> bool:
        """执行令牌桶 Lua 脚本，返回 True=允许，False=拒绝"""
        result = self._lua(
            keys=[key],
            args=[capacity, refill_rate, time.time(), ttl],
        )
        return result == 1

    def _incr_metric(self, key: str) -> None:
        """限流触发计数，保留 24 小时，供监控脚本读取"""
        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, 86400)
            pipe.execute()
        except Exception:
            pass  # 监控写入失败不影响主流程
