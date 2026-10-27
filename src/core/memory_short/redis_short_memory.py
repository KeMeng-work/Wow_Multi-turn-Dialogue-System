"""
短期记忆模块 (Redis Short Memory)
- ReAct Agent 前置模块，不属于 ReAct 工具集

设计文档：docs/short_term_memory_design.md

存储（2.0.0 起，键前缀 stm2:，与 1.x 的 session:* 键互不影响；旧键 TTL 到期自然清除）：
    stm2:{scope}:turns     Sorted Set   成员 / 分数 = 轮次序号 idx
    stm2:{scope}:t:{idx}   Hash         一轮对话：user / assistant / refused / rewritten_* / topic_id / turn_id
    stm2:{scope}:state     String(JSON) 会话状态（约束 / 话题 / 摘要 / 实体 / 事实 / 提纲），内含 version
    stm2:{scope}:seq       String       轮次序号发号器（INCR，保证并发下 idx 唯一递增）
    scope = stm_scope(user_id, session_id)

原子性：读取、追加 + 淘汰、状态写入均为 Lua 脚本，单次往返完成；状态写入为 CAS（版本不符则拒绝，由调用方重放后重试）。
敏感信息：写入前统一脱敏，只存占位符。

Author: Ke Meng
Created: 2026-03-15
Version: 2.0.0
Last Modified: 2026-10-25

变更记录：
    - 2.0.0 (2026-10-25): 生产级重构：会话状态 + 按预算组装（context_builder）+ Lua 原子写入 + CAS + 单次往返读取；
                          缓存命中也写入；拒答轮只记标记；写入失败计数（STM_METRICS）
    - 1.0.2 (2026-04-10): Hash msg key 补加 TTL；修剪逻辑计入即将写入的 2 条消息
    - 1.0.1 (2026-03-18): list 升级为 Hash + Sorted Set
    - 1.0.0 (2026-03-15): 初始版本
"""

import json
import logging
import threading
from collections import Counter
from typing import List, Optional, Tuple

from src.core.memory_short.session_state import ensure_schema, new_state
from src.core.security.pii import mask_for_storage

logger = logging.getLogger(__name__)

_TURN_FIELDS = ("turn_id", "user", "assistant", "refused", "rewritten_en", "rewritten_zh", "topic_id", "cache_hit", "ts")


class _Metrics:
    """进程内计数（/metrics 输出）：写入失败、CAS 冲突、PG 恢复、压缩结果等"""

    def __init__(self):
        self._c = Counter()
        self._lock = threading.Lock()

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._c[name] += n

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._c)


STM_METRICS = _Metrics()


def stm_scope(user_id: str, session_id: str) -> str:
    """
    短期记忆的会话作用域：{user_id}:{session_id}。
    调用方传给本类及 QueryRewriter 的 session_id 参数均为此值，使不同用户即使使用相同 session_id 也互不可见。
    """
    return f"{user_id}:{session_id}"


# KEYS: turns, state, seq    ARGV: turn_key_prefix, reserve(0/1), ttl
# 返回 {state_json|false, reserved_idx, {turn1 flat hash}, {turn2 ...}, ...}
_LUA_LOAD = """
local ids = redis.call('ZRANGE', KEYS[1], 0, -1)
local state = redis.call('GET', KEYS[2])
local reserved = 0
if ARGV[2] == '1' then
  reserved = redis.call('INCR', KEYS[3])
  redis.call('EXPIRE', KEYS[3], tonumber(ARGV[3]))
end
local out = {state or false, reserved}
for _, id in ipairs(ids) do
  local h = redis.call('HGETALL', ARGV[1] .. id)
  if #h > 0 then
    table.insert(h, 'idx'); table.insert(h, id)
    table.insert(out, h)
  end
end
return out
"""

# KEYS: turns, state, seq    ARGV: turn_key_prefix, idx, ttl, max_raw, field1, value1, ...
# 写入一轮 + 淘汰超出 max_raw 的最早轮次 + 刷新所有相关键 TTL；返回淘汰数
_LUA_APPEND = """
local key = ARGV[1] .. ARGV[2]
local ttl = tonumber(ARGV[3])
local max_raw = tonumber(ARGV[4])
local fields = {}
for i = 5, #ARGV do table.insert(fields, ARGV[i]) end
redis.call('DEL', key)
redis.call('HSET', key, unpack(fields))
redis.call('ZADD', KEYS[1], tonumber(ARGV[2]), ARGV[2])
local seq = tonumber(redis.call('GET', KEYS[3]) or '0')
if seq < tonumber(ARGV[2]) then redis.call('SET', KEYS[3], ARGV[2]) end
local evicted = 0
local n = redis.call('ZCARD', KEYS[1])
if n > max_raw then
  local old = redis.call('ZRANGE', KEYS[1], 0, n - max_raw - 1)
  for _, id in ipairs(old) do redis.call('DEL', ARGV[1] .. id) end
  redis.call('ZREMRANGEBYRANK', KEYS[1], 0, n - max_raw - 1)
  evicted = #old
end
for _, id in ipairs(redis.call('ZRANGE', KEYS[1], 0, -1)) do redis.call('EXPIRE', ARGV[1] .. id, ttl) end
redis.call('EXPIRE', KEYS[1], ttl)
redis.call('EXPIRE', KEYS[3], ttl)
if redis.call('EXISTS', KEYS[2]) == 1 then redis.call('EXPIRE', KEYS[2], ttl) end
return evicted
"""

# KEYS: state    ARGV: expected_version, new_json, ttl
# 当前版本 == expected 时写入（不存在视为版本 0）；返回 1 成功 / 0 版本冲突
_LUA_CAS = """
local cur = redis.call('GET', KEYS[1])
local ver = 0
if cur then
  local ok, obj = pcall(cjson.decode, cur)
  if ok and type(obj) == 'table' and obj['version'] then ver = tonumber(obj['version']) end
end
if ver ~= tonumber(ARGV[1]) then return 0 end
redis.call('SET', KEYS[1], ARGV[2], 'EX', tonumber(ARGV[3]))
return 1
"""


class ShortTermMemory:
    """短期记忆存储（Redis）：近期原文 + 会话状态"""

    def __init__(self, redis_client, expire_seconds: int = 86400, max_raw_turns: int = 12):
        """
        :param redis_client:   已连接的 Redis 客户端（decode_responses=True）
        :param expire_seconds: 会话过期时间（秒），与脱敏映射表一致
        :param max_raw_turns:  Redis 保留的原文轮数上限
        """
        self.redis = redis_client
        self.expire = expire_seconds
        self.max_raw = max_raw_turns
        self._load = redis_client.register_script(_LUA_LOAD)
        self._append = redis_client.register_script(_LUA_APPEND)
        self._cas = redis_client.register_script(_LUA_CAS)

    # ── 键 ────────────────────────────────────────────────────
    @staticmethod
    def _keys(scope: str) -> Tuple[str, str, str, str]:
        base = f"stm2:{scope}"
        return f"{base}:turns", f"{base}:state", f"{base}:seq", f"{base}:t:"

    # ── 读取 ──────────────────────────────────────────────────
    def load(self, scope: str, reserve_turn: bool = False) -> Tuple[Optional[dict], List[dict], int]:
        """
        一次往返读取会话状态与全部近期原文；reserve_turn=True 时同时发号本轮 idx。
        返回 (state 或 None, turns[时间正序], reserved_idx)。Redis 异常时抛出，由调用方降级。
        """
        k_turns, k_state, k_seq, prefix = self._keys(scope)
        raw = self._load(keys=[k_turns, k_state, k_seq], args=[prefix, "1" if reserve_turn else "0", self.expire])
        state = None
        if raw and raw[0]:
            try:
                state = ensure_schema(json.loads(raw[0]))
            except (ValueError, TypeError):
                logger.warning(f"会话状态解析失败，按新会话处理: {scope}")
                STM_METRICS.incr("state_corrupt")
        reserved = int(raw[1] or 0) if raw else 0
        turns = []
        for flat in (raw[2:] if raw else []):
            h = dict(zip(flat[0::2], flat[1::2]))
            try:
                h["idx"] = int(h["idx"])
            except (KeyError, ValueError):
                continue
            h["refused"] = h.get("refused") == "1"
            h["cache_hit"] = h.get("cache_hit") == "1"
            turns.append(h)
        turns.sort(key=lambda t: t["idx"])
        return state, turns, reserved

    # ── 写入 ──────────────────────────────────────────────────
    def append_turn(self, scope: str, idx: int, turn: dict) -> int:
        """原子写入一轮（脱敏后）+ 淘汰超出上限的最早轮次；返回淘汰数"""
        k_turns, k_state, k_seq, prefix = self._keys(scope)
        user, assistant, rw_en, rw_zh = mask_for_storage((
            turn.get("user", ""), turn.get("assistant", ""), turn.get("rewritten_en", ""), turn.get("rewritten_zh", "")))
        fields = {
            "turn_id": turn.get("turn_id", ""), "user": user,
            "assistant": "" if turn.get("refused") else assistant,     # 拒答轮只记标记，不存话术原文
            "refused": "1" if turn.get("refused") else "0",
            "rewritten_en": rw_en, "rewritten_zh": rw_zh,
            "topic_id": turn.get("topic_id") or "", "cache_hit": "1" if turn.get("cache_hit") else "0",
            "ts": str(turn.get("ts", "")),
        }
        args = [prefix, str(int(idx)), self.expire, self.max_raw]
        for k in _TURN_FIELDS:
            args += [k, str(fields.get(k, ""))]
        return int(self._append(keys=[k_turns, k_state, k_seq], args=args))

    def cas_state(self, scope: str, expected_version: int, state: dict) -> bool:
        """版本号 == expected_version 时写入状态（state['version'] 由本函数置为 expected + 1）；返回是否成功"""
        _, k_state, _, _ = self._keys(scope)
        st = mask_for_storage(dict(state))
        st["version"] = int(expected_version) + 1
        ok = self._cas(keys=[k_state], args=[int(expected_version), json.dumps(st, ensure_ascii=False), self.expire])
        if ok:
            state["version"] = st["version"]
        return bool(ok)

    def restore(self, scope: str, state: Optional[dict], turns: List[dict]) -> None:
        """从 PG 恢复：写回近期原文与状态（状态覆盖写，版本沿用 PG 中的版本）"""
        for t in turns:
            self.append_turn(scope, t["idx"], t)
        if state is not None:
            _, k_state, _, _ = self._keys(scope)
            self.redis.set(k_state, json.dumps(mask_for_storage(state), ensure_ascii=False), ex=self.expire)

    # ── 删除 ──────────────────────────────────────────────────
    def clear(self, scope: str) -> int:
        """删除会话的全部短期记忆（stm2 键 + 1.x 遗留键），返回删除的 key 数"""
        k_turns, k_state, k_seq, prefix = self._keys(scope)
        ids = self.redis.zrange(k_turns, 0, -1)
        keys = [k_turns, k_state, k_seq] + [prefix + i for i in ids]
        legacy_set = f"session:{scope}:history"
        keys += [legacy_set] + [f"session:{scope}:msg:{m}" for m in self.redis.zrange(legacy_set, 0, -1)]
        return int(self.redis.delete(*keys))

    @staticmethod
    def empty_state() -> dict:
        return new_state()
