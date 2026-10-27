"""
短期记忆 Redis Lua 脚本测试（服务器上对真实 Redis 运行；只读写 stm2:stmtest-* 测试键，结束后删除）

覆盖：LOAD 单次往返 + INCR 发号、APPEND 原子写入 + 淘汰 + TTL 刷新、CAS 版本校验、PG 恢复写回后发号接续、
      拒答轮不存话术、并发发号唯一、clear 删除全部键
运行（项目根目录，与 app 使用同一 .env）：
    python scripts/test_stm_redis.py
    # 容器内：docker compose exec app python scripts/test_stm_redis.py
"""

import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import redis  # noqa: E402

from config.config_path import REDIS_CONFIG  # noqa: E402
from src.core.memory_short.redis_short_memory import ShortTermMemory  # noqa: E402
from src.core.memory_short.session_state import new_state  # noqa: E402

TTL = 600


def _turn(i, **kw):
    return {"turn_id": f"tid{i}", "user": f"问题{i}", "assistant": f"回答{i}", "refused": False,
            "rewritten_en": f"question {i}", "rewritten_zh": f"问题{i}完整", "topic_id": "T1", "cache_hit": False,
            "ts": 1.0, **kw}


def main() -> int:
    r = redis.Redis(**REDIS_CONFIG)
    r.ping()
    mem = ShortTermMemory(r, expire_seconds=TTL, max_raw_turns=3)
    scope = f"stmtest-{uuid.uuid4().hex[:8]}:s"
    base = f"stm2:{scope}"
    results = []

    def check(name, cond, detail=""):
        results.append(bool(cond))
        print(f"{'PASS' if cond else 'FAIL'} {name}" + (f"  {detail}" if detail and not cond else ""))

    try:
        state, turns, idx = mem.load(scope)
        check("空会话读取", state is None and turns == [] and idx == 0)
        _, _, idx = mem.load(scope, reserve_turn=True)
        check("发号从 1 开始", idx == 1, idx)
        check("发号键带 TTL", 0 < r.ttl(f"{base}:seq") <= TTL)

        for i in range(1, 6):
            if i > 1:
                _, _, idx = mem.load(scope, reserve_turn=True)
            mem.append_turn(scope, idx, _turn(i, refused=(i == 4)))
        state, turns, _ = mem.load(scope)
        check("超出 max_raw 淘汰最早轮次", [t["idx"] for t in turns] == [3, 4, 5], [t["idx"] for t in turns])
        check("被淘汰轮次的 Hash 已删除", not r.exists(f"{base}:t:1") and not r.exists(f"{base}:t:2"))
        check("轮次字段完整往返", turns[0]["user"] == "问题3" and turns[0]["rewritten_zh"] == "问题3完整"
              and turns[0]["turn_id"] == "tid3" and turns[0]["refused"] is False)
        check("拒答轮不存话术", turns[1]["refused"] is True and turns[1]["assistant"] == "")
        check("轮次键 TTL 已刷新", all(0 < r.ttl(f"{base}:t:{i}") <= TTL for i in (3, 4, 5))
              and 0 < r.ttl(f"{base}:turns") <= TTL)

        st = new_state()
        st["current_topic"] = "T1"
        check("CAS 首次写入（版本 0）", mem.cas_state(scope, 0, st) and st["version"] == 1)
        check("CAS 旧版本被拒", not mem.cas_state(scope, 0, st))
        check("CAS 正确版本成功", mem.cas_state(scope, 1, st) and st["version"] == 2)
        state, _, _ = mem.load(scope)
        check("状态读回", state["version"] == 2 and state["current_topic"] == "T1")
        check("状态键带 TTL", 0 < r.ttl(f"{base}:state") <= TTL)

        # 并发发号：20 个线程同时 reserve，序号唯一且连续
        with ThreadPoolExecutor(20) as ex:
            ids = list(ex.map(lambda _: mem.load(scope, reserve_turn=True)[2], range(20)))
        check("并发发号唯一", len(set(ids)) == 20 and max(ids) - min(ids) == 19, sorted(ids)[:5])

        # PG 恢复：写回后发号接在恢复的最大序号之后
        scope2 = scope + "2"
        rst = new_state()
        rst["version"] = 7
        mem.restore(scope2, rst, [{**_turn(8), "idx": 8}, {**_turn(9), "idx": 9}])
        state, turns, idx = mem.load(scope2, reserve_turn=True)
        check("恢复后发号接续", idx == 10 and [t["idx"] for t in turns] == [8, 9] and state["version"] == 7, idx)
        check("恢复后 CAS 沿用版本", mem.cas_state(scope2, 7, state) and not mem.cas_state(scope2, 7, state))

        # 损坏的状态：按新会话处理，不抛异常
        r.set(f"{base}:state", "{not json", ex=TTL)
        state, turns, _ = mem.load(scope)
        check("损坏状态按 None 处理", state is None and len(turns) == 3)

        n = mem.clear(scope) + mem.clear(scope2)
        check("clear 删除全部键", n > 0 and not r.keys(f"stm2:{scope}*"), r.keys(f"stm2:{scope}*"))
    finally:
        leftover = r.keys("stm2:stmtest-*")
        if leftover:
            r.delete(*leftover)

    print(f"\n{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
