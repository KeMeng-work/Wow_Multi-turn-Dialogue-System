"""
会话状态的 PostgreSQL 持久化 (session_state) 与从全量对话恢复近期原文

设计文档：docs/short_term_memory_design.md §5.3、§10
    - Redis 是热数据，可能因 TTL 过期或 allkeys-lru 被淘汰；PG 是唯一真实数据源
    - session_state：会话状态快照（约束 / 话题 / 摘要 / 实体 / 事实 / 提纲），按版本号单调更新
    - 近期原文从 conversation_history 读取（metadata.stm 记录轮次序号与话题），不改动该表结构
"""

import json
import logging
from typing import List, Optional, Tuple

from src.core.memory_short.session_state import ensure_schema
from src.core.security.pii import mask_for_storage

logger = logging.getLogger(__name__)

SESSION_STATE_DDL = """
CREATE TABLE IF NOT EXISTS session_state (
    user_id     VARCHAR(255) NOT NULL,
    session_id  VARCHAR(255) NOT NULL,
    state       JSONB        NOT NULL,
    version     BIGINT       NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, session_id)
)
"""

# 版本只增不减：并发 / 乱序到达的旧快照不会覆盖新快照
_UPSERT_SQL = """
INSERT INTO session_state (user_id, session_id, state, version, updated_at)
VALUES ($1, $2, $3::jsonb, $4, NOW())
ON CONFLICT (user_id, session_id) DO UPDATE
   SET state = EXCLUDED.state, version = EXCLUDED.version, updated_at = NOW()
 WHERE session_state.version < EXCLUDED.version
"""


async def ensure_session_state_table(pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(SESSION_STATE_DDL)


async def save_state(pool, user_id: str, session_id: str, state: dict) -> None:
    """快照写入（写入前脱敏，只存占位符；已脱敏的状态再脱敏不变）；失败只记日志，Redis 中的状态不受影响"""
    if pool is None:
        return
    try:
        masked = mask_for_storage(state)
        async with pool.acquire() as conn:
            await conn.execute(_UPSERT_SQL, user_id, session_id, json.dumps(masked, ensure_ascii=False),
                               int(state.get("version", 0)))
    except Exception as e:
        logger.warning(f"会话状态写入 PG 失败（不影响本次请求）: {e}")


async def load_state(pool, user_id: str, session_id: str) -> Optional[dict]:
    if pool is None:
        return None
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT state FROM session_state WHERE user_id = $1 AND session_id = $2",
                                  user_id, session_id)
    if not row:
        return None
    st = row["state"]
    return ensure_schema(json.loads(st) if isinstance(st, str) else st)


async def delete_state(pool, user_id: str, session_id: str) -> None:
    if pool is None:
        return
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM session_state WHERE user_id = $1 AND session_id = $2", user_id, session_id)


def _pair_turns(rows: List[dict]) -> List[dict]:
    """
    把时间正序的消息配成轮次：一轮的 user 与 assistant 同一事务写入、created_at 相同，
    按 (created_at, id) 排序后 user 在前。metadata.stm 有轮次序号时沿用，否则按顺序编号。
    """
    turns, pending = [], None
    for r in rows:
        md = r["metadata"]
        md = json.loads(md) if isinstance(md, str) else (md or {})
        stm = md.get("stm") or {}
        if r["role"] == "user":
            pending = {"user": r["content"], "turn_id": r["message_id"], "stm": stm, "created_at": r["created_at"]}
        elif r["role"] == "assistant" and pending is not None and r["created_at"] == pending["created_at"]:
            s = pending["stm"] or stm
            turns.append({
                "idx": s.get("turn_idx"), "turn_id": s.get("turn_id") or pending["turn_id"],
                "user": pending["user"], "assistant": r["content"],
                "refused": bool(s.get("refused")), "topic_id": s.get("topic_id") or "",
                "rewritten_en": s.get("rewritten_en") or "", "rewritten_zh": s.get("rewritten_zh") or "",
                "cache_hit": bool(md.get("cache_hit")), "ts": pending["created_at"].timestamp(),
            })
            pending = None
    # 旧数据无轮次序号：按顺序补号（接在已知最大序号之后不重复）
    known = [t["idx"] for t in turns if isinstance(t["idx"], int)]
    nxt = (max(known) if known else 0)
    for i, t in enumerate(turns):
        if not isinstance(t["idx"], int):
            nxt += 1
            t["idx"] = nxt
    return turns


async def load_recent_turns(pool, user_id: str, session_id: str, limit_turns: int,
                            topic_id: Optional[str] = None) -> List[dict]:
    """
    从全量对话表读取最近 N 轮（时间正序）；按 user_id 过滤（会话 ID 可能跨用户重复）；
    topic_id 给出时只取该话题的轮次（用于按话题从原文重建摘要）。
    """
    if pool is None:
        return []
    params = [user_id, session_id, limit_turns * 2]
    topic_sql = ""
    if topic_id:
        topic_sql = "AND metadata->'stm'->>'topic_id' = $4"
        params.append(topic_id)
    sql = f"""
        SELECT role, content, metadata, message_id::text AS message_id, created_at, id
        FROM   conversation_history
        WHERE  user_id = $1 AND session_id = $2 AND role IN ('user', 'assistant') {topic_sql}
        ORDER  BY created_at DESC, id DESC
        LIMIT  $3
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    rows = sorted((dict(r) for r in rows), key=lambda r: (r["created_at"], r["id"]))
    return _pair_turns(rows)


async def load_for_restore(pool, user_id: str, session_id: str, limit_turns: int) -> Tuple[Optional[dict], List[dict]]:
    """Redis 缺失时的同步恢复：会话状态快照 + 最近 N 轮原文；会话已被用户删除时不恢复"""
    if pool is None:
        return None, []
    from src.core.session_store import is_session_deleted
    if await is_session_deleted(pool, user_id, session_id):
        return None, []
    state = await load_state(pool, user_id, session_id)
    turns = await load_recent_turns(pool, user_id, session_id, limit_turns)
    return state, turns
