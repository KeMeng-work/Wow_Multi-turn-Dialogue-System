"""
会话软删除 (session_store)

前端删除会话时只在 deleted_sessions 记一条标记，不删除 conversation_history 原始数据：
真实对话是数据飞轮挖掘 badcase / 构建评估集的来源，也便于误删恢复与审计；
conversation_history 是分区大表，单独建标记表避免 ALTER 大表。
会话列表 / 会话消息接口排除已删除会话；删除时同步清空该会话的 Redis 短期记忆。
如需按用户要求彻底删除个人数据，由管理员在数据库侧执行，不在界面开放。

Author: Ke Meng
Created: 2026-10-10
"""

DELETED_SESSIONS_DDL = """
CREATE TABLE IF NOT EXISTS deleted_sessions (
    user_id     VARCHAR(255) NOT NULL,
    session_id  VARCHAR(255) NOT NULL,
    deleted_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, session_id)
)
"""


async def ensure_deleted_sessions_table(pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(DELETED_SESSIONS_DDL)


async def soft_delete_session(pool, user_id: str, session_id: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO deleted_sessions (user_id, session_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
            user_id, session_id)


async def is_session_deleted(pool, user_id: str, session_id: str) -> bool:
    async with pool.acquire() as conn:
        return bool(await conn.fetchval(
            "SELECT 1 FROM deleted_sessions WHERE user_id = $1 AND session_id = $2", user_id, session_id))
