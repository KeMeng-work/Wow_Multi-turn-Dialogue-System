"""
用户反馈模块 (feedback)

前端每条回答下的 👍 / 👎 写入 message_feedback 表，作为数据飞轮的真实用户信号：
点踩的问答对可直接导出为 badcase 候选，点赞的可作为高质量样本。
同一用户对同一会话中的同一回答只保留最新一次评价（按回答内容哈希去重）。

Author: Ke Meng
Created: 2026-10-09
"""

import hashlib

FEEDBACK_DDL = """
CREATE TABLE IF NOT EXISTS message_feedback (
    id           BIGSERIAL    PRIMARY KEY,
    user_id      VARCHAR(255) NOT NULL,
    session_id   VARCHAR(255) NOT NULL,
    answer_hash  CHAR(16)     NOT NULL,
    question     TEXT         NOT NULL,
    answer       TEXT         NOT NULL,
    rating       SMALLINT     NOT NULL CHECK (rating IN (-1, 1)),
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, session_id, answer_hash)
)
"""


async def ensure_feedback_table(pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(FEEDBACK_DDL)


async def save_feedback(pool, user_id: str, session_id: str, question: str, answer: str, rating: int) -> None:
    answer_hash = hashlib.sha1(answer.encode("utf-8")).hexdigest()[:16]
    sql = """
        INSERT INTO message_feedback (user_id, session_id, answer_hash, question, answer, rating)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (user_id, session_id, answer_hash)
        DO UPDATE SET rating = EXCLUDED.rating, updated_at = NOW()
    """
    async with pool.acquire() as conn:
        await conn.execute(sql, user_id, session_id, answer_hash, question, answer, rating)


async def delete_feedback(pool, user_id: str, session_id: str, answer: str) -> None:
    """取消评价"""
    answer_hash = hashlib.sha1(answer.encode("utf-8")).hexdigest()[:16]
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM message_feedback WHERE user_id = $1 AND session_id = $2 AND answer_hash = $3",
            user_id, session_id, answer_hash)
