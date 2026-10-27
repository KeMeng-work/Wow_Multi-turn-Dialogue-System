"""
安全事件审计 (security_events)

记录权限拒绝、敏感信息还原、注入拦截等安全相关事件，供事后审计与统计。
只记录事件类型、用户、会话、角色与原因，**不存敏感信息原值**（detail 由调用方保证已脱敏）。

写入为后台任务，不阻塞请求；同步代码（工具线程）与异步代码都可调用 record()。
写入失败只记日志，不影响业务。

Author: Ke Meng
Created: 2026-10-17
"""

import asyncio
import json
import logging
from typing import Optional

from src.core.security.context import get_context

logger = logging.getLogger(__name__)

SECURITY_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS security_events (
    id           BIGSERIAL    PRIMARY KEY,
    event_type   VARCHAR(32)  NOT NULL,
    user_id      VARCHAR(255),
    session_id   VARCHAR(255),
    role         VARCHAR(16),
    detail       JSONB,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_security_events_type_time ON security_events (event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_security_events_user_time ON security_events (user_id, created_at DESC);
"""

_pool = None
_loop: Optional[asyncio.AbstractEventLoop] = None
_pending: set = set()


async def ensure_security_events_table(pool) -> None:
    """建表并绑定连接池与事件循环（lifespan 启动时调用）"""
    global _pool, _loop
    async with pool.acquire() as conn:
        await conn.execute(SECURITY_EVENTS_DDL)
    _pool, _loop = pool, asyncio.get_running_loop()


async def _insert(row: tuple) -> None:
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO security_events (event_type, user_id, session_id, role, detail) "
                "VALUES ($1, $2, $3, $4, $5::jsonb)", *row)
    except Exception as e:
        logger.warning(f"安全事件写入失败 {row[0]}: {e}")


def record(event_type: str, detail: Optional[dict] = None, **override) -> None:
    """
    记录一条安全事件（非阻塞）。用户、会话、角色默认取当前请求上下文，可用关键字参数覆盖。
    :param event_type: permission_denied / pii_restore / pii_mask_failed / injection_blocked / canary_leak 等
    :param detail:     事件详情（不得包含敏感信息原值）
    """
    ctx = {**get_context(), **override}
    row = (event_type, ctx.get("user_id"), ctx.get("session_id"), ctx.get("role"),
           json.dumps(detail or {}, ensure_ascii=False))
    logger.warning(f"[security] {event_type} user={row[1]} session={row[2]} role={row[3]} {row[4][:300]}")
    if _pool is None or _loop is None or _loop.is_closed():
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is _loop:
        task = _loop.create_task(_insert(row))
        _pending.add(task)
        task.add_done_callback(_pending.discard)
    else:
        asyncio.run_coroutine_threadsafe(_insert(row), _loop)
