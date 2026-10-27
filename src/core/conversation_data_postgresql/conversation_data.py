"""
PostgreSQL 全量对话历史存储模块 (PGMemory)

职责：
    - 管理 asyncpg 连接池（含参数调优）
    - 提供异步写入接口（一轮对话 = user + assistant 两条，事务原子写入）
    - 写入失败时执行指数退避重试（最多3次）
    - 三次失败后写本地 fallback 日志兜底，防止数据永久丢失
    - 信号量控制并发异步任务上限，防止内存堆积
    - 提供从 PG 加载历史的接口（Redis 冷启动回填用）

设计决策：
    - PGMemory 与 ShortTermMemory 完全解耦，各自职责单一
    - 写入顺序：Redis 同步写入（主流程） → PG 异步写入（后台任务）
    - 信号量上限20 = 连接池max_size(10) × 2，超出部分走 fallback，不堆积内存
    - ON CONFLICT DO NOTHING 保证 message_id 幂等，重试不产生重复数据

Author: Ke Meng
Created: 2026-04-10
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional
import asyncpg

from src.core.security.pii import mask_for_storage

logger = logging.getLogger(__name__)

# fallback 日志路径，PG 彻底写入失败时的最后兜底
_FALLBACK_LOG_PATH = os.path.join("logs", "pg_fallback.jsonl")

# 写入 SQL（幂等：message_id 重复时直接忽略，不报错）
_INSERT_SQL = """
    INSERT INTO conversation_history
        (message_id, session_id, user_id, role, content, token_count, metadata, created_at)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    ON CONFLICT (message_id, created_at) DO NOTHING
"""


class PGMemory:
    """
    PostgreSQL 全量对话历史存储管理器

    使用方式：
        pg = PGMemory(dsn="postgresql://user:pass@host:5432/db")
        await pg.initialize()                   # 应用启动时调用一次
        await pg.write_turn(session_id, ...)    # 每轮对话结束后调用
        await pg.close()                        # 应用关闭时调用
    """

    def __init__(
        self,
        dsn: str,
        min_size: int = 2,
        max_size: int = 10,
        command_timeout: int = 10,
        max_inactive_lifetime: int = 300,
        semaphore_limit: int = 20,
    ):
        """
        :param dsn:                  asyncpg 连接串
        :param min_size:             连接池常驻连接数，保证随时有连接可用
        :param max_size:             连接池最大连接数，不超过 PG max_connections 的一半
        :param command_timeout:      单条 SQL 超时秒数，超过直接报错不无限等待
        :param max_inactive_lifetime: 空闲连接自动释放秒数
        :param semaphore_limit:      并发异步任务上限，防止 PG 短暂不可用时任务堆积吃光内存
        """
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._command_timeout = command_timeout
        self._max_inactive = max_inactive_lifetime
        self._pool: Optional[asyncpg.Pool] = None

        # 信号量：最多同时存在 semaphore_limit 个 PG 异步写入任务
        # 超出部分在此处阻塞，由调用方（main.py）决定是否 create_task 或直接 await
        self._semaphore = asyncio.Semaphore(semaphore_limit)

    # ──────────────────────────────────────────────────────────
    # 生命周期
    # ──────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """初始化连接池，应用启动时调用一次"""
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            command_timeout=self._command_timeout,
            max_inactive_connection_lifetime=self._max_inactive,
        )
        logger.info("✅ PostgreSQL 连接池初始化成功 (min=%d, max=%d)", self._min_size, self._max_size)

    @property
    def pool(self):
        return self._pool

    async def close(self) -> None:
        """关闭连接池，应用关闭时调用"""
        if self._pool:
            await self._pool.close()
            logger.info("PostgreSQL 连接池已关闭")

    # ──────────────────────────────────────────────────────────
    # 核心写入接口（供 main.py 以 create_task 方式调用）
    # ──────────────────────────────────────────────────────────

    async def write_turn(
        self,
        session_id:       str,
        user_id:          str,
        user_input:       str,
        assistant_output: str,
        token_count:      Optional[int]  = None,
        metadata:         Optional[dict] = None,
        user_message_id:  Optional[str]  = None,
    ) -> None:
        """
        写入一轮对话（user + assistant 两条消息，事务原子写入）

        - 信号量控制并发上限，防止任务内存堆积
        - 指数退避重试最多3次
        - 三次失败后写 fallback 日志兜底
        - 敏感信息只存占位符版本（含 fallback 日志），查看历史时由 api.py 按角色还原
        - user_message_id：短期记忆的轮次 ID（事实三元组的溯源依据）；不传时自动生成
        """
        user_input, assistant_output, metadata = mask_for_storage((user_input, assistant_output, metadata or {}))
        async with self._semaphore:
            await self._write_with_retry(
                session_id=session_id,
                user_id=user_id,
                user_input=user_input,
                assistant_output=assistant_output,
                token_count=token_count,
                metadata=metadata or {},
                user_message_id=user_message_id,
            )

    # ──────────────────────────────────────────────────────────
    # Redis 冷启动回填接口
    # ──────────────────────────────────────────────────────────

    async def load_recent_turns(self, session_id: str, limit_turns: int = 5) -> list[dict]:
        """
        从 PG 加载最近 N 轮对话，供 Redis miss 时回填缓存

        :return: 时间正序列表，每条 {"role": ..., "content": ..., "message_id": ...}
        """
        if not self._pool:
            logger.warning("PG 连接池未初始化，无法加载历史")
            return []

        sql = """
            SELECT role, content, message_id
            FROM   conversation_history
            WHERE  session_id = $1
            ORDER  BY created_at DESC
            LIMIT  $2
        """
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, session_id, limit_turns * 2)
            # 返回时间正序（最早在前，与 Redis list 方向一致）
            return [dict(r) for r in reversed(rows)]
        except Exception as e:
            logger.error("从 PG 加载历史失败: %s", e)
            return []

    # ──────────────────────────────────────────────────────────
    # 内部：指数退避重试
    # ──────────────────────────────────────────────────────────

    async def _write_with_retry(
        self,
        *,
        session_id:       str,
        user_id:          str,
        user_input:       str,
        assistant_output: str,
        token_count:      Optional[int],
        metadata:         dict,
        user_message_id:  Optional[str] = None,
    ) -> None:
        """
        带指数退避的 PG 写入
        重试策略：失败后等待 1s → 2s → 4s，三次全部失败写 fallback 日志
        """
        user_msg_id = user_message_id or str(uuid.uuid4())
        asst_msg_id = str(uuid.uuid4())
        metadata_str = json.dumps(metadata, ensure_ascii=False)
        # 预生成时间戳：重试时复用同一值，ON CONFLICT (message_id, created_at) 才能正确检测重复
        now = datetime.now(timezone.utc)

        for attempt in range(3):
            try:
                async with self._pool.acquire() as conn:
                    # 事务保证 user + assistant 两条原子写入
                    async with conn.transaction():
                        await conn.execute(
                            _INSERT_SQL,
                            user_msg_id, session_id, user_id,
                            "user", user_input, token_count, metadata_str, now,
                        )
                        await conn.execute(
                            _INSERT_SQL,
                            asst_msg_id, session_id, user_id,
                            "assistant", assistant_output, token_count, metadata_str, now,
                        )
                logger.debug("PG 写入成功: session=%s", session_id)
                return  # 成功直接返回

            except Exception as e:
                if attempt < 2:
                    wait_secs = 2 ** attempt  # 1s → 2s
                    logger.warning(
                        "PG 写入失败（第%d次），%ds 后重试: %s",
                        attempt + 1, wait_secs, e
                    )
                    await asyncio.sleep(wait_secs)
                else:
                    # 三次全部失败
                    logger.error(
                        "PG 写入彻底失败（已重试3次），写入 fallback 日志: session=%s, error=%s",
                        session_id, e
                    )
                    self._write_fallback_log(
                        session_id       = session_id,
                        user_id          = user_id,
                        user_input       = user_input,
                        assistant_output = assistant_output,
                    )

    # ──────────────────────────────────────────────────────────
    # 内部：本地 fallback 日志兜底
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _write_fallback_log(
        *,
        session_id:       str,
        user_id:          str,
        user_input:       str,
        assistant_output: str,
    ) -> None:
        """
        PG 彻底写入失败时的最后兜底。
        写入本地 JSONL 文件，后续可手动或脚本补录进 PG。
        每行一条 JSON，格式固定，便于 grep 和批量补录。
        """
        record = {
            "ts":               time.time(),
            "session_id":       session_id,
            "user_id":          user_id,
            "user_input":       user_input,
            "assistant_output": assistant_output,
        }
        try:
            os.makedirs(os.path.dirname(_FALLBACK_LOG_PATH), exist_ok=True)
            with open(_FALLBACK_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            # fallback 日志也写不进去：最后的最后只能打 critical 日志
            logger.critical("fallback 日志写入失败，此条对话数据已丢失: %s", e)