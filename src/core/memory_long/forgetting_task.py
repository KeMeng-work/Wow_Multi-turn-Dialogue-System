"""
遗忘机制 - 定时任务 (Forgetting Task)

独立于主流程的后台定时任务，负责自动淘汰低价值记忆，防止记忆库无限膨胀。

两种遗忘策略（互补，分别触发）：
1. 分数衰减淘汰（score_decay）：
   定时扫描全库，综合分低于阈值的活跃记忆标记为 is_active=FALSE。
   触发时机：每天定时执行一次（推荐凌晨低峰期）。

2. 容量上限淘汰（capacity_cap）：
   每次 process_and_store 完成后检查单用户记忆数量，
   超过上限时软删除综合分最低的记忆。
   触发时机：每次记忆存储后由 long_term_memory.py 调用。

遗忘公式与 long_term_memory.py 中的动态评分一致：
  decay_score = importance_score × 0.6
              + EXP(-0.01 × days_since_update) × 0.3
              + LEAST(access_count/10.0, 1.0) × 0.1

Author: Ke Meng
Created: 2026-04-12
Version: 1.0.0
"""

import logging
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)

# 遗忘阈值：decay_score 低于此值的记忆将被软删除
_DEFAULT_SCORE_THRESHOLD = 0.1

# 每用户记忆上限（超过则淘汰最低分记忆）
_DEFAULT_CAPACITY_PER_USER = 200

# 分数衰减淘汰时，episodic 类型记忆额外加速衰减（事件类记忆时效性更短）
# 通过缩短半衰期实现：λ=0.03（约 23 天半衰，约为 fact 类的 3 倍）
_EPISODIC_DECAY_LAMBDA = 0.03
_DEFAULT_DECAY_LAMBDA = 0.01


class ForgettingTask:
    """
    遗忘机制定时任务。

    使用外部传入的 asyncpg.Pool，不自行管理连接生命周期。
    可单独实例化后独立测试，与 LongTermMemory 完全解耦。

    Usage（定时任务）：
        task = ForgettingTask(pool)
        # 每天执行：
        await task.run_score_decay_forgetting()
        # 每次存储后执行：
        await task.run_capacity_forgetting(user_id="xxx")
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
        capacity_per_user: int = _DEFAULT_CAPACITY_PER_USER,
    ):
        self.pool = db_pool
        self.score_threshold = score_threshold
        self.capacity_per_user = capacity_per_user

    # ═══════════════════════════ 公开接口 ═══════════════════════════

    async def run_score_decay_forgetting(self) -> int:
        """
        策略1：分数衰减淘汰（每天定时调用）。

        扫描全库活跃记忆，计算 decay_score，低于阈值的标记为 is_active=FALSE。
        episodic 类型记忆使用更高的衰减系数（λ=0.03），加速淘汰过时事件。

        Returns:
            本次淘汰的记忆条数
        """
        async with self.pool.acquire() as conn:
            await register_vector(conn)
            result = await conn.execute(
                """
                UPDATE memories SET
                    is_active  = FALSE,
                    updated_at = NOW()
                WHERE is_active = TRUE
                  AND (
                      -- fact / preference 类型：λ=0.01，约 70 天半衰
                      (memory_type != 'episodic' AND
                       importance_score * 0.6
                       + EXP(-$1::float * EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400.0) * 0.3
                       + LEAST(access_count::float / 10.0, 1.0) * 0.1 < $3::float)
                      OR
                      -- episodic 类型：λ=0.03，约 23 天半衰
                      (memory_type = 'episodic' AND
                       importance_score * 0.6
                       + EXP(-$2::float * EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400.0) * 0.3
                       + LEAST(access_count::float / 10.0, 1.0) * 0.1 < $3::float)
                  )
                """,
                _DEFAULT_DECAY_LAMBDA,
                _EPISODIC_DECAY_LAMBDA,
                self.score_threshold,
            )

        # asyncpg execute 返回 "UPDATE N" 字符串
        count = int(result.split()[-1]) if result else 0
        logger.info(f"[遗忘-分数衰减] 本次淘汰 {count} 条记忆（阈值={self.score_threshold}）")
        return count

    async def run_capacity_forgetting(
        self,
        user_id: str,
        capacity: Optional[int] = None,
    ) -> int:
        """
        策略2：容量上限淘汰（每次记忆存储后调用）。

        检查指定用户的活跃记忆数量，超过 capacity 时软删除 decay_score 最低的记忆。

        Args:
            user_id:  目标用户 ID
            capacity: 单用户记忆上限，不传则使用实例默认值

        Returns:
            本次淘汰的记忆条数
        """
        cap = capacity or self.capacity_per_user

        async with self.pool.acquire() as conn:
            await register_vector(conn)

            # 先检查数量，不超上限直接返回（避免不必要的 UPDATE）
            count_row = await conn.fetchrow(
                "SELECT COUNT(*) AS cnt FROM memories WHERE user_id = $1 AND is_active = TRUE",
                user_id,
            )
            current_count = count_row["cnt"]

            if current_count <= cap:
                logger.debug(f"[遗忘-容量] user={user_id} 记忆数={current_count}，未超上限 {cap}")
                return 0

            excess = current_count - cap
            logger.info(f"[遗忘-容量] user={user_id} 记忆数={current_count}，超出 {excess} 条，开始淘汰")

            # 找出 decay_score 最低的 excess 条，软删除
            result = await conn.execute(
                """
                UPDATE memories SET
                    is_active  = FALSE,
                    updated_at = NOW()
                WHERE id IN (
                    SELECT id FROM memories
                    WHERE user_id = $1 AND is_active = TRUE
                    ORDER BY (
                        importance_score * 0.6
                        + EXP(-$2::float * EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400.0) * 0.3
                        + LEAST(access_count::float / 10.0, 1.0) * 0.1
                    ) ASC
                    LIMIT $3
                )
                """,
                user_id,
                _DEFAULT_DECAY_LAMBDA,
                excess,
            )

        淘汰数 = int(result.split()[-1]) if result else 0
        logger.info(f"[遗忘-容量] user={user_id} 淘汰 {淘汰数} 条")
        return 淘汰数

    async def get_forgetting_stats(self) -> dict:
        """
        获取遗忘统计信息（用于监控和调试）。

        Returns:
            {
              total_active:   全库活跃记忆数
              total_inactive: 全库已遗忘记忆数
              by_type:        各类型活跃记忆数分布
              near_threshold: decay_score 接近阈值（< threshold × 2）的记忆数
            }
        """
        async with self.pool.acquire() as conn:
            await register_vector(conn)

            total = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) FILTER (WHERE is_active = TRUE)  AS active,
                    COUNT(*) FILTER (WHERE is_active = FALSE) AS inactive
                FROM memories
                """
            )

            by_type = await conn.fetch(
                """
                SELECT memory_type, COUNT(*) AS cnt
                FROM memories WHERE is_active = TRUE
                GROUP BY memory_type
                """
            )

            near = await conn.fetchrow(
                """
                SELECT COUNT(*) AS cnt FROM memories
                WHERE is_active = TRUE
                  AND (
                      importance_score * 0.6
                      + EXP(-$1::float * EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400.0) * 0.3
                      + LEAST(access_count::float / 10.0, 1.0) * 0.1
                  ) < $2::float
                """,
                _DEFAULT_DECAY_LAMBDA,
                self.score_threshold * 2,
            )

        return {
            "total_active": total["active"],
            "total_inactive": total["inactive"],
            "by_type": {r["memory_type"]: r["cnt"] for r in by_type},
            "near_threshold": near["cnt"],
        }