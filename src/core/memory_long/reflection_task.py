"""
反思机制定时任务 (Reflection Task)

独立的后台定时任务，定期将用户的碎片化低层记忆聚合提炼为高价值洞察（reflection 类型记忆）。

核心价值：
  解决"记忆碎片化"问题。用户的零散记忆（如多次问健身、提到在减肥、说不喜欢跑步）
  单条看价值有限，但聚合后可生成高质量洞察：
  "用户正在减肥，偏好低强度运动，关注饮食控制" → importance=0.9 的 reflection 记忆。

触发条件（满足任一）：
  - 用户活跃记忆数 >= reflection_trigger_count（默认 20 条）
  - 上次反思距今 >= reflection_interval_days（默认 7 天）

反思流程（per user）：
  1. 查询该用户最近的低层记忆（importance < 0.7，排除已有 reflection）
  2. 若可反思的记忆数 < min_memories_for_reflection（默认 5），跳过
  3. 打包送给 Qwen3-4B 归纳，返回 1~3 条高价值洞察
  4. 将洞察存为 reflection 类型记忆（importance=0.9，source_ids 溯源）
  5. 更新 user_profiles.stats 中的 last_reflection_at

Author: Ke Meng
Created: 2026-04-12
Version: 1.0.0
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import asyncpg
import httpx
from pgvector.asyncpg import register_vector
from src.core.embedder import ONNXEmbedder
from src.core.vllm_client import vllm_generate_async

logger = logging.getLogger(__name__)

# 送给 Qwen3-4B 反思的最大记忆条数（避免 prompt 过长）
_MAX_MEMORIES_FOR_REFLECTION = 20

# 触发反思的最小记忆数量（低于此值跳过，碎片太少无需归纳）
_MIN_MEMORIES_FOR_REFLECTION = 8

_REFLECTION_PROMPT = """\
你是一个用户画像分析师。以下是某用户的若干条零散记忆，请从中归纳出 1~3 条高价值的综合洞察。

【要求】
- 每条洞察必须是对多条原始记忆的规律性归纳，不是单条记忆的复述
- 用第三人称简洁表达（如"用户..."），不超过 60 字
- 只输出真正有价值的规律，宁可少不可滥
- 为每条洞察评估重要性（importance_score，范围 0.7~1.0）：
    0.7~0.8：规律存在但不够显著
    0.8~0.9：规律明确，对理解用户有较高价值
    0.9~1.0：高度概括性洞察，对用户画像极有价值
- 返回 JSON 对象，格式为：
  {{"reflections": [
    {{"content": "洞察内容", "importance_score": 0.85}},
    ...
  ]}}

【用户零散记忆】
{memories_text}

返回 JSON：
"""


class ReflectionTask:
    """
    反思机制定时任务。

    使用外部传入的 asyncpg.Pool 和 httpx.AsyncClient，不自行管理连接生命周期。
    可单独实例化后独立测试，与 LongTermMemory 完全解耦。

    Usage（定时任务）：
        task = ReflectionTask(pool, http_client, embedder, vllm_base_url="http://localhost:8001/v1")
        # 对单个用户执行反思：
        await task.run_for_user(user_id="xxx")
        # 对所有满足条件的用户批量执行：
        await task.run_all_eligible_users()
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        http_client: httpx.AsyncClient,
        embedder: ONNXEmbedder,
        vllm_base_url: str = "http://localhost:8001/v1",
        vllm_model: str = "qwen3-4b",
        api_key: str = "EMPTY",
        reflection_timeout: float = 60.0,
        reflection_trigger_count: int = 50,
        reflection_interval_days: int = 7,
        min_memories_for_reflection: int = _MIN_MEMORIES_FOR_REFLECTION,
        max_memories_for_reflection: int = _MAX_MEMORIES_FOR_REFLECTION,
    ):
        self.pool = db_pool
        self.http = http_client
        self.embedder = embedder
        self.vllm_base_url = vllm_base_url
        self.vllm_model = vllm_model
        self.api_key = api_key
        self.reflection_timeout = reflection_timeout
        self.reflection_trigger_count = reflection_trigger_count
        self.reflection_interval_days = reflection_interval_days
        self.min_memories = min_memories_for_reflection
        self.max_memories = max_memories_for_reflection

    # ═══════════════════════════ 公开接口 ═══════════════════════════

    async def run_for_user(self, user_id: str) -> int:
        """
        对单个用户执行反思（可直接调用，也可由 run_all_eligible_users 内部调用）。

        按 memory_type 分组归纳：每种类型的碎片独立送模型，主题一致，洞察质量更高。
        混合不同类型会导致归纳主题散乱，reflection 向量语义不集中，检索时难以命中。

        Returns:
            本次生成的 reflection 记忆条数（0 表示条件不满足或无有效洞察）
        """
        logger.info(f"[反思] 开始处理 user={user_id}")

        # Step 1: 按类型分组获取可反思的低层记忆
        candidates_by_type = await self._get_reflection_candidates(user_id)
        if not candidates_by_type:
            logger.info(f"[反思] user={user_id} 无可反思的低层记忆，跳过")
            return 0

        stored_count = 0

        # Step 2: 对每种类型独立归纳
        for memory_type, candidates in candidates_by_type.items():
            if len(candidates) < self.min_memories:
                logger.info(
                    f"[反思] user={user_id} [{memory_type}] 候选仅 {len(candidates)} 条，"
                    f"低于阈值 {self.min_memories}，跳过"
                )
                continue

            reflections = await self._generate_reflections(candidates)
            if not reflections:
                logger.info(f"[反思] user={user_id} [{memory_type}] Qwen3-4B 未生成有效洞察")
                continue

            source_ids = [str(r["id"]) for r in candidates]
            for reflection in reflections:
                try:
                    await self._store_reflection(
                        user_id,
                        reflection["content"],
                        source_ids,
                        reflection["importance_score"],
                    )
                    stored_count += 1
                except Exception as e:
                    logger.error(
                        f"[反思] user={user_id} [{memory_type}] 存储洞察失败: "
                        f"{reflection.get('content', '')[:50]} | {e}"
                    )

        # Step 3: 有任意类型产生洞察才更新反思时间
        if stored_count > 0:
            await self._update_last_reflection_time(user_id)

        logger.info(f"[反思] user={user_id} 共生成并存储 {stored_count} 条反思记忆")
        return stored_count

    async def run_all_eligible_users(self) -> Dict[str, int]:
        """
        批量对所有满足反思条件的用户执行反思。

        触发条件（满足任一）：
        - 活跃记忆数 >= reflection_trigger_count
        - 上次反思距今 >= reflection_interval_days

        Returns:
            {user_id: 生成的 reflection 条数} 字典
        """
        eligible_users = await self._get_eligible_users()
        logger.info(f"[反思] 本次扫描到 {len(eligible_users)} 个用户满足反思条件")

        results: Dict[str, int] = {}
        for user_id in eligible_users:
            try:
                count = await self.run_for_user(user_id)
                results[user_id] = count
            except Exception as e:
                logger.error(f"[反思] user={user_id} 反思任务异常: {e}", exc_info=True)
                results[user_id] = 0

        return results

    # ═══════════════════════════ 私有方法 ═══════════════════════════

    async def _get_eligible_users(self) -> List[str]:
        """
        查询满足反思触发条件的用户列表。

        条件（满足任一）：
        1. 活跃记忆数 >= reflection_trigger_count
        2. user_profiles.stats 中 last_reflection_at 超过 interval_days 天前，
           或从未做过反思（无 last_reflection_at 字段）
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.reflection_interval_days)

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT m.user_id
                FROM memories m
                LEFT JOIN user_profiles p ON p.user_id = m.user_id
                WHERE m.is_active = TRUE
                  AND m.memory_type != 'reflection'
                GROUP BY m.user_id, p.stats
                HAVING
                    -- 条件1：活跃记忆数超过触发阈值
                    COUNT(*) >= $1
                    OR
                    -- 条件2：从未反思 或 上次反思超过间隔天数
                    (
                        (p.stats->>'last_reflection_at') IS NULL
                        OR (p.stats->>'last_reflection_at')::timestamptz < $2
                    )
                """,
                self.reflection_trigger_count,
                cutoff,
            )

        return [r["user_id"] for r in rows]

    async def _get_reflection_candidates(self, user_id: str) -> Dict[str, List[Dict]]:
        """
        按 memory_type 分组获取低层记忆候选。

        分组归纳的原因：混合不同类型的碎片送模型会导致归纳主题散乱，
        生成的 reflection 向量语义不集中，检索时余弦相似度难以精准命中。
        每种类型独立归纳，主题一致，洞察质量和检索精度均显著提升。

        过滤条件：
        - is_active = TRUE
        - memory_type in (fact, preference, episodic)，不对 reflection 二次归纳
        - importance_score < 0.7（低重要性记忆才需要聚合提炼）
        - 按 created_at 降序（优先归纳最近的记忆）

        Returns:
            {"fact": [...], "preference": [...], "episodic": [...]}
            仅返回有记录的类型，空类型不包含在内
        """
        result: Dict[str, List[Dict]] = {}
        async with self.pool.acquire() as conn:
            for memory_type in ("fact", "preference", "episodic"):
                rows = await conn.fetch(
                    """
                    SELECT id, content, memory_type, importance_score, created_at
                    FROM memories
                    WHERE user_id = $1
                      AND is_active = TRUE
                      AND memory_type = $2
                      AND importance_score < 0.7
                    ORDER BY created_at DESC
                    LIMIT $3
                    """,
                    user_id,
                    memory_type,
                    self.max_memories,
                )
                if rows:
                    result[memory_type] = [dict(r) for r in rows]
        return result

    async def _generate_reflections(self, candidates: List[Dict]) -> List[Dict]:
        """
        调用 Qwen3-4B 从候选记忆中归纳洞察。

        Returns:
            洞察列表（1~3 条），每条为 {content: str, importance_score: float}
        """
        memories_text = "\n".join(
            f"- [{r['memory_type']}] {r['content']}"
            for r in candidates
        )
        prompt = _REFLECTION_PROMPT.format(memories_text=memories_text)

        raw = await vllm_generate_async(
            http_client=self.http,
            base_url=self.vllm_base_url,
            model=self.vllm_model,
            prompt=prompt,
            api_key=self.api_key,
            timeout=self.reflection_timeout,
            temperature=0.3,
            max_tokens=512,
            json_mode=True,
            thinking=True,
        )
        if raw is None:
            return []
        raw = self._strip_think_tags(raw)

        try:
            data = json.loads(raw)
            reflections = data.get("reflections", [])
        except json.JSONDecodeError:
            logger.warning(f"Qwen3-4B 反思输出解析失败: {raw[:200]}")
            return []

        valid = []
        for r in reflections[:3]:
            if not isinstance(r, dict):
                continue
            content = str(r.get("content", "")).strip()
            if not content:
                continue
            try:
                score = max(0.7, min(1.0, float(r.get("importance_score", 0.85))))
            except (TypeError, ValueError):
                score = 0.85
            valid.append({"content": content, "importance_score": score})
        return valid

    async def _store_reflection(
        self,
        user_id: str,
        insight: str,
        source_ids: List[str],
        importance_score: float = 0.85,
    ) -> None:
        """
        将一条反思洞察向量化后存入 memories 表。

        - memory_type = 'reflection'
        - importance_score = 模型评估值（0.7~1.0），不再固定为 0.9
        - source_ids = 被归纳的原始记忆 UUID 列表
        """
        embedding = self.embedder.embed_query(insight)

        async with self.pool.acquire() as conn:
            await register_vector(conn)
            await conn.execute(
                """
                INSERT INTO memories
                    (user_id, content, embedding, memory_type, importance_score, source_ids)
                VALUES ($1, $2, $3, 'reflection', $4, $5::uuid[])
                """,
                user_id,
                insight,
                embedding,
                importance_score,
                source_ids,
            )
        logger.info(f"[反思] user={user_id} 存储洞察(score={importance_score}): {insight[:60]}")

    async def _update_last_reflection_time(self, user_id: str) -> None:
        """
        更新 user_profiles.stats.last_reflection_at 为当前时间。
        使用 JSONB || 运算符原子合并，不覆盖其他 stats 字段。
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_profiles (user_id, stats)
                VALUES ($1, $2::jsonb)
                ON CONFLICT (user_id) DO UPDATE SET
                    stats      = user_profiles.stats || EXCLUDED.stats,
                    updated_at = NOW()
                """,
                user_id,
                json.dumps({"last_reflection_at": now_iso}),
            )

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """移除 Qwen3 thinking 模式输出的 <think>...</think> 内容。"""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()