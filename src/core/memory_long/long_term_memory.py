"""
长期记忆模块 (Long Term Memory) - 生产级重构版
- 存储后端：PostgreSQL + pgvector（替换 Milvus）
- 记忆提取：Qwen3-1.7B（via Ollama，最多返回 3 条）
- 合并决策：Qwen3-4B（via Ollama，add/update/delete/skip）
- 向量化：nomic-embed-text（via Ollama）
- 框架：Mem0 完全移除（替换为 asyncpg + pgvector）

核心流程：
1. 检索（retrieve_relevant）：
   向量化查询 → pgvector 余弦候选检索 → 动态综合评分重排 → 返回 top_k → 后台异步更新 access_count

2. 存储（process_and_store）：
   前置门槛（原话无自称且无"记住"→不抽取）→ Qwen3-1.7B 提取（≤3条，附原话依据 evidence）
   → 写入前校验（依据须在用户原话中、拒绝提问记录 / 占位符 / 未要求记住的敏感信息）→ 对每条：向量化 → 检索相似旧记忆
   → Qwen3-4B 决策 → PG 事务执行（add/update/软删除/skip）

动态综合评分公式：
   final_score = cosine_sim   × 0.5
               + importance   × 0.2   （Qwen3-1.7B 静态评分）
               + recency      × 0.2   （e^(-0.01 × days_since_update)，约70天半衰）
               + frequency    × 0.1   （min(access_count/10, 1.0)）

Author: Ke Meng
Created: 2026-04-12
Version: 2.0.0
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

import asyncpg
import httpx
from pgvector.asyncpg import register_vector
from src.core.memory_long.forgetting_task import ForgettingTask
from src.core.embedder import ONNXEmbedder
from src.core.security.pii import contains_pii
from src.core.vllm_client import vllm_generate_async

logger = logging.getLogger(__name__)

# ─────────────────────────── Prompts ────────────────────────────

# Prefix Caching 设计：固定指令作为 system_prompt 独立传入，变化内容作为 user prompt。
# vLLM 以 system 消息前缀的 hash 作为 cache key，固定前缀只计算一次 KV Cache。

_EXTRACTION_SYSTEM = """\
你是一个个人信息组织者，只从【用户自己说的话】中提取"关于用户本人、长期有效"的信息。宁可不提取，也不要提取错。

【输出格式】
返回 JSON 对象，包含 "memories" 键，值为列表（最多 3 条）。每条记忆包含：
- content:          简洁一句话，第三人称，以"用户"开头（如"用户是…""用户喜欢…"），不超过 50 字
- content_en:       content 的英文翻译，以 "The user" 开头（用于向量检索，专有名词保留常用英文名）
- memory_type:      只能是 fact / preference / episodic 之一
- importance_score: 0.0~1.0 浮点数
- evidence:         从用户原话中逐字摘录的、支持这条记忆的片段（必须是用户原话的连续子串，包含"我"等自称）

【只提取】用户以第一人称陈述的关于自己的信息：
- 身份 / 职业 / 学历 / 所在地，如"我是一名开发工程师""我在上海工作"
- 偏好 / 习惯 / 风格，如"我喜欢古典音乐""我喜欢简洁的回答"
- 本人的计划 / 经历 / 正在做的事，如"我下个月要去阿根廷""我在备考雅思"

【不要提取】
- 用户向助手提问、查询、请求帮助这件事本身："用户询问…""用户想知道…""用户想了解…"都不是记忆
- 助手回答里的任何内容；知识性内容（定义、原因、数据、历史等）
- 用户对外部世界的断言或转述别人的话（如"记住：某公司总部在纽约""我朋友说…"），这不是关于用户本人的信息
- 寒暄、致谢、确认（"好的""谢谢"），以及只与本次提问有关的临时信息
- 健康、疾病、收入、存款等敏感信息，除非用户明确说"记住"
- 一句话里既有提问又有个人信息时，只提取个人信息；content 只写个人信息本身，不要附带"想知道 / 想了解…"的提问内容
用户没有透露任何个人信息时返回 {"memories": []}
注意："记住"后面是关于用户自己的信息（如过敏、饮食禁忌、称呼）要提取；"记住"后面是外部世界的事则不提取。

【类型判断】
fact:       客观稳定事实（职业、城市、身份）
preference: 主观偏好（喜好、习惯、风格）
episodic:   用户本人做过或计划做的事（不包括和助手的这次对话）

【示例】
用户：我是一名开发工程师
输出：{"memories": [
  {"content": "用户是一名开发工程师", "content_en": "The user is a software developer", "memory_type": "fact", "importance_score": 0.8, "evidence": "我是一名开发工程师"}
]}

用户：我是 AI 工程师，平时喜欢看硬科幻电影，尤其是星际穿越
输出：{"memories": [
  {"content": "用户职业是 AI 工程师", "content_en": "The user works as an AI engineer", "memory_type": "fact", "importance_score": 0.9, "evidence": "我是 AI 工程师"},
  {"content": "用户喜欢硬科幻电影，尤其是星际穿越", "content_en": "The user likes hard science fiction movies, especially Interstellar", "memory_type": "preference", "importance_score": 0.8, "evidence": "我是 AI 工程师，平时喜欢看硬科幻电影"}
]}

用户：阿根廷葡萄酒产量排第几？我下个月要去阿根廷旅行
输出：{"memories": [
  {"content": "用户计划下个月去阿根廷旅行", "content_en": "The user plans to travel to Argentina next month", "memory_type": "episodic", "importance_score": 0.7, "evidence": "我下个月要去阿根廷旅行"}
]}

用户：我在准备考研，线性代数的特征值怎么理解？
输出：{"memories": [
  {"content": "用户在准备考研", "content_en": "The user is preparing for the graduate school entrance exam", "memory_type": "episodic", "importance_score": 0.7, "evidence": "我在准备考研"}
]}

用户：记住，我对海鲜过敏
输出：{"memories": [
  {"content": "用户对海鲜过敏", "content_en": "The user is allergic to seafood", "memory_type": "fact", "importance_score": 0.9, "evidence": "我对海鲜过敏"}
]}

用户：我比较喜欢带表格的回答
输出：{"memories": [
  {"content": "用户喜欢带表格的回答", "content_en": "The user prefers answers with tables", "memory_type": "preference", "importance_score": 0.7, "evidence": "我比较喜欢带表格的回答"}
]}

用户：双相情感障碍 II 型的定义是什么？
输出：{"memories": []}

用户：我想知道钢琴是谁发明的
输出：{"memories": []}

用户：帮我查一下丹麦的首都
输出：{"memories": []}

用户：记住：好时公司的总部在纽约。
输出：{"memories": []}

用户：好的，谢谢
输出：{"memories": []}"""

_EXTRACTION_USER = "请分析以下对话并返回 JSON：\n{dialogue}"

_MERGE_SYSTEM = """\
你是一个记忆管理器。给定一条新记忆和若干条相似旧记忆，决定如何处理。

【决策规则】
- add:    新记忆与所有旧记忆描述的不是同一件事（如职业 vs 爱好、过敏 vs 旅行），直接新增；不同话题不得合并成一条
- update: 新记忆与某条旧记忆描述的是同一属性或同一件事（如"开发工程师"细化为"Python 开发工程师"），合并更新该条
- delete: 新记忆说明某条旧记忆已过时或错误，软删旧记忆并新增新记忆
- skip:   新记忆与某条旧记忆高度重复，无需任何操作（相似度 > 0.95 且语义相同）

【输出格式】
返回 JSON，包含：
- action:         "add" / "update" / "delete" / "skip"
- target_id:      update/delete 时填目标旧记忆的 id 字符串，否则填 null
- merged_content:    update 时填合并后的中文内容，以"用户"开头（只写记忆正文，不要附带类型、重要性等标注），否则填 null
- merged_content_en: update 时填 merged_content 的英文翻译，以 "The user" 开头，否则填 null"""

_MERGE_USER = """\
【新记忆】
{new_memory}

【相似旧记忆（按相似度降序）】
{existing_memories}

返回 JSON："""


_META_SUFFIX = re.compile(r"\s*[|｜]\s*(类型|重要性|importance|type)\s*[:：].*$", re.IGNORECASE)
_CONTENT_PREFIX = re.compile(r"^\s*内容\s*[:：]\s*")


# nomic-embed-text 官方要求的任务前缀：检索查询 / 被检索文档。仅用于长期记忆链路，知识库（Milvus）不受影响
_QUERY_PREFIX = "search_query: "
_DOC_PREFIX = "search_document: "
_CJK = re.compile(r"[\u4e00-\u9fff]")


# ─────────────────────── 写入门槛与写入前校验 ───────────────────────
# 原则：只存用户亲口说的、关于自己的、长期有效的信息；宁可漏存，不可错存。
#   L0 前置门槛：原话没有自称、也没有"记住"类指令 → 不调用 1.7B（纯知识提问属于对话历史，不属于长期记忆）
#   L2 写入前校验：evidence 须为用户原话子串且所在句含自称；拒绝"用户询问…"类提问记录、占位符 / 敏感信息（未明确要求记住）
_SELF_REF_RE = re.compile(r"我|本人|咱|俺|作为一名|作为一个|身为|\b(?:I|I'm|I've|I'll|I'd|my|me|mine|myself)\b", re.I)
_REMEMBER_RE = re.compile(r"记住|记一下|记下|帮我记|别忘了|remember", re.I)
_QUESTION_LOG_ZH = re.compile(r"^用户(?:曾经?|多次|刚才|之前|在)?(?:询问|问了?|问过|想知道|想了解|咨询|查询|请教|提问|让助手|请助手|要求助手)")
_QUESTION_LOG_EN = re.compile(r"^the user (?:has |had )?(?:asked|asks|inquired|inquires|wants? to know|wanted to know|"
                              r"wants? to learn|is asking|was asking|queried|requested|is curious)", re.I)
_SENSITIVE_RE = re.compile(r"疾病|病史|确诊|焦虑|抑郁|癌|糖尿病|高血压|怀孕|过敏|服药|心理|收入|月薪|年薪|工资|存款|负债|贷款|欠款|"
                           r"身份证|护照|宗教|信仰|政治|性取向|salary|income|diagnos|allerg|religio", re.I)
_SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]")
_NORM_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def _norm(text: str) -> str:
    return _NORM_RE.sub("", (text or "").lower())


def should_extract(user_input: str) -> bool:
    """L0：原话含自称或明确的"记住"指令才值得抽取"""
    return bool(_SELF_REF_RE.search(user_input or "") or _REMEMBER_RE.search(user_input or ""))


def _evidence_ok(evidence: str, user_input: str) -> bool:
    """evidence 须为用户原话的子串（忽略空白与标点），且所在句子含自称"""
    ne = _norm(evidence)
    if not ne or ne not in _norm(user_input):
        return False
    sentences = [s for s in _SENT_SPLIT_RE.split(user_input) if _norm(s)]
    holders = [s for s in sentences if ne in _norm(s)]
    if holders:
        return any(_SELF_REF_RE.search(s) for s in holders)
    return bool(_SELF_REF_RE.search(evidence))          # 跨句摘录：摘录本身须含自称


def memory_reject_reason(mem: Dict, user_input: str) -> Optional[str]:
    """L2：返回拒绝原因；None 表示可以写入。清理脚本复用其中只看正文的规则（user_input 传 None）"""
    content, content_en = mem.get("content") or "", mem.get("content_en") or ""
    if not content.startswith("用户") or not content_en.lower().startswith("the user"):
        return "format"
    if _QUESTION_LOG_ZH.search(content) or _QUESTION_LOG_EN.search(content_en):
        return "question_log"
    if contains_pii(content) or contains_pii(content_en):
        return "pii"                                     # 占位符跨会话无法还原，存了也无用
    if user_input is None:
        return None
    if _SENSITIVE_RE.search(content) or _SENSITIVE_RE.search(content_en):
        if not _REMEMBER_RE.search(user_input):
            return "sensitive"
    if not _evidence_ok(str(mem.get("evidence") or ""), user_input):
        return "no_evidence"
    return None


def _mostly_chinese(text: str) -> bool:
    """改写降级时 en_query 会退化为中文原话：中文占比高则判定为非英文查询"""
    text = text.strip()
    return bool(text) and len(_CJK.findall(text)) / len(text) > 0.2


def _clean_memory_text(text: str) -> str:
    """去掉模型照抄的"内容：… | 类型：fact | 重要性：0.8"格式标注，只保留记忆正文"""
    text = _CONTENT_PREFIX.sub("", text.strip())
    return _META_SUFFIX.sub("", text).strip()


MEMORY_SCHEMA_MIGRATION = "ALTER TABLE memories ADD COLUMN IF NOT EXISTS content_en TEXT"


async def ensure_memory_schema(pool) -> None:
    """启动时执行：memories 增加 content_en（英文记忆正文，用于向量化）。
    旧记忆 content_en 为空、向量为中文生成，不参与召回与合并比对"""
    async with pool.acquire() as conn:
        await conn.execute(MEMORY_SCHEMA_MIGRATION)


class LongTermMemory:
    """
    生产级长期记忆管理器。

    实例化时需要外部传入已初始化的 asyncpg.Pool 和 httpx.AsyncClient，
    由调用方（main.py / api.py）统一管理生命周期，避免模块内部重复创建连接。

    Attributes:
        pool:                   asyncpg 连接池
        http:                   httpx 异步客户端
        embedder:               ONNXEmbedder 实例（nomic-embed-text ONNX CPU 推理）
        extract_base_url:       vLLM 提取模型地址（qwen3-1.7b，端口 8002）
        extract_model:          记忆提取模型名
        merge_base_url:         vLLM 合并模型地址（qwen3-4b，端口 8001）
        merge_model:            合并决策模型名
        api_key:                vLLM 鉴权 key
        top_k:                  最终返回的记忆条数（默认 3）
        max_candidates:         动态评分前的候选数量（默认 10）
        similar_existing_limit: 送给 merge 模型决策的旧记忆数量（默认 3）
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        http_client: httpx.AsyncClient,
        embedder: ONNXEmbedder,
        extract_base_url: str = "http://localhost:8002/v1",
        extract_model: str = "qwen3-1.7b",
        extract_timeout: float = 30.0,
        merge_base_url: str = "http://localhost:8001/v1",
        merge_model: str = "qwen3-4b",
        merge_timeout: float = 30.0,
        api_key: str = "EMPTY",
        top_k: int = 3,
        max_candidates: int = 10,
        similar_existing_limit: int = 3,
        forgetting_task: Optional["ForgettingTask"] = None,
        min_similarity: float = 0.0,
        merge_min_similarity: float = 0.78,
    ):
        self.pool = db_pool
        self.http = http_client
        self.embedder = embedder
        self.extract_base_url = extract_base_url
        self.extract_model = extract_model
        self.extract_timeout = extract_timeout
        self.merge_base_url = merge_base_url
        self.merge_model = merge_model
        self.merge_timeout = merge_timeout
        self.api_key = api_key
        self.top_k = top_k
        self.max_candidates = max_candidates
        self.similar_existing_limit = similar_existing_limit
        self.merge_min_similarity = merge_min_similarity
        self.forgetting_task = forgetting_task
        self.min_similarity = min_similarity

    # ═══════════════════════════ 公开接口 ═══════════════════════════

    async def retrieve_relevant(
        self,
        user_id: str,
        query: str,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        检索与用户问题语义相关的长期记忆。

        Args:
            user_id:     用户 ID
            query:       查询文本（本轮英文改写 en_query；记忆向量基于英文 content_en）
            memory_type: 可选，按类型过滤（fact/preference/episodic/reflection）

        召回规则：余弦相似度 ≥ min_similarity 的记忆才进入候选，再按动态综合评分取前 top_k；
        没有达标记忆时返回空，达标不足 top_k 时只返回达标的条数。

        Returns:
            List[Dict]，每条包含：
            - content:         记忆正文
            - memory_type:     记忆类型
            - importance_score: 静态重要性评分
            - final_score:     动态综合评分
            - metadata:        {id, access_count, created_at}
        """
        # 记忆向量基于英文（content_en）生成，查询须为英文改写；改写降级为中文原话时跳过，避免跨语言误召回
        if _mostly_chinese(query):
            logger.warning(f"[{user_id}] 长期记忆召回跳过：查询非英文（改写可能已降级）query={query[:40]}")
            return []
        query_embedding = await self._embed(_QUERY_PREFIX + query)

        async with self.pool.acquire() as conn:
            await register_vector(conn)
            results = await self._search_with_dynamic_score(
                conn, user_id, query_embedding, memory_type
            )

        if not results:
            logger.info(f"[{user_id}] 未检索到相关长期记忆")
            return []

        # 只返回相似度达到阈值的记忆，最多 top_k 条（达标 1 条就只返回 1 条）
        top_results = results[: self.top_k]
        logger.info(f"[{user_id}] 命中 {len(top_results)} 条长期记忆（阈值 {self.min_similarity}），相似度 "
                    f"{[round(float(r['similarity']), 3) for r in top_results]}")

        # 后台更新 access 统计，不阻塞返回
        hit_ids = [str(r["id"]) for r in top_results]
        asyncio.create_task(self._update_access_stats(hit_ids))

        return [self._format_result(r) for r in top_results]

    async def process_and_store(
        self,
        user_id: str,
        user_input: str,
        assistant_output: str,
    ) -> None:
        """
        提取并存储长期记忆（在主程序生成回复后异步调用，不阻塞主流程）。

        Args:
            user_id:          用户 ID
            user_input:       本轮用户输入
            assistant_output: 本轮助手回复
        """
        new_memories = await self._extract_memories(user_input, assistant_output)

        if new_memories is None:
            # 调用失败（超时 / 400 等）与"无需提取"区分开，避免静默丢失记忆
            logger.error(f"[{user_id}] 长期记忆提取失败：Qwen3-1.7B 调用未返回结果，本轮记忆未写入")
            return
        if not new_memories:
            logger.info(f"[{user_id}] 本轮对话无需提取长期记忆")
            return

        logger.info(f"[{user_id}] 提取到 {len(new_memories)} 条新记忆，开始并行处理")

        # 并行处理：最多3条同时跑，各自独立持有连接，错误不中断其他条目
        results = await asyncio.gather(
            *[self._process_single_memory(user_id, mem) for mem in new_memories],
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"[{user_id}] 处理记忆失败 | content={new_memories[i].get('content', '')[:40]} | {result}",
                    exc_info=True,
                )
        if self.forgetting_task is not None:
            try:
                await self.forgetting_task.run_capacity_forgetting(user_id)
            except Exception as e:
                logger.error(f"[{user_id}] 容量淘汰失败（不影响主流程）: {e}")

    async def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        读取用户核心画像（每次对话前直接主键查询，无向量检索开销）。

        Returns:
            {core_facts, preferences, stats} 或 None（用户无画像时）
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT core_facts, preferences, stats FROM user_profiles WHERE user_id = $1",
                user_id,
            )
        if not row:
            return None
        return {
            "core_facts": row["core_facts"] or {},
            "preferences": row["preferences"] or {},
            "stats": row["stats"] or {},
        }

    async def upsert_user_profile(
        self,
        user_id: str,
        core_facts: Optional[Dict] = None,
        preferences: Optional[Dict] = None,
        stats_patch: Optional[Dict] = None,
    ) -> None:
        """
        原子更新用户核心画像（INSERT ON CONFLICT DO UPDATE，JSONB 增量合并）。

        Args:
            core_facts:   合并到 core_facts 的键值（不传则不变）
            preferences:  合并到 preferences 的键值（不传则不变）
            stats_patch:  合并到 stats 的键值
        """
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO user_profiles (user_id, core_facts, preferences, stats)
                VALUES ($1, $2::jsonb, $3::jsonb, $4::jsonb)
                ON CONFLICT (user_id) DO UPDATE SET
                    core_facts  = user_profiles.core_facts  || EXCLUDED.core_facts,
                    preferences = user_profiles.preferences || EXCLUDED.preferences,
                    stats       = user_profiles.stats       || EXCLUDED.stats,
                    updated_at  = NOW()
                """,
                user_id,
                json.dumps(core_facts or {}),
                json.dumps(preferences or {}),
                json.dumps(stats_patch or {}),
            )

    # ═══════════════════════════ 私有方法 ═══════════════════════════

    async def _process_single_memory(self, user_id: str, new_mem: Dict) -> None:
        """
        单条记忆的完整处理流程：
        向量化 → 检索相似旧记忆 → Qwen3-4B 决策 → 预计算合并向量 → 事务执行

        注意：HTTP 调用（embed/decide）全部在事务外完成，
        事务内只做纯 SQL 操作，保证事务时间短。
        """
        # Step 1: 向量化新记忆（英文 + 文档前缀，与召回的英文查询处于同一语义空间）
        embedding = await self._embed(_DOC_PREFIX + new_mem["content_en"])

        async with self.pool.acquire() as conn:
            await register_vector(conn)

            # Step 2: 检索相似旧记忆（事务外查询，只读）
            existing = await self._get_similar_existing(conn, user_id, embedding)

            # Step 3: Qwen3-4B 决策（事务外，HTTP 调用）
            decision = await self._decide_action(new_mem, existing)

            # Step 4: 如果是 update，提前计算合并后内容的向量（事务外）
            merged_embedding: Optional[List[float]] = None
            if decision["action"] == "update":
                merged_zh = _clean_memory_text(decision.get("merged_content") or "")
                merged_en = _clean_memory_text(decision.get("merged_content_en") or "")
                if merged_zh.startswith("用户") and merged_en.lower().startswith("the user"):
                    decision["merged_content"], decision["merged_content_en"] = merged_zh, merged_en
                    merged_embedding = await self._embed(_DOC_PREFIX + merged_en)
                else:
                    # 合并结果缺语言版本或语言错位（如中文栏写成英文）：不覆盖旧记忆，改为新增
                    logger.info(f"[{user_id}] 合并结果不合规，改为 ADD: {merged_zh[:40]} / {merged_en[:40]}")
                    decision = {"action": "add", "target_id": None, "merged_content": None}

            # Step 5: 事务执行（纯 SQL，无 HTTP 调用，保证短事务）
            await self._execute_decision(
                conn, user_id, new_mem, embedding, decision, merged_embedding
            )

    async def _embed(self, text: str) -> List[float]:
        """ONNX CPU 嵌入。计算是同步的（单次 ~5-15ms），并发下会累积阻塞事件循环，
        因此放入线程池执行（ONNX 推理释放 GIL，可与其他请求真正并行）。"""
        embedding = await asyncio.to_thread(self.embedder.embed_query, text)
        if not embedding:
            raise ValueError(f"embed 返回空向量，text={text[:50]}")
        return embedding

    async def _extract_memories(
        self, user_input: str, assistant_output: str
    ) -> Optional[List[Dict]]:
        """
        抽取并校验后可写入的记忆，最多 3 条；模型调用失败时返回 None（区别于"无需提取"的空列表）
        """
        out = await self.extract_candidates(user_input, assistant_output)
        if out["skipped"]:
            logger.debug(f"长期记忆抽取跳过（{out['skipped']}）")
        for r in out["rejected"]:
            logger.info(f"丢弃抽取结果（{r['reason']}）: {r['content'][:40]}")
        return out["memories"]

    async def extract_candidates(self, user_input: str, assistant_output: str) -> Dict[str, Any]:
        """
        L0 前置门槛 → Qwen3-1.7B 抽取 → L2 写入前校验。
        返回 {"skipped": 跳过原因或 None, "memories": 可写入的记忆（调用失败为 None）, "rejected": [{content, reason}]}
        """
        if not should_extract(user_input):
            return {"skipped": "no_self_reference", "memories": [], "rejected": []}
        # 助手回答只截取开头作为背景：回答多为知识性内容，全文送入会让小模型把知识当成用户事实抽取
        dialogue = f"用户：{user_input}\n助手（节选，仅供理解语境，不从中提取知识）：{assistant_output[:150]}"

        raw = await vllm_generate_async(
            http_client=self.http,
            base_url=self.extract_base_url,
            model=self.extract_model,
            prompt=_EXTRACTION_USER.format(dialogue=dialogue),
            system_prompt=_EXTRACTION_SYSTEM,
            api_key=self.api_key,
            timeout=self.extract_timeout,
            temperature=0.1,
            max_tokens=512,
            json_mode=False,
        )
        if raw is None:
            return {"skipped": None, "memories": None, "rejected": []}
        raw = self._strip_think_tags(raw)

        try:
            data = json.loads(raw)
            raw_memories = data.get("memories", []) if isinstance(data, dict) else []
        except json.JSONDecodeError:
            logger.warning(f"Qwen3-1.7B 输出无法解析为 JSON: {raw[:200]}")
            return {"skipped": "bad_json", "memories": [], "rejected": []}

        # 校验字段 + 写入前校验 + 强制上限 3 条
        valid: List[Dict] = []
        rejected: List[Dict] = []
        for m in raw_memories[:3] if isinstance(raw_memories, list) else []:
            if not isinstance(m, dict) or not str(m.get("content", "")).strip():
                continue
            memory_type = m.get("memory_type", "fact")
            if memory_type not in ("fact", "preference", "episodic"):
                memory_type = "fact"
            try:
                importance = max(0.0, min(1.0, float(m.get("importance_score", 0.5))))
            except (TypeError, ValueError):
                importance = 0.5
            mem = {
                "content": _clean_memory_text(str(m["content"])),
                "content_en": _clean_memory_text(str(m.get("content_en") or "")),
                "memory_type": memory_type,
                "importance_score": importance,
                "evidence": str(m.get("evidence") or "").strip(),
            }
            reason = memory_reject_reason(mem, user_input)
            if reason:
                rejected.append({"content": mem["content"], "reason": reason})
                continue
            valid.append(mem)

        return {"skipped": None, "memories": valid, "rejected": rejected}

    async def _get_similar_existing(
        self,
        conn: asyncpg.Connection,
        user_id: str,
        embedding: List[float],
    ) -> List[Dict]:
        """
        从 pgvector 检索余弦相似度最高的旧记忆，供 Qwen3-4B 决策使用。
        只取相似度 ≥ merge_min_similarity 的候选：低于门槛视为不同话题，直接新增、不调用 4B，
        避免把"职业"和"爱好"这类无关记忆合并成一条。

        返回字段包含 id（字符串化），供决策结果中引用。
        """
        rows = await conn.fetch(
            """
            SELECT id::text, content, content_en, memory_type, importance_score,
                   ROUND((1 - (embedding <=> $1))::numeric, 4) AS similarity
            FROM memories
            WHERE user_id = $2 AND is_active = TRUE AND content_en IS NOT NULL
              AND (1 - (embedding <=> $1)) >= $4
            ORDER BY embedding <=> $1
            LIMIT $3
            """,
            embedding,
            user_id,
            self.similar_existing_limit,
            self.merge_min_similarity,
        )
        return [dict(r) for r in rows]

    async def _decide_action(
        self,
        new_mem: Dict,
        existing: List[Dict],
    ) -> Dict:
        """
        调用 Qwen3-4B 决策新记忆的处理方式。

        无相似旧记忆时直接返回 add，跳过 LLM 调用。

        Returns:
            {action: str, target_id: str|None, merged_content: str|None}
        """
        if not existing:
            return {"action": "add", "target_id": None, "merged_content": None}

        existing_text = "\n".join(
            f"[id: {r['id']}] [相似度: {r['similarity']}] {r['content_en']}（中文：{r['content']}）"
            for r in existing
        )

        raw = await vllm_generate_async(
            http_client=self.http,
            base_url=self.merge_base_url,
            model=self.merge_model,
            prompt=_MERGE_USER.format(
                new_memory=f"{new_mem['content_en']}（中文：{new_mem['content']}）",
                existing_memories=existing_text,
            ),
            system_prompt=_MERGE_SYSTEM,
            api_key=self.api_key,
            timeout=self.merge_timeout,
            temperature=0.1,
            max_tokens=256,
            json_mode=True,
        )
        if raw is None:
            logger.warning("vLLM 合并决策返回 None，fallback 到 add")
            return {"action": "add", "target_id": None, "merged_content": None}
        raw = self._strip_think_tags(raw)

        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Qwen3-4B 决策解析失败: {raw[:200]}，fallback 到 add")
            return {"action": "add", "target_id": None, "merged_content": None}

        action = decision.get("action", "add")
        if action not in ("add", "update", "delete", "skip"):
            logger.warning(f"Qwen3-4B 返回非法 action: {action}，fallback 到 add")
            action = "add"

        # skip 时模型通常不返回 target_id，取相似度最高的旧记忆 id 供后续更新 access_count
        target_id = decision.get("target_id")
        if action == "skip" and not target_id and existing:
            target_id = str(existing[0]["id"])

        return {
            "action": action,
            "target_id": target_id,
            "merged_content": decision.get("merged_content"),
            "merged_content_en": decision.get("merged_content_en"),
        }

    async def _execute_decision(
        self,
        conn: asyncpg.Connection,
        user_id: str,
        new_mem: Dict,
        embedding: List[float],
        decision: Dict,
        merged_embedding: Optional[List[float]],
    ) -> None:
        """
        在单个 PG 事务中执行 Qwen3-4B 的决策。

        事务内只有纯 SQL 操作，无任何 HTTP 调用（embedding 已在事务外预计算）。
        软删除（is_active=FALSE）替代物理删除，保留溯源能力。
        """
        action = decision["action"]

        async with conn.transaction():

            if action == "skip":
                # 新旧记忆高度重复，更新旧记忆命中次数，延长其生命周期避免被遗忘机制过早淘汰
                if decision.get("target_id"):
                    await conn.execute(
                        """
                        UPDATE memories SET
                            access_count     = access_count + 1,
                            last_accessed_at = NOW()
                        WHERE id = $1::uuid AND user_id = $2 AND is_active = TRUE
                        """,
                        decision["target_id"],
                        user_id,
                    )
                logger.info(f"[{user_id}] SKIP: {new_mem['content'][:50]}")

            elif action == "add":
                await conn.execute(
                    """
                    INSERT INTO memories
                        (user_id, content, content_en, embedding, memory_type, importance_score)
                    VALUES ($1, $2, $6, $3, $4, $5)
                    """,
                    user_id,
                    new_mem["content"],
                    embedding,
                    new_mem["memory_type"],
                    new_mem["importance_score"],
                    new_mem["content_en"],
                )
                logger.info(f"[{user_id}] ADD: {new_mem['content'][:50]}")

            elif action == "update" and decision.get("target_id"):
                merged_content = decision["merged_content"] or new_mem["content"]
                final_embedding = merged_embedding or embedding
                result = await conn.execute(
                    """
                    UPDATE memories SET
                        content          = $1,
                        content_en       = $6,
                        embedding        = $2,
                        importance_score = GREATEST(importance_score, $3),
                        updated_at       = NOW()
                    WHERE id = $4::uuid
                      AND user_id = $5
                      AND is_active = TRUE
                    """,
                    merged_content,
                    final_embedding,
                    new_mem["importance_score"],
                    decision["target_id"],
                    user_id,
                    decision["merged_content_en"],
                )
                if result == "UPDATE 0":
                    # target_id 不存在或已被软删（模型幻觉），降级为 add 防止记忆静默丢失
                    logger.warning(
                        f"[{user_id}] UPDATE target_id={decision['target_id']} 影响0行，降级为 ADD"
                    )
                    await conn.execute(
                        """
                        INSERT INTO memories
                            (user_id, content, content_en, embedding, memory_type, importance_score)
                        VALUES ($1, $2, $6, $3, $4, $5)
                        """,
                        user_id,
                        new_mem["content"],
                        embedding,
                        new_mem["memory_type"],
                        new_mem["importance_score"],
                        new_mem["content_en"],
                    )
                else:
                    logger.info(f"[{user_id}] UPDATE: {merged_content[:50]}")

            elif action == "delete" and decision.get("target_id"):
                # CTE 原子执行：软删旧记忆 + 插入新记忆，两步合并为一次网络往返
                await conn.execute(
                    """
                    WITH soft_del AS (
                        UPDATE memories SET is_active = FALSE
                        WHERE id = $1::uuid AND user_id = $2 AND is_active = TRUE
                        RETURNING id
                    )
                    INSERT INTO memories
                        (user_id, content, content_en, embedding, memory_type, importance_score)
                    VALUES ($3, $4, $8, $5, $6, $7)
                    """,
                    decision["target_id"],
                    user_id,
                    user_id,
                    new_mem["content"],
                    embedding,
                    new_mem["memory_type"],
                    new_mem["importance_score"],
                    new_mem["content_en"],
                )
                logger.info(f"[{user_id}] DELETE+ADD: {new_mem['content'][:50]}")

            else:
                # 兜底：action 字段异常时执行 add
                logger.warning(
                    f"[{user_id}] 未知决策 action={action}，兜底执行 ADD"
                )
                await conn.execute(
                    """
                    INSERT INTO memories
                        (user_id, content, content_en, embedding, memory_type, importance_score)
                    VALUES ($1, $2, $6, $3, $4, $5)
                    """,
                    user_id,
                    new_mem["content"],
                    embedding,
                    new_mem["memory_type"],
                    new_mem["importance_score"],
                    new_mem["content_en"],
                )

    async def _search_with_dynamic_score(
        self,
        conn: asyncpg.Connection,
        user_id: str,
        embedding: List[float],
        memory_type: Optional[str],
    ) -> List[Dict]:
        """
        动态综合评分检索。

        评分公式（在 SQL 中实时计算，不持久化）：
          final_score = cosine_sim   × 0.5   （1 - 余弦距离）
                      + importance   × 0.2   （Qwen3-1.7B 静态评分）
                      + recency      × 0.2   （e^(-0.01 × days_since_update)）
                      + frequency    × 0.1   （min(access_count/10, 1.0)）

        时间衰减参数 λ=0.01，约 70 天后 recency 降至 0.5。
        """
        if memory_type:
            rows = await conn.fetch(
                """
                SELECT id, content, memory_type, importance_score,
                       access_count, last_accessed_at, created_at, updated_at,
                       (1.0 - (embedding <=> $1)) * 0.5
                       + importance_score * 0.2
                       + EXP(-0.01 * EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400.0) * 0.2
                       + LEAST(access_count::float / 10.0, 1.0) * 0.1
                       AS final_score,
                       1.0 - (embedding <=> $1) AS similarity
                FROM memories
                WHERE user_id = $2
                  AND is_active = TRUE
                  AND content_en IS NOT NULL
                  AND (1.0 - (embedding <=> $1)) >= $5
                  AND memory_type = $4
                ORDER BY final_score DESC
                LIMIT $3
                """,
                embedding,
                user_id,
                self.max_candidates,
                memory_type,
                self.min_similarity,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, content, memory_type, importance_score,
                       access_count, last_accessed_at, created_at, updated_at,
                       (1.0 - (embedding <=> $1)) * 0.5
                       + importance_score * 0.2
                       + EXP(-0.01 * EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400.0) * 0.2
                       + LEAST(access_count::float / 10.0, 1.0) * 0.1
                       AS final_score,
                       1.0 - (embedding <=> $1) AS similarity
                FROM memories
                WHERE user_id = $2
                  AND is_active = TRUE
                  AND content_en IS NOT NULL
                  AND (1.0 - (embedding <=> $1)) >= $4
                ORDER BY final_score DESC
                LIMIT $3
                """,
                embedding,
                user_id,
                self.max_candidates,
                self.min_similarity,
            )

        return [dict(r) for r in rows]

    async def _update_access_stats(self, memory_ids: List[str]) -> None:
        """
        后台更新命中记忆的访问统计（asyncio.create_task 调用，不阻塞检索返回）。
        失败只记录警告，不影响主流程。
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE memories SET
                        access_count     = access_count + 1,
                        last_accessed_at = NOW()
                    WHERE id = ANY($1::uuid[])
                    """,
                    memory_ids,
                )
        except Exception as e:
            logger.warning(f"更新 access_count 失败（不影响主流程）: {e}")

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """移除 Qwen3 thinking 模式输出的 <think>...</think> 内容，以及包裹 JSON 的 Markdown 代码块标记。
        （压测发现约 1% 的提取输出被 ```json ... ``` 包裹，导致 json.loads 失败、记忆丢失）"""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        fenced = re.match(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", text, flags=re.DOTALL)
        return fenced.group(1).strip() if fenced else text

    @staticmethod
    def _format_result(row: Dict) -> Dict[str, Any]:
        """将 DB 行格式化为统一的检索返回格式。"""
        return {
            "content": row["content"],
            "memory_type": row["memory_type"],
            "importance_score": float(row["importance_score"]),
            "final_score": round(float(row["final_score"]), 4),
            "metadata": {
                "id": str(row["id"]),
                "access_count": row["access_count"],
                "created_at": (
                    row["created_at"].isoformat() if row.get("created_at") else None
                ),
            },
        }