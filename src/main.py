"""
系统主入口模块 (Main)

RAG + ReAct Agent 多轮对话系统核心入口。

版本记录：
    - 1.0.0 (2026-03-17): 初始版本。
    - 2.0.0 (2026-04-11): 协同 agent.py 重构：
        · 长期记忆改为字符串注入 AgentState.long_term_memory（进入 system prompt 占位符）
        · 短期记忆保持消息列表格式，直接拼入 messages 前段，边界清晰
        · ainvoke 补全所有 AgentState 必填字段（query、long_term_memory）
        · 高频缓存命中时不写入短期记忆

Author: Ke Meng
Created: 2026-03-17
Last Modified: 2026-04-11
"""

import time
import redis
import random
from datetime import datetime
import asyncio
import httpx
from typing import Optional, AsyncGenerator
from pymilvus import Collection, connections
from langchain_openai import ChatOpenAI
from src.core.reasoning_chat import ReasoningChatOpenAI
from src.core.embedder import ONNXEmbedder
from langchain_core.messages import AIMessage, HumanMessage
import opentelemetry.trace as otel_trace
from langfuse import observe, get_client, propagate_attributes, LangfuseOtelSpanAttributes
from langfuse.langchain import CallbackHandler

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========== 敏感信息观测脱敏（须在任何 Langfuse 客户端创建之前）==========
# 日志：根处理器加过滤器，f-string 拼进日志的用户输入 / 检索预览统一换成占位符
# Langfuse：客户端 mask 钩子，所有观测的 input / output / metadata 上报前脱敏（get_client() 复用此设置）
from langfuse import Langfuse
from src.core.security.pii import (
    PIILogFilter, PIIVault, contains_pii, langfuse_mask, mask_for_storage,
)
for _h in logging.getLogger().handlers:
    _h.addFilter(PIILogFilter())
Langfuse(mask=langfuse_mask)

# ========== 配置导入 ==========
from config.config_path import (
    REDIS_CONFIG,
    VLLM_CONFIG,
    EMBED_CONFIG,
    MILVUS_CONFIG,
    MILVUS_COLLECTION,
    API_CONFIG,
    NEO4J_CONFIG,
    KG_CONFIG,
    RETRIEVAL_CONFIG,
    TRANSLATOR_CONFIG,
    RERANKER_CONFIG,
    CACHE_CONFIG,
    STM_CONFIG,
    REWRITER_CONFIG,
    LTM_CONFIG,
    PG_CONFIG,
    FORGETTING_CONFIG,
    RAGAS_CONFIG,
    WEB_SEARCH_CONFIG,
    EVALUATOR_CONFIG,
    POSTPROCESS_CONFIG,
)

# ========== 模块导入 ==========
from src.core.ReAct_Agent.tools.agent import (
    create_react_agent, check_kr_param_quality, create_direct_chain, _retrieval_outcome, _evidence_level, SP_CANARY,
)
from src.core.security.events import record as record_security_event

# SP 金丝雀被触发（系统提示疑似被诱导泄露）时，整条回答替换为固定话术
CANARY_BLOCK_TEXT = "该回答因安全策略被拦截，请换个问法。"
from src.core.query_rewrite.query_rewriter import QueryRewriter, _fallback as _rewrite_fallback_result
from src.core.high_frequency_query_cache.redis_bloom import HighFreqCache, advanced_normalize
from src.core.memory_short.redis_short_memory import ShortTermMemory, stm_scope
from src.core.memory_short.compactor import Compactor
from src.core.memory_short.stm_service import STMService
from src.core.security.context import set_context
from src.core.memory_long.long_term_memory import LongTermMemory
from src.core.memory_long.recall_gate import RecallGate
from src.core.conversation_data_postgresql.conversation_data import PGMemory
from src.core.high_frequency_query_cache.rate_limiter import RateLimiter, RateLimitExceeded
from src.core.high_frequency_query_cache.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from src.core.memory_long.forgetting_task import ForgettingTask
from src.core.ReAct_Agent.tools.retriever import RetrieverTool, TEIReranker, SufficiencyEvaluator
from src.core.postprocess.detectors import detect as detect_answer_issues
from src.core.ReAct_Agent.tools.relation_verifier import RelationVerifier, create_relation_verifier
from src.core.ReAct_Agent.tools.translate import create_translator_tool
from src.core.ReAct_Agent.tools.web_search import create_web_search_tool
from src.core.evaluation.ragas_evaluator import RagasEvaluator
from src.core.resilience import REWRITE_BULKHEAD, BulkheadFull

# ========== 环境变量 ==========
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path, override=True)

# ========== 全局变量（延迟初始化） ==========
background_tasks = set()
# 后台 LLM 任务（长期记忆抽取/合并）并发上限：避免挤占 qwen3-4b 主链路（查询改写 / 充分性判定）
_background_llm_sem = asyncio.Semaphore(POSTPROCESS_CONFIG['background_llm_concurrency'])
pg_memory: Optional[PGMemory] = None
long_term_memory: Optional[LongTermMemory] = None
http_client: Optional[httpx.AsyncClient] = None
forgetting_task: Optional[ForgettingTask] = None
ragas_evaluator: Optional[RagasEvaluator] = None

# ========== 1. Redis ==========
redis_client = redis.Redis(**REDIS_CONFIG)
try:
    redis_client.ping()
    logger.info("Redis 连接成功")
except redis.ConnectionError:
    raise RuntimeError("Redis 连接失败")

# ========== 2. 嵌入模型（ONNX CPU）==========
embeddings = ONNXEmbedder(
    model_path=EMBED_CONFIG['model_path'],
    onnx_file=EMBED_CONFIG['onnx_file'],
    providers=EMBED_CONFIG['providers'],
    intra_op_num_threads=EMBED_CONFIG['intra_op_num_threads'],
    inter_op_num_threads=EMBED_CONFIG['inter_op_num_threads'],
    max_length=EMBED_CONFIG['max_length'],
)

# ========== 3. RAGAS 评估器（复用 embeddings 实例，judge LLM 独立）==========
if RAGAS_CONFIG['enabled']:
    ragas_evaluator = RagasEvaluator(
        judge_model=RAGAS_CONFIG['judge_model'],
        api_key=RAGAS_CONFIG['api_key'],
        base_url=RAGAS_CONFIG['base_url'],
        embedder=embeddings,
    )

# ========== 4. Milvus ==========
connections.connect(**MILVUS_CONFIG)
milvus_collection = Collection(MILVUS_COLLECTION)

# ========== 4. 重排器（TEI GPU HTTP）==========
reranker = TEIReranker(
    tei_url=RERANKER_CONFIG['tei_url'],
    timeout=RERANKER_CONFIG['timeout'],
    max_length=RERANKER_CONFIG['max_length'],
    max_client_batch_size=RERANKER_CONFIG['max_client_batch_size'],
)

# ========== 5. 主 LLM（云端 qwen3.5-35b-a3b，用于 ReAct 检索推理路径）==========
# ReasoningChatOpenAI：把百炼流式 delta 中的 reasoning_content 透传到 additional_kwargs（langchain-openai 原生会丢弃），前端"思考过程"依赖它
llm = ReasoningChatOpenAI(
    model=API_CONFIG['model'],
    openai_api_key=API_CONFIG['api_key'],
    openai_api_base=API_CONFIG['base_url'],
    temperature=API_CONFIG['temperature'],
    model_kwargs=API_CONFIG.get('model_kwargs', {}),
    extra_body=API_CONFIG.get('extra_body', {}),
    streaming=True,
    # 流式时请求 stream_options.include_usage，最后一个 chunk 携带 token 用量 → Langfuse 可统计成本
    stream_usage=True,
    # 默认重试 2 次；过载时重试会放大云端调用量，压测前优化改为 1（LLM_MAX_RETRIES 可调）
    max_retries=int(os.getenv('LLM_MAX_RETRIES', 1)),
)

# ========== 5b. Direct LLM（本地 qwen3-4b vLLM :8001，用于 direct 意图快速路径）==========
# direct 意图（闲聊/身份/简单问答）无需云 API，复用已运行的 qwen3-4b 实例，成本接近零
direct_llm = ChatOpenAI(
    model=VLLM_CONFIG['rewrite_model'],
    openai_api_key=VLLM_CONFIG['api_key'],
    openai_api_base=VLLM_CONFIG['rewrite_base_url'],
    temperature=0.7,
    streaming=True,
    # vLLM 未配置 reasoning parser，思考模式下 <think> 内容会混入 content 直接推送给用户
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

# ========== 6. 检索工具 ==========
# 低置信档充分性判定器（默认复用本地 qwen3-4b；EVALUATOR_MODE=off 时不创建）
sufficiency_evaluator = None
if EVALUATOR_CONFIG['mode'] != 'off':
    sufficiency_evaluator = SufficiencyEvaluator(
        base_url=EVALUATOR_CONFIG['base_url'],
        model=EVALUATOR_CONFIG['model'],
        api_key=EVALUATOR_CONFIG['api_key'],
        timeout=EVALUATOR_CONFIG['timeout'],
        max_tokens=EVALUATOR_CONFIG['max_tokens'],
        passage_chars=EVALUATOR_CONFIG['passage_chars'],
    )

retriever = RetrieverTool(
    embedding_model=embeddings,
    milvus_collection=milvus_collection,
    reranker=reranker,
    vector_top_k=RETRIEVAL_CONFIG['vector_top_k'],
    bm25_top_k=RETRIEVAL_CONFIG['bm25_top_k'],
    rrf_k=RETRIEVAL_CONFIG['rrf_k'],
    rrf_final_top_k=RETRIEVAL_CONFIG['rrf_final_top_k'],
    bm25_weight=RETRIEVAL_CONFIG.get('bm25_weight', 1.0),
    tau_high=RETRIEVAL_CONFIG['tau_high'],
    tau_low=RETRIEVAL_CONFIG['tau_low'],
    alpha=RETRIEVAL_CONFIG['alpha'],
    ambiguous_max=RETRIEVAL_CONFIG['ambiguous_max'],
    max_k=RETRIEVAL_CONFIG['max_k'],
    evaluator=sufficiency_evaluator,
    evaluator_mode=EVALUATOR_CONFIG['mode'],
    enforce_refusal=RETRIEVAL_CONFIG['enforce_refusal'],
    threshold_version=RETRIEVAL_CONFIG['threshold_version'],
    trace_top_n=RETRIEVAL_CONFIG['trace_top_n'],
    vector_search_params={"metric_type": "COSINE", "params": {"nprobe": 128}}
)
retriever_tool = retriever.as_tool()

# ========== 7. 关系验证工具（知识图谱）==========
# 验证器实例单独保留：Agent 工具、multi_hop 预取与 /tools/verify（MCP 服务）共用同一实例与 Neo4j 连接
# Neo4j 不可用时不阻断启动，调用时返回空结果
relation_verifier = create_relation_verifier(
    neo4j_url=NEO4J_CONFIG['url'],
    neo4j_user=NEO4J_CONFIG['user'],
    neo4j_password=NEO4J_CONFIG['password'],
    database=NEO4J_CONFIG['database'],
    config=KG_CONFIG,
)
verifier_tool = relation_verifier.as_tool()

# ========== 8. 翻译工具（ONNX CPU）==========
translator_tool = create_translator_tool(
    model_path=TRANSLATOR_CONFIG['model_path'],
    encoder_onnx_file=TRANSLATOR_CONFIG['encoder_onnx_file'],
    decoder_onnx_file=TRANSLATOR_CONFIG['decoder_onnx_file'],
    providers=TRANSLATOR_CONFIG['providers'],
    intra_op_num_threads=TRANSLATOR_CONFIG['intra_op_num_threads'],
    inter_op_num_threads=TRANSLATOR_CONFIG['inter_op_num_threads'],
    max_length=TRANSLATOR_CONFIG['max_length'],
)

# ========== 8b. 在线搜索工具（百度千帆）==========
web_search_tool = create_web_search_tool(
    api_key=WEB_SEARCH_CONFIG['api_key'],
    top_k=WEB_SEARCH_CONFIG['top_k'],
    max_snippet_chars=WEB_SEARCH_CONFIG['max_snippet_chars'],
    timeout=WEB_SEARCH_CONFIG['timeout'],
)

# ========== 9. Agent（只传入业务工具，不含记忆工具）==========
# 职责分离：长期记忆读取/写入由 main.py 独立负责，agent 只做推理和工具调用
tools = [retriever_tool, translator_tool]
if KG_CONFIG['enabled']:     # 关闭时作为评估对照组：不预取，Agent 也看不到该工具
    tools.insert(1, verifier_tool)
if WEB_SEARCH_CONFIG['enabled']:
    tools.append(web_search_tool)
agent_app = create_react_agent(
    llm=llm,
    tools=tools,
    multihop_budget=RETRIEVAL_CONFIG['multihop_budget'],
    refusal_text=POSTPROCESS_CONFIG['refusal_text'],
)

# ========== 9b. Direct chain（无工具，用于 direct 意图快速路径）==========
direct_chain = create_direct_chain(llm=direct_llm)

# ========== 10. 短期记忆存储 ==========
memory = ShortTermMemory(
    redis_client=redis_client,
    expire_seconds=STM_CONFIG['expire_seconds'],
    max_raw_turns=STM_CONFIG['max_raw_turns'],
)

# ========== 11. 查询改写（4B 熔断：持续失败时前台直接走原话兜底，不等超时；后台压缩只读其状态）==========
rewrite_breaker = CircuitBreaker(
    redis_client=redis_client,
    name="rewrite_4b",
    failure_threshold=0.5,
    min_samples=10,
    window_seconds=60,
    open_duration=30,
    half_open_max=2,
)
rewriter = QueryRewriter(
    memory=memory,
    stm_config=STM_CONFIG,
    vllm_base_url=VLLM_CONFIG['rewrite_base_url'],
    vllm_model=VLLM_CONFIG['rewrite_model'],
    api_key=VLLM_CONFIG['api_key'],
    rewrite_timeout=REWRITER_CONFIG['timeout'],
    max_retries=REWRITER_CONFIG['max_retries'],
    max_tokens=REWRITER_CONFIG['max_tokens'],
    temperature=REWRITER_CONFIG['temperature'],
    circuit_breaker=rewrite_breaker,
    enable_warmup=True,
)

# ========== 12. 高频缓存（键带版本命名空间与会话约束指纹）==========
cache = HighFreqCache(
    redis_client=redis_client,
    threshold=CACHE_CONFIG['threshold'],
    window_seconds=CACHE_CONFIG['window_seconds'],
    bloom_capacity=CACHE_CONFIG['bloom_capacity'],
    bloom_error_rate=CACHE_CONFIG['bloom_error_rate'],
    namespace=CACHE_CONFIG['namespace'],
)


# ========== 12b. 短期记忆编排（会话状态 / 异步压缩 / PG 恢复）==========
def _kg_canonicalize(name: str) -> Optional[dict]:
    """实体表归一：知识库已知实体映射为图谱标准名（只查内存词典，不访问 Neo4j）"""
    if not KG_CONFIG['enabled']:
        return None
    try:
        return relation_verifier.linker.canonical(name)
    except Exception:
        return None


def _pg_pool():
    return pg_memory.pool if pg_memory is not None else None


stm_compactor = Compactor(
    memory=memory,
    cfg=STM_CONFIG,
    base_url=VLLM_CONFIG['rewrite_base_url'],
    model=VLLM_CONFIG['rewrite_model'],
    api_key=VLLM_CONFIG['api_key'],
    get_http_client=lambda: http_client,
    get_pg_pool=_pg_pool,
    background_sem=_background_llm_sem,     # 与长期记忆后台任务共用 4B 后台并发上限
    circuit_breaker=rewrite_breaker,
)
stm = STMService(
    memory=memory,
    cfg=STM_CONFIG,
    compactor=stm_compactor,
    get_pg_pool=_pg_pool,
    canonicalize=_kg_canonicalize,
    background_tasks=background_tasks,
)

# ========== 12c. 长期记忆召回门（1.7B 判断是否需要个人背景，不占 4B）==========
ltm_gate = RecallGate(
    get_http_client=lambda: http_client,
    base_url=VLLM_CONFIG['extract_base_url'],
    model=VLLM_CONFIG['extract_model'],
    api_key=VLLM_CONFIG['api_key'],
    timeout=LTM_CONFIG['recall_gate_timeout'],
) if LTM_CONFIG['recall_gate_enabled'] else None


async def _recall_ltm(user_id: str, zh_query: str, en_query: str) -> tuple:
    """召回门 → 长期记忆召回；返回 (memories, 观测字段)。门用改写后的中文问题（代词已还原）"""
    t0 = time.perf_counter()
    need, src = (await ltm_gate.decide(zh_query)) if ltm_gate else (True, "off")
    gate_ms = round((time.perf_counter() - t0) * 1000, 1)
    memories = []
    if need:
        memories = await long_term_memory.retrieve_relevant(
            user_id=user_id,
            query=en_query,   # 记忆向量基于英文 content_en（nomic 对中文区分度不足），召回用英文改写
            memory_type=None,
        )
    return memories, {"ltm_gate": need, "ltm_gate_source": src, "latency_ltm_gate_ms": gate_ms}

# ========== 13. 限流器 ==========
# 阈值可通过环境变量调整（压测时临时放开全局 / 单条查询限流），默认值即线上配置
rate_limiter = RateLimiter(
    redis_client=redis_client,
    global_rate=int(os.getenv('RATE_LIMIT_GLOBAL', 5)),
    user_rate=int(os.getenv('RATE_LIMIT_USER', 20)),
    query_rate=int(os.getenv('RATE_LIMIT_QUERY', 3)),
)

# ========== 14. 熔断器 ==========
circuit_breaker = CircuitBreaker(
    redis_client=redis_client,
    name="agent",
    failure_threshold=0.5,
    min_samples=5,
    window_seconds=60,
    open_duration=30,
    half_open_max=2,
)

# ========== 15. Langfuse 客户端（后台任务独立 Trace）==========
langfuse_client = get_client()


# ==================== 辅助函数 ====================

def _format_long_term_memory(memories: list) -> str:
    """
    将长期记忆列表格式化为 system prompt 可读字符串。
    注入 AgentState.long_term_memory → 渲染进 SYSTEM_PROMPT 的 {long_term_memory} 占位符。

    示例输出：
        - 用户偏好科幻类型电视剧
        - 上次查询涉及增值税减免政策（财税[2023]1号）
    """
    if not memories:
        return "暂无长期记忆"
    return "\n".join([f"- {m['content']}" for m in memories])


def _answer_policy(state: Optional[dict]) -> dict:
    """
    根据 Agent 终态决定回答后处理策略（确定性，不依赖 system prompt）：
      - refused：检索层拒答提前结束（final_answer 已是固定话术）
      - append_notice：本题检索结果全部为低置信 → 答案末尾追加缺失声明
      - persist：是否写入高频缓存与长期记忆（拒答 / 含低置信资料的答案不写入，避免固化低质量回答）
    """
    state = state or {}
    refused = bool(state.get("short_circuit"))
    evidence = state.get("evidence_level") or "none"
    return {
        "refused": refused,
        "evidence_level": evidence,
        "append_notice": (not refused) and evidence == "low",
        "persist": (not refused) and evidence not in ("low", "mixed"),
    }


def _score_answer_issues(lf, trace_id: str, answer: str, contexts: list, policy: dict) -> dict:
    """规则检测器结果写入 Langfuse 打分（只监控，不改写答案）"""
    issues = detect_answer_issues(answer, contexts)
    if trace_id:
        try:
            for name in ("meta_phrase", "english_span", "unverified_name"):
                lf.create_score(trace_id=trace_id, name=f"pp_{name}", value=issues[name],
                                comment=str(issues["details"][name])[:300])
            lf.create_score(trace_id=trace_id, name="retrieval_refused", value=int(policy["refused"]))
        except Exception as _e:
            logger.warning(f"后处理检测打分写入失败: {_e}")
    return issues


_init_lock = asyncio.Lock()


async def init_resources() -> None:
    """
    初始化 PG 连接池、HTTP 客户端、遗忘任务与长期记忆（只执行一次）。
    原为首个请求懒加载：并发首批请求时，后到的请求会在 initialize() 尚未完成时拿到未就绪的 pg_memory。
    现由 api.py lifespan 启动阶段调用；请求路径保留调用（CLI 模式），双重检查 + 锁保证只初始化一次。
    """
    global pg_memory, long_term_memory, http_client, forgetting_task
    if long_term_memory is not None:
        return
    async with _init_lock:
        if long_term_memory is not None:
            return
        _pg = PGMemory(
            dsn=PG_CONFIG["dsn"],
            min_size=PG_CONFIG.get("pool_min_size", 2),
            max_size=PG_CONFIG.get("pool_max_size", 10),
            semaphore_limit=PG_CONFIG.get("pool_max_size", 10) * 2,   # 写入任务上限 = 连接池 × 2，随连接池联动
        )
        await _pg.initialize()
        pg_memory = _pg
        http_client = httpx.AsyncClient(timeout=120.0)
        forgetting_task = ForgettingTask(
            db_pool=pg_memory.pool,
            score_threshold=FORGETTING_CONFIG['score_threshold'],
            capacity_per_user=FORGETTING_CONFIG['capacity_per_user'],
        )
        long_term_memory = LongTermMemory(
            db_pool=pg_memory.pool,
            http_client=http_client,
            embedder=embeddings,
            extract_base_url=VLLM_CONFIG['extract_base_url'],
            extract_model=VLLM_CONFIG['extract_model'],
            extract_timeout=VLLM_CONFIG['extract_timeout'],
            merge_base_url=VLLM_CONFIG['rewrite_base_url'],
            merge_model=VLLM_CONFIG['rewrite_model'],
            merge_timeout=VLLM_CONFIG['rewrite_timeout'],
            api_key=VLLM_CONFIG['api_key'],
            top_k=LTM_CONFIG['top_k'],
            max_candidates=LTM_CONFIG['max_candidates'],
            similar_existing_limit=LTM_CONFIG['similar_existing_limit'],
            min_similarity=LTM_CONFIG['min_similarity'],
            merge_min_similarity=LTM_CONFIG['merge_min_similarity'],
            forgetting_task=forgetting_task,
        )


def load_vault(user_id: str, session_id: str) -> PIIVault:
    """会话级脱敏映射表（Redis pii:{user_id}:{session_id}，按用户隔离，TTL 与短期记忆一致）"""
    return PIIVault(redis_client, user_id, session_id)


def _rewrite_fallback(user_input: str) -> dict:
    """改写舱壁满额超时时的降级：与改写器自身失败时一致，直接使用原始输入（不更新约束与实体、不进缓存）"""
    return _rewrite_fallback_result(user_input, reason="bulkhead_full")


async def _rewrite_guarded(user_input: str, session_id: str, view: Optional[dict] = None) -> dict:
    """查询改写：舱壁限制 4B 前台并发 → 线程池执行同步改写（不阻塞事件循环）；舱壁满额超时则降级"""
    try:
        async with REWRITE_BULKHEAD:
            return await asyncio.to_thread(rewriter.rewrite, user_input, session_id, view)
    except BulkheadFull as e:
        logger.warning(f"改写降级为原始输入: {e}")
        return _rewrite_fallback(user_input)


def _history_messages(ctx: dict) -> list:
    """短期记忆近期原文 → 消息历史（预算内、已按裁剪规则处理）"""
    msgs = []
    for u, a in ctx["history"]:
        msgs += [HumanMessage(content=u), AIMessage(content=a)]
    return msgs


def _cache_admissible(rw: dict, policy: dict, memories: list, answer: str, applied: dict) -> bool:
    """
    高频缓存准入（只缓存与会话无关、证据充分的通用回答）：
      - 原问题可脱离上下文独立理解（standalone），且改写成功
      - 证据等级 high（联网 / 直答 / 低置信 / 拒答不缓存）
      - 未注入长期记忆（个性化回答会串号）、不含敏感信息占位符
      - 没有待确认的约束冲突（回答可能包含确认提问）
    """
    return bool(rw.get("ok") and rw.get("standalone") and policy["persist"]
                and policy["evidence_level"] == "high" and not memories
                and not contains_pii(answer) and not stm.has_pending_conflict(applied))


def _stm_trace_fields(view: dict, rw: dict, applied: dict, ctx: Optional[dict], cache_variant: str) -> dict:
    """短期记忆与改写的观测字段（Langfuse metadata）"""
    ev = applied.get("events") or {}
    cev = ev.get("constraints") or {}
    out = {
        "stm_source": view.get("source"),
        "stm_turn_idx": view.get("idx"),
        "rewrite_ok": bool(rw.get("ok")),
        "rewrite_fallback_reason": rw.get("fallback_reason"),
        "rewrite_standalone": bool(rw.get("standalone")),
        "rewrite_context_tokens": (rw.get("context_stats") or {}).get("tokens"),
        "rewrite_unclassified_directive": bool(rw.get("unclassified_directive")),
        "topic_shift": (ev.get("topic") or {}).get("shift"),
        "topic_id": applied["state"].get("current_topic"),
        "constraints_set": len(cev.get("set") or []),
        "constraints_conflict": len(cev.get("conflict") or []),
        "constraints_rejected": len(cev.get("rejected") or []),
        "cache_variant": bool(cache_variant),
    }
    if ctx is not None:
        st = ctx["stats"]
        out.update({"stm_context_tokens": st["tokens"], "stm_history_turns": st["history_turns"],
                    "stm_trim_steps": st["trim_steps"], "stm_constraints": st["constraints"],
                    "stm_conflicts": st["conflicts"], "stm_has_summary": st["has_summary"]})
    return out


def _spawn_pg_write(session_id: str, user_id: str, user_input: str, answer: str, metadata: dict,
                    message_id: Optional[str]) -> None:
    pg_task = asyncio.create_task(
        pg_memory.write_turn(
            session_id=session_id, user_id=user_id,
            user_input=user_input, assistant_output=answer,
            metadata=metadata, user_message_id=message_id,
        )
    )
    background_tasks.add(pg_task)
    pg_task.add_done_callback(background_tasks.discard)


async def _ltm_write_traced(
    user_id: str,
    user_input: str,
    assistant_output: str,
    origin_trace_id: str,
) -> None:
    """长期记忆后台写入，独立 Trace，通过 user_id 关联（LTM 是用户级，非会话级）。敏感信息只存占位符版本。"""
    user_input, assistant_output = mask_for_storage((user_input, assistant_output))
    lf = get_client()
    with propagate_attributes(
        user_id=user_id,
        trace_name="ltm_write",
        metadata={"origin_trace_id": origin_trace_id},
    ):
        with lf.start_as_current_observation(
            name="process_and_store",
            as_type="span",
            input={"user_input": user_input[:500]},
        ) as span:
            try:
                async with _background_llm_sem:   # 后台 LLM 并发上限
                    await long_term_memory.process_and_store(
                        user_id=user_id,
                        user_input=user_input,
                        assistant_output=assistant_output,
                    )
                span.update(output={"status": "success"})
            except Exception as e:
                span.update(level="ERROR", status_message=str(e))
                logger.error(f"LTM write traced failed: {e}")
    # 不在此处 lf.flush()：flush 是同步网络上报，会阻塞事件循环；
    # Langfuse 后台线程自动批量上报，进程退出时由 api.py lifespan 统一 flush


# ==================== 工具直调（/tools/*，供 MCP 服务使用）====================
# 只暴露只读能力，不经主模型、不读写缓存与短期 / 长期记忆；Trace 以 source=mcp 标记，与对话请求区分
TOOL_QUESTION_TYPES = ("simple", "reasoning", "multi_hop")


async def tool_retrieve(query: str, user_id: str, question_type: Optional[str] = None) -> dict:
    """
    知识库检索直调：限流 → 4B 改写（中文问题转英文检索词，知识库为英文）→ 检索 → 证据分级。
    改写使用独立会话 ID，不读取用户的对话历史；调用方未指定题型时采用改写给出的题型。
    """
    rate_limiter.check(user_id=user_id, norm_query=advanced_normalize(query))   # 超限抛 RateLimitExceeded → 429
    lf = get_client()
    with propagate_attributes(user_id=user_id, trace_name="tool_retrieve", metadata={"source": "mcp"}):
        with lf.start_as_current_observation(
            name="tool_retrieve", as_type="span", input={"query": query[:500], "question_type": question_type},
        ) as span:
            rewritten = await _rewrite_guarded(query, f"mcp-tools:{user_id}")
            en_query = rewritten.get("en_query") or query
            qtype = question_type if question_type in TOOL_QUESTION_TYPES else (rewritten.get("question_type") or "simple")
            result = await asyncio.to_thread(retriever.retrieve, en_query, qtype)
            outcome = _retrieval_outcome(result)
            passages = [
                {"content": it["content"], "low_relevance": bool(it.get("low_relevance"))}
                for it in (result if isinstance(result, list) else [])
                if isinstance(it, dict) and it.get("content")
            ]
            out = {
                "query": query,
                "en_query": en_query,
                "question_type": qtype,
                "evidence_level": _evidence_level([outcome]),   # high / low / none
                "refused": outcome == "refuse",
                "passages": passages,
            }
            span.update(output={k: v for k, v in out.items() if k != "passages"} | {"passage_count": len(passages)})
            return out


async def tool_verify(entities: list, user_id: str, max_hops: int = 3) -> dict:
    """知识图谱关系校验直调：实体名使用知识库中的英文原名（也可用中文名），不做改写（专有名词翻译易出错）"""
    rate_limiter.check(user_id=user_id, norm_query=advanced_normalize(" ".join(entities)))
    lf = get_client()
    with propagate_attributes(user_id=user_id, trace_name="tool_verify", metadata={"source": "mcp"}):
        with lf.start_as_current_observation(
            name="tool_verify", as_type="span", input={"entities": entities, "max_hops": max_hops},
        ) as span:
            out = await asyncio.to_thread(relation_verifier.verify, entities, max_hops)
            span.update(output={"stats": out.get("stats"), "not_in_graph": out.get("not_in_graph")})
            return out


def merge_graph_evidence(retrieved: list, graph: list) -> list:
    """答案依据 = 检索段落 + 图谱证据句；证据句已包含在某个检索段落中的不再单独列出（避免前端与 RAGAS 重复计算）"""
    merged = list(retrieved or [])
    for ev in graph or []:
        if ev and not any(ev in c for c in merged):
            merged.append(ev)
    return merged


async def graph_prefetch(question_type: str, intent: str, texts: list) -> tuple:
    """
    multi_hop 题知识图谱预取（HippoRAG 式：问题中的实体作为种子，一次查出多跳路径）。
    返回 (AgentState 增量字段, 预取结果)；未触发 / 未命中 / 出错时返回 ({}, None)，主流程不受影响。
    """
    if not (KG_CONFIG['enabled'] and KG_CONFIG['prefetch']) or question_type != "multi_hop" or intent == "direct":
        return {}, None
    lf = get_client()
    _t = time.perf_counter()
    with lf.start_as_current_observation(name="graph_prefetch", as_type="span", input={"texts": texts}) as span:
        try:
            result = await asyncio.wait_for(asyncio.to_thread(relation_verifier.prefetch, texts),
                                            timeout=KG_CONFIG['timeout_s'] * 3)
        except Exception as e:
            logger.warning(f"知识图谱预取失败，跳过: {type(e).__name__}: {e}")
            span.update(output={"hit": False, "error": f"{type(e).__name__}: {e}"[:200]})
            return {}, None
        ms = round((time.perf_counter() - _t) * 1000, 1)
        if not result:
            logger.info(f"🕸️ 知识图谱预取未命中（{ms}ms）")
            span.update(output={"hit": False, "ms": ms})
            return {}, None
        logger.info(f"🕸️ 知识图谱预取命中：实体 {[l['name'] for l in result['linked']]}，"
                    f"事实 {len(result['facts'])}，路径 {len(result['paths'])}（{ms}ms）")
        span.update(output={"hit": True, "ms": ms, "linked": result["linked"], "stats": result["stats"]})
        return {"graph_context": RelationVerifier.render(result),
                "graph_contexts": RelationVerifier.evidence_of(result)}, result


# ==================== 核心请求处理 ====================

@observe
async def handle_request_async(
    user_input: str,
    session_id: str,
    user_id: str = "default_user",
    role: str = "user",
    vault: Optional[PIIVault] = None,
) -> str:
    """
    异步处理用户请求，完整流程：
      限流检查 → 查询改写 → 高频缓存 → 互斥锁
      → 短期记忆加载 → 长期记忆预取 → AgentState 构造
      → 熔断器包裹 Agent 调用 → 异步记忆/PG 写入 → 缓存/短期记忆更新

    ⚠️ API 层（FastAPI/api.py）请直接 await 此函数。
       不要通过 handle_request 间接调用（会产生事件循环冲突）。
    """
    global pg_memory, long_term_memory, http_client, forgetting_task
    lf = get_client()
    req_start = time.perf_counter()
    logger.info(f"开始处理请求: user_input='{user_input}', session_id='{session_id}'")
    # 脱敏映射表：api.py 传入同一实例（展示时按角色还原需与本次脱敏编号一致）；CLI 路径在此创建
    vault = vault or load_vault(user_id, session_id)
    set_context(user_id, session_id, role, vault)
    stm_sid = stm_scope(user_id, session_id)   # 短期记忆按用户隔离：拿到别人的 session_id 也读不到其上下文

    # 标记当前 Trace 的基础维度（session + user），通过 OTel span attribute 写入
    _cur_span = otel_trace.get_current_span()
    _cur_span.set_attribute(LangfuseOtelSpanAttributes.TRACE_USER_ID, user_id)
    _cur_span.set_attribute(LangfuseOtelSpanAttributes.TRACE_SESSION_ID, session_id)
    lf.set_current_trace_io(input=user_input)

    # ── 资源初始化（api lifespan 启动时已执行；CLI 路径在此首次执行，加锁防并发重复初始化）──
    await init_resources()

    # ── Step A: 限流检查（所有业务逻辑之前）──────────────────────────────────
    norm_for_rl = advanced_normalize(user_input)
    try:
        rate_limiter.check(user_id=user_id, norm_query=norm_for_rl)
    except Exception:
        lf.update_current_span(metadata={"rate_limit": "reject"})
        raise
    # 触发时抛出 RateLimitExceeded → api.py 捕获 → HTTP 429

    # ── Step 0: 读取短期记忆（Redis 一次往返并发号本轮；Redis 缺失时从 PG 同步恢复）──
    view = await stm.prepare(user_id, session_id)
    stm_load_ms = view["load_ms"]

    # ── Step 1: 查询改写（会话状态作为上下文；同时输出 standalone / 约束 / 实体 / 话题）──
    _t = time.perf_counter()
    # 同步改写（4B HTTP）在线程池执行，避免阻塞事件循环；舱壁限制 4B 前台并发
    rewritten_dict = await _rewrite_guarded(user_input, stm_sid, view)
    rewrite_ms =round((time.perf_counter() - _t) * 1000, 1)
    en_query = rewritten_dict["en_query"]                        # agent 检索用（英文）
    zh_query = rewritten_dict["zh_query"]                        # 长期记忆检索用（中文语义更准）
    intent = rewritten_dict.get("intent", "local")               # 意图分类：local / web_search / direct
    question_type = rewritten_dict.get("question_type", "simple")  # 题型分类：simple / reasoning / multi_hop
    logger.info(f"查询改写: en='{en_query}', zh='{zh_query}', intent='{intent}', question_type='{question_type}'")
    # 改写结果落到会话状态（内存）：本轮的话题切换、约束、实体立即生效；回答后再持久化
    applied = stm.apply(view, rewritten_dict, user_input)
    cache_variant = stm.cache_variant(applied)                  # 影响答案的约束指纹，进入缓存键

    # ── Step 2: 高频缓存检查（命中也写入短期记忆，否则下一轮指代会丢失这一轮）──────
    _t = time.perf_counter()
    cached = cache.get(en_query, cache_variant)
    cache_lookup_ms = round((time.perf_counter() - _t) * 1000, 1)
    if cached is not None:
        logger.info("缓存命中，直接返回")
        stm_out = await stm.finalize(view, applied, rewritten_dict, user_input, cached, refused=False,
                                     cache_hit=True, user_id=user_id, session_id=session_id)
        lf.set_current_trace_io(output=cached)
        lf.update_current_span(metadata={
            "cache_hit": True,
            "e2e_latency_ms": round((time.perf_counter() - req_start) * 1000, 1),
            **_stm_trace_fields(view, rewritten_dict, applied, None, cache_variant), **stm_out,
        })
        _spawn_pg_write(session_id, user_id, user_input, cached,
                        {"cache_hit": True, "stm": stm.pg_metadata(view, applied, rewritten_dict, False)},
                        view["turn_id"])
        return cached

    # ── Step B: 互斥锁（防缓存击穿）─────────────────────────────────────────
    lock_acquired = False
    if cache.is_high_freq(en_query, cache_variant):
        lock_acquired = cache.try_lock(en_query, cache_variant)
        if not lock_acquired:
            logger.info("未拿到互斥锁，等待缓存写入...")
            for _ in range(3):
                await asyncio.sleep(0.3)
                cached = cache.get(en_query, cache_variant)
                if cached is not None:
                    logger.info("等待后缓存命中")
                    stm_out = await stm.finalize(view, applied, rewritten_dict, user_input, cached, refused=False,
                                                 cache_hit=True, user_id=user_id, session_id=session_id)
                    lf.update_current_span(metadata={
                        "cache_hit": True, "mutex_wait": True,
                        **_stm_trace_fields(view, rewritten_dict, applied, None, cache_variant), **stm_out,
                    })
                    _spawn_pg_write(session_id, user_id, user_input, cached,
                                    {"cache_hit": True, "mutex_wait": True,
                                     "stm": stm.pg_metadata(view, applied, rewritten_dict, False)},
                                    view["turn_id"])
                    return cached
            logger.warning("互斥锁等待超时，放行到 Agent")

    # ── Step 3: 组装短期记忆上下文（按 Agent 预算：会话记忆段 + 近期原文 + 当前问题前的约束提示）──
    stm_ctx = stm.agent_inputs(view, applied)
    short_term_msgs = _history_messages(stm_ctx)
    stm_turns = len(stm_ctx["history"])
    logger.info(f"短期记忆: {stm_turns} 轮原文，上下文约 {stm_ctx['stats']['tokens']} token，"
                f"裁剪 {stm_ctx['stats']['trim_steps'] or '无'}")

    # ── Step 4: 长期记忆（召回门判断需要个人背景才召回；格式化为字符串注入 system prompt 占位符）──
    _t = time.perf_counter()
    memories, ltm_gate_fields = await _recall_ltm(user_id, zh_query, en_query)
    ltm_retrieve_ms = round((time.perf_counter() - _t) * 1000, 1)
    long_term_memory_str = _format_long_term_memory(memories)
    ltm_count = len(memories)
    logger.info(f"长期记忆: {ltm_count} 条，预览: {long_term_memory_str[:100]}")
    lf.update_current_span(metadata={
        "stm_turns": stm_turns,
        "ltm_count": ltm_count,
        "ltm_empty": ltm_count == 0,
        **ltm_gate_fields,
    })

    # ── Step 5: 构造 AgentState ───────────────────────────────────────────────
    # messages  = 短期记忆近期原文 + 当前用户原始输入（中文，话题级 / 本问约束拼在前面）
    # en_query  通过 AgentState.query 注入 system prompt {en_query} 占位符，供 knowledge_retriever 使用
    # long_term_memory / session_memory = 格式化字符串 → 进 system prompt 占位符
    initial_messages = short_term_msgs + [HumanMessage(content=stm_ctx["question_prefix"] + user_input)]

    # ── Step 6: 熔断器包裹 Agent/Direct 调用 ──────────────────────────────────
    logger.info(f"调用路由: intent='{intent}'")
    final_answer = None
    cb_handler = CallbackHandler()  # 自动绑定当前 Trace，捕获 LLM token/延迟
    _agent_start = time.perf_counter()
    try:
        circuit_breaker.before_call()  # OPEN 状态时抛出 CircuitBreakerOpen

        if intent == "direct":
            # direct 路径：跳过 Agent ReAct 循环，直接调 LLM
            response = await direct_chain.ainvoke(
                {
                    "messages": initial_messages,
                    "current_date": datetime.now().strftime("%Y年%m月%d日"),
                    "long_term_memory": long_term_memory_str,
                    "session_memory": stm_ctx["memory"],
                },
                config={"callbacks": [cb_handler]},
            )
            final_answer = response.content.strip() or "抱歉，我无法回答这个问题。"
            step_count = 1
            tool_call_stats = {}
        else:
            # local / web_search 路径：走 ReAct Agent（multi_hop 先做知识图谱预取，未命中时为空）
            graph_state, _ = await graph_prefetch(question_type, intent, [en_query, user_input, zh_query])
            result = await agent_app.ainvoke(
                {
                    **graph_state,                             # graph_context → {graph_rule}；graph_contexts → 答案依据
                    "query": en_query,                         # AgentState.query → {en_query} 占位符
                    "user_input": user_input,                  # AgentState.user_input → web_search query 原始输入
                    "messages": initial_messages,              # 短期记忆 + 当前问题（原始中文）
                    "step_count": 0,
                    "max_steps": {"simple": 3, "reasoning": 5, "multi_hop": 7}.get(question_type, 5),
                    "final_answer": None,
                    "long_term_memory": long_term_memory_str,  # 注入 system prompt 占位符
                    "session_memory": stm_ctx["memory"],      # 会话记忆段（用户要求 / 话题摘要 / 对话事实）
                    "current_date": datetime.now().strftime("%Y年%m月%d日"),
                    "intent": intent,                          # 意图指令注入 → {intent_rule}
                    "question_type": question_type,            # 题型注入 → {question_type_rule}
                },
                config={"callbacks": [cb_handler]},
            )
            final_answer = result.get("final_answer", "抱歉，我无法回答这个问题。")
            step_count = result.get("step_count", 0)
            tool_call_stats = result.get("tool_call_stats") or {}

        policy = _answer_policy(result if intent != "direct" else None)
        if policy["append_notice"]:
            final_answer = f"{final_answer}\n\n{POSTPROCESS_CONFIG['low_evidence_notice']}"
        canary_hit = False
        if SP_CANARY in (final_answer or ""):
            # 注入防护第 4 层：系统提示疑似泄露 → 拦截整条回答，不写缓存与长期记忆
            record_security_event("canary_leak", {"path": "chat", "intent": intent})
            final_answer = CANARY_BLOCK_TEXT
            policy = {**policy, "persist": False}
            canary_hit = True

        answer_empty = not bool(final_answer and final_answer.strip())
        agent_ms = round((time.perf_counter() - _agent_start) * 1000, 1)
        logger.info(f"调用完成: {final_answer[:100]}")

        lf.set_current_trace_io(output=final_answer)
        lf.update_current_span(metadata={
            "cache_hit": False,
            "intent": intent,
            "agent_steps": step_count,
            "answer_empty": answer_empty,
            "tool_call_stats": tool_call_stats,
            "retrieval_refused": policy["refused"],
            "evidence_level": policy["evidence_level"],
            **_stm_trace_fields(view, rewritten_dict, applied, stm_ctx, cache_variant),
            # ── 全链路延迟分解 ──────────────────────────────────────
            "e2e_latency_ms":           round((time.perf_counter() - req_start) * 1000, 1),
            "latency_rewrite_ms":       rewrite_ms,
            "latency_cache_lookup_ms":  cache_lookup_ms,
            "latency_stm_load_ms":      stm_load_ms,
            "latency_ltm_retrieve_ms":  ltm_retrieve_ms,
            "latency_agent_ms":         agent_ms,
        })

        circuit_breaker.on_success()

    except CircuitBreakerOpen:
        logger.warning("熔断器断开，返回降级响应")
        lf.update_current_span(metadata={"error_type": "circuit_breaker_open"})
        if lock_acquired:
            cache.release_lock(en_query, cache_variant)
        return "系统暂时繁忙，请稍后再试。"

    except Exception as e:
        circuit_breaker.on_failure(e)
        lf.update_current_span(
            level="ERROR",
            metadata={
                "error_type": type(e).__name__,
                "error_msg": str(e)[:500],
            },
        )
        if lock_acquired:
            cache.release_lock(en_query, cache_variant)
        raise

    # ── Step 7: 异步更新长期记忆（独立 Trace，user_id 关联；拒答 / 低置信不写入）────
    origin_trace_id = lf.get_current_trace_id() or ""
    if policy["persist"]:
        ltm_task = asyncio.create_task(
            _ltm_write_traced(
                user_id=user_id,
                user_input=user_input,
                assistant_output=final_answer,
                origin_trace_id=origin_trace_id,
            )
        )
        background_tasks.add(ltm_task)
        ltm_task.add_done_callback(background_tasks.discard)

    # ── Step 8: 高频缓存准入（见 _cache_admissible）+ 短期记忆写入（原子追加 + 状态 CAS + 必要时后台压缩）
    if _cache_admissible(rewritten_dict, policy, memories, final_answer, applied):
        cache.update(en_query, final_answer, cache_variant)
    if lock_acquired:
        cache.release_lock(en_query, cache_variant)  # 写完缓存后释放锁
    stm_refused = policy["refused"] or canary_hit   # 拒答 / 被拦截的回答只记标记，不进入后续上下文
    stm_out = await stm.finalize(view, applied, rewritten_dict, user_input, final_answer, refused=stm_refused,
                                 cache_hit=False, user_id=user_id, session_id=session_id)
    lf.update_current_span(metadata=stm_out)
    logger.info(f"缓存和短期记忆已更新: {stm_out}")

    # ── Step 9: PG 异步写入全量对话数据 ─────────────────────────────────────
    _spawn_pg_write(session_id, user_id, user_input, final_answer,
                    {"cache_hit": False, "stm": stm.pg_metadata(view, applied, rewritten_dict, stm_refused)},
                    view["turn_id"])

    # ── Step 10a: 工具参数质量评分（无 LLM，规则判断，同步写入）────────────
    kr_pq_scores = (tool_call_stats.get("knowledge_retriever") or {}).get("param_quality_scores", [])
    if kr_pq_scores and origin_trace_id:
        pq_avg = sum(kr_pq_scores) / len(kr_pq_scores)
        try:
            lf.create_score(
                trace_id=origin_trace_id,
                name="tool_param_quality",
                value=pq_avg,
                comment=f"knowledge_retriever query质量（{len(kr_pq_scores)}次调用均值）",
            )
            logger.info(f"工具参数质量: {pq_avg:.2f}（{len(kr_pq_scores)}次）→ Langfuse ✅")
        except Exception as _e:
            logger.warning(f"工具参数质量写入失败: {_e}")

    # ── Step 10b: RAGAS 在线评估（采样 15%，异步后台，不阻塞响应）────────────
    # 答案依据 = 检索段落 + 知识图谱证据原文（两者同为知识库内容，均参与后处理检测与 RAGAS 评估）
    _retrieved_contexts = (
        merge_graph_evidence(result.get("retrieved_contexts"), result.get("graph_contexts"))
        if intent != "direct" else None
    )
    if intent != "direct":
        _score_answer_issues(lf, origin_trace_id, final_answer, _retrieved_contexts or [], policy)
    if (
        ragas_evaluator is not None
        and RAGAS_CONFIG['enabled']
        and _retrieved_contexts
        and random.random() < RAGAS_CONFIG['sample_rate']
    ):
        ragas_task = asyncio.create_task(
            ragas_evaluator.evaluate_and_log(
                # judge 模型同样在云端：问题与检索原文脱敏后再发出（回答本身来自主 LLM，已是占位符）
                question=mask_for_storage(user_input),
                answer=final_answer,
                contexts=mask_for_storage(list(_retrieved_contexts)),
                trace_id=origin_trace_id,
            )
        )
        background_tasks.add(ragas_task)
        ragas_task.add_done_callback(background_tasks.discard)
        logger.info("RAGAS 评估任务已提交（后台）")

    return final_answer


# ==================== 流式请求处理 ====================

async def handle_request_stream(
    user_input: str,
    session_id: str,
    user_id: str = "default_user",
    role: str = "user",
    vault: Optional[PIIVault] = None,
) -> AsyncGenerator[dict, None]:
    """
    流式处理用户请求，yield 结构化事件供 SSE 推送：
      {"type": "rewrite",   "zh_query": "...", ...}          — 查询改写结果（指代消解后的问题、意图、题型）
      {"type": "memory",    "history_turns": N, "summarized_until": M, "items": [...]}
                                                            — 带入的上下文轮数、新形成摘要覆盖到的轮次（0 表示无）、召回的长期记忆
      {"type": "thinking",  "content": "..."}               — 原生思考片段（每步实时推送）
      {"type": "tool_call", "name": "...", "status": "..."}  — 工具调用事件（start/end）
      {"type": "answer",    "content": "..."}               — 最终答案 token
      {"type": "done"}                                       — 流结束
      {"type": "error",     "code": N, "detail": "..."}     — 错误
    长短期记忆、PG 全量存储皆仅存 final_answer，不含 thinking。
    """
    global pg_memory, long_term_memory, http_client, forgetting_task
    lf = get_client()
    req_start = time.perf_counter()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    output_chunks: int = 0   # 流式 answer chunk 数，近似等于输出 token 数（1 chunk ≈ 1 token）
    logger.info(f"[stream] 开始处理: user_input='{user_input}', session_id='{session_id}'")
    # 脱敏映射表：api.py 传入同一实例（展示时按角色还原需与本次脱敏编号一致）；CLI 路径在此创建
    vault = vault or load_vault(user_id, session_id)
    set_context(user_id, session_id, role, vault)
    stm_sid = stm_scope(user_id, session_id)   # 短期记忆按用户隔离：拿到别人的 session_id 也读不到其上下文

    # ── 资源初始化（api lifespan 启动时已执行；CLI 路径在此首次执行，加锁防并发重复初始化）──
    await init_resources()

    with lf.start_as_current_observation(name="handle_request_stream", as_type="span") as stream_span:
        # 设置 Trace 基础维度（user_id / session_id 通过 OTel attribute 写入 trace）
        _cur_otel_span = otel_trace.get_current_span()
        _cur_otel_span.set_attribute(LangfuseOtelSpanAttributes.TRACE_USER_ID, user_id)
        _cur_otel_span.set_attribute(LangfuseOtelSpanAttributes.TRACE_SESSION_ID, session_id)
        lf.set_current_trace_io(input=user_input)

        # ── Step A: 限流检查 ──────────────────────────────────────────────
        norm_for_rl = advanced_normalize(user_input)
        try:
            rate_limiter.check(user_id=user_id, norm_query=norm_for_rl)
        except RateLimitExceeded as e:
            lf.update_current_span(metadata={"rate_limit": "reject", "error_type": "RateLimitExceeded"})
            yield {"type": "error", "code": 429, "detail": e.reason}
            return

        # ── Step 0: 读取短期记忆（Redis 一次往返并发号本轮；缺失时从 PG 同步恢复）──
        view = await stm.prepare(user_id, session_id)
        s_stm_load_ms = view["load_ms"]

        # ── Step 1: 查询改写 ──────────────────────────────────────────────
        _t = time.perf_counter()
        # 同步改写（4B HTTP）在线程池执行，避免阻塞事件循环；舱壁限制 4B 前台并发
        rewritten_dict = await _rewrite_guarded(user_input, stm_sid, view)
        s_rewrite_ms = round((time.perf_counter() - _t) * 1000, 1)
        en_query = rewritten_dict["en_query"]
        zh_query = rewritten_dict["zh_query"]
        intent = rewritten_dict.get("intent", "local")
        question_type = rewritten_dict.get("question_type", "simple")
        logger.info(f"[stream] 查询改写: en='{en_query}', zh='{zh_query}', intent='{intent}', question_type='{question_type}'")
        # 前端"理解问题"步骤：展示指代消解后的改写结果
        yield {"type": "rewrite", "zh_query": zh_query, "en_query": en_query,
               "intent": intent, "question_type": question_type, "rewrite_ms": s_rewrite_ms}
        # 前端处理过程（改写 / 记忆 / 工具 / 思考 / 证据 / 原文）随回答写入 PG metadata，历史会话可还原
        ui_trace: dict = {"rewrite": {"zh_query": zh_query, "en_query": en_query,
                                      "intent": intent, "question_type": question_type},
                          "thinking": "", "tools": []}
        # 改写结果落到会话状态（内存）：本轮的话题切换、约束、实体立即生效；回答后再持久化
        applied = stm.apply(view, rewritten_dict, user_input)
        cache_variant = stm.cache_variant(applied)

        # ── Step 2: 高频缓存（命中也写入短期记忆）──────────────────────────
        _t = time.perf_counter()
        cached = cache.get(en_query, cache_variant)
        s_cache_lookup_ms = round((time.perf_counter() - _t) * 1000, 1)
        if cached is not None:
            logger.info("[stream] 缓存命中")
            yield {"type": "answer", "content": cached}
            stm_out = await stm.finalize(view, applied, rewritten_dict, user_input, cached, refused=False,
                                         cache_hit=True, user_id=user_id, session_id=session_id)
            lf.set_current_trace_io(output=cached)
            lf.update_current_span(metadata={
                "cache_hit": True,
                "e2e_latency_ms": round((time.perf_counter() - req_start) * 1000, 1),
                **_stm_trace_fields(view, rewritten_dict, applied, None, cache_variant), **stm_out,
            })
            _spawn_pg_write(session_id, user_id, user_input, cached,
                            {"cache_hit": True, "trace": ui_trace,
                             "stm": stm.pg_metadata(view, applied, rewritten_dict, False)},
                            view["turn_id"])
            yield {"type": "done"}
            return

        # ── Step B: 互斥锁（防缓存击穿）─────────────────────────────────
        lock_acquired = False
        if cache.is_high_freq(en_query, cache_variant):
            lock_acquired = cache.try_lock(en_query, cache_variant)
            if not lock_acquired:
                logger.info("[stream] 未拿到互斥锁，等待缓存写入...")
                for _ in range(3):
                    await asyncio.sleep(0.3)
                    cached = cache.get(en_query, cache_variant)
                    if cached is not None:
                        yield {"type": "answer", "content": cached}
                        stm_out = await stm.finalize(view, applied, rewritten_dict, user_input, cached, refused=False,
                                                     cache_hit=True, user_id=user_id, session_id=session_id)
                        lf.set_current_trace_io(output=cached)
                        lf.update_current_span(metadata={
                            "cache_hit": True, "mutex_wait": True,
                            **_stm_trace_fields(view, rewritten_dict, applied, None, cache_variant), **stm_out,
                        })
                        _spawn_pg_write(session_id, user_id, user_input, cached,
                                        {"cache_hit": True, "mutex_wait": True, "trace": ui_trace,
                                         "stm": stm.pg_metadata(view, applied, rewritten_dict, False)},
                                        view["turn_id"])
                        yield {"type": "done"}
                        return
                logger.warning("[stream] 互斥锁等待超时，放行到 Agent")

        # ── Step 3: 组装短期记忆上下文（按 Agent 预算）────────────────────────
        stm_ctx = stm.agent_inputs(view, applied)
        short_term_msgs = _history_messages(stm_ctx)
        logger.info(f"[stream] 短期记忆: {len(stm_ctx['history'])} 轮原文，上下文约 {stm_ctx['stats']['tokens']} token，"
                    f"裁剪 {stm_ctx['stats']['trim_steps'] or '无'}")

        # ── Step 4: 长期记忆（召回门判断需要个人背景才召回）──────────────
        _t = time.perf_counter()
        memories, ltm_gate_fields = await _recall_ltm(user_id, zh_query, en_query)
        s_ltm_retrieve_ms = round((time.perf_counter() - _t) * 1000, 1)
        long_term_memory_str = _format_long_term_memory(memories)
        # 前端"记忆"步骤：本轮带入的上下文轮数与召回的长期记忆
        _mem_event = {"type": "memory", "history_turns": len(stm_ctx["history"]),
                      "summarized_until": stm.new_summary_until(view),     # 压缩后的下一轮提示一次"早期对话已形成摘要"
                      "items": [{"content": m.get("content", ""), "memory_type": m.get("memory_type", "")}
                                for m in (memories or [])][:5]}
        ui_trace["memory"] = {k: _mem_event[k] for k in ("history_turns", "summarized_until", "items")}
        yield _mem_event

        # ── Step 5: 构造 AgentState / direct 输入 ────────────────────────────
        # HumanMessage 携带原始中文（话题级 / 本问约束拼在前面）；en_query 通过 AgentState.query 注入系统提示 {en_query}
        initial_messages = short_term_msgs + [HumanMessage(content=stm_ctx["question_prefix"] + user_input)]
        initial_state = {
            "query": en_query,
            "user_input": user_input,
            "messages": initial_messages,
            "step_count": 0,
            "max_steps": {"simple": 3, "reasoning": 5, "multi_hop": 7}.get(question_type, 5),
            "final_answer": None,
            "long_term_memory": long_term_memory_str,
            "session_memory": stm_ctx["memory"],
            "current_date": datetime.now().strftime("%Y年%m月%d日"),
            "thinking_steps": [],
            "intent": intent,           # 意图指令注入 → {intent_rule}
            "question_type": question_type,  # 题型注入 → {question_type_rule}
        }

        # ── Step 6: 流式推理（按意图路由）────────────────────────────────
        accumulated_answer = ""
        stream_tool_stats: dict = {}   # {tool_name: {success, failure}}
        stream_agent_steps: int = 0
        stream_retrieved_contexts: list = []  # knowledge_retriever 返回的 chunk，供 RAGAS 评估
        stream_param_quality_scores: list = []  # knowledge_retriever query 参数质量得分
        stream_final_state: Optional[dict] = None  # Agent 图终态（short_circuit / evidence_level）
        # 前端"回答依据"标签：本轮是否查过知识库；在线搜索状态 None(未调用) / ok / empty / failed
        stream_kb_searched = False
        stream_web_status: Optional[str] = None
        canary_hit = False   # SP 金丝雀出现在思考或回答中 → 拦截整条回答，之后的输出全部丢弃
        cb_handler = CallbackHandler()  # LangGraph LLM 子调用追踪
        _agent_start = time.perf_counter()
        try:
            circuit_breaker.before_call()

            if intent == "direct":
                # direct 路径：跳过 Agent 循环，直接流式输出
                async for chunk in direct_chain.astream(
                    {
                        "messages": initial_messages,
                        "current_date": datetime.now().strftime("%Y年%m月%d日"),
                        "long_term_memory": long_term_memory_str,
                        "session_memory": stm_ctx["memory"],
                    },
                    config={"callbacks": [cb_handler]},
                ):
                    rc = chunk.additional_kwargs.get("reasoning_content", "")
                    if rc:
                        ui_trace["thinking"] += rc
                        yield {"type": "thinking", "content": rc}
                    if chunk.content:
                        _now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = _now
                        last_token_time = _now
                        output_chunks += 1
                        yield {"type": "answer", "content": chunk.content}
                        accumulated_answer += chunk.content
                stream_agent_steps = 1

            else:
                # local / web_search 路径：multi_hop 先做知识图谱预取，命中时推送工具事件（前端显示"知识图谱"）
                graph_state, graph_result = await graph_prefetch(question_type, intent, [en_query, user_input, zh_query])
                if graph_result:
                    initial_state.update(graph_state)
                    ui_trace["tools"].append("relation_verifier")
                    stream_tool_stats["relation_verifier"] = {"success": 1, "failure": 0, "prefetch": 1}
                    yield {"type": "tool_call", "name": "relation_verifier", "status": "start"}
                    yield {"type": "tool_call", "name": "relation_verifier", "status": "end"}
                # ReAct Agent astream_events
                async for event in agent_app.astream_events(
                    initial_state,
                    version="v2",
                    config={"callbacks": [cb_handler]},
                ):
                    etype = event["event"]

                    if etype == "on_chat_model_stream":
                        if canary_hit:
                            continue
                        chunk = event["data"]["chunk"]
                        rc = chunk.additional_kwargs.get("reasoning_content", "")
                        content = chunk.content if isinstance(chunk.content, str) else ""
                        # 金丝雀检查在推送之前（跨块拼接：已累积文本末尾 + 本块）
                        _probe = ui_trace["thinking"][-len(SP_CANARY):] + rc + "\x00" + accumulated_answer[-len(SP_CANARY):] + content
                        if SP_CANARY in _probe:
                            canary_hit = True
                            record_security_event("canary_leak", {"path": "chat_stream", "intent": intent})
                            ui_trace["thinking"] = ""
                            accumulated_answer = CANARY_BLOCK_TEXT
                            yield {"type": "blocked", "content": CANARY_BLOCK_TEXT}
                            continue
                        if rc:
                            ui_trace["thinking"] += rc
                            yield {"type": "thinking", "content": rc}
                        if content:
                            _now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = _now
                            last_token_time = _now
                            output_chunks += 1
                            yield {"type": "answer", "content": content}
                            accumulated_answer += content

                    elif etype == "on_chain_end" and not event.get("parent_ids"):
                        # 根图结束事件：data.output 为 Agent 终态
                        _out = event["data"].get("output")
                        if isinstance(_out, dict):
                            stream_final_state = _out

                    elif etype == "on_tool_start":
                        t_start_name = event.get("name", "tool")
                        t_input = event["data"].get("input") or {}
                        if t_start_name == "knowledge_retriever":
                            stream_kb_searched = True
                            pq = check_kr_param_quality(str(t_input.get("query", "")))
                            stream_param_quality_scores.append(pq)
                            yield {"type": "tool_call", "name": t_start_name, "status": "start"}
                        elif t_start_name == "web_search":
                            yield {"type": "tool_call", "name": t_start_name, "status": "start",
                                   "query": t_input.get("query", "")}
                        else:
                            yield {"type": "tool_call", "name": t_start_name, "status": "start"}

                    elif etype == "on_tool_end":
                        t_name = event.get("name", "tool")
                        t_error = event["data"].get("error")
                        if t_name not in stream_tool_stats:
                            stream_tool_stats[t_name] = {"success": 0, "failure": 0}
                        if t_error:
                            stream_tool_stats[t_name]["failure"] += 1
                            if t_name == "web_search":
                                stream_web_status = "failed"
                        else:
                            stream_tool_stats[t_name]["success"] += 1
                            if t_name == "knowledge_retriever":
                                output = event["data"].get("output")
                                if isinstance(output, list):
                                    stream_retrieved_contexts.extend(
                                        item["content"] for item in output
                                        if isinstance(item, dict) and item.get("content")
                                    )
                            elif t_name == "web_search":
                                # 工具 artifact：{"status": ok / empty / failed, "sources": [{url, domain, title, status, reason, ...}]}
                                # 来源逐条带安全状态（ok / cleaned / blocked），前端展示 ✅ / 🧹 / ⛔
                                _art = getattr(event["data"].get("output"), "artifact", None) or {}
                                _ws = _art.get("status") or "failed"
                                if _ws == "ok":
                                    stream_web_status = "ok"
                                else:
                                    stream_web_status = stream_web_status or _ws
                                _sources = _art.get("sources") or []
                                if _sources:
                                    urls = [x["url"] for x in _sources if x.get("status") != "blocked" and x.get("url")]
                                    ui_trace["web_urls"] = urls
                                    ui_trace["web_sources"] = _sources
                                    yield {"type": "web_sources", "urls": urls, "sources": _sources}
                        stream_agent_steps += 1
                        if t_name not in ui_trace["tools"]:
                            ui_trace["tools"].append(t_name)
                        yield {"type": "tool_call", "name": t_name, "status": "end"}

            # 知识图谱证据（预取 + Agent 调用 relation_verifier）同为知识库原文：计入后处理检测、RAGAS 与前端依据
            stream_retrieved_contexts = merge_graph_evidence(
                stream_retrieved_contexts, (stream_final_state or {}).get("graph_contexts"))

            # ── 回答后处理（确定性）：拒答固定话术 / 低置信缺失声明 ──────────
            s_policy = _answer_policy(stream_final_state if intent != "direct" else None)
            if canary_hit:
                s_policy = {**s_policy, "persist": False}
            elif s_policy["refused"]:
                _refusal = (stream_final_state or {}).get("final_answer") or POSTPROCESS_CONFIG['refusal_text']
                _chunk = ("\n\n" if accumulated_answer.strip() else "") + _refusal
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                last_token_time = time.perf_counter()
                yield {"type": "answer", "content": _chunk}
                accumulated_answer += _chunk
            elif s_policy["append_notice"] and accumulated_answer.strip():
                _chunk = "\n\n" + POSTPROCESS_CONFIG['low_evidence_notice']
                yield {"type": "answer", "content": _chunk}
                accumulated_answer += _chunk
            if intent != "direct":
                yield {"type": "retrieval_meta",
                       "refused": s_policy["refused"],
                       "evidence_level": s_policy["evidence_level"],
                       "kb_searched": stream_kb_searched,
                       "web_status": stream_web_status,
                       "retrieval_outcomes": (stream_final_state or {}).get("retrieval_outcomes") or []}

            circuit_breaker.on_success()

        except CircuitBreakerOpen:
            if lock_acquired:
                cache.release_lock(en_query, cache_variant)
            lf.update_current_span(metadata={"error_type": "circuit_breaker_open"})
            yield {"type": "error", "code": 503, "detail": "系统暂时繁忙，请稍后再试。"}
            return

        except (asyncio.CancelledError, GeneratorExit):
            # 客户端提前断开（断连取消 or aclose）：记录已积累的部分输出后重新抛出
            # 不能在此 yield，只做同步记录；re-raise 让生成器正常关闭
            if lock_acquired:
                cache.release_lock(en_query, cache_variant)
            _partial = accumulated_answer.strip()
            if _partial:
                lf.set_current_trace_io(output=_partial)
            lf.update_current_span(metadata={
                "cancelled": True,
                "output_chunks": output_chunks,
                "partial_answer_len": len(_partial),
            })
            # 不 flush：过载时断连增多，同步上报会阻塞事件循环；由后台线程批量上报
            raise

        except Exception as e:
            circuit_breaker.on_failure(e)
            if lock_acquired:
                cache.release_lock(en_query, cache_variant)
            lf.update_current_span(
                level="ERROR",
                metadata={"error_type": type(e).__name__, "error_msg": str(e)[:500]},
            )
            logger.error(f"[stream] Agent 异常: {e}")
            yield {"type": "error", "code": 500, "detail": str(e)}
            return

        s_agent_ms = round((time.perf_counter() - _agent_start) * 1000, 1)
        stream_agent_steps += 1  # 最后一步：LLM 生成最终答案（无工具调用）
        final_answer = accumulated_answer.strip() or "抱歉，无法获得有效答案，请稍后重试。"
        answer_empty = not bool(accumulated_answer.strip())
        _end = time.perf_counter()
        ttft_ms = round((first_token_time - req_start) * 1000, 1) if first_token_time else None
        # TPOT = (最后一个 token 时间 - 第一个 token 时间) / 输出 chunk 数
        tpot_ms = (
            round((last_token_time - first_token_time) * 1000 / output_chunks, 2)
            if (first_token_time and last_token_time and output_chunks > 1)
            else None
        )
        e2e_ms = round((_end - req_start) * 1000, 1)
        logger.info(
            f"[stream] 完成，答案长度: {len(final_answer)}, "
            f"TTFT={ttft_ms}ms, TPOT={tpot_ms}ms/token, E2E={e2e_ms}ms"
        )

        # 记录所有流式指标到当前 span
        lf.set_current_trace_io(output=final_answer)
        lf.update_current_span(metadata={
            "cache_hit": False,
            "streaming": True,
            "intent": intent,
            "answer_empty": answer_empty,
            "agent_steps": stream_agent_steps,
            "tool_call_stats": stream_tool_stats,
            "retrieval_refused": s_policy["refused"],
            "evidence_level": s_policy["evidence_level"],
            "stm_turns": len(stm_ctx["history"]),
            **_stm_trace_fields(view, rewritten_dict, applied, stm_ctx, cache_variant),
            "ltm_count": len(memories),
            "ltm_empty": len(memories) == 0,
            **ltm_gate_fields,
            # ── 流式延迟指标 ──────────────────────────────────────────
            "ttft_ms":                  ttft_ms,
            "tpot_ms":                  tpot_ms,
            "output_chunks":            output_chunks,
            "e2e_latency_ms":           e2e_ms,
            # ── 全链路各阶段延迟 ──────────────────────────────────────
            "latency_rewrite_ms":       s_rewrite_ms,
            "latency_cache_lookup_ms":  s_cache_lookup_ms,
            "latency_stm_load_ms":      s_stm_load_ms,
            "latency_ltm_retrieve_ms":  s_ltm_retrieve_ms,
            "latency_agent_ms":         s_agent_ms,
        })

        # ── Step 7: 异步更新长期记忆（独立 Trace，user_id 关联；拒答 / 低置信不写入）
        if s_policy["persist"]:
            ltm_task = asyncio.create_task(
                _ltm_write_traced(
                    user_id=user_id,
                    user_input=user_input,
                    assistant_output=final_answer,
                    origin_trace_id=lf.get_current_trace_id() or "",
                )
            )
            background_tasks.add(ltm_task)
            ltm_task.add_done_callback(background_tasks.discard)

        # ── Step 8: 缓存（准入见 _cache_admissible）+ 短期记忆写入（仅存 final_answer，不含 thinking）
        if _cache_admissible(rewritten_dict, s_policy, memories, final_answer, applied):
            cache.update(en_query, final_answer, cache_variant)
        if lock_acquired:
            cache.release_lock(en_query, cache_variant)
        stm_refused = s_policy["refused"] or canary_hit   # 拒答 / 被拦截的回答只记标记，不进入后续上下文
        stm_out = await stm.finalize(view, applied, rewritten_dict, user_input, final_answer, refused=stm_refused,
                                     cache_hit=False, user_id=user_id, session_id=session_id)
        lf.update_current_span(metadata=stm_out)

        # ── Step 9: PG 全量对话写入（仅存 final_answer，不含 thinking）──
        _spawn_pg_write(session_id, user_id, user_input, final_answer,
                        {"cache_hit": False, "streaming": True, "trace": {
                            **ui_trace,
                            "thinking": ui_trace["thinking"][:8000],
                            "evidence_level": s_policy["evidence_level"],
                            "kb_searched": stream_kb_searched,
                            "web_status": stream_web_status,
                            "contexts": stream_retrieved_contexts[:8],
                        }, "stm": stm.pg_metadata(view, applied, rewritten_dict, stm_refused)},
                        view["turn_id"])

        # ── Step 10a: 工具参数质量评分（流式路径）────────────────────────
        s_trace_id = lf.get_current_trace_id() or ""
        if stream_param_quality_scores and s_trace_id:
            s_pq_avg = sum(stream_param_quality_scores) / len(stream_param_quality_scores)
            try:
                lf.create_score(
                    trace_id=s_trace_id,
                    name="tool_param_quality",
                    value=s_pq_avg,
                    comment=f"knowledge_retriever query质量（{len(stream_param_quality_scores)}次调用均值）",
                )
                logger.info(f"[stream] 工具参数质量: {s_pq_avg:.2f}（{len(stream_param_quality_scores)}次）→ Langfuse ✅")
            except Exception as _e:
                logger.warning(f"[stream] 工具参数质量写入失败: {_e}")

        # ── Step 10a': 回答规则检测（只监控，写入 Langfuse 打分）──────────
        if intent != "direct":
            _score_answer_issues(lf, s_trace_id, final_answer, stream_retrieved_contexts, s_policy)

        # ── Step 10b: RAGAS 在线评估（采样 15%，异步后台，不阻塞响应）────
        if (
            ragas_evaluator is not None
            and RAGAS_CONFIG['enabled']
            and stream_retrieved_contexts
            and random.random() < RAGAS_CONFIG['sample_rate']
        ):
            ragas_task = asyncio.create_task(
                ragas_evaluator.evaluate_and_log(
                    # judge 模型同样在云端：问题与检索原文脱敏后再发出（回答本身来自主 LLM，已是占位符）
                    question=mask_for_storage(user_input),
                    answer=final_answer,
                    contexts=mask_for_storage(list(stream_retrieved_contexts)),
                    trace_id=s_trace_id,
                )
            )
            background_tasks.add(ragas_task)
            ragas_task.add_done_callback(background_tasks.discard)
            logger.info("RAGAS 评估任务已提交（后台，流式路径）")

        if stream_retrieved_contexts:
            yield {"type": "contexts", "data": stream_retrieved_contexts}
        yield {"type": "done"}


# ==================== CLI 测试用同步包装器 ====================

def handle_request(
    user_input: str,
    session_id: str,
    user_id: str = "default_user"
) -> str:
    """
    同步包装器，仅用于命令行测试（python main.py）。

    ⚠️ 不要在 FastAPI / api.py 中调用此函数。
       API 层已有运行中的事件循环，asyncio.new_event_loop() 会与之冲突。
       API 层请直接：result = await handle_request_async(...)
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            handle_request_async(user_input, session_id, user_id)
        )
        # 等待后台任务（最多 10 秒），防止进程退出前数据丢失
        if background_tasks:
            done, pending = loop.run_until_complete(
                asyncio.wait(background_tasks, timeout=10)
            )
            if pending:
                logger.warning(f"{len(pending)} 个后台任务未在 10 秒内完成")
        return result
    finally:
        if http_client is not None:
            loop.run_until_complete(http_client.aclose())
        if pg_memory is not None:
            loop.run_until_complete(pg_memory.close())
        loop.close()


# ==================== CLI 测试入口 ====================

if __name__ == "__main__":
    test_query = "有什么好看的科幻电视剧推荐吗？"
    test_session = "test_001"
    test_user = "test_01"
    answer = handle_request(test_query, test_session, test_user)
    logger.info(f"最终答案: {answer}")