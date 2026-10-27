"""
ReAct Agent 检索工具集 (Retrieval Tool)

该模块提供面向知识库的高效检索能力，是 RAG系统的核心组件之一。它通过多路召回、融合排序和深度重排，确保从海量文档中准确定位最相关信息。

主要功能：
    1. 向量检索：基于 Milvus 2.6+ 的 IVF_RABITQ 索引，对查询嵌入在单句（cs_text）上做语义相似性搜索，召回 Top_N 单句。
    2. 全文检索：利用 Milvus 内置的 BM25 Function 对单句做全文检索，弥补纯向量检索在关键词匹配上的不足，召回 Top_N 单句。
    3. 段落级 RRF 融合（small-to-big）：每一路先把命中单句归并到所属父段落（段落名次 = 该段在本路首次出现的次序），
       再按段落做 RRF，得到最多 rrf_final_top_k 个独立父段落；同时保留每个段落在各路命中的单句。
    4. MaxP 重排：父段落整段与命中单句在一次 TEI 请求中打分，段落分 = max(整段分, 命中单句最高分)，
       避免父段落由多篇文章拼接导致相关句被稀释。
    5. 分档判定（CRAG 式，阈值在代码/配置中，与 system prompt 分离），按最高分 s1：
         - 置信  s1 ≥ tau_high：返回 score ≥ max(tau_high, alpha × s1) 的块，不超过题型上限；
         - 低置信 tau_low ≤ s1 < tau_high：≥ tau_low 的块逐块送充分性判定器（本地小模型），
           pass 的块带 low_relevance 返回；全部 fail 视为拒答；判定器超时/失败时保留这些块并标记（降级）；
         - 拒答  s1 < tau_low：返回空列表，由 Agent 层输出固定话术；
         - 重排失败：按 RRF 顺序取前 3 块（降级）。
       多跳题另受整题段落预算约束，跨次检索去重。

索引设计：
1、向量索引: IVF_RABITQ (milvus2.6 版本新索引), 支撑后续向量检索
    - 内存占用极低（最高32倍压缩）
    - 查询性能快（超越 HNSW、ivf_sq8 ）
    - 召回精度高（超越 HNSW、ivf_sq8 ）

2、全文索引: sparse字段 + BM25 function（Milvus 2.6 + 版本新功能）
    - 对 cs_text 字段的文本内容进行分词并计算 BM25稀疏向量，存入 sparse字段
    - 为 sparse 字段创建 BM25索引，检索时能计算 BM25分数，支撑后续系统全文检索
    - 支撑 BM25 全文检索，避免了引入 Elasticsearch 增加数据库维护成本

输入输出：
    - 输入：英文检索查询 `query`（模型填写）；`question_type` / `exclude_contents` / `budget_remaining`
      由 tool_node 注入（InjectedToolArg，不出现在模型可见的工具 schema 中）。
    - 输出：JSON 列表，每个元素只含 `content`；低置信档返回的块带 `low_relevance: True`；拒答时为空列表。
      向量/BM25/重排分数、分档与判定结果只写入日志（RETRIEVAL_TRACE）和 Langfuse，不返回给模型。

Author: Ke Meng
Created: 2026-01-20
Version: 1.2.0
Last Modified: 2026-10-04

变更记录：
    - 1.2.0 (2026-10-04):
                        改动1、去掉 k_min 保底（trace 显示保底块几乎全是噪声），改为双阈值分档判定
                        改动2、低置信档新增充分性判定器（SufficiencyEvaluator，参考 CRAG 检索评估器），逐块判定
                        改动3、拒答由检索层判定、Agent 层输出固定话术，不依赖 system prompt
                        改动4、trace 新增 tier / s1 / s2 / gap / 判定结果 / 阈值版本
    - 1.1.0 (2026-10-02):
                        改动1、RRF 由按单句 chunk_id 计分改为按父段落归并后计分，避免同段多句挤占候选池
                        改动2、重排改为 MaxP：整段 + 命中单句同一请求打分，段落分取最大值；重排失败降级为 RRF 排序
                        改动3、固定阈值 0.45 + 全低留 top1 改为 k_min 保底 + 相对/绝对阈值扩展 + 题型上限
                        改动4、多跳题整题段落预算 + 跨次检索去重（由 tool_node 注入参数）
                        改动5、返回结果去掉 score；新增 RETRIEVAL_TRACE 结构化日志与 Langfuse 候选明细
    - 1.0.2 (2026-08-30): BGEReranker 从 sentence-transformers 迁移至裸 ONNX Runtime（后改为 TEI HTTP）
    - 1.0.1 (2026-03-15):
                        改动1、将system prompt 有关检索工具集的内容分离到工具集的 description 中，
                              Agent在调用工具前已经获取所有工具集的自然语言描述。
                        改动2、新建支持全文检索的 Milvus 知识库集合，与长期记忆集合逻辑分离
                        改动3、修改重排后对于候选文档评分小于0.5的固定阈值判断，只设置一个最低分数（-5.0），
                              超过分数皆返回文档（设置上限Top5）
                        改动4、引入 RRF 融合算法进一步缩减候选文档数量
    - 1.0.0 (2026-01-25): 初始版本

依赖：
    - requests: 调用 TEI /rerank 接口。
    - pymilvus: Milvus 向量数据库 Python SDK。
    - langchain-core: 用于嵌入模型接口和工具创建。
    - langfuse: 检索 span 与候选明细上报。
"""

import re
import json
import time
import logging
import concurrent.futures
from typing import List, Any, Dict, Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.embeddings import Embeddings
from langchain_core.tools import InjectedToolArg
import requests
from requests.adapters import HTTPAdapter
from langfuse import observe, get_client
from src.core.ReAct_Agent.tools.base import create_tool
from src.core.resilience import EVALUATOR_BULKHEAD, RERANK_BULKHEAD


# 检索工具在线程池（64）中并发执行；requests 默认每主机仅 10 条连接，
# 超出时会"Connection pool is full, discarding connection"并反复重建 TCP 连接
_HTTP_POOL_SIZE = 64


def _pooled_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=4, pool_maxsize=_HTTP_POOL_SIZE)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


logger =logging.getLogger(__name__)

# 单句 chunk_id = {父段落 id}_s{序号}
_SENTENCE_SUFFIX = re.compile(r"_s\d+$")

# 分档名称（写入 trace，供 Langfuse 统计）
TIER_CONFIDENT = "confident"
TIER_AMBIGUOUS_PASS = "ambiguous_pass"
TIER_AMBIGUOUS_FAIL = "ambiguous_fail"
TIER_AMBIGUOUS_DEGRADED = "ambiguous_degraded"
TIER_REFUSE = "refuse"
TIER_RERANK_FAILED = "rerank_failed"

# 跨次检索去重后没有新段落时返回给模型的提示
RERANK_FAILED_TOP_N = 3   # 重排失败时按 RRF 顺序返回的段落数


def _r4(x):
    return None if x is None else round(x, 4)


NO_NEW_RESULT_NOTICE = "本次检索未发现新的文档，相关文档均已在此前的检索结果中返回，请基于已有资料作答。"


def parent_id_of(chunk_id: str) -> str:
    """由单句 chunk_id 得到父段落 id（去掉 _s{序号} 后缀）。"""
    return _SENTENCE_SUFFIX.sub("", chunk_id)


# ==================== 重排器封装（TEI HTTP，GPU 推理） ====================
class TEIReranker:
    """
    BGE Reranker via TEI (Text Embeddings Inference) HTTP API。
    将本地 ONNX CPU 推理替换为 GPU 加速的独立 HTTP 服务。
    TEI /rerank 接口：输入 query + texts，返回按 score 降序的 [{index, score}] 列表（sigmoid 0–1）。
    """

    def __init__(self, tei_url: str, timeout: int = 30, max_length: int = 1024,
                 max_client_batch_size: int = 128):
        self.tei_url = tei_url
        self.timeout = timeout
        self.max_length = max_length
        # 单次请求文本数上限，须 ≤ TEI 启动参数 --max-client-batch-size，超出会被 TEI 拒绝
        self.max_client_batch_size = max_client_batch_size
        self._session = _pooled_session()
        logger.info(f"TEIReranker 初始化完成，服务地址: {tei_url}，单次请求上限: {max_client_batch_size}")

    def score(self, query: str, texts: List[str]) -> List[float]:
        """调用 TEI /rerank 接口，返回与 texts 一一对应（按输入下标）的相关性分数。"""
        if not texts:
            return []

        payload = {
            "query": query,
            "texts": texts,
            "truncate": True,
            "truncation_direction": "Right",
        }
        # 舱壁：TEI 并发上限，满额等待超时抛 BulkheadFull → 调用方降级为 RRF 排序
        with RERANK_BULKHEAD:
            response = self._session.post(self.tei_url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        # TEI 返回: [{"index": int, "score": float}, ...] 已按 score 降序，这里还原为输入顺序
        scores: List[Optional[float]] = [None] * len(texts)
        for item in response.json():
            scores[item["index"]] = float(item["score"])
        if any(s is None for s in scores):
            raise ValueError("TEI 返回的分数数量与输入文本数量不一致")
        return scores


# ==================== 充分性判定器（低置信档） ====================
class SufficiencyEvaluator:
    """
    判断检索段落能否回答问题（参考 CRAG 的检索评估器）。
    一次请求逐块判定，返回 pass 的段落下标；超时、HTTP 错误或输出无法解析时抛异常，由调用方降级。
    后端为 OpenAI 兼容接口（默认本地 vLLM qwen3-4b），关闭思考模式并要求 JSON 输出。
    """

    SYSTEM_PROMPT = (
        "You judge whether retrieved passages can answer a question. "
        "A passage PASSES only if it explicitly states information that answers the question "
        "or is necessary evidence for the answer. Sharing the topic or keywords is NOT enough. "
        'Reply with JSON only: {"answerable": [numbers of passing passages]}; '
        'use {"answerable": []} if none pass.'
    )

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 timeout: float = 3.0, max_tokens: int = 32, passage_chars: int = 1500):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.passage_chars = passage_chars
        self._session = _pooled_session()
        self._headers = {"Authorization": f"Bearer {api_key}"}
        logger.info(f"SufficiencyEvaluator 初始化完成: {self.url} model={model} timeout={timeout}s")

    def judge(self, query: str, passages: List[str]) -> List[int]:
        """返回 pass 的段落下标（0 起）。"""
        numbered = "\n\n".join(
            f"[{i + 1}] {p[:self.passage_chars]}" for i, p in enumerate(passages)
        )
        user = f"/no_think\nQuestion: {query}\n\nPassages:\n{numbered}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        # 舱壁：4B 前台判定并发上限，满额等待超时抛 BulkheadFull → 调用方降级为保留并标记
        with EVALUATOR_BULKHEAD:
            resp = self._session.post(self.url, json=payload, headers=self._headers, timeout=self.timeout)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"] or ""
        return self.parse(text, len(passages))

    @staticmethod
    def parse(text: str, n: int) -> List[int]:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        try:
            nums = json.loads(text)["answerable"]
        except Exception:
            m = re.search(r"\[([\d,\s]*)\]", text)
            if not m:
                raise ValueError(f"判定器输出无法解析: {text[:100]}")
            nums = [int(x) for x in m.group(1).replace(" ", "").split(",") if x]
        if not isinstance(nums, list):
            raise ValueError(f"判定器输出格式错误: {text[:100]}")
        return sorted({int(x) - 1 for x in nums if 1 <= int(x) <= n})


# ==================== 段落级 RRF 融合 ====================
def fuse_by_parent(
    results_list: List[List[Dict[str, Any]]],
    k: int = 60,
    weights: List[float] = None,
) -> List[Dict[str, Any]]:
    """
    small-to-big 段落级 RRF：每一路先把单句命中归并到父段落，再按段落计 RRF 分。
    段落在某一路的名次 = 该段落在这一路结果中首次出现的次序（1 起，按段落稠密编号），
    同一段落在同一路的多个命中单句只计一次分，但全部记入 sentences 供重排使用。

    :param results_list: 每一路的单句检索结果（已按该路得分降序），元素含 id/parent_id/content/cs_text/score/rank/source
    :param k: RRF 常数，较大 k 平滑名次差异，较小 k 更强调前几名
    :param weights: 各路权重（默认全 1.0）
    :return: 按 rrf_score 降序的段落列表，元素含 parent_id/content/rrf_score/ranks/sentences
    """
    if weights is None:
        weights = [1.0] * len(results_list)

    parents: Dict[str, Dict[str, Any]] = {}

    for results, weight in zip(results_list, weights):
        parent_rank: Dict[str, int] = {}
        for hit in results:
            pid = hit["parent_id"]
            if pid not in parents:
                parents[pid] = {
                    "parent_id": pid,
                    "content": hit["content"],
                    "rrf_score": 0.0,
                    "ranks": {},
                    "sentences": {},
                }
            p = parents[pid]
            if pid not in parent_rank:
                parent_rank[pid] = len(parent_rank) + 1
                p["rrf_score"] += weight * (1.0 / (k + parent_rank[pid]))
                p["ranks"][hit["source"]] = parent_rank[pid]

            sent = p["sentences"].setdefault(hit["id"], {
                "chunk_id": hit["id"],
                "cs_text": hit.get("cs_text") or "",
                "best_rank": hit["rank"],
            })
            sent["best_rank"] = min(sent["best_rank"], hit["rank"])
            sent[f"{hit['source']}_rank"] = hit["rank"]
            sent[f"{hit['source']}_score"] = hit["score"]

    fused = sorted(parents.values(), key=lambda x: x["rrf_score"], reverse=True)
    for p in fused:
        p["sentences"] = sorted(p["sentences"].values(), key=lambda s: s["best_rank"])
    return fused


# ==================== 检索工具输入 Schema ====================
class RetrieverInput(BaseModel):
    query: str = Field(description="英文检索查询，按系统提示'通用规则'中关于knowledge_retriever 的 query 规定填写")
    # 以下参数由 tool_node 按 AgentState 注入，不出现在模型可见的工具 schema 中
    question_type: Annotated[Optional[str], InjectedToolArg] = Field(default=None, description="题型（simple/reasoning/multi_hop）")
    exclude_contents: Annotated[Optional[List[str]], InjectedToolArg] = Field(default=None, description="本题此前已返回的段落内容，跨次检索去重")
    budget_remaining: Annotated[Optional[int], InjectedToolArg] = Field(default=None, description="多跳题剩余段落预算")


# ==================== 主工具类 ====================
class RetrieverTool:
    """
    生产级检索工具：单句双路召回 → 段落级 RRF → MaxP 重排 → 分档判定（置信 / 低置信+充分性判定 / 拒答）。
    返回给模型的只有段落内容（低置信档带 low_relevance 标记），分数与分档只进日志与 Langfuse。
    """
    def __init__(
        self,
        embedding_model: Embeddings,
        milvus_collection,
        reranker: TEIReranker,
        vector_top_k: int = 30,
        bm25_top_k: int = 30,
        rrf_k: int = 60,
        rrf_final_top_k: int = 25,
        bm25_weight: float = 1.0,
        tau_high: float = 0.5,
        tau_low: float = 0.1,
        alpha: float = 0.5,
        ambiguous_max: int = 3,
        max_k: Optional[Dict[str, int]] = None,
        evaluator: Optional[SufficiencyEvaluator] = None,
        evaluator_mode: str = "ambiguous",
        enforce_refusal: bool = True,
        threshold_version: str = "",
        trace_top_n: int = 10,
        vector_search_params: dict = None,
    ):
        self.embedding_model = embedding_model
        self.collection = milvus_collection
        self.collection.load()
        self.reranker = reranker
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k
        self.rrf_final_top_k = rrf_final_top_k
        self.bm25_weight = bm25_weight
        self.tau_high = tau_high                # 置信档下限
        self.tau_low = tau_low                  # 拒答线
        self.alpha = alpha                      # 置信档相对阈值系数：块需 ≥ alpha × 最高分
        self.ambiguous_max = ambiguous_max      # 低置信档送判定器的块数上限
        self.max_k = max_k or {"simple": 5, "reasoning": 5, "multi_hop": 3}
        self.evaluator = evaluator
        self.evaluator_mode = evaluator_mode if evaluator is not None else "off"
        self.enforce_refusal = enforce_refusal  # False = 影子模式：只记录"本应拒答"，照常返回
        self.threshold_version = threshold_version
        self.trace_top_n = trace_top_n
        self.vector_search_params = vector_search_params or {
            "metric_type": "COSINE",
            "params": {"nprobe": 128}
        }

    @staticmethod
    def _parse_hits(results, source: str) -> List[Dict[str, Any]]:
        output = []
        for hits in results:  # 只有一个 query
            for rank, hit in enumerate(hits, start=1):
                chunk_id = hit.entity.get("chunk_id") or hit.entity.get("content")[:100]
                output.append({
                    "id": chunk_id,
                    "parent_id": parent_id_of(chunk_id),
                    "content": hit.entity.get("content"),
                    "cs_text": hit.entity.get("cs_text"),
                    "score": float(hit.score),
                    "rank": rank,
                    "source": source,
                })
        return output

    def _vector_search(self, query: str) -> List[Dict[str, Any]]:
        """向量检索（单句 cs_text 向量）"""
        # nomic-embed-text 查询侧前缀，与入库侧 search_document: 对齐
        query_embedding = self.embedding_model.embed_query("search_query: " + query)
        results = self.collection.search(
            data=[query_embedding],
            anns_field="vector",
            param=self.vector_search_params,
            limit=self.vector_top_k,
            output_fields=["content", "chunk_id", "cs_text"]
        )
        return self._parse_hits(results, "vector")

    def _bm25_search(self, query: str) -> List[Dict[str, Any]]:
        """BM25 全文检索，利用 Milvus 2.6+ Function 生成的 sparse 向量"""
        results = self.collection.search(
            data=[query],  # BM25 检索直接传入文本
            anns_field="sparse",  # 必须是在 schema 中用 Function 定义的稀疏字段
            param={"metric_type": "BM25"},
            limit=self.bm25_top_k,
            output_fields=["content", "chunk_id", "cs_text"]
        )
        return self._parse_hits(results, "bm25")

    def _maxp_rerank(self, query: str, candidates: List[Dict[str, Any]]) -> int:
        """
        整段 + 命中单句一次请求打分，原地写入 parent_score / sent_max / final_score，返回送 TEI 的文本数。
        单句按召回名次（两路中较好者）择优填充，总文本数不超过 TEI 单次上限；相同文本只送一次。
        """
        texts: List[str] = []
        index_of: Dict[str, int] = {}

        def add(text: str) -> Optional[int]:
            key = text.strip()
            if not key:
                return None
            if key not in index_of:
                if len(texts) >= self.reranker.max_client_batch_size:
                    return None
                index_of[key] = len(texts)
                texts.append(text)
            return index_of[key]

        for c in candidates:
            c["_parent_idx"] = add(c["content"])

        # 所有候选段落的命中单句按召回名次统一排序，名次好的优先占用剩余名额
        all_sents = sorted(
            ((s, c) for c in candidates for s in c["sentences"]),
            key=lambda x: x[0]["best_rank"],
        )
        for s, _ in all_sents:
            s["_idx"] = add(s["cs_text"])

        scores = self.reranker.score(query, texts)

        for c in candidates:
            c["parent_score"] = scores[c["_parent_idx"]] if c["_parent_idx"] is not None else None
            sent_scores = []
            for s in c["sentences"]:
                s["score"] = scores[s["_idx"]] if s.get("_idx") is not None else None
                if s["score"] is not None:
                    sent_scores.append(s["score"])
            c["sent_max"] = max(sent_scores) if sent_scores else None
            parts = [x for x in (c["parent_score"], c["sent_max"]) if x is not None]
            c["final_score"] = max(parts) if parts else 0.0
        return len(texts)

    def _trace(self, payload: Dict[str, Any]) -> None:
        """结构化检索日志（每次检索一行，便于 grep RETRIEVAL_TRACE 后离线分析）"""
        try:
            logger.info("RETRIEVAL_TRACE " + json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"RETRIEVAL_TRACE 序列化失败: {e}")

    @observe(name="knowledge_retrieval")
    def retrieve(
        self,
        query: str,
        question_type: Optional[str] = None,
        exclude_contents: Optional[List[str]] = None,
        budget_remaining: Optional[int] = None,
    ) -> Any:
        """
        主检索入口：单句双路召回 → 段落级 RRF → MaxP 重排 → 分档判定。
        返回 [{"content": ...}, ...]：置信档不带标记；低置信档（判定 pass 或判定器降级）带 "low_relevance": True；
        拒答（两路无结果 / s1 < tau_low / 判定全 fail）返回空列表；本次候选全部已在此前返回过时，返回提示字符串。
        """
        qtype = question_type if question_type in self.max_k else "simple"
        cap = self.max_k[qtype]
        if budget_remaining is not None:
            cap = min(cap, budget_remaining)
        excluded = {c.strip() for c in (exclude_contents or []) if c}
        logger.info(f"开始检索: query='{query}', question_type={qtype}, cap={cap}, 已返回段落={len(excluded)}")

        trace: Dict[str, Any] = {
            "query": query,
            "question_type": qtype,
            "params": {
                "vector_top_k": self.vector_top_k, "bm25_top_k": self.bm25_top_k,
                "rrf_k": self.rrf_k, "rrf_final_top_k": self.rrf_final_top_k,
                "tau_high": self.tau_high, "tau_low": self.tau_low, "alpha": self.alpha,
                "ambiguous_max": self.ambiguous_max, "evaluator_mode": self.evaluator_mode,
                "enforce_refusal": self.enforce_refusal,
                "cap": cap, "budget_remaining": budget_remaining,
            },
            "threshold_version": self.threshold_version,
        }

        # ── 1 + 2：双路并行召回 + 段落级 RRF（整体计时）──────────────────────
        dual_rrf_start = time.perf_counter()
        vector_results: List[Dict[str, Any]] = []
        bm25_results: List[Dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_vector = executor.submit(self._vector_search, query)
            future_bm25 = executor.submit(self._bm25_search, query)

            try:
                vector_results = future_vector.result(timeout=30)
            except concurrent.futures.TimeoutError:
                logger.error("向量检索超时")
            except Exception as e:
                logger.error(f"向量检索失败: {e}")

            try:
                bm25_results = future_bm25.result(timeout=30)
            except concurrent.futures.TimeoutError:
                logger.error("BM25 检索超时")
            except Exception as e:
                logger.error(f"BM25 检索失败: {e}")

        trace["n_vector"] = len(vector_results)
        trace["n_bm25"] = len(bm25_results)

        if not vector_results and not bm25_results:
            logger.warning("两路检索均无结果")
            dual_rrf_ms = round((time.perf_counter() - dual_rrf_start) * 1000, 1)
            trace.update({"result": "both_paths_empty", "tier": TIER_REFUSE,
                          "latency_dual_retrieval_rrf_ms": dual_rrf_ms})
            self._trace(trace)
            get_client().update_current_span(metadata={
                "retrieval_empty": True,
                "tier": TIER_REFUSE,
                "reason": "both_paths_empty",
                "latency_dual_retrieval_rrf_ms": dual_rrf_ms,
            })
            return []

        fused = fuse_by_parent(
            [vector_results, bm25_results],
            k=self.rrf_k,
            weights=[1.0, self.bm25_weight],
        )

        # 3、排除此前已返回的段落 + 精确 content 去重兜底，截取前 rrf_final_top_k 个独立段落
        candidates: List[Dict[str, Any]] = []
        seen_content: set = set()
        n_excluded = 0
        for rrf_rank, p in enumerate(fused, start=1):
            key = (p["content"] or "").strip()
            if not key:
                continue
            if key in excluded:
                n_excluded += 1
                continue
            if key in seen_content:
                continue
            seen_content.add(key)
            p["rrf_rank"] = rrf_rank
            candidates.append(p)
            if len(candidates) >= self.rrf_final_top_k:
                break
        dual_rrf_ms = round((time.perf_counter() - dual_rrf_start) * 1000, 1)
        trace["n_parents_fused"] = len(fused)
        trace["n_excluded"] = n_excluded
        trace["n_candidates"] = len(candidates)
        trace["latency_dual_retrieval_rrf_ms"] = dual_rrf_ms
        logger.info(f"段落级 RRF：{len(fused)} 个父段落，排除已返回 {n_excluded} 个，送重排 {len(candidates)} 个")

        if not candidates:
            trace["result"] = "all_excluded" if n_excluded else "no_candidates"
            self._trace(trace)
            get_client().update_current_span(metadata={
                "retrieval_empty": True,
                "reason": trace["result"],
                "latency_dual_retrieval_rrf_ms": dual_rrf_ms,
            })
            return NO_NEW_RESULT_NOTICE if n_excluded else []

        # ── 4：MaxP 重排（单独计时；失败降级为 RRF 排序）─────────────────────
        rerank_start = time.perf_counter()
        rerank_failed = False
        n_texts = 0
        try:
            n_texts = self._maxp_rerank(query, candidates)
            ranked = sorted(candidates, key=lambda c: c["final_score"], reverse=True)
        except Exception as e:
            rerank_failed = True
            logger.error(f"重排失败，降级为 RRF 排序并取前 {RERANK_FAILED_TOP_N} 个段落: {e}")
            ranked = candidates
        rerank_ms = round((time.perf_counter() - rerank_start) * 1000, 1)
        trace["rerank_failed"] = rerank_failed
        trace["rerank_texts"] = n_texts
        trace["latency_rerank_ms"] = rerank_ms

        # ── 5：分档判定 ───────────────────────────────────────────────────────
        s1 = None if rerank_failed else ranked[0]["final_score"]
        s2 = None if rerank_failed or len(ranked) < 2 else ranked[1]["final_score"]
        gap = None if s1 is None or s2 is None else s1 - s2
        decision = self._decide(query, ranked, cap, s1, rerank_failed)
        tier = decision["tier"]
        selected = decision["selected"]
        flag_low = decision["flag_low"]

        result: List[Dict[str, Any]] = []
        for c in selected:
            item = {"content": c["content"]}
            if flag_low:
                item["low_relevance"] = True
            result.append(item)

        logger.info(
            f"📦 检索输出: tier={tier}，返回 {len(result)} 个段落，s1={_r4(s1)}，gap={_r4(gap)}，"
            f"low_relevance={flag_low}，判定={decision.get('evaluator')}，重排 {rerank_ms}ms"
        )

        # ── 6：日志与 Langfuse（分数只在这里出现，不返回给模型）──────────────
        top_candidates = []
        for pos, c in enumerate(ranked[:self.trace_top_n], start=1):
            top_candidates.append({
                "pos": pos,
                "parent_id": c["parent_id"],
                "rrf_rank": c["rrf_rank"],
                "rrf_score": _r4(c["rrf_score"]),
                "vector_prank": c["ranks"].get("vector"),
                "bm25_prank": c["ranks"].get("bm25"),
                "parent_score": _r4(c.get("parent_score")),
                "sent_max": _r4(c.get("sent_max")),
                "final_score": _r4(c.get("final_score")),
                "selected": c.get("selected"),
                "preview": (c["content"] or "")[:80],
                "sentences": [
                    {
                        "chunk_id": s["chunk_id"],
                        "score": _r4(s.get("score")),
                        "vector_rank": s.get("vector_rank"),
                        "vector_score": _r4(s.get("vector_score")),
                        "bm25_rank": s.get("bm25_rank"),
                        "bm25_score": _r4(s.get("bm25_score")),
                        "text": s["cs_text"][:80],
                    }
                    for s in c["sentences"]
                ],
            })
        trace.update({
            "result": "ok",
            "tier": tier,
            "s1": _r4(s1),
            "s2": _r4(s2),
            "gap": _r4(gap),
            "cut": _r4(decision.get("cut")),
            "low_relevance": flag_low,
            "evaluator": decision.get("evaluator"),
            "would_refuse": decision.get("would_refuse", False),
            "returned": [c["parent_id"] for c in selected],
            "returned_reasons": [c.get("selected") for c in selected],
            "candidates": top_candidates,
        })
        self._trace(trace)

        get_client().update_current_span(metadata={
            "retrieval_empty": len(result) == 0,
            "tier": tier,
            "result_count": len(result),
            "s1": _r4(s1),
            "gap": _r4(gap),
            "low_relevance": flag_low,
            "evaluator": decision.get("evaluator"),
            "would_refuse": decision.get("would_refuse", False),
            "rerank_failed": rerank_failed,
            "question_type": qtype,
            "params": trace["params"],
            "threshold_version": self.threshold_version,
            "n_candidates": len(candidates),
            "n_excluded": n_excluded,
            "rerank_texts": n_texts,
            "candidates": top_candidates,
            # 核心延迟分解：两段独立可观测，便于定位瓶颈
            "latency_dual_retrieval_rrf_ms": dual_rrf_ms,   # Milvus 双路并行召回 + 段落级 RRF
            "latency_rerank_ms": rerank_ms,                  # TEI BGE Reranker GPU 推理（整段 + 单句）
        })
        return result

    def classify(self, s1: Optional[float], rerank_failed: bool = False) -> str:
        """按最高分分档（判定规则集中在此处；以后加入 gap 等信号只改这里与配置）"""
        if rerank_failed:
            return TIER_RERANK_FAILED
        if s1 >= self.tau_high:
            return TIER_CONFIDENT
        if s1 >= self.tau_low:
            return "ambiguous"
        return TIER_REFUSE

    def _evaluate(self, query: str, pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        """调用充分性判定器；返回 {"passed": [...], "info": {...}}，失败时 passed 为 None。"""
        t0 = time.perf_counter()
        try:
            idx = self.evaluator.judge(query, [c["content"] for c in pool])
            info = {"status": "ok", "n_judged": len(pool), "passed_idx": idx,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
            return {"passed": [pool[i] for i in idx], "info": info}
        except Exception as e:
            logger.error(f"充分性判定失败，降级为保留并标记: {e}")
            info = {"status": "error", "error": str(e)[:200], "n_judged": len(pool),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
            return {"passed": None, "info": info}

    def _decide(self, query: str, ranked: List[Dict[str, Any]], cap: int,
                s1: Optional[float], rerank_failed: bool) -> Dict[str, Any]:
        """按分档选出返回的段落，并给出是否整体标记 low_relevance。"""
        tier = self.classify(s1, rerank_failed)

        if tier == TIER_RERANK_FAILED:
            selected = ranked[:min(RERANK_FAILED_TOP_N, cap)]
            for c in selected:
                c["selected"] = "rrf_fallback"
            return {"tier": tier, "selected": selected, "flag_low": False}

        if tier == TIER_CONFIDENT:
            cut = max(self.tau_high, self.alpha * s1)
            selected = [c for c in ranked if c["final_score"] >= cut][:cap]
            for c in selected:
                c["selected"] = "confident"
            decision = {"tier": tier, "selected": selected, "flag_low": False, "cut": cut}
            if self.evaluator_mode == "all" and selected:
                ev = self._evaluate(query, selected)
                decision["evaluator"] = ev["info"]
                if ev["passed"] is not None:
                    decision["selected"] = ev["passed"]
                    if not ev["passed"]:
                        return self._refuse(ranked, cap, TIER_AMBIGUOUS_FAIL, decision)
            return decision

        if tier == "ambiguous":
            pool = [c for c in ranked if c["final_score"] >= self.tau_low][:min(self.ambiguous_max, cap)]
            if self.evaluator_mode == "off":
                for c in pool:
                    c["selected"] = "ambiguous_unjudged"
                return {"tier": TIER_AMBIGUOUS_DEGRADED, "selected": pool, "flag_low": True,
                        "cut": self.tau_low, "evaluator": {"status": "off"}}
            ev = self._evaluate(query, pool)
            if ev["passed"] is None:
                for c in pool:
                    c["selected"] = "ambiguous_degraded"
                return {"tier": TIER_AMBIGUOUS_DEGRADED, "selected": pool, "flag_low": True,
                        "cut": self.tau_low, "evaluator": ev["info"]}
            if ev["passed"]:
                for c in ev["passed"]:
                    c["selected"] = "ambiguous_pass"
                return {"tier": TIER_AMBIGUOUS_PASS, "selected": ev["passed"], "flag_low": True,
                        "cut": self.tau_low, "evaluator": ev["info"]}
            return self._refuse(ranked, cap, TIER_AMBIGUOUS_FAIL,
                                {"cut": self.tau_low, "evaluator": ev["info"]})

        return self._refuse(ranked, cap, TIER_REFUSE, {"cut": self.tau_low})

    def _refuse(self, ranked: List[Dict[str, Any]], cap: int, tier: str,
                extra: Dict[str, Any]) -> Dict[str, Any]:
        """拒答：强制模式返回空；影子模式记录 would_refuse 并返回前几块（带标记）。"""
        decision = {**extra, "tier": tier, "would_refuse": True}
        if self.enforce_refusal:
            decision.update({"selected": [], "flag_low": False})
            return decision
        shadow = ranked[:min(self.ambiguous_max, cap)]
        for c in shadow:
            c["selected"] = "shadow"
        decision.update({"selected": shadow, "flag_low": True})
        return decision

    def as_tool(self) -> Any:
        """转换为 LangChain 工具（question_type / exclude_contents / budget_remaining 由 tool_node 注入）"""
        return create_tool(
            name="knowledge_retriever",
            description=(
                "每次回答时首先调用此工具，检索本地知识库。\n"
                "输入：`query` 参数按系统提示'通用规则'中关于 knowledge_retriever 的 query 规定填写，即默认原样使用其中给定的英文查询。\n"
                "返回：JSON 列表，每个元素包含 `content`（文档内容），已按相关性排序和过滤，同一问题内不会重复返回此前已返回的文档。"
                "文档带有 `low_relevance: true` 时，表示这些文档与问题的相关性不确定；返回空列表表示知识库中没有相关资料。"
            ),
            args_schema=RetrieverInput,
            func=self.retrieve,
            return_direct=False,
        )
