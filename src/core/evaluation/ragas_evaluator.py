"""
RAGAS 在线评估模块

在生产流量中按采样率异步执行 RAG 质量评估，评估结果回写到 Langfuse 同一条 trace 的 score。

指标（reference-free，无需 ground truth，适合在线场景）：
  - Faithfulness       答案是否只包含检索上下文支持的信息（衡量幻觉风险）
  - AnswerRelevancy    答案是否切题（衡量跑题风险）
  - ContextPrecision   检索到的上下文是否精准（衡量检索质量）

运行方式：asyncio 后台任务，不阻塞主响应链路。
Judge LLM：qwen-max（独立于被评估模型，避免自评自判）
Embedding：本地 ONNX nomic-embed-text INT8（已有 LangChain 接口，直接复用）

Author: Ke Meng
"""

import asyncio
import logging
import math
import sys
from types import ModuleType
from typing import List, Optional

# ragas 硬编码导入了已被 langchain-community >=0.3 移除的 vertexai 模块
# 注入空壳模块让 ragas 顺利加载，我们实际不使用 Vertex AI
class _AnyAttr(ModuleType):
    def __getattr__(self, name):
        return None

for _mod in (
    "langchain_community.chat_models.vertexai",
    "langchain_community.chat_models",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = _AnyAttr(_mod)

from langchain_openai import ChatOpenAI
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langfuse import get_client

logger = logging.getLogger(__name__)


class RagasEvaluator:
    """
    RAGAS 在线评估器。

    单例使用：main.py 全局初始化一次，所有请求共享 judge LLM 和 embedding 实例。
    evaluate_and_log() 为异步方法，内部用 asyncio.to_thread 驱动同步 RAGAS evaluate()。
    """

    def __init__(self, judge_model: str, api_key: str, base_url: str, embedder):
        """
        :param judge_model: qwen-max 或其他强模型，作为评估裁判
        :param api_key:     阿里百炼 API Key
        :param base_url:    阿里百炼 OpenAI-compatible endpoint
        :param embedder:    ONNXEmbedder 实例（已实现 LangChain Embeddings 接口）
        """
        judge_llm = ChatOpenAI(
            model=judge_model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,      # 评估需要确定性，不需要创造力
            max_tokens=2048,
        )
        self._ragas_llm = LangchainLLMWrapper(judge_llm)
        self._ragas_embeddings = LangchainEmbeddingsWrapper(embedder)
        # 覆盖英文 question_generation prompt，强制生成中文问题，避免跨语言 AR 虚低
        ar_metric = AnswerRelevancy(llm=self._ragas_llm, embeddings=self._ragas_embeddings)
        ar_metric.question_generation.instruction = (
            "根据给定的答案生成一个问题，并判断答案是否具有承诺性。"
            "如果答案模糊、回避或不确定（例如"我不知道"、"不确定"），noncommittal 返回 1，否则返回 0。"
            "生成的问题必须使用中文。"
        )
        self._metrics = [
            Faithfulness(llm=self._ragas_llm),
            ar_metric,
            ContextPrecision(llm=self._ragas_llm),
        ]
        logger.info(f"RagasEvaluator 初始化完成，judge={judge_model}")

    def _evaluate_sync(
        self, question: str, answer: str, contexts: List[str]
    ) -> dict:
        """同步执行 RAGAS evaluate，由 asyncio.to_thread 在线程池中运行。"""
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
        )
        dataset = EvaluationDataset(samples=[sample])
        result = evaluate(dataset=dataset, metrics=self._metrics)
        df = result.to_pandas()
        row = df.iloc[0]
        return {
            "faithfulness":      row.get("faithfulness"),
            "answer_relevancy":  row.get("answer_relevancy"),
            "context_precision": row.get("context_precision"),
        }

    async def evaluate_and_log(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        trace_id: str,
    ) -> None:
        """
        异步评估并将分数写回 Langfuse 指定 trace。

        :param question:  用户原始问题
        :param answer:    系统最终答案
        :param contexts:  knowledge_retriever 返回的原始 chunk 文本列表
        :param trace_id:  Langfuse trace ID（从 OTel span context 获取）
        """
        if not contexts:
            logger.info("RAGAS 跳过：无检索上下文（高频缓存命中或检索为空）")
            return
        if not answer or not answer.strip():
            logger.info("RAGAS 跳过：答案为空")
            return

        try:
            logger.info(f"RAGAS 开始评估，contexts={len(contexts)} 条，trace={trace_id[:8]}...")
            scores = await asyncio.to_thread(
                self._evaluate_sync, question, answer, contexts
            )

            lf = get_client()
            for metric_name, score_value in scores.items():
                if score_value is None or (isinstance(score_value, float) and math.isnan(score_value)):
                    logger.warning(f"RAGAS {metric_name} 返回 NaN，跳过写入")
                    continue
                lf.create_score(
                    trace_id=trace_id,
                    name=metric_name,
                    value=float(score_value),
                    comment="RAGAS online evaluation",
                )
                logger.info(f"RAGAS {metric_name}: {score_value:.3f} → Langfuse ✅")

        except Exception as e:
            logger.error(f"RAGAS 评估失败（不影响主流程）: {e}", exc_info=True)
