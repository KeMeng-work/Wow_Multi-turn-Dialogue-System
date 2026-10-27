"""
ONNX Runtime CPU模型部署 - 嵌入器 (ONNX Embedder)

基于 ONNX Runtime 的 nomic-embed-text 嵌入模型推理。
实现 LangChain Embeddings 接口，可直接替换项目中所有 OllamaEmbeddings 调用。

使用场景（在线推理路径）：
  1. main.py          → Milvus 向量检索（embed_query）
  2. long_term_memory → pgvector 记忆写入 / 检索向量化（embed_query）
  3. reflection_task  → 反思记忆向量化（embed_query）

关键后处理（nomic-embed-text 必须，否则向量空间不一致）：
  Step 1：均值池化（Mean Pooling）
          ONNX 原始输出是 last_hidden_state，shape (batch, seq_len, 768)
          需对 token 维度做加权平均（padding token 权重为 0），得到 (batch, 768)
  Step 2：L2 归一化
          将向量模长置为 1，保证余弦相似度计算正确
          与 Milvus 集合 VECTOR(768) 和 pgvector memories.embedding VECTOR(768) 严格对齐

Author: Ke Meng
Created: 2026-08-30
Version: 1.0.0
"""

import os
import logging
from typing import List

import numpy as np
import onnxruntime as ort
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class ONNXEmbedder(Embeddings):
    """
    nomic-embed-text ONNX Runtime 嵌入器。

    实现 LangChain Embeddings 接口（embed_query / embed_documents），
    可无缝替换 OllamaEmbeddings，调用方无需修改业务逻辑。

    推理链：
      tokenizer → session.run() → mean_pool → l2_normalize → 768 维向量
    """

    def __init__(
        self,
        model_path: str,
        onnx_file: str = 'onnx/model.onnx',
        providers: list = None,
        intra_op_num_threads: int = 4,
        inter_op_num_threads: int = 1,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        """
        :param model_path:           模型根目录（含 tokenizer 文件和 onnx/ 子目录）
        :param onnx_file:            相对 model_path 的 ONNX 文件路径
        :param providers:            ONNX Runtime 执行后端
        :param intra_op_num_threads: 单算子内并行线程数（建议 = CPU 核数 / 2）
        :param inter_op_num_threads: 算子间并行线程数（Transformer 顺序架构保持 1）
        :param max_length:           tokenizer 截断长度（查询和记忆均远小于 512）
        :param batch_size:           embed_documents 分批上限（embed_query 不受影响）
        """
        if providers is None:
            providers = ['CPUExecutionProvider']

        self.max_length = max_length
        self.batch_size = batch_size

        # nomic-embed-text 基于改良 BERT 架构，trust_remote_code 保证 tokenizer 正确加载
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

        onnx_path = os.path.join(model_path, onnx_file)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = intra_op_num_threads
        sess_options.inter_op_num_threads = inter_op_num_threads
        self.session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        # 动态获取输入名，兼容有 / 无 token_type_ids 的模型变体
        self._input_names = {inp.name for inp in self.session.get_inputs()}

        self._warmup()

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _warmup(self):
        """触发 ONNX Runtime 图优化，消除首次推理延迟。"""
        try:
            self.embed_query("warmup")
            logger.info("ONNXEmbedder 预热完成")
        except Exception as e:
            logger.warning(f"ONNXEmbedder 预热失败（不影响使用）: {e}")

    def _mean_pool(
        self, hidden_states: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        均值池化：对 token 维度做加权平均。

        hidden_states:  (batch, seq_len, hidden_size) ← ONNX last_hidden_state 输出
        attention_mask: (batch, seq_len)              ← padding token 处为 0
        returns:        (batch, hidden_size)
        """
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)  # (batch, seq_len, 1)
        sum_emb  = (hidden_states * mask).sum(axis=1)   # (batch, hidden_size)
        sum_mask = mask.sum(axis=1).clip(min=1e-9)      # (batch, 1)  防止除以零
        return sum_emb / sum_mask

    def _l2_normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2 归一化：将向量模长置为 1。

        保证输出向量与 Ollama nomic-embed-text 的余弦相似度空间一致，
        Milvus / pgvector 余弦检索结果不偏移。
        """
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True).clip(min=1e-9)
        return embeddings / norms

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        对一批文本做一次完整的嵌入推理。

        返回 shape: (len(texts), 768)，每行已完成均值池化 + L2 归一化。
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='np',
        )
        inputs = {
            'input_ids':      encoded['input_ids'].astype(np.int64),
            'attention_mask': encoded['attention_mask'].astype(np.int64),
        }
        # token_type_ids：BERT 系有，RoBERTa / nomic 系通常无，动态判断避免报错
        if 'token_type_ids' in self._input_names and 'token_type_ids' in encoded:
            inputs['token_type_ids'] = encoded['token_type_ids'].astype(np.int64)

        # ONNX 输出：last_hidden_state，shape (batch, seq_len, 768)
        hidden_states = self.session.run(None, inputs)[0]

        pooled     = self._mean_pool(hidden_states, encoded['attention_mask'])
        normalized = self._l2_normalize(pooled)
        return normalized  # (batch, 768)，float32

    # ── LangChain Embeddings 接口 ─────────────────────────────────────────────

    def embed_query(self, text: str) -> List[float]:
        """
        单条文本嵌入，用于查询向量化。

        调用方：main.py RetrieverTool（Milvus 向量检索）
                long_term_memory（pgvector 检索查询向量化）
        """
        return self._embed_batch([text])[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量文本嵌入，按 batch_size 分批推理，防止大批量 OOM。

        调用方：LangChain 内部（如 VectorStore.add_texts）
        """
        results: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            embeddings = self._embed_batch(batch)
            results.extend(embeddings.tolist())
        return results
