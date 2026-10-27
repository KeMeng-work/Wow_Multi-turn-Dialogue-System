"""
消息队列 - 生产者（Producer）：
读取已处理好的知识库数据文件，向量化后推入 Redis Stream（Milvus 入库）。
知识图谱走独立的 kg_pipeline，见 producer_kg.py。
"""

import json
import os
import sys
import redis
from tqdm import tqdm
from pathlib import Path
from config.paths import paths
from config.config_path import EMBED_CONFIG, REDIS_CONFIG, MILVUS_COLLECTION
from src.core.embedder import ONNXEmbedder

# ========== 配置 ==========
REDIS_STREAM = "knowledge_pipeline"
BATCH_SIZE = 50  # 每批推送的消息数量，避免一次性加载过大
INPUT_FILE = paths.processed_data / "chunks" / "wow_ner_keywords_merged.jsonl"
INCREMENT_FILE = paths.processed_data / "chunks" / "wow_increment_330.json"

# nomic-embed-text 指令前缀：告知模型当前是文档入库，与查询侧 search_query 对齐
NOMIC_DOC_PREFIX = "search_document: "

# ONNX 嵌入模型（量化版，与线上推理路径保持一致）
embedder = ONNXEmbedder(
    model_path=EMBED_CONFIG['model_path'],
    onnx_file=EMBED_CONFIG['onnx_file'],
    providers=EMBED_CONFIG['providers'],
    intra_op_num_threads=EMBED_CONFIG['intra_op_num_threads'],
    inter_op_num_threads=EMBED_CONFIG['inter_op_num_threads'],
    max_length=EMBED_CONFIG['max_length'],
)


def connect_redis():
    """连接 Redis，测试连通性"""
    r = redis.Redis(**REDIS_CONFIG)
    try:
        r.ping()
        print("Redis 连接成功")
        return r
    except redis.ConnectionError:
        print("无法连接到 Redis，请检查服务")
        raise


def extract_keywords_text(keywords_list):
    """提取关键词文本列表（去重），兼容字符串列表和字典列表两种格式"""
    texts = []
    for kw in keywords_list:
        if isinstance(kw, str):
            if kw:
                texts.append(kw)
        elif isinstance(kw, dict):
            if kw.get("normalized_text"):
                texts.append(kw["normalized_text"])
            elif kw.get("text"):
                texts.append(kw["text"].replace(" ", "_"))
    return list(set(texts))


def extract_entities_text(entities_list):
    """提取实体文本列表（去重）"""
    texts = []
    for ent in entities_list:
        if ent.get("text"):
            texts.append(ent["text"])
    return list(set(texts))


def generate_embedding(text):
    """生成向量，使用 ONNX 量化模型，与线上 embed_query 路径完全一致"""
    return embedder.embed_query(text)


def get_checked_sentences(item) -> tuple[list[str], str]:
    """
    从 metadata.original_data.dialog 中提取所有 checked_sentence。
    返回 (individual_sentences, full_block)：
      - individual_sentences: 用于向量化的单句列表（small-to-big 小端）
      - full_block: 全部句子拼接文本，作为 content 存入 Milvus（BM25 + LLM 大端）
    """
    dialog = item.get("metadata", {}).get("original_data", {}).get("dialog", [])
    sentences = []
    for turn in dialog:
        cs = turn.get("checked_sentence", {})
        if not isinstance(cs, dict):
            continue
        for v in cs.values():
            if isinstance(v, str):
                v = v.strip()
                if v and v != "no_passages_used":
                    sentences.append(v)
            elif isinstance(v, list):
                sentences.extend(s.strip() for s in v
                                 if isinstance(s, str) and s.strip() and s.strip() != "no_passages_used")
    return sentences, " ".join(sentences)


def _ensure_redis_compatible(value):
    """将值转换为 Redis 可接受的类型（字符串、整数、浮点数）"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, (list, dict)):
        # 理论上这些应该在之前已被 json.dumps 处理，但若遗漏则转为 JSON 字符串
        return json.dumps(value)
    return value  # 字符串原样返回


def build_messages(item) -> list[dict]:
    """
    根据 item 类型构建消息列表：
    - 完整数据（含 content + metadata）：small-to-big 策略，每条 checked_sentence 生成一条消息
      vector = embed(search_document: 单句)，content = 全部句子拼接（BM25 + LLM）
    - 纯知识图谱数据（仅含 chunk_id + knowledge_graph）：保持单条消息不变
    """
    # ── KG-only 路径（无 content 字段）──────────────────────────────────────
    if "content" not in item:
        msg = {}
        if "chunk_id" in item:
            msg["chunk_id"] = item["chunk_id"]
        elif "metadata" in item and "chunk_id" in item["metadata"]:
            msg["chunk_id"] = item["metadata"]["chunk_id"]
        else:
            raise ValueError("条目缺少 chunk_id")
        msg["knowledge_graph"] = json.dumps(item.get("knowledge_graph", {}))
        msg["content"] = ""
        msg["vector"] = json.dumps([])
        for field in ["source", "document_id", "chunk_seq", "content_length", "content_hash",
                      "dialog_id", "dialog_index", "topic", "turn_count", "speaker_role",
                      "wizard_eval_score", "has_checked_evidence", "total_retrieved_passages",
                      "total_retrieved_topics", "evidence_density", "prev_chunk_id",
                      "next_chunk_id", "split_type"]:
            msg[field] = None
        msg["keywords_text"] = json.dumps([])
        msg["ner_entities_text"] = json.dumps([])
        msg["keywords_json"] = json.dumps([])
        msg["ner_entities_json"] = json.dumps([])
        msg["relations_json"] = json.dumps([])
        msg["retry_count"] = 0
        return [{k: _ensure_redis_compatible(v) for k, v in msg.items()}]

    # ── 完整数据路径：small-to-big ────────────────────────────────────────────
    metadata = item.get("metadata", {})
    base_chunk_id = metadata.get("chunk_id") or item.get("chunk_id")
    if not base_chunk_id:
        raise ValueError("条目缺少 chunk_id")

    sentences, full_block = get_checked_sentences(item)

    # 无 checked_sentences 时降级：用原始 content 整体入库（加前缀）
    if not sentences:
        sentences = [item["content"]]
        full_block = item["content"]

    keywords_list = metadata.get("keywords", [])
    entities_list = metadata.get("ner_entities", [])
    kw_text  = json.dumps(extract_keywords_text(keywords_list))
    kw_json  = json.dumps(keywords_list)
    ent_text = json.dumps(extract_entities_text(entities_list))
    ent_json = json.dumps(entities_list)
    rel_json = json.dumps(metadata.get("relations", []))

    scalar_fields = ["source", "document_id", "chunk_seq", "content_hash",
                     "dialog_id", "dialog_index", "topic", "turn_count", "speaker_role",
                     "wizard_eval_score", "has_checked_evidence", "total_retrieved_passages",
                     "total_retrieved_topics", "evidence_density", "prev_chunk_id",
                     "next_chunk_id", "split_type"]

    messages = []
    for idx, sentence in enumerate(sentences):
        msg = {}
        msg["chunk_id"] = f"{base_chunk_id}_s{idx}"
        # 向量：单句 + nomic 文档前缀（精准语义定位）
        msg["vector"] = json.dumps(generate_embedding(f"{NOMIC_DOC_PREFIX}{sentence}"))
        # content：全部 checked_sentences 拼接（返回 LLM，small-to-big 大端）
        msg["content"] = full_block
        # cs_text：当前单句 checked_sentence（BM25 细粒度检索，与向量化颗粒度对齐）
        msg["cs_text"] = sentence
        msg["knowledge_graph"] = json.dumps({})
        for field in scalar_fields:
            msg[field] = metadata.get(field)
        msg["content_length"] = len(full_block)
        msg["keywords_text"] = kw_text
        msg["keywords_json"] = kw_json
        msg["ner_entities_text"] = ent_text
        msg["ner_entities_json"] = ent_json
        msg["relations_json"] = rel_json
        msg["retry_count"] = 0
        messages.append({k: _ensure_redis_compatible(v) for k, v in msg.items()})

    return messages


def read_items(file_path):
    """
    读取文件，自动识别 JSON 数组或 JSON Lines 格式。
    返回生成器，每次 yield 一个数据项（字典）。
    """
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        # 读取第一个非空字符，判断是否为数组
        first_char = None
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first_char = ch
                break
        f.seek(0)  # 重置指针

        if first_char == '[':
            # JSON 数组格式
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield item
            else:
                raise ValueError("文件根元素是数组，但解析后不是列表")
        else:
            # JSON Lines 格式
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def push_batch(redis_client, stream_name, batch):
    """将一批消息推入 Redis Stream，MAXLEN 限制防止无限堆积"""
    for msg in batch:
        # maxlen=10000：INPUT_FILE 100条 + INCREMENT 330条，拆分后约 2000 条消息
        # 设为 10000 留足 5 倍余量，防止 consumer 短暂滞后时被裁剪
        redis_client.xadd(stream_name, msg, maxlen=10000, approximate=True)


def process_and_push(redis_client, items, data_type):
    """通用处理并推送函数"""
    batch = []
    for idx, item in enumerate(tqdm(items, desc=f"处理 {data_type}")):
        try:
            msgs = build_messages(item)
            batch.extend(msgs)
            if len(batch) >= BATCH_SIZE:
                push_batch(redis_client, REDIS_STREAM, batch)
                batch = []
        except Exception as e:
            chunk_id = item.get("chunk_id") or item.get("metadata", {}).get("chunk_id", "unknown")
            print(f"处理失败 {chunk_id}: {e}")
            continue
    if batch:
        push_batch(redis_client, REDIS_STREAM, batch)
    print(f"{data_type} 推送完成")


def check_consumer_lag(r, stream=REDIS_STREAM, group="group_milvus", warn_threshold=100):
    """检查消费者组积压量，积压过多说明 consumer 可能离线"""
    try:
        info = r.xinfo_groups(stream)
        for g in info:
            name = g.get("name") or g.get(b"name", b"").decode()
            if name == group:
                lag = g.get("lag") or g.get(b"lag", 0)
                pending = g.get("pending") or g.get(b"pending", 0)
                if lag > warn_threshold or pending > warn_threshold:
                    print(f"⚠️  消费者组 {group} 积压 lag={lag} pending={pending}，请确认 consumer 正在运行")
                    ans = input("继续入库？(y/N): ").strip().lower()
                    if ans != "y":
                        print("已取消")
                        sys.exit(1)
                else:
                    print(f"✅ 消费者组 {group} 健康 lag={lag} pending={pending}")
                return
        print(f"⚠️  消费者组 {group} 不存在，请先启动 consumer_milvus")
    except Exception as e:
        print(f"⚠️  无法检查消费者状态: {e}（如首次运行可忽略）")


def main():
    increment_only = "--increment-only" in sys.argv
    input_only     = "--input-only"     in sys.argv

    # --supplement-file <path>：只入库指定的补充文件，跳过所有其他路径
    supplement_file = None
    if "--supplement-file" in sys.argv:
        idx = sys.argv.index("--supplement-file")
        if idx + 1 < len(sys.argv):
            supplement_file = Path(sys.argv[idx + 1])

    r = connect_redis()

    # supplement 模式或 rebuild 模式跳过健康检查（环境由调用方保障）
    skip_check = supplement_file is not None or os.environ.get("SKIP_LAG_CHECK") == "1"
    if not skip_check:
        check_consumer_lag(r)

    if supplement_file is not None:
        if supplement_file.exists():
            print(f"处理补充数据文件: {supplement_file}")
            sup_items = list(read_items(supplement_file))
            process_and_push(r, sup_items, f"补充数据({supplement_file.name})")
        else:
            print(f"补充文件不存在: {supplement_file}")
        return

    # 处理原有合并文件（含 content）
    if not increment_only:
        if INPUT_FILE.exists():
            print(f"处理合并数据文件: {INPUT_FILE}")
            items = list(read_items(INPUT_FILE))
            process_and_push(r, items, "合并数据")
        else:
            print(f"合并文件不存在: {INPUT_FILE}")

    # 处理增量数据文件（--input-only 时跳过）
    if not input_only:
        if INCREMENT_FILE.exists():
            print(f"处理增量数据文件: {INCREMENT_FILE}")
            inc_items = list(read_items(INCREMENT_FILE))
            process_and_push(r, inc_items, "增量数据(330 dialogs)")
        else:
            print(f"增量文件不存在，跳过: {INCREMENT_FILE}")


if __name__ == "__main__":
    main()