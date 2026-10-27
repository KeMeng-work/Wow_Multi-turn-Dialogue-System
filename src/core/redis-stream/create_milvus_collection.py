"""
Milvus 知识库集合索引创建（Create Milvus Collection）
独立运行一次，用于创建 Milvus 知识库集合和索引。
同时支撑向量检索和全文检索，避免了引入 Elasticsearch 增加数据库维护成本

索引设计：
- 1、向量索引: IVF_RABITQ (milvus2.6 版本新索引), 支撑后续向量检索
    - 内存占用极低（最高32倍压缩）
    - 查询性能快（超越 HNSW、ivf_sq8 ）
    - 召回精度高（超越 HNSW、ivf_sq8 ）

- 2、标量索引: 倒排索引, 支撑后续全文检索

- 3、位图索引：存储空间极小, 查询速度快

- 4、sparse字段 + BM25 function（Milvus 2.6 + 版本新功能）
    - 对 content字段的文本内容进行分词并计算 BM25稀疏向量，存入 sparse字段
    - 为 sparse 字段创建 BM25索引，检索时能计算 BM25分数，支撑后续系统全文检索
    - 支撑 BM25 全文检索，避免了引入 Elasticsearch 增加数据库维护成本
    - 添加了集合迁移代码，在当前目录同级 migrate_milvus.py中
    - 已通过测试用例 migrate_milvus_test.py，足以支撑后续 Agent系统的 BM25 全文检索。

"""
from pymilvus import MilvusClient, DataType, Function, FunctionType

# ========== 配置 ==========
MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "wow_knowledge_base_v2"
VECTOR_DIM = 768


def create_schema():
    """定义集合的 schema"""
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)

    # 主键
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=200, is_primary=True)
    # 向量
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    # 内容（多个 checked_sentences 拼接，供 LLM 阅读，small-to-big 大端）
    schema.add_field(
        "content",
        DataType.VARCHAR,
        max_length=65535,
    )
    # 单个 checked_sentence 文本（BM25 细粒度检索字段，与向量化颗粒度对齐）
    schema.add_field(
        "cs_text",
        DataType.VARCHAR,
        max_length=2000,
        enable_analyzer=True,
        enable_match=True,
    )
    # BM25 稀疏向量字段（由 BM25 Function 自动填充，不需要手动写入）
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

    # BM25 Function：将 cs_text 单句文本转换为 sparse 稀疏向量（细粒度匹配）
    schema.add_function(Function(
        name="bm25",
        input_field_names=["cs_text"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    ))
    # 标量字段
    schema.add_field("source", DataType.VARCHAR, max_length=50)
    schema.add_field("document_id", DataType.VARCHAR, max_length=50)
    schema.add_field("chunk_seq", DataType.INT64)
    schema.add_field("content_length", DataType.INT64)
    schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
    schema.add_field("dialog_id", DataType.VARCHAR, max_length=50)
    schema.add_field("dialog_index", DataType.INT64)
    schema.add_field("topic", DataType.VARCHAR, max_length=100)
    schema.add_field("turn_count", DataType.INT64)
    schema.add_field("speaker_role", DataType.VARCHAR, max_length=50)
    schema.add_field("wizard_eval_score", DataType.INT64)
    schema.add_field("has_checked_evidence", DataType.BOOL)
    schema.add_field("total_retrieved_passages", DataType.INT64)
    schema.add_field("total_retrieved_topics", DataType.INT64)
    schema.add_field("evidence_density", DataType.FLOAT)
    schema.add_field("prev_chunk_id", DataType.VARCHAR, max_length=200)
    schema.add_field("next_chunk_id", DataType.VARCHAR, max_length=200)
    schema.add_field("split_type", DataType.VARCHAR, max_length=20)
    # 数组字段
    schema.add_field(
        "keywords_text",
        DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_length=100,
        max_capacity=100,
    )
    schema.add_field(
        "ner_entities_text",
        DataType.ARRAY,
        element_type=DataType.VARCHAR,
        max_length=100,
        max_capacity=100,
    )
    # JSON 备份字段
    schema.add_field("keywords_json", DataType.JSON)
    schema.add_field("ner_entities_json", DataType.JSON)
    schema.add_field("relations_json", DataType.JSON)

    return schema


def create_index_params():
    """准备索引参数"""
    index_params = MilvusClient.prepare_index_params()

    # 向量索引：IVF_RABITQ + refine
    index_params.add_index(
        field_name="vector",
        index_type="IVF_RABITQ",
        metric_type="COSINE",
        params={"nlist": 1024, "refine": True, "refine_type": "SQ8"},
    )

    # 标量倒排索引
    scalar_fields = [
        "source",
        "document_id",
        "chunk_seq",
        "dialog_id",
        "topic",
        "speaker_role",
        "wizard_eval_score",
        "total_retrieved_passages",
    ]
    for f in scalar_fields:
        index_params.add_index(f, "INVERTED", index_name=f"idx_{f}")

    # 数组倒排索引
    index_params.add_index(
        "keywords_text", "INVERTED", index_name="idx_keywords_text"
    )
    index_params.add_index(
        "ner_entities_text", "INVERTED", index_name="idx_ner_entities_text"
    )

    # 位图索引（低基数字段）
    index_params.add_index(
        "has_checked_evidence", "BITMAP", index_name="bitmap_has_checked_evidence"
    )
    index_params.add_index("split_type", "BITMAP", index_name="bitmap_split_type")

    # BM25 稀疏向量索引（支撑全文检索）
    index_params.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"bm25_k1": 1.2, "bm25_b": 0.75},
        index_name="idx_sparse_bm25",
    )

    return index_params


def main():
    client = MilvusClient(uri=MILVUS_URI)

    # 删除已有集合（开发时使用，生产请谨慎）
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"已删除现有集合: {COLLECTION_NAME}")

    # schema 和 index_params 必须同时传入 create_collection
    # BM25 Function 只有在建集合时一并注册才能生效，分开调用 create_index 不会注册 Function
    schema = create_schema()
    index_params = create_index_params()
    client.create_collection(COLLECTION_NAME, schema=schema, index_params=index_params)
    print(f"集合 {COLLECTION_NAME} 创建成功（含 BM25 Function 和全部索引）")

    # 加载集合（查询前需要）
    client.load_collection(COLLECTION_NAME)
    print("集合已加载")


if __name__ == "__main__":
    main()
