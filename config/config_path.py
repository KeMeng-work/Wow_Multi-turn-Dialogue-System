# config.py
import os
from dotenv import load_dotenv
load_dotenv()
from config.paths import paths


# ==================== Redis 配置 ====================
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6380)),
    'db': int(os.getenv('REDIS_DB', 0)),
    'decode_responses': True,
    'socket_connect_timeout': 5,
    'socket_timeout': 5,
    'password': os.getenv('REDIS_PASSWORD'),
}

REDIS_PREFIX = {
    'session': 'session:',
    'counter': 'cnt:',
    'cache': 'cache:',
    'bloom': 'bf:',
    'pref': 'pref:',
    'token': 'token:',
}


# ==================== Milvus 配置 ====================
MILVUS_CONFIG = {
    'uri': os.getenv('MILVUS_URI', 'http://localhost:19530'),
}

MILVUS_COLLECTION = os.getenv('MILVUS_COLLECTION', 'wow_knowledge_base_v2')


# ==================== vLLM 配置（替换 Ollama，OpenAI 兼容接口）====================
# qwen3-4b 承担：查询改写 / 记忆合并决策 / 反思生成（端口 8001）
# qwen3-1.7b 承担：长期记忆提取（端口 8002）
# 两个 vLLM 实例分别部署，各自独占 GPU 显存分区
VLLM_CONFIG = {
    # qwen3-4b（端口 8001）：查询改写 / 记忆合并决策 / 反思生成
    'rewrite_base_url': os.getenv('VLLM_REWRITE_URL', 'http://localhost:8001/v1'),
    'rewrite_model':    os.getenv('VLLM_REWRITE_MODEL', 'qwen3-4b'),
    # qwen3-1.7b（端口 8002）：长期记忆提取
    'extract_base_url': os.getenv('VLLM_EXTRACT_URL', 'http://localhost:8002/v1'),
    'extract_model':    os.getenv('VLLM_EXTRACT_MODEL', 'qwen3-1.7b'),
    'api_key':          os.getenv('VLLM_API_KEY', 'EMPTY'),
    # 任务级超时：A10 AWQ 推理，各留 3 倍余量
    'rewrite_timeout':    int(os.getenv('VLLM_REWRITE_TIMEOUT', 30)),    # 改写/合并决策 ~150 token → 实际 1-2s
    'extract_timeout':    int(os.getenv('VLLM_EXTRACT_TIMEOUT', 30)),    # 记忆提取 ~200 token → 实际 1-2s
    'reflection_timeout': int(os.getenv('VLLM_REFLECTION_TIMEOUT', 60)), # 反思生成 ~500 token → 实际 5-8s
}


# ==================== LLM API 配置（qwen3.5-35b-a3b，云端阿里百炼）====================
API_CONFIG = {
    # 压测时可通过环境变量指向替身服务（loadtest/mock_llm.py）；不设置时为百炼线上配置
    'model': os.getenv('LLM_MODEL', 'qwen3.5-35b-a3b'),
    'api_key': os.getenv('OPENAI_API_KEY'),
    'base_url': os.getenv('LLM_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
    'temperature': 0.3,  # ReAct Agent需要稳定工具调用和事实推理，避免温度过高格式错误、推理不忠于检索结果
    'model_kwargs': {},
    'extra_body': {'enable_thinking': True},  # 启用原生思考过程，reasoning_content 独立字段，content 为干净答案
}


# ==================== Neo4j 配置 ====================
NEO4J_CONFIG = {
    'url': os.getenv('NEO4J_URL', 'bolt://localhost:7687'),
    'user': os.getenv('NEO4J_USER', 'neo4j'),
    'password': os.getenv('NEO4J_PASSWORD', 'NEO4J_PASSWORD'),
    'database': os.getenv('NEO4J_DATABASE', 'neo4j'),
}

# ==================== PostgreSQL 配置 ====================
PG_CONFIG = {
    "dsn": (
        f"postgresql://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
        f"@{os.getenv('PG_HOST', 'localhost')}:{os.getenv('PG_PORT', '5432')}"
        f"/{os.getenv('PG_DB')}"
    ),
    "pool_min_size": 2,
    "pool_max_size": int(os.getenv("PG_POOL_MAX_SIZE", 30)),  # 压测前 10；对话记录 + 长期记忆共用，PG max_connections 默认 100
}
DATABASE_URL = PG_CONFIG['dsn']
PG_POOL_MIN_SIZE = PG_CONFIG['pool_min_size']
PG_POOL_MAX_SIZE = PG_CONFIG['pool_max_size']


# ==================== ONNX 嵌入模型配置 ====================
# 在线推理路径：用户提问向量化 / 长期记忆提取向量化 / 长期记忆检索 pgvector 向量化（保持同纬度）
# 输出维度：768（与 Milvus 集合和 pgvector memories 表 VECTOR(768) 严格对齐）
EMBED_CONFIG = {
    'model_path':           str(paths.nomic_embed_model),
    'onnx_file':            'onnx/model_int8.onnx',
    'providers':            ['CPUExecutionProvider'],
    'intra_op_num_threads': int(os.getenv('EMBED_THREADS', 2)),  # 单次推理内并行线程数（8 核服务器）；压测前为 4，并发下多个推理同时运行时 4 线程会争抢 CPU，改为 2
    'inter_op_num_threads': int(os.getenv('EMBED_INTER_THREADS', 1)),  # transformer模型标准设置
    'max_length':           int(os.getenv('EMBED_MAX_LENGTH', 512)),  # 超过512token的文本自动截断（保守值，可下调至256）
    'batch_size':           int(os.getenv('EMBED_BATCH_SIZE', 32)),  # 上限，实际运行中为单独向量转换
}


# ==================== 重排器配置（TEI HTTP服务，GPU部署）====================
# bge-reranker-v2-m3 FP16，由 tei-reranker 容器提供，通过 /rerank 接口调用
RERANKER_CONFIG = {
    'tei_url':         os.getenv('TEI_RERANKER_URL', 'http://tei-reranker:80/rerank'),
    'timeout':         int(os.getenv('RERANKER_TIMEOUT', 30)),
    'max_length':      int(os.getenv('RERANKER_MAX_LENGTH', 1024)),
    # 单次 /rerank 请求的文本上限，须 ≤ TEI 启动参数 --max-client-batch-size（docker-compose 中为 128）
    'max_client_batch_size': int(os.getenv('RERANKER_MAX_CLIENT_BATCH_SIZE', 128)),
}


# ==================== 翻译工具配置（ONNX Runtime，CPU部署，MarianMT seq2seq）====================
# seq2seq 导出为两个文件：encoder + decoder_merged（含 past_key_values）
TRANSLATOR_CONFIG = {
    'model_path':           str(paths.translator_model),
    'encoder_onnx_file':    'onnx/encoder_model.onnx',
    'decoder_onnx_file':    'onnx/decoder_model_int8.onnx',
    'providers':            ['CPUExecutionProvider'],
    'intra_op_num_threads': int(os.getenv('TRANSLATOR_THREADS', 2)),
    'inter_op_num_threads': int(os.getenv('TRANSLATOR_INTER_THREADS', 1)),
    'max_length':           int(os.getenv('TRANSLATOR_MAX_LENGTH', 512)),
}


# ==================== 检索工具参数 ====================
RETRIEVAL_CONFIG = {
    # 召回：两路各取前 N 个单句（cs_text），再按父段落归并
    'vector_top_k':    int(os.getenv('VECTOR_TOP_K', 30)),
    'bm25_top_k':      int(os.getenv('BM25_TOP_K', 30)),
    'rrf_k':           int(os.getenv('RRF_K', 60)),
    'rrf_final_top_k': int(os.getenv('RRF_FINAL_TOP_K', 25)), # 段落级 RRF 后送入重排的独立父段落上限
    # BM25 在 RRF 融合时的权重系数（向量默认 1.0）；cs_text 优化后两路精度对齐，权重对等
    'bm25_weight':     float(os.getenv('BM25_WEIGHT', 1.0)),
    # 分档判定（重排分为 sigmoid 0–1，按最高分 s1 判定）：
    #   s1 ≥ tau_high          → 置信：返回 score ≥ max(tau_high, alpha × s1) 的块，不超过题型上限
    #   tau_low ≤ s1 < tau_high → 低置信：≥ tau_low 的块（≤ ambiguous_max）逐块送充分性判定器，pass 的块带 low_relevance 返回
    #   s1 < tau_low 或判定全 fail → 拒答
    # 阈值与重排模型版本绑定，更换模型/精度/TEI 版本须重新标定并更新 threshold_version
    'tau_high':        float(os.getenv('RETRIEVAL_TAU_HIGH', 0.3)),
    'tau_low':         float(os.getenv('RETRIEVAL_TAU_LOW', 0.1)),
    'alpha':           float(os.getenv('RETRIEVAL_ALPHA', 0.5)),
    'ambiguous_max':   int(os.getenv('RETRIEVAL_AMBIGUOUS_MAX', 3)),
    'enforce_refusal': os.getenv('RETRIEVAL_ENFORCE_REFUSAL', 'true').lower() == 'true',  # false = 影子模式：只记录不拒答
    'threshold_version': os.getenv('RETRIEVAL_THRESHOLD_VERSION', 'bge-reranker-v2-m3_fp16_tei_2026-10'),
    'max_k': {
        'simple':    int(os.getenv('RETRIEVAL_MAX_K_SIMPLE', 5)),
        'reasoning': int(os.getenv('RETRIEVAL_MAX_K_REASONING', 5)),
        'multi_hop': int(os.getenv('RETRIEVAL_MAX_K_MULTI_HOP', 3)),     # 多跳单次调用上限
    },
    'multihop_budget': int(os.getenv('RETRIEVAL_MULTIHOP_BUDGET', 12)),  # 多跳整题不同段落总预算
    'trace_top_n':     int(os.getenv('RETRIEVAL_TRACE_TOP_N', 10)),      # 日志/Langfuse 记录的候选段落数
}

# ==================== 充分性判定器（低置信档，CRAG 式检索评估）====================
# 默认复用本地 qwen3-4b（:8001）；切换到 1.7B 或云端模型只需改环境变量
EVALUATOR_CONFIG = {
    'mode':          os.getenv('EVALUATOR_MODE', 'ambiguous'),   # off / ambiguous / all
    'base_url':      os.getenv('EVALUATOR_BASE_URL', os.getenv('VLLM_REWRITE_URL', 'http://localhost:8001/v1')),
    'model':         os.getenv('EVALUATOR_MODEL', os.getenv('VLLM_REWRITE_MODEL', 'qwen3-4b')),
    'api_key':       os.getenv('EVALUATOR_API_KEY', os.getenv('VLLM_API_KEY', 'EMPTY')),
    'timeout':       float(os.getenv('EVALUATOR_TIMEOUT', 3)),
    'max_tokens':    int(os.getenv('EVALUATOR_MAX_TOKENS', 32)),
    'passage_chars': int(os.getenv('EVALUATOR_PASSAGE_CHARS', 1500)),
}

# ==================== 回答后处理配置 ====================
POSTPROCESS_CONFIG = {
    # 检索层拒答时直接返回的固定话术（不经过主模型）
    'refusal_text': os.getenv(
        'REFUSAL_TEXT',
        '知识库中没有找到与这个问题相关的资料，暂时无法回答。可以换个问法，或补充更具体的人名、作品名等信息后再试。'),
    # 本题全部检索结果均为低置信时，在答案末尾追加的声明
    'low_evidence_notice': os.getenv(
        'LOW_EVIDENCE_NOTICE', '（说明：以上回答所依据的知识库资料与问题的相关性较低，仅供参考。）'),
    # 后台 LLM 任务（长期记忆抽取/合并）并发上限，避免挤占 qwen3-4b 主链路（改写/判定）
    'background_llm_concurrency': int(os.getenv('BACKGROUND_LLM_CONCURRENCY', 2)),
}


# ==================== 高频缓存配置 ====================
CACHE_CONFIG = {
    'threshold':       int(os.getenv('CACHE_THRESHOLD', 5)),
    'window_seconds':  int(os.getenv('CACHE_WINDOW', 1800)),
    'bloom_capacity':  int(os.getenv('BLOOM_CAPACITY', 1000000)),
    'bloom_error_rate': float(os.getenv('BLOOM_ERROR_RATE', 0.01)),
    # 缓存键版本命名空间：知识库 / 提示词 / 主模型任一变化时，旧缓存自然失效（键与布隆过滤器均带命名空间）
    # 知识库重建或系统提示词改动后，递增 CACHE_KB_VERSION / CACHE_PROMPT_VERSION
    'namespace': os.getenv('CACHE_NAMESPACE') or ":".join([
        os.getenv('CACHE_KB_VERSION', 'kb1'),
        os.getenv('CACHE_PROMPT_VERSION', 'p1'),
        os.getenv('LLM_MODEL', 'qwen3.5-35b-a3b'),
    ]),
}


# ==================== 短期记忆配置 ====================
MEMORY_CONFIG = {
    'max_history_turns': int(os.getenv('MAX_HISTORY_TURNS', 5)),
    'expire_seconds':    int(os.getenv('MEMORY_EXPIRE', 86400)),
}

# ==================== 短期记忆（会话状态）配置 ====================
# 设计文档：docs/short_term_memory_design.md。token 数为保守估算（中文按 1 字 1 token、其余按 3 字符 1 token）
STM_CONFIG = {
    'expire_seconds':        MEMORY_CONFIG['expire_seconds'],
    'max_raw_turns':         int(os.getenv('STM_MAX_RAW_TURNS', 12)),        # Redis 保留的原文轮数上限（超出由压缩与截断兜底）
    'rewriter_budget':       int(os.getenv('STM_REWRITER_BUDGET', 1400)),    # 改写器记忆部分总预算
    'rewriter_recent_budget': int(os.getenv('STM_REWRITER_RECENT_BUDGET', 800)),  # 其中近期轮次部分
    'agent_budget':          int(os.getenv('STM_AGENT_BUDGET', 3000)),       # Agent 记忆部分总预算（不含工具结果）
    'compact_trigger_ratio': float(os.getenv('STM_COMPACT_TRIGGER', 0.7)),   # 未压缩原文超过 Agent 预算的比例时触发异步压缩
    'compact_keep_ratio':    float(os.getenv('STM_COMPACT_KEEP', 0.4)),      # 压缩后保留的未压缩原文比例（避免每轮都触发）
    'compact_keep_turns':    int(os.getenv('STM_COMPACT_KEEP_TURNS', 6)),     # 压缩后保留的未压缩原文轮数上限（短回答时按轮数触发，见 compactor）
    'rebuild_every':         int(os.getenv('STM_REBUILD_EVERY', 3)),         # 每累计 N 次压缩，用 PG 原文重建该话题摘要
    'max_constraints':       15,
    'max_other_constraints': 3,
    'max_topics':            5,                                               # 保留完整摘要的话题数
    'max_entities':          30,
    'max_facts':             20,
    'max_outlines':          5,
    'summary_budget':        300,                                             # 单个话题摘要上限
    'conflict_ttl_turns':    2,                                               # 冲突约束未被确认时，N 轮后失效
    'pg_restore_turns':      int(os.getenv('STM_PG_RESTORE_TURNS', 12)),    # Redis 缺失时从 PG 同步恢复的轮数；与 max_raw_turns 一致，
                                                                              # 否则水位线之后、最近 N 轮之前的未摘要轮次会在恢复后缺失
    'compact_timeout_s':     float(os.getenv('STM_COMPACT_TIMEOUT', 20)),
    'compact_backlog_max':   int(os.getenv('STM_COMPACT_BACKLOG', 30)),      # 待执行压缩任务超过该值时丢弃新任务（背压）
}

# ==================== 长期记忆配置 ====================
LTM_CONFIG = {
    'top_k':                 int(os.getenv('LTM_TOP_K', 3)),
    'max_candidates':        int(os.getenv('LTM_MAX_CANDIDATES', 10)),
    'similar_existing_limit': int(os.getenv('LTM_SIMILAR_LIMIT', 3)),
    # 召回相关性门槛：英文记忆 content_en 与英文改写 en_query 的余弦相似度（nomic，带 search_document/search_query 前缀）。
    # scripts/calibrate_ltm_threshold.py 实测（16 相关 / 16 不相关）：中文两组重叠无法区分；英文+前缀在 0.38 时相关保留 88%、不相关放过 12%（0.36：94% / 19%；0.42：75% / 0%）
    # 2026-10-26 上调至 0.42：线上无关记忆以 0.388 / 0.39 越过 0.38 被召回（"英超兴趣"→"丹麦地理位置"）；
    # 0.42 处不相关放过 0%、相关保留 75%（宁可少召回，不可误召回；误召回还会挡住缓存写入）
    'min_similarity':        float(os.getenv('LTM_MIN_SIMILARITY', 0.42)),
    # 召回门（门 1）：1.7B 按改写后的问题判断"回答是否需要用户个人背景"，不需要则跳过召回（src/core/memory_long/recall_gate.py）
    'recall_gate_enabled':   os.getenv('LTM_RECALL_GATE', 'true').lower() == 'true',
    'recall_gate_timeout':   float(os.getenv('LTM_RECALL_GATE_TIMEOUT', 2.0)),
    # 合并候选门槛：新旧记忆（均为 search_document 前缀英文）余弦相似度，低于此值直接新增、不送 4B 合并。
    # 2026-10-10 实测：同一属性（职业细化、同一爱好换说法、过敏同义）0.80–0.92；不同话题（职业 vs 英超等）0.61–0.73
    'merge_min_similarity':  float(os.getenv('LTM_MERGE_MIN_SIMILARITY', 0.78)),
}

# ==================== 长期记忆 - 遗忘机制配置 ====================
FORGETTING_CONFIG = {
    'score_threshold':     float(os.getenv('FORGET_SCORE_THRESHOLD', 0.1)),
    'capacity_per_user':   int(os.getenv('FORGET_CAPACITY_PER_USER', 200)),   # 只存用户自述的个人信息后，单用户实际几十条
    'decay_lambda_default': float(os.getenv('FORGET_DECAY_LAMBDA', 0.01)),
    'episodic_decay_lambda': float(os.getenv('FORGET_EPISODIC_LAMBDA', 0.03)),
    'score_decay_enabled': os.getenv('FORGET_SCORE_DECAY_ENABLED', 'true').lower() == 'true',
    'capacity_enabled':    os.getenv('FORGET_CAPACITY_ENABLED', 'true').lower() == 'true',
}

# ==================== 长期记忆 - 反思机制配置 ====================
REFLECTION_CONFIG = {
    'trigger_count':          int(os.getenv('REFLECTION_TRIGGER_COUNT', 20)),
    'interval_days':          int(os.getenv('REFLECTION_INTERVAL_DAYS', 7)),
    'min_memories':           int(os.getenv('REFLECTION_MIN_MEMORIES', 5)),
    'max_memories':           int(os.getenv('REFLECTION_MAX_MEMORIES', 20)),
    'importance_threshold':   float(os.getenv('REFLECTION_IMPORTANCE_THRESHOLD', 0.7)),
    'reflection_importance':  float(os.getenv('REFLECTION_IMPORTANCE', 0.9)),
    'model': VLLM_CONFIG['rewrite_model'],  # 反思用 qwen3-4b，与改写模型共用同一 vLLM 实例
}

# ==================== RAGAS 在线评估配置 ====================
RAGAS_CONFIG = {
    'enabled':      os.getenv('RAGAS_ENABLED', 'true').lower() == 'true',
    'sample_rate':  float(os.getenv('RAGAS_SAMPLE_RATE', 0.15)),   # 15% 生产流量采样
    'judge_model':  os.getenv('RAGAS_JUDGE_MODEL', 'qwen-max'),    # 独立 judge LLM，避免自评自判
    'api_key':      os.getenv('OPENAI_API_KEY'),                    # 与主 LLM 共用同一阿里百炼 key
    'base_url':     'https://dashscope.aliyuncs.com/compatible-mode/v1',
}

# ==================== 过载保护配置（压测第 2 步 P2，见 src/core/resilience.py）====================
OVERLOAD_CONFIG = {
    # 准入控制：全局在途请求上限；≤0 表示不限（压测并发阶梯时设 0，测系统自然拐点）
    # 定稿 20（压测第 5 步）：阶梯拐点给出 24（30 会话在途 P95=22 达标、40 会话 26 超标）；
    # 突增 10→60→10 对比 24/1.0 与 20/0.5：后者冲击期放行请求 TTFE P95 6.01→5.25s、达标比例 69%→90%、goodput +12%。
    # 真实百炼下稳定并发约 8 会话（在途约 5–6），正常负载触达不到 20，故按过载表现取 20；低于百炼 TPM 约 40 在途
    'admission_max_inflight':      int(os.getenv('ADMISSION_MAX_INFLIGHT', 20)),
    # 满额时最多排队秒数，超时返回 503；排队时间计入 TTFE，由 2.0 降为 0.5（被拒请求 0.5s 内得到答复）
    'admission_queue_timeout_s':   float(os.getenv('ADMISSION_QUEUE_TIMEOUT', 0.5)),
    # 请求级总截止时间（流式与非流式），超时返回 504
    'request_deadline_s':          float(os.getenv('REQUEST_DEADLINE_S', 60)),
    # 依赖隔离：4B 前台 = 改写 12 + 判定 6，加后台 LLM 并发 2 = 20 = 4B 的 --max-num-seqs，请求不在 vLLM 内部排队
    'bulkhead_rewrite':            int(os.getenv('BULKHEAD_REWRITE', 12)),
    'bulkhead_rewrite_timeout_s':  float(os.getenv('BULKHEAD_REWRITE_TIMEOUT', 5.0)),   # 超时降级为原始输入
    'bulkhead_evaluator':          int(os.getenv('BULKHEAD_EVALUATOR', 6)),
    'bulkhead_evaluator_timeout_s': float(os.getenv('BULKHEAD_EVALUATOR_TIMEOUT', 1.0)),  # 超时按判定失败降级（保留并标记）
    'bulkhead_rerank':             int(os.getenv('BULKHEAD_RERANK', 16)),
    'bulkhead_rerank_timeout_s':   float(os.getenv('BULKHEAD_RERANK_TIMEOUT', 5.0)),    # 超时按重排失败降级（RRF 排序）
    # 重试预算：10s 窗口内重试次数 ≤ max(3, 10% × 请求数)
    'retry_budget_ratio':          float(os.getenv('RETRY_BUDGET_RATIO', 0.1)),
    'retry_budget_min':            int(os.getenv('RETRY_BUDGET_MIN', 3)),
    'retry_budget_window_s':       float(os.getenv('RETRY_BUDGET_WINDOW', 10)),
}

# ==================== 查询改写器配置 ====================
REWRITER_CONFIG = {
    'max_history_turns': MEMORY_CONFIG['max_history_turns'],
    'timeout':           VLLM_CONFIG['rewrite_timeout'],  # 与 vLLM 改写超时对齐，不单独配置
    'max_retries':       int(os.getenv('REWRITER_MAX_RETRIES', 2)),  # 总尝试次数（含首发）；压测前为 3，过载时重试会放大流量，另受重试预算约束
    # 结构化 JSON 输出（改写 + 分类 + standalone + 约束 + 实体 + 话题），256 不够
    'max_tokens':        int(os.getenv('REWRITER_MAX_TOKENS', 512)),
    # 温度 0：改写结果与缓存键稳定（上线前用 scripts/eval_stm.py 确认改写质量不降）
    'temperature':       float(os.getenv('REWRITER_TEMPERATURE', 0.0)),
}

# ==================== 登录认证配置（前端 /ui，见 src/core/auth.py）====================
# 不开放注册：账号由 scripts/create_user.py 创建。
# AUTH_ENABLED=false 时对话接口不校验登录，user_id 取请求体（压测 / 离线评估脚本使用）。
# JWT_SECRET 未设置时启动随机生成，重启后已登录用户需重新登录。
AUTH_CONFIG = {
    'enabled':       os.getenv('AUTH_ENABLED', 'true').lower() == 'true',
    'jwt_secret':    os.getenv('JWT_SECRET', ''),
    'jwt_expire_h':  int(os.getenv('JWT_EXPIRE_HOURS', 24)),
}

# ==================== 敏感信息可逆脱敏 ====================
# 人名不用 NER（知识库里大量演员 / 角色名会被误伤），只匹配受保护人员名单；企业场景对应 HR 花名册或 CRM 客户主数据
PII_CONFIG = {
    'enabled':         os.getenv('PII_MASK_ENABLED', 'true').lower() == 'true',
    'protected_names': [n for n in os.getenv('PII_PROTECTED_NAMES', '孟可,Ke Meng,Meng Ke').split(',') if n.strip()],
    'vault_ttl_s':     int(os.getenv('PII_VAULT_TTL_S', 86400)),   # 映射表有效期，与短期记忆一致
}

# 角色 → 允许使用的 Agent 工具（"*" 表示全部；也可写工具名列表，如 ["knowledge_retriever", "translator"]）
#   admin / user：对话；ops：/chat_eval 离线评估走完整推理，需要与线上一致的工具集
#   需要限制某个角色时只改这里，工具节点执行前按此检查
ROLE_TOOL_POLICY = {
    'admin': '*',
    'user':  '*',
    'ops':   '*',
}

# ==================== 在线搜索配置（百度千帆）====================
WEB_SEARCH_CONFIG = {
    'enabled':           os.getenv('WEB_SEARCH_ENABLED', 'true').lower() == 'true',
    'api_url':           'https://qianfan.baidubce.com/v2/ai_search/web_summary',
    'api_key':           os.getenv('QIANFAN_API_KEY', ''),
    'top_k':             int(os.getenv('WEB_SEARCH_TOP_K', 3)),
    'max_snippet_chars': int(os.getenv('WEB_SEARCH_SNIPPET_CHARS', 300)),
    'timeout':           int(os.getenv('WEB_SEARCH_TIMEOUT', 15)),
}

# ==================== 知识图谱（GraphRAG / 实体关系验证）====================
# enabled=false：不预取、relation_verifier 也不进 Agent 工具列表（评估对照组）
# prefetch=false：只关多跳题的代码预取，工具仍由 Agent 自行调用
KG_CONFIG = {
    'enabled':            os.getenv('KG_ENABLED', 'true').lower() == 'true',
    'prefetch':           os.getenv('KG_PREFETCH', 'true').lower() == 'true',
    'max_hops':           int(os.getenv('KG_MAX_HOPS', 3)),
    'timeout_s':          float(os.getenv('KG_TIMEOUT_S', 2.0)),     # 单次 Cypher 查询超时
    'max_facts':          int(os.getenv('KG_MAX_FACTS', 15)),        # 单实体邻域展开最多返回的事实数
    'max_paths_per_pair': int(os.getenv('KG_MAX_PATHS_PER_PAIR', 3)),
    'max_pairs':          int(os.getenv('KG_MAX_PAIRS', 10)),
    'hub_degree':         int(os.getenv('KG_HUB_DEGREE', 8)),        # 关联实体数达到该值视为枢纽，路径途经时降权
    'evidence_chars':     int(os.getenv('KG_EVIDENCE_CHARS', 300)),
    'reload_interval_s':  int(os.getenv('KG_RELOAD_INTERVAL_S', 300)),  # 词典版本检查间隔（图谱重新入库后自动重载）
    'fulltext_min_score': float(os.getenv('KG_FULLTEXT_MIN_SCORE', 1.0)),
}
