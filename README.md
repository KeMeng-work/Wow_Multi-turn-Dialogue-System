## 📖 项目简介

**ReAct Agent** + **Rag** 开发的**生产级多轮对话系统**，包含以下核心功能：

  - ReAct 多工具编排 / 多模型部署 / GraphRAG / 数据飞轮Agent / 并发压测 / MCP封装 / 前端页面等
    

  - 独立数据飞轮 Agent： https://github.com/Jkler879/WOW-Round3
  
  - 生产级并发压测： https://github.com/Jkler879/Wow-Production-Grade-Stress-Testing-Module


  **演示录频**：https://my.feishu.cn/wiki/MXPiwoSUoiXH8Rk6ytgcD0gvnbd
  
  **在线演示**：项目完整代码均已同步 Github，如需在线体验前端界面请提前告知（服务器非 7 * 24 小时运行）
  
  **Email_address** :  kemeng_work@outlook.com

## 🖼️ 架构图

<p align="center">
  <a href="https://Jkler879.github.io/Wow_Multi-turn-Dialogue-System/architecture.html" target="_blank" rel="noopener noreferrer">
    <img src="./architecture_dark_v3.svg" alt="WOW多轮对话系统 RAG + ReAct Agent 系统架构图" width="100%">
  </a>
</p>

<p align="center"><sub>点击图片可放大观看（支持滚轮缩放、拖拽平移）</sub></p>

## ✨ 核心亮点

### 1、多轮对话与分层记忆
- **短期记忆**：`Redis` 热存储 + `PostgreSQL` 持久化 + 分层注入 + 摘要压缩
  | 模块 | 实现方式 |
  |---|---|
  | **热存储** | `Redis` 按 session_id 存储结构化会话状态 + 最近 12 轮对话原文，每轮对话自动续期，24 小时无对话 TTL 淘汰  |
  | **持久化** | `PostgreSQL` 存储全量对话数据（按月分区）+ 结构化会话状态；Redis 过期后自动读取 PG 中的会话状态与 12轮原文并写回 Redis，会话无缝衔接 |
  | **结构化会话状态** | 话题摘要、用户约束（会话级 / 话题级 / 本问级）、话题列表、实体表、事实三元组 |
  | **分层注入** | 用户提问后，会话状态注入系统提示词，近期原文注入消息历史，话题级与本问级约束拼在当前问题前 |
  | **Token 预算** | 短期记忆上限 3000 token，通常包含约 6–10轮历史对话原文 + 结构化会话状态；超出时先裁早期原文，生效约束永不裁剪 |
  | **后台异步压缩** | 对话满 10 轮或历史消息超过 70% 水位时，按话题递归生成摘要；轮数与 token双条件触发，避免短对话永远达不到水位而不压缩 |
  | **并发安全** | `Redis Lua` 单次往返原子读写 + 版本号 `CAS`，防止并发请求互相覆盖 |
  | **生产级恢复** | `AOF/RDB` 混合持久化 + `Docker` 数据卷挂载 + `PostgreSQL`全量对话存储，系统异常重启后短期记忆不丢失 |

      
- **长期记忆（跨会话）**：`PostgreSQL` + `pgvector` 存储，后台异步执行，不阻塞回答

  | 模块 | 实现方式 |
  |---|---|
  | **新记忆抽取** | 每轮回复后取完整问答对：代码先做前置过滤（原话里没有自称、也没有"记住"指令就跳过）→ `Qwen3-1.7B` 判断是否有高价值记忆，最多抽取3 条（事实 / 偏好 / 经历 + 重要性）→ 校验证据必须是用户原话；误存率 16% → 0% |
  | **新旧记忆合并** | 新记忆与该用户已有记忆比对，相似度 ≥ 0.78 的旧记忆（最多 3 条）交给 `Qwen3-4B`，输出新增 / 更新 / 删除 / 跳过的结构化决策，由代码在 `PostgreSQL` 事务中执行，不让 LLM 生成 SQL |
  | **长期记忆召回** | 用户提问后，`Qwen3-1.7B` 先判断问题是否需要个人背景，需要才检索；英文向量召回（相似度门槛 0.42，经标定）→ 按相似度、重要性、时效、频率加权重排，最多注入 3 条；每次被召回，访问次数 +1，提高后续排序 |
  | **遗忘（写入触发）** | 每次写入后检查单用户容量（200 条），超过上限按价值分（重要性 × 0.6 + 时效 × 0.3 + 频率 × 0.1）淘汰最低的记忆 |
  | **遗忘（分数衰减）** | 价值分随时间衰减，低于 0.1 的记忆失效（软删除） |
  | **反思（条件触发）** | 活跃记忆 ≥ 20 条或距上次反思 ≥ 7 天时触发：每类记忆取重要性 < 0.7 的最近 20 条，满 5 条由 `Qwen3-4B` 归纳为 1–3 条高层洞察，记录来源记忆可溯源 |


- **查询改写**：本地 `Qwen3-4B` 读取短期记忆（约 1400 token）+ `few-shot prompt`，输出 `JSON Schema` 约束的结构化结果

  | 模块 | 实现方式 |
  |---|---|
  | **指代消解与双语改写** | 把"它 / 他 / 刚才那个"等指代词还原为具体实体，生成独立的中文问句 + 英文检索问句（对齐英文知识库） |
  | **意图分类：direct** | 打招呼等简单问题降级为本地 4B 模型推理，跳过 ReAct Agent 推理 |
  | **意图分类：local** | 全链路执行 |
  | **意图分类：web_search** | 需时效性信息，全链路执行 + 定制化 System Prompt 意图注入 + 联网搜索工具调用 |
  | **题型分类：simple / reasoning** | 全链路执行 + 定制化 System Prompt 题型注入，步数上限 3 / 5 |
  | **题型分类：multi_hop** | 知识图谱预取 + 定制化 System Prompt 题型注入，步数上限 7 |


### 2、ReAct Agent 自主规划与工具调用
- **ReAct Agent 中枢**：`Qwen3.5-35b-A3b` （API）
    
    - 选型原因：MOE 架构 3b 激活参数; ReAct每题需多次调用模型，思考模式额外生成大量推理 token，激活参数少明显降低单次请求的延迟和成本。
    - RPM：600，TPM：1000000，单台 A10 服务器并发压测后稳定 30 并发
      
- **Agent 框架：LangGraph**
  -  LangGraph 实现 ReAct 循环（思考-行动-观察），覆盖检索/知识图谱/在线搜索等工具集，工具调用全部采用原生 Function Calling，免去文本解析带来的格式错误与参数丢失
    
  - **MCP 封装**：对外暴露 3 个只读工具（知识库检索、图谱关系验证、完整 Agent 问答），令牌鉴权，供 Claude Code等客户端远程调用

- **工具集**：
  - **检索工具**：`双路召回（ 向量检索 / BM25全文检索）+ RRF 融合 + BGE-reranker-v2 模型重排`，输出 Top-5 候选文档并附带重排评分，确保生成高可信度结果。
  - **实体关系验证工具**：实体链接 + 参数化 Cypher 模板，在 `Neo4j` 中做 3 跳以内的路径查询
  - **翻译工具**：本地模型 `Helsinki-NLP/opus-mt-en-zh` 英转中模型将英文检索答案翻译为中文，支撑中文回复。

### 3、检索系统深度优化
- **向量检索（混合检索）**：`Milvus 2.6`+ 版本新索引 `IVF_RABITQ` 支撑向量检索 ，RaBitQ 将 768 维 float32 向量量化为二进制码本，内存占用降至原始 1/32；配合 nlist=1024 倒排聚类，进一步减少搜索时的距离计算量。
  - **对比备选方案**：IVF 系列训练仅需 k-means 聚类，远快于 HNSW 的逐点插入。若选 HNSW，内存高 3-4 倍且构建慢；若选纯 IVF_FLAT，无量化压缩、大数据集延迟高

  - **Milvus优势**：存算分离，增量更新时索引构建在专属资源池异步执行，不占用查询的 CPU/内存。即使全量重建 IVF_RABITQ 索引，写入延迟和查询 P99 几乎不受影响。

  - **业务价值**：数万条数据规模下，Milvus standalone 单机内存可承载百万级向量，无需 GPU 或分布式集群。增量更新的可见性延迟为秒级。

  
- **全文检索（混合检索）**：`Milvus 2.6`+ 版本原生 BM25，通过 `sparse` 字段 + `BM25 Function` 实现**原生全文检索**，无需额外引入Elasticsearch，增加数据库维护成本。
  - **业务价值**：纯原生索引，运维复杂度从"双系统同步"降为单库管理，agent检索从双路召回变为单路双召回，消除事务不一致性。

- **RRF加权融合**
- **BGE rerank 重排**

### 4、数据预处理与知识图谱增强
- **本体建模**：用 OWL 2 / RDF 定义 25 类实体、27 种关系及互斥约束；SHACL 校验规则和 Neo4j约束都由本体自动生成；本体做版本化管理

- **版本化入库**：Redis Stream 消息队列入库，按 ID幂等写入、失败重试、异常消息进死信队列
  
- **图增强检索**：
  - Aho-Corasick 词典做实体链接，别名覆盖中英文，弱别名单独处理，图上连通性作为判断依据
  - 参数化只读 Cypher 模板做 3 跳以内的路径查询，避免LLM生成Cypher产生的注入与语法错误
  - 多跳题强制代码执行图谱搜索，每条关系附带证据句并回链向量库切片；其他题型Agent按需调用

### 5、本地模型量化与微调
- **模型量化**：

  **GPU 推理**（vLLM，OpenAI 兼容接口）

  | 模型 | 部署方式 | 量化方式 | 显存占用 | 说明 |
  |------|---------|---------|---------|------|
  | Qwen3-4B | vLLM 端口 8001 | AWQ int4 | ~8GB | 查询改写 / 意图分类器 /长期记忆新旧记忆比对 |
  | Qwen3-1.7B | vLLM 端口 8002 | AWQ int4 | ~3.2GB | 长期记忆新记忆提取：提取每轮问答对有价值的记忆 |
  | BGE-Reranker-v2-M3 | TEI | 原版FP32部署，TEI转FP16 | ~1.2GB | 检索后重排 |

  **CPU 推理**（ONNX Runtime，三个轻量模型）

  | 模型 | 量化方式 | 原始大小 | 量化后大小 | 压缩比 | 单次延迟 (p50) | 内存占用 |
  |------|---------|---------|-----------|--------|---------------|---------|
  | nomic-embed-text-v1 | ONNX导出 + INT8动态量化 | 523 MB | 132 MB | ↓75% | 13.9 ms | ~285 MB |
  | Helsinki-NLP/opus-mt-en-zh | ONNX 导出 + INT8 动态量化（encoder + decoder 分别量化） | 552 MB | 139 MB | ↓75% | 66.7 ms | ~256 MB |

- **部署**
    - 容器化：使用 `Docker Compose` 编排所有服务（Redis、Milvus、Neo4j、Ollama），实现一键启动完整环境，保证环境一致性。
    - API 服务：`FastAPI` 提供高并发异步接口，将长期记忆模块异步化不阻塞 Agent 主程序。

## 📚 知识库选型：Wizard of Wikipedia (英文数据集)
  本系统采用 Facebook AI 发布的 Wizard of Wikipedia (WoW) 数据集作为核心知识库。包含 22k+ 多轮对话（训练集 18k、验证集 2k、测试集 2k），共覆盖 1.3k+个维基百科主题。

**选型理由**：
  - 跨领域覆盖：主题涵盖历史、科技、艺术、娱乐等领域，考验系统在不同类型主题查询下的检索与生成能力。
  - RAG 基准：作为学术界和工业界广泛使用的 RAG 评估黄金基准。
  - 高轮次：单条数据平均`9.15轮`对话，考验系统在多轮对话中的短期记忆连贯性与指代消解能力。
  - 知识密集：单条数据平均 `750-1000 tokens` 的适中长度，考验模型的长上下文处理能力。

**预处理后数据集特征**：
  - 分块后平均轮次：`3.0轮`
  - 分块后平均大小：`724字符`
  - 分块后目标范围（800-1800字符）：`25.5%`数据落于目标范围内，`74.5%`数据小于目标范围（源数据本身包含大量短文本），无数据超过目标范围。
  - 每100条数据关系密度：经`NER抽取`和`RE抽取`后，`每100条数据总实体数量1917`，`总关系数345`，`18%关系密度`合理。

  ## 🛠️ 技术栈与选型理由

| 组件 | 选型 | 理由 |
|------|------|------|
| **Agent 框架** | LangGraph | 原生状态图编排可精细控制"思考-行动-观察"循环，可水平或纵向进行模块扩展，ReAct Agent标配|
| **ReAct Agent 模型** | Qwen3-30b-a3b (API) | 大参数模型确保支撑ReAct Agent能精准解析输出、稳定执行工具调用。百炼平台提供百万token Free额度，零成本支撑当前阶段开发与测试，后期用Qwen3-30b-a3b本地部署替换 |
| **重排模型** | BGE Reranker-v2-m3 (本地ONNX) | Hugging Face 最强开源重排模型之一，26年被 Qwen3-rerank-0.6B性能超越，因项目早期于2026年年初设计选型沿用。|
| **翻译模型** | Helsinki-NLP (本地ONNX) | Hugging Face 专业开源英转中翻译模型（模型精度、速度在过往项目中实际部署验证过） |
| **嵌入模型** | nomic-embed-text-v1 (本地ONNX) | 支持前缀注入。知识库入库数据为英文数据集，参考MTEB榜单评分，nomic为英文场景开源嵌入模型的性能标杆。需统一用于知识库向量化入库、用户查询向量化、长期记忆向量化，确保模型维度对齐 |
| **改写模型** | Qwen3-4B (本地vLLM) | 参考查询改写主流模型选型经验(4-8B)，分别在Qwen3-1.7b、4B、8b测试，4B为精度与速度平衡最佳 |
| **长期记忆模型** | qwen3-1.7b/4b (本地vLLM) | 参考Mem0长期记忆官方文档，记忆提取部分 Small size模型为黄金选择，记忆更新部分使用4B模型，确保新旧记忆增删改决策的准确性 |
| **长期记忆框架** | PGvector + PostGreSQL | Qwen3-1.7B/Qwen3-4B分别负责双阶段处理，通过Qwen3-1.7B提取新记忆，Qwen3-4B负责新旧记忆对比，通过function call 接收记忆的新增/更新/删除/合并决策，保证入库格式正确。前期使用LangMem 和 Mem0 开发，过于demo化、操作过程黑盒，代码重构升级为当前架构 |
| **API 服务** | FastAPI | 异步原生，支撑长期记忆异步写入不阻塞agent主程序 |
| **向量数据库** | Milvus 2.6 |IVF_RABITQ 新索引内存占用低且召回率高，sparse + BM25function实现原生全文检索，避免额外引入Elasticsearch降低数据库运维成本 |
| **图数据库** | Neo4j | GraphRag主流选择，支撑知识图谱存储、实体关系验证、多跳路径验证 |
| **缓存+短期记忆数据库** | Redis-Stack | 内存数据库，低延迟读写适合短期记忆存取，通过内置的布隆过滤器支撑高频查询缓存过滤，限流熔断器 |
| **消息队列** | Redis Stream | 基于现有 Redis 基础设施，轻量级解耦知识库与知识图谱的异步入库，避免引入外部中间件Kafka增加系统复杂度 |
| **模型量化** | ONNXRUNTIME | 相比Optimum方法更底层、可定制化量化范围更广 |
| **日志监控** | LangFuse + RAGAS | 自动收集所有由LangGraph开发的内置监控 + 自定义Langfuse监控 + RAGAS在线/离线指标 + 本地logger日志 |


## 🚀 后续优化方向
  - **推理引擎升级**：
      - 2、Qlora 微调 Qwen3-4b 模型，新增一轮飞轮验证查询改写、意图分类、题型分类的指标变化。
      - 3、基于当前服务器配置，将 Qwen3-30b-a3b api 替换为本地部署 Qwen3-14B int4 量化版，移除 API 费用。
      - 4、多 Agent 协作，改造原有 LangGraph 单 Agent 模式。
      - 5、添加K3S部署，替代原有Docker Compose。
 

## 📁 项目代码结构

```text
Wow_Multi-turn-Dialogue-System/
├── config/ - 配置
│   ├── config_path.py  # 集成配置管理
│   ├── paths.py  # 集成路径管理
│   └── redis_conf  # Redis 配置文件（bloom、内存管理、AOF/RDB等）
│
├── init/ - PostgreSQL 初始化脚本（容器首次启动自动执行）
│   ├── 01_schema_conversation.sql  # 主表：全量对话数据存储（ pg_partmen自动按月分区、冥等写入、敏感信息脱敏）
│   ├── 02_schema_vector.sql  # pgvector扩展（ HNSW向量索引 ）
│   └── 03_schema_long_term_memory.sql  # 长期记忆（memories主表 + user_profile用户画像）
│
├── ontology/ - 知识图谱本体建模（OWL 2 / RDF）
│   ├── wow_ontology.ttl  # 定义实体类型、关系与互斥约束
│   └── wow_shapes.ttl  # SHACL 校验规则
│
├── src/ - 核心源代码
│   ├── main.py  # 核心代码，全链路编排
│   ├── api.py  # FastAPI 异步服务入口（对话 / 会话 / 记忆 / 反馈 / 工具接口）
│   ├── eval_handler.py  # 飞轮评估专用接口
│   │
│   ├── web/ - 交互前端
│   │   ├── index.html  # SSE 流式展示推理过程、工具步骤与回答依据
│   │   └── static/  # Markdown 渲染、XSS 过滤
│   │
│   ├── mcp_server/ - MCP 服务
│   │   └── server.py  # 对外暴露 3 个只读工具（知识检索 / 图谱验证 / 完整 Agent 问答），令牌鉴权
│   │
│   └── core/
│       ├── auth.py  # 登录认证（bcrypt + JWT）
│       ├── session_store.py  # 会话软删除
│       ├── feedback.py  # 用户点赞 / 点踩，回流数据飞轮
│       ├── resilience.py  # 过载保护（准入控制、舱壁隔离）
│       ├── reasoning_chat.py  # 支持原生思考过程的 ChatOpenAI 扩展（百炼主模型）
│       ├── vllm_client.py  # 本地 vLLM 
│       │
│       ├── query_rewrite/ - 查询改写
│       │   ├── query_rewriter.py  # 指代消解、中英双语改写、意图与题型分类
│       │   └── query_rewrite_test_case.py  # 模块测试用例
│       │
│       ├── high_frequency_query_cache/ - 限流、熔断、高频查询缓存
│       │   ├── redis_bloom.py  # 高频查询缓存
│       │   ├── rate_limiter.py  # 三级令牌桶限流
│       │   ├── circuit_breaker.py  # 滑动窗口熔断器
│       │   └── redis_bloom_test_case.py  # 模块测试用例
│       │
│       ├── memory_short/ - 短期记忆（Redis 热存储 + PostgreSQL 快照）
│       │   ├── stm_service.py  # 短期记忆服务入口
│       │   ├── session_state.py  # 结构化会话状态（约束 / 话题 / 摘要 / 实体 / 事实三元组）
│       │   ├── context_builder.py  # 按 token 预算分层组装上下文（改写器 / Agent）
│       │   ├── compactor.py  # 后台异步压缩
│       │   ├── redis_short_memory.py  # Redis 存储（Lua 原子读写 + 版本号 CAS）
│       │   ├── session_store_pg.py  # 会话状态 PG 快照与恢复
│       │   └── stm_test.py  # 模块测试用例（34项）
│       │
│       ├── memory_long/ - 长期记忆（PostgreSQL + pgvector）
│       │   ├── long_term_memory.py  # 长期记忆检索、新记忆抽取、新旧记忆合并决策、PGvector 事务写入
│       │   ├── recall_gate.py  # 长期记忆召回判断
│       │   ├── forgetting_task.py  # 长期记忆遗忘（每日定时脚本）
│       │   ├── reflection_task.py  # 长期记忆反思（每日定时脚本 + 条件触发）
│       │   └── ltm_write_gate_test.py  # 模块测试用例
│       │
│       ├── conversation_data_postgresql/ - 全量对话历史
│       │   └── conversation_data.py  # asyncpg 连接池、分区表、指数退避重试
│       │
│       ├── ReAct_Agent/ - Agent 开发
│       │   └── tools/ - 工具集
│       │       ├── agent.py  # Agent 图构建与节点（LangGraph）
│       │       ├── base.py  # 工具工厂
│       │       ├── retriever.py  # 检索工具：混合检索 + RRF 融合 + MaxP 重排 + 证据分级
│       │       ├── relation_verifier.py  # 实体关系验证工具（知识图谱）
│       │       ├── translate.py  # 翻译工具
│       │       └── web_search.py  # 在线搜索工具（含安全注入防护）
│       │
│       ├── kg/ - 知识图谱
│       │   ├── linker.py  # 实体链接（Aho-Corasick 词典 + 消歧）
│       │   └── ontology.py  # 本体加载，生成校验规则与 Neo4j 约束
│       │
│       ├── security/ - 安全管控
│       │   ├── context.py  # 请求安全上下文（用户 / 会话 / 角色）
│       │   ├── pii.py  # 可逆脱敏
│       │   ├── injection.py  # 注入检测
│       │   └── events.py  # 安全事件审计
│       │
│       ├── evaluation/ - 在线评估
│       │   └── ragas_evaluator.py  # 按采样率异步评估，结果回写 Langfuse
│       │
│       ├── postprocess/ - 回答后处理
│       │   └── detectors.py  # 规则检测器
│       │
│       ├── embedder/ - embedding 推理
│       │   └── onnx_embedder.py  
│       │
│       ├── redis-stream/ - 消息队列异步入库
│       │   ├── producer.py  # 生产者：知识库数据推送至 Redis Stream
│       │   ├── producer_kg.py  # 生产者：知识图谱数据推送至 Redis Stream
│       │   ├── consumer_milvus.py  # 消费者：写入 Milvus
│       │   ├── consumer_neo4j.py  # 消费者：写入 Neo4j
│       │   ├── create_milvus_collection.py  # 创建 Milvus 知识库集合及索引（初始化运行一次）
│       │   ├── long_memory_collection.py  # 创建 pgvector 长期记忆集合及索引（初始化运行一次）
│       │   ├── migrate_milvus.py  # Milvus 集合迁移（新增 BM25 全文检索）
│       │   └── migrate_milvus_test.py  # 迁移测试
│       │
│       └── chunking/ - 数据预处理（文档分块 + 元数据注入 + NER抽取 + RE抽取）
│
├── scripts/ - 评测、标定、知识库构建与运维脚本
│
├── models/ - 本地模型（大文件不上传）
├── data/ - 源数据（大文件不上传）
├── .env.example - 环境变量模板
├── .gitignore - Git 忽略文件
├── .dockerignore - Docker 构建忽略文件
├── docker-compose.yml - Docker Compose 编排文件（ 13 个异构服务一键部署，按负责分配内存配额）
├── Dockerfile - 应用镜像构建文件（分层构建，代码更新秒级重建容器）
├── requirements.txt - Python 依赖列表
└── README.md - 项目说明文档
```
