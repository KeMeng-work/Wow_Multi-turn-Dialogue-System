-- ============================================================
-- 长期记忆模块 Schema
-- 依赖：PostgreSQL >= 15, pgvector >= 0.5.0
-- 建表内容: 核心记忆表 + 用户画像表
-- Author: Ke Meng
-- Created: 2026-04-12
-- ============================================================

-- ============================================================
-- 表1：memories（核心记忆表）
-- 职责：存储所有长期记忆的内容、向量、元数据
-- 向量检索 + 动态评分 + 遗忘机制 + 反思溯源 均在此表操作
-- ============================================================
CREATE TABLE IF NOT EXISTS memories (

    -- 主键
    id                  UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- 归属用户
    user_id             TEXT        NOT NULL,

    -- 记忆正文（干净文本，不含任何标签或元数据）
    content             TEXT        NOT NULL,

    -- 英文记忆正文（1.7B 抽取时同步生成）：向量基于它计算，召回用英文改写 en_query 匹配
    -- nomic-embed-text 对中文区分度不足，英文链路实测可区分相关 / 不相关（见 scripts/calibrate_ltm_threshold.py）
    content_en          TEXT,

    -- 向量列（nomic-embed-text 输出，768 维，余弦距离）
    embedding           VECTOR(768) NOT NULL,

    -- 记忆类型：fact / preference / episodic / reflection
    -- fact:       客观事实，如"用户叫 Ke"
    -- preference: 偏好，如"用户喜欢硬科幻电影"
    -- episodic:   事件，如"用户昨天去了故宫"
    -- reflection: 反思聚合，由 Qwen3-4B 从多条低层记忆归纳生成
    memory_type         TEXT        NOT NULL DEFAULT 'fact'
                            CHECK (memory_type IN ('fact', 'preference', 'episodic', 'reflection')),

    -- 静态重要性评分（Qwen3-1.7B 提取时赋值，0.0~1.0，存入后不再由 LLM 修改）
    -- 动态综合评分在查询时实时计算，不持久化
    importance_score    FLOAT       NOT NULL DEFAULT 0.5
                            CHECK (importance_score >= 0.0 AND importance_score <= 1.0),

    -- 被检索命中的累计次数（每次 retrieve_relevant 命中 +1）
    -- 用于动态评分的 frequency 分量
    access_count        INT         NOT NULL DEFAULT 0,

    -- 最后一次被检索命中的时间（用于频次分析，不用于时间衰减）
    -- 时间衰减基于 updated_at 计算，反映记忆本身的新鲜度
    last_accessed_at    TIMESTAMPTZ,

    -- 记录时间戳
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),  -- update 操作时同步更新，驱动时间衰减

    -- 软删除标志（遗忘机制标记为 FALSE，不物理删除，保留溯源能力）
    is_active           BOOLEAN     NOT NULL DEFAULT TRUE,

    -- 反思溯源（reflection 类型专用）
    -- 记录本条反思记忆是由哪些原始记忆 id 聚合生成的，便于追溯
    source_ids          UUID[],

    -- 预留扩展字段（不参与任何查询条件，用于未来扩展）
    metadata            JSONB       NOT NULL DEFAULT '{}'
);

-- ── 索引设计 ──────────────────────────────────────────────────

-- 1. 向量检索索引（HNSW，余弦距离）
--    m=16: 每节点最大连接数，影响图密度和精度
--    ef_construction=128: 构建时搜索宽度，值越大精度越高、构建越慢
--    记忆量级（每用户数百条），当前参数完全够用
CREATE INDEX IF NOT EXISTS idx_memories_embedding
    ON memories USING HNSW (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- 2. 最高频查询：某用户的全部活跃记忆（retrieve_relevant 主过滤条件）
CREATE INDEX IF NOT EXISTS idx_memories_user_active
    ON memories (user_id, is_active);

-- 3. 按类型过滤检索（retrieve_relevant 可选参数 memory_type）
CREATE INDEX IF NOT EXISTS idx_memories_user_type_active
    ON memories (user_id, memory_type, is_active);

-- 4. 遗忘定时任务扫描：找出长期未更新的活跃记忆
--    部分索引（WHERE is_active = TRUE）：只索引需要扫描的记忆，减小索引体积
CREATE INDEX IF NOT EXISTS idx_memories_updated_at_active
    ON memories (updated_at)
    WHERE is_active = TRUE;

-- 5. 容量淘汰：找出某用户 access_count 最低的记忆
--    部分索引同上
CREATE INDEX IF NOT EXISTS idx_memories_user_access_count
    ON memories (user_id, access_count)
    WHERE is_active = TRUE;

-- 6. 反思任务：找出低重要性活跃记忆（importance_score < 0.7）
CREATE INDEX IF NOT EXISTS idx_memories_user_importance
    ON memories (user_id, importance_score)
    WHERE is_active = TRUE;


-- ============================================================
-- 表2：user_profiles（用户核心画像表）
-- 职责：持久化高频读、结构化、不需要向量检索的核心用户信息
-- 每次对话前直接主键查询，无向量检索开销
-- ============================================================
CREATE TABLE IF NOT EXISTS user_profiles (

    -- 用户 ID，主键（TEXT 类型，与 memories.user_id 一致）
    user_id         TEXT        PRIMARY KEY,

    -- 核心事实（高确定性、长期稳定的用户信息）
    -- 示例：{"name": "Ke", "age": 28, "city": "上海", "occupation": "AI工程师"}
    core_facts      JSONB       NOT NULL DEFAULT '{}',

    -- 偏好摘要（高频使用的用户偏好，每次对话注入系统提示）
    -- 示例：{"language_style": "简洁直接", "topics": ["AI", "音乐"], "food": "不吃香菜"}
    preferences     JSONB       NOT NULL DEFAULT '{}',

    -- 统计信息（由系统维护，不由 LLM 生成）
    -- 示例：{"total_turns": 128, "memory_count": 47, "last_reflection_at": "2026-04-10T..."}
    stats           JSONB       NOT NULL DEFAULT '{}',

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── 索引设计 ──────────────────────────────────────────────────

-- GIN 索引支持 JSONB 的包含查询（@> 运算符）
-- 例：WHERE core_facts @> '{"name": "Ke"}' 或 WHERE preferences @> '{"topics": ["AI"]}'
CREATE INDEX IF NOT EXISTS idx_user_profiles_core_facts
    ON user_profiles USING GIN (core_facts);

CREATE INDEX IF NOT EXISTS idx_user_profiles_preferences
    ON user_profiles USING GIN (preferences);


-- ============================================================
-- 注释说明
-- ============================================================
COMMENT ON TABLE memories IS '长期记忆主表：存储用户记忆的内容、向量、元数据，支持动态评分检索和软删除遗忘';
COMMENT ON TABLE user_profiles IS '用户核心画像表：高频读的结构化用户信息，不走向量检索，每次对话前直接注入';
COMMENT ON COLUMN memories.embedding IS 'nomic-embed-text 生成的 768 维向量，使用余弦距离（<=>）进行检索';
COMMENT ON COLUMN memories.importance_score IS 'Qwen3-1.7B 提取时赋予的静态评分，后续由时间衰减和访问频次动态加权';
COMMENT ON COLUMN memories.access_count IS '累计被检索命中次数，用于动态评分 frequency 分量：LEAST(access_count/10.0, 1.0)';
COMMENT ON COLUMN memories.updated_at IS '记忆内容最后更新时间，用于时间衰减：EXP(-0.01 × days_since_update)';
COMMENT ON COLUMN memories.is_active IS '软删除标志，FALSE 表示已被遗忘机制淘汰，保留记录用于溯源，不参与任何检索';
COMMENT ON COLUMN memories.source_ids IS 'reflection 类型记忆专用，记录被聚合的原始记忆 UUID 列表';
COMMENT ON COLUMN user_profiles.stats IS '系统维护的统计字段，包含 total_turns / memory_count / last_reflection_at 等';