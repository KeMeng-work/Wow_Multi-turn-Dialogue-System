-- ============================================================
-- 对话历史全量存储 - 数据库初始化脚本
-- 执行方式：一次性运行，重复部署用 IF NOT EXISTS 保护
-- ============================================================

-- 启用 UUID 生成扩展
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 启用慢查询统计扩展（监控用）
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ============================================================
-- 主表：按月范围分区
-- ============================================================
CREATE TABLE IF NOT EXISTS conversation_history (
    id           BIGSERIAL,
    message_id   UUID         NOT NULL DEFAULT gen_random_uuid(),  -- 幂等锚点，防重复写入
    session_id   VARCHAR(255) NOT NULL,
    user_id      VARCHAR(255) NOT NULL,
    role         VARCHAR(50)  NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content      TEXT         NOT NULL,
    token_count  INT,                                               -- 预留：上下文窗口管理
    metadata     JSONB        NOT NULL DEFAULT '{}',                -- 扩展字段：模型名/耗时/评分等
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- 分区表主键必须包含分区键
    PRIMARY KEY (id, created_at),

    -- message_id 全局唯一，异步重试时 ON CONFLICT DO NOTHING 依赖此约束
    CONSTRAINT uq_message_id UNIQUE (message_id,created_at)

) PARTITION BY RANGE (created_at);

-- ============================================================
-- 索引（建在父表上，子分区自动继承）
-- ============================================================

-- 核心查询索引：session 历史按时间倒序（最常用查询路径）
CREATE INDEX IF NOT EXISTS idx_conv_session_time
    ON conversation_history (session_id, created_at DESC);

-- 用户维度查询索引
CREATE INDEX IF NOT EXISTS idx_conv_user_time
    ON conversation_history (user_id, created_at DESC);

-- JSONB 元数据全文检索索引
CREATE INDEX IF NOT EXISTS idx_conv_metadata
    ON conversation_history USING GIN (metadata);

-- ============================================================
-- 分区表：手动创建2026年全年（pg_partman 接入前的过渡方案）
-- 按月分区，超出范围的写入会报错，必须提前建好
-- ============================================================
CREATE TABLE IF NOT EXISTS conversation_history_2026_01
    PARTITION OF conversation_history
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');

CREATE TABLE IF NOT EXISTS conversation_history_2026_02
    PARTITION OF conversation_history
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');

CREATE TABLE IF NOT EXISTS conversation_history_2026_03
    PARTITION OF conversation_history
    FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');

CREATE TABLE IF NOT EXISTS conversation_history_2026_04
    PARTITION OF conversation_history
    FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');

CREATE TABLE IF NOT EXISTS conversation_history_2026_05
    PARTITION OF conversation_history
    FOR VALUES FROM ('2026-05-01') TO ('2026-06-01');

CREATE TABLE IF NOT EXISTS conversation_history_2026_06
    PARTITION OF conversation_history
    FOR VALUES FROM ('2026-06-01') TO ('2026-07-01');

CREATE TABLE IF NOT EXISTS conversation_history_2026_07
    PARTITION OF conversation_history
    FOR VALUES FROM ('2026-07-01') TO ('2026-08-01');

CREATE TABLE IF NOT EXISTS conversation_history_2026_08
    PARTITION OF conversation_history
    FOR VALUES FROM ('2026-08-01') TO ('2026-09-01');

CREATE TABLE IF NOT EXISTS conversation_history_2026_09
    PARTITION OF conversation_history
    FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');

CREATE TABLE IF NOT EXISTS conversation_history_2026_10
    PARTITION OF conversation_history
    FOR VALUES FROM ('2026-10-01') TO ('2026-11-01');

CREATE TABLE IF NOT EXISTS conversation_history_2026_11
    PARTITION OF conversation_history
    FOR VALUES FROM ('2026-11-01') TO ('2026-12-01');

CREATE TABLE IF NOT EXISTS conversation_history_2026_12
    PARTITION OF conversation_history
    FOR VALUES FROM ('2026-12-01') TO ('2027-01-01');

CREATE TABLE IF NOT EXISTS conversation_history_2027_01
    PARTITION OF conversation_history
    FOR VALUES FROM ('2027-01-01') TO ('2027-02-01');

-- ============================================================
-- 字段注释
-- ============================================================
COMMENT ON TABLE  conversation_history                IS '全量对话历史，按月分区';
COMMENT ON COLUMN conversation_history.message_id    IS '全局唯一消息ID，异步写入幂等控制';
COMMENT ON COLUMN conversation_history.token_count   IS 'LLM token 数，预留用于上下文窗口裁剪';
COMMENT ON COLUMN conversation_history.metadata      IS '扩展字段，结构示例: {"model":"gpt-4","latency_ms":320}';

-- ============================================================
-- pg_partman 自动化分区管理（需先安装扩展，可选升级项）
-- 取消注释后执行一次即可，之后自动创建新分区、归档旧分区
-- ============================================================
-- CREATE EXTENSION IF NOT EXISTS pg_partman;
--
-- SELECT partman.create_parent(
--     p_parent_table => 'public.conversation_history',
--     p_control      => 'created_at',
--     p_type         => 'range',
--     p_interval     => 'monthly',
--     p_premake      => 3
-- );
--
-- UPDATE partman.part_config
-- SET    retention              = '6 months',
--        retention_keep_table   = false,
--        automatic_maintenance  = 'on'
-- WHERE  parent_table = 'public.conversation_history';
