-- 创建 pgvector 扩展
CREATE EXTENSION IF NOT EXISTS vector;

-- (可选) 启用其他可能需要的扩展
CREATE EXTENSION IF NOT EXISTS hstore;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
