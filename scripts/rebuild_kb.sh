#!/usr/bin/env bash
# rebuild_kb.sh — 知识库完整重建脚本（生产级）
#
# 用法：
#   docker compose exec app bash scripts/rebuild_kb.sh
#   docker compose exec app bash scripts/rebuild_kb.sh --with-supplement data/processed/chunks/wow_supplement_badcase6.json
#
# 流程：
#   1. 清空 Redis Stream（消除历史积压）
#   2. 删除旧消费者组（确保 consumer 以 "$" 重建）
#   3. 重建 Milvus 集合
#   4. 启动 consumer_milvus（后台）
#   5. 运行 producer（全量入库）
#   6. 等待消费完成（轮询 pending=0）
#   7. 可选：补充文件入库
#   8. 最终验证条目数
#   9. 清空 Stream（管道使命结束）

set -euo pipefail

STREAM="knowledge_pipeline"
GROUP="group_milvus"
SUPPLEMENT_FILE=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-supplement) SUPPLEMENT_FILE="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "  知识库完整重建开始"
echo "========================================"

# 1. 清空 Redis Stream
echo "[1/9] 清空 Redis Stream..."
python3 -c "
import redis, os
from config.config_path import REDIS_CONFIG
r = redis.Redis(**REDIS_CONFIG)
deleted = r.delete('$STREAM')
print(f'  Stream 已清空（deleted={deleted}）')
"

# 2. 删除旧消费者组（下次 consumer 启动时以 \$ 重建）
echo "[2/9] 清理旧消费者组..."
python3 -c "
import redis
from config.config_path import REDIS_CONFIG
r = redis.Redis(**REDIS_CONFIG)
try:
    r.xgroup_destroy('$STREAM', '$GROUP')
    print('  消费者组已删除')
except Exception as e:
    print(f'  消费者组不存在或已清理: {e}')
" || true

# 3. 重建 Milvus 集合
echo "[3/9] 重建 Milvus 集合..."
python3 src/core/redis-stream/create_milvus_collection.py

# 4. 启动 consumer_milvus（后台）
echo "[4/9] 启动 consumer_milvus..."
python3 src/core/redis-stream/consumer_milvus.py &
CONSUMER_PID=$!
echo "  consumer PID=$CONSUMER_PID"
sleep 3  # 等待 consumer 完成初始化和消费者组创建

# 5. 全量入库
echo "[5/9] 运行 producer（全量）..."
SKIP_LAG_CHECK=1 python3 src/core/redis-stream/producer.py

# 6. 等待消费完成（最多等 300 秒）
echo "[6/9] 等待 consumer 消费完成..."
python3 -c "
import redis, time
from config.config_path import REDIS_CONFIG
r = redis.Redis(**REDIS_CONFIG)
for i in range(60):
    try:
        groups = r.xinfo_groups('$STREAM')
        for g in groups:
            name = g.get('name') or g.get(b'name', b'').decode()
            if name == '$GROUP':
                pending = g.get('pending') or g.get(b'pending', 0)
                lag    = g.get('lag')    or g.get(b'lag', 0)
                print(f'  [{i*5}s] pending={pending} lag={lag}')
                if pending == 0 and lag == 0:
                    print('  ✅ 消费完成')
                    exit(0)
    except Exception as e:
        print(f'  检查失败: {e}')
    time.sleep(5)
print('  ⚠️  超时300s，请手动检查')
exit(1)
"

# 7. 补充文件（可选）
if [[ -n "$SUPPLEMENT_FILE" ]]; then
    echo "[7/9] 入库补充文件: $SUPPLEMENT_FILE"
    python3 src/core/redis-stream/producer.py --supplement-file "$SUPPLEMENT_FILE"
    echo "  等待补充文件消费..."
    sleep 10
else
    echo "[7/9] 无补充文件，跳过"
fi

# 8. 验证条目数
echo "[8/9] 验证 Milvus 条目数..."
python3 -c "
from pymilvus import Collection, connections
connections.connect(uri='http://localhost:19530')
col = Collection('wow_knowledge_base_v2')
col.flush()
count = col.num_entities
print(f'  ✅ Milvus 总条目数: {count}')
"

# 9. 清空 Stream（管道使命结束）
echo "[9/9] 清空 Redis Stream..."
python3 -c "
import redis
from config.config_path import REDIS_CONFIG
r = redis.Redis(**REDIS_CONFIG)
r.delete('$STREAM')
print('  Stream 已清空')
"

kill $CONSUMER_PID 2>/dev/null || true
echo ""
echo "========================================"
echo "  知识库重建完成"
echo "========================================"
