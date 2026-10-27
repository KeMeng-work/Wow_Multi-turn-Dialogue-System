#!/bin/bash
# Qwen3-1.7B vLLM 启动脚本（V1 适配版）
# 职责：长期记忆提取（端口 8002）
# 关键：等待 8001 完全就绪 + 清理显存碎片后再启动，避免 V1 连续显存分配失败

source /home/ecs-user/vllm-env/bin/activate

MODEL_PATH="/home/ecs-user/services/my_project/src/models/qwen3-1.7b/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
LOG_DIR="/home/ecs-user/services/my_project/logs"
mkdir -p "$LOG_DIR"

# ========== 时序控制：等待 8001 完全就绪 ==========
echo "⏳ 等待 qwen3-4b (8001) 就绪..."
MAX_WAIT=120
WAITED=0
while ! curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "❌ 等待 8001 超时 (${MAX_WAIT}s)，退出"
        exit 1
    fi
    sleep 3
    WAITED=$((WAITED + 3))
    echo "   已等待 ${WAITED}s..."
done
echo "✅ qwen3-4b (8001) 已就绪"

# ========== 显存碎片整理 ==========
echo "🧹 清理 GPU 显存碎片..."
python3 -c "import torch; torch.cuda.empty_cache(); print('   empty_cache 完成')"

# 额外等待 2s，让 CUDA runtime 完成内存合并
sleep 2

# ========== 启动 vLLM 1.7B ==========
echo "🚀 启动 Qwen3-1.7B (端口 8002)..."
# max-model-len 1024 → 2048（2026-10-07 压测发现）：提取请求 max_tokens=512，
# 1024 时输入只剩 512 token，系统提示 + 较长答案即超限，约 11% 提取被 vLLM 以 400 拒绝
nohup vllm serve "$MODEL_PATH" \
    --served-model-name qwen3-1.7b \
    --port 8002 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.22 \
    --max-model-len 2048 \
    --max-num-seqs 15 \
    --max-num-batched-tokens 2048 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    > "$LOG_DIR/vllm_1b7.log" 2>&1 &

echo "PID: $!"
echo $! > "$LOG_DIR/vllm_1b7.pid"
echo "日志: $LOG_DIR/vllm_1b7.log"

# ========== 等待 8002 就绪 ==========
echo "⏳ 等待 qwen3-1.7b (8002) 就绪..."
WAITED=0
while ! curl -sf http://localhost:8002/v1/models > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "❌ 等待 8002 超时，请检查日志: $LOG_DIR/vllm_1b7.log"
        exit 1
    fi
    sleep 3
    WAITED=$((WAITED + 3))
done
echo "✅ qwen3-1.7b (8002) 已就绪，双实例启动完成"
