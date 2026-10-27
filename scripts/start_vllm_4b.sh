#!/bin/bash
# Qwen3-4B-AWQ vLLM 启动脚本（V1 适配版）
# 职责：查询改写 / 记忆合并决策 / 反思生成（端口 8001）
# 关键：4b 必须先于 1b7 完全就绪，start_vllm_1b7.sh 依赖 8001 的就绪信号

source /home/ecs-user/vllm-env/bin/activate

MODEL_PATH="/home/ecs-user/services/my_project/src/models/qwen3-4b-awq/snapshots/74d4bd2bd4bff9cafc9345221320bffb08b406a3"
LOG_DIR="/home/ecs-user/services/my_project/logs"
mkdir -p "$LOG_DIR"

# 上下文 8192：短期记忆改写上下文 ~1400 + 提示词与 few-shot + 压缩任务输入（话题原文 + 旧摘要）需要余量
# max-num-batched-tokens 保持 4096：已开启 chunked prefill，长输入分块预填充，不受 max-model-len 约束
# ========== 启动 vLLM 4B ==========
echo "🚀 启动 Qwen3-4B-AWQ (端口 8001)..."
nohup vllm serve "$MODEL_PATH" \
    --served-model-name qwen3-4b \
    --port 8001 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --quantization awq \
    --dtype auto \
    --gpu-memory-utilization 0.50 \
    --max-model-len 8192 \
    --max-num-seqs 20 \
    --max-num-batched-tokens 4096 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    > "$LOG_DIR/vllm_4b.log" 2>&1 &

echo "PID: $!"
echo $! > "$LOG_DIR/vllm_4b.pid"
echo "日志: $LOG_DIR/vllm_4b.log"

# ========== 等待 8001 就绪（含 KV Cache 初始化完成）==========
# 此步骤确保 profiling forward pass 结束、临时显存释放回 PyTorch 缓存
# start_vllm_1b7.sh 将在此之后执行 empty_cache 并申请连续显存块
echo "⏳ 等待 qwen3-4b (8001) 就绪（含 KV Cache 初始化）..."
MAX_WAIT=300
WAITED=0
while ! curl -sf http://localhost:8001/v1/models > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "❌ 等待 8001 超时 (${MAX_WAIT}s)，请检查日志: $LOG_DIR/vllm_4b.log"
        exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "   已等待 ${WAITED}s..."
done
echo "✅ qwen3-4b (8001) 已就绪，可运行 start_vllm_1b7.sh"
