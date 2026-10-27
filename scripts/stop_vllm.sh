#!/bin/bash
# 停止所有 vLLM 实例

LOG_DIR="/home/ecs-user/services/my_project/logs"

for pid_file in "$LOG_DIR"/vllm_*.pid; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        name=$(basename "$pid_file" .pid)
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "已停止 $name (PID $pid)"
        else
            echo "$name (PID $pid) 已不在运行"
        fi
        rm -f "$pid_file"
    fi
done
