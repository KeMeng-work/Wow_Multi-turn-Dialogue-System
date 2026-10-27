#!/bin/bash
# 服务器初始化脚本 - 新服务器上线时执行一次
# 适用：阿里云 Linux (RHEL/CentOS 系)
# 用法：bash scripts/setup_server.sh

set -e
echo "=========================================="
echo "  服务器初始化脚本"
echo "=========================================="

# ── 1. 安装 Docker ────────────────────────────────────────
echo "[1/4] 安装 Docker..."
if command -v docker &>/dev/null; then
    echo "  Docker 已安装，跳过"
else
    sudo yum install -y yum-utils
    sudo yum-config-manager --add-repo \
        https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
    sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker "$USER"
    echo "  Docker 安装完成"
fi

# ── 2. 配置 Docker 国内镜像源 ────────────────────────────
echo "[2/4] 配置 Docker 镜像源..."
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  }
}
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
echo "  镜像源配置完成"

# ── 3. 创建项目目录结构 ───────────────────────────────────
echo "[3/4] 创建项目目录..."
PROJECT_DIR="/home/ecs-user/services/my_project"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/src/models"
echo "  目录创建完成：$PROJECT_DIR"

# ── 4. 生成 Langfuse 密钥（若 .env 中还是占位符则自动填入）──
echo "[4/4] 检查 Langfuse 密钥..."
ENV_FILE="$PROJECT_DIR/.env"
if [ -f "$ENV_FILE" ]; then
    if grep -q "your_nextauth_secret_here" "$ENV_FILE"; then
        SECRET=$(openssl rand -base64 32)
        SALT=$(openssl rand -base64 32)
        sed -i "s|your_nextauth_secret_here|$SECRET|" "$ENV_FILE"
        sed -i "s|your_salt_here|$SALT|" "$ENV_FILE"
        echo "  Langfuse 密钥已自动生成并写入 .env"
    else
        echo "  Langfuse 密钥已存在，跳过"
    fi
else
    echo "  未找到 .env 文件，请先同步项目文件再运行此脚本"
fi

echo ""
echo "=========================================="
echo "  初始化完成！"
echo "  如果是首次安装 Docker，请重新登录 SSH 使 docker 用户组生效"
echo "  然后进入项目目录执行：docker compose up -d <服务名>"
echo "=========================================="
