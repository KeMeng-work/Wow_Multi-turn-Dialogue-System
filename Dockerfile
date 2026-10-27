# 前置要求：服务器需配置 Docker 镜像加速器（国内环境），否则 FROM python:3.11-slim 拉取会走向dockerhub超时
# 假设是阿里云服务器，访问以下网站：https://cr.console.aliyun.com/cn-shanghai/instances/mirrors，获取专属加速器url（free）
# 进入服务器后执行以下命令，填写你的专属url地址，即可配置成功：
# sudo tee /etc/docker/daemon.json <<EOF
  #  {
  #    "registry-mirrors": ["https://<你的阿里云专属镜像账号>.mirror.aliyuncs.com"]
  #  }
  #  EOF
  #  sudo systemctl daemon-reload
  #  sudo systemctl restart docker

# Tips：因为阿里云镜像加速不稳定，如果嫌配置麻烦，直接本地拉取 python3.11-slim（150MB），打包成 tar 传到服务器即可跳过以上配置

FROM python:3.11-slim
WORKDIR /app
RUN rm -f /etc/apt/sources.list.d/*.sources && \
    echo "deb https://mirrors.aliyun.com/debian $(. /etc/os-release && echo $VERSION_CODENAME) main" > /etc/apt/sources.list && \
    echo "deb https://mirrors.aliyun.com/debian-security $(. /etc/os-release && echo $VERSION_CODENAME)-security main" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# pip 下载缓存挂载到 BuildKit 缓存目录：改依赖时只下载新增/变更的包，缓存不进镜像
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
ENV PYTHONPATH=/app
COPY . .
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
