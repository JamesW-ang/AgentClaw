# ============================================================================
# AgentClaw AI Agent - Docker 镜像
# ============================================================================
FROM python:3.13-slim

# 设置时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# 安装系统依赖（ChromaDB 需要 libssl, sqlite3 等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存层加速构建
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data /app/memory_db /app/knowledge_base

# 环境变量
ENV AGENT_MODE=single \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 暴露端口
EXPOSE 8000 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# 启动命令
CMD ["python", "main.py"]
