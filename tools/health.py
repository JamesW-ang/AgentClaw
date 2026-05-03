# ============================================================
# 健康检查模块
# ============================================================
"""
系统健康检查端点实现

该模块提供了一个综合的健康检查接口，用于监控系统的各个关键组件:
    1. ChromaDB 数据库连接状态
    2. LLM API 可达性
    3. 系统内存使用情况

返回值:
    - 200 OK: 所有组件正常
    - 503 Service Unavailable: 有组件异常
    - JSON 响应包含详细的检查结果和时间戳

使用场景:
    - Kubernetes 存活性探针 (liveness probe)
    - Kubernetes 就绪性探针 (readiness probe)
    - 监控系统告警
    - 负载均衡器健康检查
"""

import time

import psutil
from fastapi.responses import JSONResponse

# ============================================================
# 健康检查端点
# ============================================================

def health_check() -> JSONResponse:
    """
    综合健康检查接口

    执行流程:
        1. 检查 ChromaDB 数据库连接
        2. 检查 LLM API 可达性
        3. 检查系统内存使用
        4. 返回整体健康状态

    返回:
        JSONResponse: 包含以下信息的 JSON 响应
        {
            "status": "healthy" | "degraded",
            "timestamp": 当前 Unix 时间戳,
            "checks": {
                "chromadb": {...},
                "llm_api": {...},
                "memory": {...}
            }
        }

    状态码:
        200: 健康 (所有检查通过)
        503: 降级 (至少一个检查失败)

    注意:
        - ChromaDB 检查使用 heartbeat() 方法，不消耗额外资源
        - LLM API 检查使用 HEAD 请求，不消耗 token
        - 内存检查基于系统 /proc/meminfo (Linux) 或系统 API (其他系统)
    """

    # 存储所有检查结果的字典
    checks = {}

    # 标志位：表示所有检查是否都通过
    all_healthy = True

    # ========== 检查 1: ChromaDB 连接 ==========
    try:
        # 导入 ChromaDB 客户端库
        import chromadb

        # 创建持久化客户端，连接到本地数据库
        client = chromadb.PersistentClient(path="./data/chroma_db")

        # 发送心跳信号，验证连接有效性
        # heartbeat() 方法会在连接失败时抛出异常
        client.heartbeat()

        # 记录成功状态
        checks["chromadb"] = {"status": "ok"}

    except Exception as e:
        # 捕获任何连接错误
        checks["chromadb"] = {
            "status": "error",
            "error": str(e)
        }
        # 标记整体状态为不健康
        all_healthy = False

    # ========== 检查 2: LLM API 可达性 ==========
    try:
        # 导入 HTTP 客户端库
        import httpx

        # 发送 HEAD 请求到 DeepSeek API
        # HEAD 请求只获取响应头，不获取响应体，避免消耗 token
        resp = httpx.head(
            "https://api.deepseek.com",
            timeout=5.0  # 5 秒超时
        )

        # 记录 API 状态和 HTTP 状态码
        checks["llm_api"] = {
            "status": "ok",
            "status_code": resp.status_code
        }

    except Exception as e:
        # 捕获任何网络或连接错误
        checks["llm_api"] = {
            "status": "error",
            "error": str(e)
        }
        # 标记整体状态为不健康
        all_healthy = False

    # ========== 检查 3: 系统内存使用 ==========
    # 获取系统内存信息
    mem = psutil.virtual_memory()

    # 根据内存使用百分比判断状态
    # - < 85%: 正常 (status: ok)
    # - >= 85%: 警告 (status: warning，但不影响整体健康)
    checks["memory"] = {
        "status": "ok" if mem.percent < 85 else "warning",
        "percent": mem.percent,                              # 内存使用百分比
        "available_mb": mem.available // (1024 * 1024)      # 可用内存 (MB)
    }

    # ========== 返回结果 ==========
    # 根据整体状态确定 HTTP 状态码
    status_code = 200 if all_healthy else 503

    # 构建并返回 JSON 响应
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": time.time(),     # Unix 时间戳
            "checks": checks              # 详细的检查结果
        }
    )
