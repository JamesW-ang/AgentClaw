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

from core.logger import get_logger

logger = get_logger("HealthCheck")

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

    checks = {}
    all_healthy = True

    # ========== 检查 1: ChromaDB 连接 ==========
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./data/chroma_db")
        client.heartbeat()
        checks["chromadb"] = {"status": "ok"}
        logger.debug("ChromaDB 健康检查通过")
    except Exception as e:
        checks["chromadb"] = {"status": "error", "error": str(e)}
        all_healthy = False
        logger.warning(f"ChromaDB 健康检查失败: {e}")

    # ========== 检查 2: LLM API 可达性 ==========
    try:
        import httpx
        resp = httpx.head("https://api.deepseek.com", timeout=5.0)
        checks["llm_api"] = {"status": "ok", "status_code": resp.status_code}
        logger.debug(f"LLM API 可达性检查通过 (HTTP {resp.status_code})")
    except Exception as e:
        checks["llm_api"] = {"status": "error", "error": str(e)}
        all_healthy = False
        logger.warning(f"LLM API 可达性检查失败: {e}")

    # ========== 检查 3: 系统内存使用 ==========
    mem = psutil.virtual_memory()
    mem_status = "ok" if mem.percent < 85 else "warning"
    checks["memory"] = {
        "status": mem_status,
        "percent": mem.percent,
        "available_mb": mem.available // (1024 * 1024),
    }
    if mem.percent >= 85:
        logger.warning(f"内存使用率偏高: {mem.percent:.1f}% (可用 {mem.available // (1024 * 1024)}MB)")

    status_code = 200 if all_healthy else 503
    status_text = "healthy" if all_healthy else "degraded"
    logger.info(f"健康检查完成: {status_text} (ChromaDB={checks['chromadb']['status']}, LLM={checks['llm_api']['status']}, Mem={checks['memory']['status']})")

    return JSONResponse(
        status_code=status_code,
        content={
            "status": status_text,
            "timestamp": time.time(),
            "checks": checks,
        }
    )
