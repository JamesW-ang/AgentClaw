# api_server.py - 集成日志 + 健康检查（完整版）
from dotenv import load_dotenv
load_dotenv()

import time
import psutil
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from core.logger import get_logger

logger = get_logger("APIServer")

# Phase 1: TraceChain integration
try:
    from core.trace_chain import TraceChain, span_context, SpanKind, SpanStatus
    _trace_chain = TraceChain(
        persist_dir='data/traces',
        max_memory=200,
        persist_enabled=True,
        console_enabled=False,  # Don't spam console in production
    )
    logger.info("TraceChain 已初始化")
except Exception as e:
    _trace_chain = None
    logger.warning(f"TraceChain 初始化失败 (不影响主功能): {e}")

START_TIME = time.time()
VERSION = "6.1.0"

app = FastAPI(title="AgentClaw API", version=VERSION)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)

from tools.security import SecurityMiddleware
from core.config import settings
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class APIKeyMiddleware(BaseHTTPMiddleware):
    SKIP_PATHS = {"/health", "/docs", "/openapi.json", "/health/detailed"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        api_key = settings.API_KEY
        if not api_key:
            return await call_next(request)

        client_key = request.headers.get("X-API-Key", "")
        if client_key != api_key:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key"}
            )
        return await call_next(request)

app.add_middleware(SecurityMiddleware)
app.add_middleware(APIKeyMiddleware)

# 集成健康检查模块
from tools.health import health_check as _detailed_health_check

class Question(BaseModel):
    question: str
    session_id: str = "default"

class Answer(BaseModel):
    answer: str
    usage: dict

logger.info(f"AgentClaw API v{VERSION} 启动中...")

@app.post("/ask", response_model=Answer)
async def ask(req: Question):
    """主问答接口，集成安全检查、日志记录和请求追踪"""
    logger.info(f"收到请求: session={req.session_id}, q={req.question[:50]}")
    start = time.time()

    # Phase 1: Start TraceChain trace if available
    trace = None
    if _trace_chain is not None:
        trace = _trace_chain.start_trace(
            request_text=req.question,
            session_id=req.session_id,
        )

    try:
        from agent.core import get_react_agent
        agent_app = get_react_agent()

        if trace is not None:
            with span_context(trace, "agent.react", SpanKind.AGENT) as ctx:
                config = {"configurable": {"thread_id": req.session_id}}
                result = await agent_app.ainvoke(
                    {"messages": [HumanMessage(content=req.question)]}, config
                )
                last_msg = result["messages"][-1]
                ctx.set_output(last_msg.content[:500])
        else:
            config = {"configurable": {"thread_id": req.session_id}}
            result = await agent_app.ainvoke(
                {"messages": [HumanMessage(content=req.question)]}, config
            )
            last_msg = result["messages"][-1]

        elapsed = time.time() - start

        logger.info(f"请求完成: 耗时{elapsed:.2f}s")

        # Phase 1: Finish trace
        if trace is not None:
            trace.finish(status=SpanStatus.OK, response_text=last_msg.content)
            _trace_chain.end_trace(trace)

        usage_data = {
            "thread_id": req.session_id,
            "elapsed": f"{elapsed:.2f}s",
        }
        if trace is not None:
            usage_data["trace_id"] = trace.id

        return Answer(answer=last_msg.content, usage=usage_data)
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"请求失败: {e}")

        # Phase 1: Record error in trace
        if trace is not None:
            trace.finish(status=SpanStatus.ERROR, error=str(e))
            _trace_chain.end_trace(trace)

        usage_data = {
            "thread_id": req.session_id,
            "elapsed": f"{elapsed:.2f}s",
            "error": str(e),
        }
        if trace is not None:
            usage_data["trace_id"] = trace.id

        return Answer(answer=f"处理失败: {e}", usage=usage_data)

@app.get("/health")
async def health():
    """基础健康检查：返回服务状态、运行时间、资源使用"""
    uptime = time.time() - START_TIME
    memory = psutil.Process().memory_info().rss / (1024 * 1024)
    return {
        "status": "ok",
        "service": "agent-api",
        "version": VERSION,
        "uptime_seconds": round(uptime),
        "memory_mb": round(memory, 1),
    }

@app.get("/health/detailed")
async def detailed_health():
    """详细健康检查：ChromaDB + LLM API + 系统内存"""
    return _detailed_health_check()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
