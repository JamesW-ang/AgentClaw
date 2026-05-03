# AgentClaw API 参考

> 版本：6.1.0 | 更新：2026-05

---

## 1. 概述

| 项目 | 说明 |
|------|------|
| 基础 URL | `http://localhost:8000` |
| 协议 | HTTP/1.1 |
| 内容类型 | `application/json` |
| 认证方式 | `X-API-Key` 请求头（可选，取决于是否设置了 `API_KEY` 环境变量） |
| 超时 | 无显式 HTTP 超时，工具级超时由 `TOOL_TIMEOUT` 控制（默认 30s） |

---

## 2. 认证

通过环境变量 `API_KEY` 控制：

- **未设置** `API_KEY`：所有端点无需认证
- **已设置** `API_KEY`：除 `/health`、`/health/detailed`、`/metrics`、`/docs`、`/openapi.json` 外的所有端点需要提供 `X-API-Key` 请求头

```
X-API-Key: your-secret-key
```

认证失败返回 `401 Unauthorized`。

---

## 3. 安全中间件

所有请求经过 `SecurityMiddleware` 过滤：

| 规则 | 违规响应 |
|------|---------|
| 每 IP 30 次/60 秒 | `429 Too Many Requests` |
| POST 正文 > 5000 字符 | `413 Payload Too Large` |
| SQL 注入模式检测 | `400 Bad Request` |
| XSS 模式检测 | `400 Bad Request` |
| 敏感数据检测（password、secret、api_key 等） | `400 Bad Request` |

---

## 4. 端点

### 4.1 `POST /ask` — 同步问答

发送一个问题，返回 Agent 的完整回答。

**请求体：**

```json
{
  "question": "string (必填)",
  "session_id": "string (可选, 默认 \"default\")"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | 是 | 用户问题 |
| `session_id` | string | 否 | 会话标识，用于保持对话历史 |

**响应（200 OK）：**

```json
{
  "answer": "string",
  "usage": {
    "thread_id": "string",
    "elapsed": "string (如 \"2.45s\")",
    "trace_id": "string (可选, TraceChain 启用时返回)"
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` | string | Agent 的完整回答文本 |
| `usage.thread_id` | string | 会话线程 ID（同 session_id） |
| `usage.elapsed` | string | 处理耗时 |
| `usage.trace_id` | string | 链路追踪 ID（仅当 TraceChain 启用） |

**注意：** 处理异常时仍返回 `200 OK`，错误信息包含在 `answer` 字段中。

**示例：**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"question": "搜索 Python 3.13 的新特性", "session_id": "user-001"}'
```

```json
{
  "answer": "Python 3.13 的主要新特性包括：\n1. 新的交互式解释器...",
  "usage": {
    "thread_id": "user-001",
    "elapsed": "3.21s",
    "trace_id": "trace-a1b2c3"
  }
}
```

---

### 4.2 `POST /ask/stream` — SSE 流式问答

发送一个问题，通过 Server-Sent Events 实时流式返回回答。

**请求体：** 同 `POST /ask`

**响应：** `text/event-stream`

每个事件格式为 `data: <JSON>\n\n`，共 4 种事件类型：

#### token — 文本令牌

```json
{"type": "token", "content": "Pyth"}
```

逐个返回 Agent 生成的文本片段。

#### tool_call — 工具调用

```json
{"type": "tool_call", "id": "call_abc123", "name": "web_search", "args": "{\"query\": \"...\"}", "index": 0}
```

Agent 调用工具时实时推送。`args` 为 JSON 字符串。

#### done — 流结束

```json
{"type": "done"}
```

流式传输正常完成。

#### error — 错误

```json
{"type": "error", "content": "error message"}
```

处理过程中发生错误。

**示例：**

```bash
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"question": "1+1等于几"}'
```

```
data: {"type": "token", "content": "1"}

data: {"type": "token", "content": "+"}

data: {"type": "token", "content": "1"}

data: {"type": "token", "content": "等于"}

data: {"type": "token", "content": "2"}

data: {"type": "done"}
```

**Python 客户端示例：**

```python
import httpx
import json

async def stream_ask(question: str, api_key: str = ""):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/ask/stream",
            json={"question": question, "session_id": "demo"},
            headers=headers,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    if event["type"] == "token":
                        print(event["content"], end="", flush=True)
                    elif event["type"] == "tool_call":
                        print(f"\n[调用工具: {event['name']}]", flush=True)
                    elif event["type"] == "done":
                        print("\n--- 完成 ---")
                    elif event["type"] == "error":
                        print(f"\n错误: {event['content']}")

import asyncio
asyncio.run(stream_ask("搜索AgentClaw项目"))
```

---

### 4.3 `GET /health` — 基础健康检查

**响应（200 OK）：**

```json
{
  "status": "ok",
  "service": "agent-api",
  "version": "6.1.0",
  "uptime_seconds": 12345,
  "memory_mb": 156.7
}
```

| 字段 | 说明 |
|------|------|
| `status` | 固定 `"ok"` |
| `service` | 服务名称 |
| `version` | API 版本 |
| `uptime_seconds` | 进程运行时长（秒） |
| `memory_mb` | 当前进程 RSS 内存（MB） |

**示例：**

```bash
curl http://localhost:8000/health
```

---

### 4.4 `GET /health/detailed` — 详细健康检查

检查下游依赖状态。

**响应（200 OK — 所有检查通过）：**

```json
{
  "status": "healthy",
  "timestamp": 1712345678.123,
  "checks": {
    "chromadb": {"status": "ok"},
    "llm_api": {"status": "ok", "status_code": 200},
    "memory": {"status": "ok", "percent": 45.2, "available_mb": 8192}
  }
}
```

**响应（503 Service Unavailable — 至少一项关键检查失败）：**

```json
{
  "status": "degraded",
  "timestamp": 1712345678.123,
  "checks": {
    "chromadb": {"status": "error", "error": "Connection refused"},
    "llm_api": {"status": "ok", "status_code": 200},
    "memory": {"status": "ok", "percent": 45.2, "available_mb": 8192}
  }
}
```

**子检查说明：**

| 检查项 | 方式 | 故障条件 |
|--------|------|---------|
| `chromadb` | `PersistentClient` 心跳到 `./data/chroma_db` | 连接失败 |
| `llm_api` | `HEAD https://api.deepseek.com`（5s 超时） | 非 2xx 或超时 |
| `memory` | `psutil.virtual_memory()` | ≥ 85%（warning，不触发 degraded） |

---

### 4.5 `GET /metrics` — Prometheus 指标

暴露 Prometheus 格式的指标数据。

**可用指标族：**

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `agentclaw_requests_total` | Counter | 请求总数 |
| `agentclaw_request_duration_seconds` | Histogram | 请求延迟分布 |
| `agentclaw_llm_calls_total` | Counter | LLM 调用次数（按 model、status） |
| `agentclaw_llm_errors_total` | Counter | LLM 错误次数（按 error_type） |
| `agentclaw_llm_duration_seconds` | Histogram | LLM 调用延迟 |
| `agentclaw_tool_calls_total` | Counter | 工具调用次数（按 tool、status） |
| `agentclaw_tool_duration_seconds` | Histogram | 工具调用延迟 |
| `agentclaw_span_total` | Counter | Span 数量 |
| `agentclaw_system_info` | Gauge | 系统信息（工具数量等） |

**示例：**

```bash
curl http://localhost:8000/metrics
```

```
# HELP agentclaw_tool_calls_total Total tool calls
# TYPE agentclaw_tool_calls_total counter
agentclaw_tool_calls_total{tool="web_search",status="success"} 142.0
agentclaw_tool_calls_total{tool="calculator",status="success"} 89.0
agentclaw_tool_calls_total{tool="web_search",status="error"} 3.0
...
```

---

## 5. 错误处理

| HTTP 状态码 | 场景 |
|-------------|------|
| `200` | 始终 — 业务错误也返回 200，错误信息嵌入响应体 |
| `400` | 安全中间件检测到恶意内容 |
| `401` | API Key 缺失或错误 |
| `413` | 请求体超过 5000 字符 |
| `429` | 触发频率限制 |
| `503` | `/health/detailed` 检测到关键依赖不可用 |

---

## 6. CORS

所有端点的 CORS 配置为完全开放：

```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: *
Access-Control-Allow-Headers: *
```

---

## 7. 完整 curl 示例

```bash
# 同步问答
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "你好，介绍一下你自己"}' | jq .

# 流式问答
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "计算 (3+5)*12"}'

# 基础健康
curl -s http://localhost:8000/health | jq .

# 详细健康
curl -s http://localhost:8000/health/detailed | jq .

# 指标
curl -s http://localhost:8000/metrics | head -20
```

---

## 8. 相关文档

- [快速入门教程](quick-start-tutorial.md) — 从零启动 API 服务
- [配置参考](configuration-reference.md) — API_PORT、API_KEY 等配置项
- [架构设计](architecture.md) — API 在六层架构中的位置
