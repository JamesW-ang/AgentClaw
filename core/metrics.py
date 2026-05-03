"""
Prometheus 指标定义 — 用于整个系统的可观测性
"""
import time

from prometheus_client import Counter, Gauge, Histogram

# === 请求级别指标 ===

agent_request_duration_seconds = Histogram(
    "agent_request_duration_seconds",
    "Request latency in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

agent_requests_total = Counter(
    "agent_requests_total",
    "Total number of requests",
    labelnames=["status"],
)

agent_tokens_total = Counter(
    "agent_tokens_total",
    "Total token usage",
    labelnames=["direction"],
)

# === LLM 级别指标 ===

agent_llm_calls_total = Counter(
    "agent_llm_calls_total",
    "Total LLM calls",
    labelnames=["model", "status"],
)

agent_llm_errors_total = Counter(
    "agent_llm_errors_total",
    "Total LLM errors by type",
    labelnames=["error_type"],
)

agent_llm_latency_seconds = Histogram(
    "agent_llm_latency_seconds",
    "LLM call latency in seconds",
    labelnames=["model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)

# === 工具级别指标 ===

agent_tool_calls_total = Counter(
    "agent_tool_calls_total",
    "Total tool calls",
    labelnames=["tool", "status"],
)

agent_tool_latency_seconds = Histogram(
    "agent_tool_latency_seconds",
    "Tool call latency in seconds",
    labelnames=["tool"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# === Span/Trace 级别指标 ===

agent_spans_total = Counter(
    "agent_spans_total",
    "Total spans by kind and status",
    labelnames=["kind", "status"],
)

# === 系统级别 Gauge ===

agent_uptime_seconds = Gauge("agent_uptime_seconds", "Service uptime in seconds")
agent_memory_mb = Gauge("agent_memory_mb", "Process RSS memory in MB")
agent_tool_count = Gauge("agent_tool_count", "Number of registered tools")
agent_active_sessions = Gauge("agent_active_sessions", "Number of active sessions")

_start_time = time.time()


def observe_request(status: str, duration: float):
    agent_requests_total.labels(status=status).inc()
    agent_request_duration_seconds.observe(duration)


def observe_llm_call(model: str, status: str, duration: float, tokens_in: int = 0, tokens_out: int = 0):
    agent_llm_calls_total.labels(model=model, status=status).inc()
    agent_llm_latency_seconds.labels(model=model).observe(duration)
    if tokens_in:
        agent_tokens_total.labels(direction="in").inc(tokens_in)
    if tokens_out:
        agent_tokens_total.labels(direction="out").inc(tokens_out)


def observe_llm_error(error_type: str):
    agent_llm_errors_total.labels(error_type=error_type).inc()


def observe_tool_call(tool: str, status: str, duration: float):
    agent_tool_calls_total.labels(tool=tool, status=status).inc()
    agent_tool_latency_seconds.labels(tool=tool).observe(duration)


def observe_span(kind: str, status: str):
    agent_spans_total.labels(kind=kind, status=status).inc()


def update_system_gauges(tool_count: int = 0, active_sessions: int = 0):
    import psutil
    agent_uptime_seconds.set(time.time() - _start_time)
    agent_memory_mb.set(psutil.Process().memory_info().rss / (1024 * 1024))
    if tool_count:
        agent_tool_count.set(tool_count)
    if active_sessions:
        agent_active_sessions.set(active_sessions)
