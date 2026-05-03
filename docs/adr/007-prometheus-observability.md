# ADR-007: Prometheus Metrics for Runtime Observability

**状态**：已采纳 | **日期**：2026-05 | **决定者**：架构组

## 背景

系统需要实时了解运行时状态：请求延迟、LLM 调用成功率、工具执行分布、系统资源使用。日志和追踪提供事后分析，但实时指标需要专门的度量系统。

## 方案

使用 **Prometheus (`prometheus_client`)** 作为指标系统，定义 9 个指标族，通过 `/metrics` 端点暴露。

## 考虑过的选项

### 选项 A：自建日志解析 + 统计
- 优点：无额外依赖
- 缺点：延迟高（分钟级）、不可靠（日志格式变更）、无聚合

### 选项 B：Prometheus（选定）
- 优点：行业标准、Grafana 集成、低开销、拉模式稳定

### 选项 C：OpenTelemetry Metrics
- 优点：多后端、标准化
- 缺点：依赖重、本项目指标相对简单

## 指标设计

```python
# 请求级
agent_request_duration_seconds  # Histogram: 请求总延迟
agent_requests_total            # Counter:  请求计数 (status=ok/error)

# LLM 级
agent_llm_calls_total           # Counter:  LLM 调用 (model, status)
agent_llm_errors_total          # Counter:  LLM 错误 (error_type)
agent_llm_latency_seconds       # Histogram: LLM 延迟 (model)

# 工具级
agent_tool_calls_total          # Counter:  工具调用 (tool, status)
agent_tool_latency_seconds      # Histogram: 工具延迟 (tool)

# 系统级
agent_spans_total               # Counter:  追踪 span (kind, status)
agent_system_info               # Gauge:    uptime, memory, tool_count
```

## 采集点分布

| 采集点 | 指标 | 触发 |
|--------|------|------|
| `api/server.py` | request_duration, requests_total | 每次 POST /ask 完成 |
| `core/llm_guard.py` | llm_calls, llm_errors, llm_latency | 每次 LLM chat() 调用 |
| `tools/registry.py` | tool_calls, tool_latency | 每次 registry.execute() |
| `core/trace_chain.py` | spans_total | 每次 span 结束 |
| 定时器 | system_info | 每 15 秒 |

## 后果

- 正：Grafana 面板可实时监控系统健康
- 正：指标驱动告警（如 P99 延迟 > 10s → 告警）
- 正：9 个指标覆盖请求→LLM→工具→系统全栈
- 负：`/metrics` endpoint 增加 ~5ms 的采集开销

## 关联

- ADR-003：三条链都通过 Metrics 暴露运行时状态
- `api/server.py`：`make_asgi_app()` 挂载 `/metrics`
- `core/metrics.py`：指标定义和辅助函数
