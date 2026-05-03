# AgentClaw 数据流

> 版本：6.2 | 2026-05

---

## 1. 主请求数据流 (POST /ask)

```
 ┌──────┐    ┌──────────┐    ┌───────────┐    ┌──────────────┐
 │Client│    │FastAPI   │    │Security   │    │agent/core.py │
 │      │    │server.py │    │middleware │    │(Facade)      │
 └──┬───┘    └────┬─────┘    └─────┬─────┘    └──────┬───────┘
    │              │                │                  │
    │ POST /ask    │                │                  │
    │──────────────>                │                  │
    │              │                │                  │
    │              │ 1. API Key     │                  │
    │              │    验证         │                  │
    │              │                │                  │
    │              │ 2. 限流检查    │                  │
    │              │───────────────>│                  │
    │              │                │ TokenBucket      │
    │              │<───────────────│ consume()        │
    │              │                │                  │
    │              │ 3. 创建 Trace  │                  │
    │              │─────────────────────────────────>│
    │              │                │                  │
    │              │                │                  │
    │              │ 4. invoke      │                  │
    │              │─────────────────────────────────>│
    │              │                │                  │
    │              │                │            ┌─────┴──────┐
    │              │                │            │ ReAct Loop │
    │              │                │            │            │
    │              │                │            │ tool call   │
    │              │                │            │──────>     │
    │              │                │            │ retry?     │
    │              │                │            │<──────     │
    │              │                │            │ LLM call   │
    │              │                │            │──────>     │
    │              │                │            │<──────     │
    │              │                │            └────────────┘
    │              │                │                  │
    │              │ 5. Response    │                  │
    │              │<──────────────────────────────────│
    │              │                │                  │
    │              │ 6. Finalize    │                  │
    │              │ Trace          │                  │
    │<─────────────│                │                  │
    │              │                │                  │
```

### 详细请求生命周期

```
POST /ask { "text": "搜索 LangGraph 最新版本" }
  │
  ├─ [1] FastAPI 路由: ask()
  │     ├─ trace_id = TraceChain.create_trace(user_message)
  │     └─ span = TraceChain.start_span("agent_reasoning")
  │
  ├─ [2] SecurityMiddleware 检查
  │     ├─ TokenBucket.consume("llm")  → 60 RPM 限流
  │     ├─ SQL注入检测                  → request.text
  │     └─ XSS检测                      → request.text
  │
  ├─ [3] agent.get_react_agent()
  │     ├─ GuardedChatModel._generate()  → LLMGuard.chat()
  │     │     ├─ LLMGuard 超时控制 (timeout=30s)
  │     │     ├─ LLMRetryPolicy 重试 (max_attempts=3)
  │     │     │     └─ 指数退避: 1s → 2s → 4s + jitter
  │     │     ├─ LLMErrorClassifier 分类错误
  │     │     ├─ 成功 → LLMCache.put() 写入缓存
  │     │     └─ 全部失败 → FallbackConfig 降级消息
  │     │
  │     ├─ LLM 决定调用工具 → 返回 tool_calls
  │     ├─ RegistryAdapter 包装
  │     │     ├─ registry.execute(tool_name, args)
  │     │     │     ├─ ErrorChain.execute() 包裹
  │     │     │     │     ├─ ErrorClassifier 分类
  │     │     │     │     ├─ CircuitBreaker 检查
  │     │     │     │     ├─ RetryPolicy 重试
  │     │     │     │     └─ Fallback 回调 / 降级值
  │     │     │     │
  │     │     │     ├─ rate_limiter.consume() 限流
  │     │     │     ├─ 实际执行工具函数
  │     │     │     └─ 记录 feedback 信号
  │     │     │
  │     │     └─ check_anti_pattern(tool_name)  # 学习系统
  │     │
  │     └─ 循环直到 LLM 给出最终答案
  │
  ├─ [4] 最终响应
  │     ├─ TraceChain.finish_span(span)
  │     ├─ TraceChain.finalize_trace(trace)
  │     ├─ metrics.observe_request(duration, status)
  │     └─ 返回 { "response": "...", "trace_id": "..." }
  │
  └─ [5] (后台) EvolutionManager 周期执行
        ├─ 收集 200 条 feedback
        ├─ ExperienceLearner.learn() → 策略/反模式
        ├─ AdaptiveOptimizer.optimize() → 权重/提示词
        └─ EvolutionManager.evaluate() → 进化报告
```

---

## 2. 工具执行数据流 (Registry.execute)

```
 RegistryAdapter  ──→  Registry.execute ──→  ErrorChain.execute ──→  实际函数
       │                     │                      │                    │
       │  工具名 + 参数      │                      │                    │
       │────────────────────>│                      │                    │
       │                     │                      │                    │
       │              ┌──────┴──────┐               │                    │
       │              │ Step 1:     │               │                    │
       │              │ 参数验证    │               │                    │
       │              │ (required,  │               │                    │
       │              │  type, enum)│               │                    │
       │              └─────────────┘               │                    │
       │                     │                      │                    │
       │              ┌──────┴──────┐               │                    │
       │              │ Step 2:     │               │                    │
       │              │ 速率限制    │               │                    │
       │              │ (consume)   │               │                    │
       │              └─────────────┘               │                    │
       │                     │                      │                    │
       │                     │   ErrorChain 包裹    │                    │
       │                     │─────────────────────>│                    │
       │                     │                      │                    │
       │                     │               ┌──────┴──────┐            │
       │                     │               │ Step 3a:    │            │
       │                     │               │ Circuit     │            │
       │                     │               │ Breaker     │            │
       │                     │               │ (closed?)   │            │
       │                     │               └─────────────┘            │
       │                     │                      │                    │
       │                     │               ┌──────┴──────┐            │
       │                     │               │ Step 3b:    │            │
       │                     │               │ Execute     │───────────>│
       │                     │               │ (with retry)│            │
       │                     │               └─────────────┘            │
       │                     │                      │                    │
       │                     │               ┌──────┴──────┐            │
       │                     │               │ Step 3c:    │            │
       │                     │               │ if Error:   │            │
       │                     │               │ 分类 + 重试  │            │
       │                     │               │ 或降级       │            │
       │                     │               └─────────────┘            │
       │                     │                      │                    │
       │                     │   Result / Error     │                    │
       │                     │<─────────────────────│                    │
       │                     │                      │                    │
       │              ┌──────┴──────┐               │                    │
       │              │ Step 4:     │               │                    │
       │              │ 统计更新    │               │                    │
       │              │ (call_count, │               │                    │
       │              │  latency)   │               │                    │
       │              └─────────────┘               │                    │
       │                     │                      │                    │
       │              ┌──────┴──────┐               │                    │
       │              │ Step 5:     │               │                    │
       │              │ Feedback    │               │                    │
       │              │ 记录         │               │                    │
       │              └─────────────┘               │                    │
       │                     │                      │                    │
       │  结果              │                      │                    │
       │<────────────────────│                      │                    │
```

---

## 3. 学习系统反馈闭环

```
                     ┌──────────────────────────────┐
                     │     每轮工具执行               │
                     │                              │
                     │ registry_adapter              │
                     │  .make_wrapper()              │
                     └──────────────┬───────────────┘
                                    │
                                    │ FeedbackSignal
                                    │ (tool_name, success,
                                    │  latency, error_type)
                                    ▼
                     ┌──────────────────────────────┐
                     │   FeedbackCollector           │
                     │   (环形缓冲区, max 10,000)    │
                     │                              │
                     │   每 50 条 → JSONL 持久化     │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │  EvolutionManager (后台线程)   │
                     │  周期: 1 小时 (默认)           │
                     │                              │
                     │  1. collect(200, feedbacks)   │
                     │  2. learn(strategies)         │
                     │  3. optimize(weights)         │
                     │  4. evaluate()                │
                     └──────┬───────────┬───────────┘
                            │           │
                            ▼           ▼
               ┌──────────────────┐  ┌────────────────────┐
               │ ExperienceLearner│  │ AdaptiveOptimizer  │
               │                  │  │                    │
               │ 策略挖掘:        │  │ 路由权重:           │
               │ TF-IDF + Wilson  │  │ α = 1/(1+0.1n)    │
               │                  │  │                    │
               │ 反模式检测        │  │ 提示词优化          │
               │ (failure_path)   │  │ (failure_category) │
               │                  │  │                    │
               │ 时间加权衰减      │  │ 工具偏好矩阵        │
               │ (7天半衰期)       │  │ (context→tool)     │
               │                  │  │                    │
               │ 策略修剪          │  │                    │
               │ (>30d, eff<0.3)  │  │                    │
               └────────┬─────────┘  └────────┬───────────┘
                        │                     │
                        ▼                     ▼
               ┌──────────────────────────────────────────┐
               │         回写至 Agent 运行时                │
               │                                          │
               │ • Top-3 策略注入 Agent Prompt              │
               │   (effectiveness > 0.5)                   │
               │ • 反模式 → RegistryAdapter                │
               │   check_anti_pattern()                    │
               │ • 权重 → ToolRegistry 排序                 │
               │ • 进化报告 → /metrics + demo              │
               └──────────────────────────────────────────┘
```

---

## 4. AOI 闭环工作流数据流

```
 ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
 │Stage 1:  │    │Stage 2:  │    │Stage 3:  │    │Stage 4:  │
 │Defect    │───>│Param     │───>│Config    │───>│Verifier  │
 │Analyst   │    │Optimizer │    │Executor  │    │          │
 └──────────┘    └──────────┘    └──────────┘    └──────────┘
      │               │               │               │
      ▼               ▼               ▼               ▼
 ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
 │AOI       │    │RAG       │    │XML       │    │LLM       │
 │Engine    │    │Search    │    │Config    │    │Compare   │
 │(检测)     │    │(历史案例) │    │(写参数)   │    │(前后对比) │
 └──────────┘    └──────────┘    └──────────┘    └──────────┘

Stage 1 (defect_analyst):
  输入: PCB image path
  处理: AOIEngine.detect(path) → Defect[]
        LLM 分析缺陷 + 判断是否需要调参
  决策: should_tune? → True: Stage 2 | False: 返回结果

Stage 2 (param_optimizer):
  输入: 当前缺陷报告 + XML 配置
  处理: RAG 搜索 aoi_cases.json 相似案例
        LLM 推荐最优参数 (曝光/阈值/边缘检测参数)
  输出: 推荐参数字典

Stage 3 (config_executor):
  输入: 推荐参数
  处理: xml_config_write(params) → 写入 XML
        AOIEngine.detect(path) → 新检测结果
  输出: 新检测结果 + 配置快照

Stage 4 (verifier):
  输入: 旧结果 + 新结果
  处理: LLM 评估改善程度
        规则指标: defect_count_ratio, severity_change
  输出: Verdict (PASS/FAIL/PARTIAL) + 改善报告
```

---

## 5. 三链联动数据流

```
                    ┌─────────────────────┐
                    │    用户请求          │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ TraceChain          │
                    │ 创建一个 Trace       │
                    │ 包含 N 个 Span       │
                    │ 每个 Phase 一个 Span │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ LLMGuard            │
                    │ 容错 LLM 调用        │
                    │                     │
                    │ 成功 │ 降级 │ 失败   │
                    └──┬──┘──┬───┘──┬────┘
                       │     │      │
          ┌────────────┘     │      └────────────┐
          ▼                  ▼                   ▼
   ┌──────────────┐  ┌──────────────┐  ┌────────────────┐
   │ TraceChain   │  │ ErrorChain   │  │ ErrorChain     │
   │ Span OK      │  │ 重试 / 降级   │  │ CircuitBreaker │
   │              │  │              │  │ open → 熔断    │
   │ metrics      │  │ TraceChain   │  │                │
   │ observe_ok   │  │ Span Warning │  │ TraceChain     │
   └──────────────┘  └──────────────┘  │ Span Error     │
                                        │                │
                                        │ metrics        │
                                        │ observe_error  │
                                        └────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Response (带 trace_id)│
                    └─────────────────────┘
```

---

## 6. 指标采集数据流

```
 ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
 │ TraceChain      │     │ LLMGuard         │     │ ToolRegistry    │
 │                 │     │                  │     │                 │
 │ end_trace() ───>│     │ chat() ─────────>│     │ execute() ─────>│
 │ observe_span()  │     │ observe_llm_call │     │ observe_tool    │
 └────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
          │                      │                         │
          ▼                      ▼                         ▼
 ┌───────────────────────────────────────────────────────────────┐
 │                    core/metrics.py                            │
 │                                                               │
 │  agent_request_duration_seconds  (Histogram)                  │
 │  agent_requests_total             (Counter, status)           │
 │  agent_tokens_total               (Counter, direction)        │
 │  agent_llm_calls_total            (Counter, model/status)     │
 │  agent_llm_errors_total           (Counter, error_type)       │
 │  agent_llm_latency_seconds        (Histogram, model)          │
 │  agent_tool_calls_total           (Counter, tool/status)      │
 │  agent_tool_latency_seconds       (Histogram, tool)           │
 │  agent_spans_total                (Counter, kind/status)      │
 │                                                               │
 │  System gauges: uptime, memory_rss, tool_count,               │
 │                 active_sessions, chromadb_health               │
 └─────────────────────────────────────┬─────────────────────────┘
                                       │
                                       ▼
 ┌───────────────────────────────────────────────┐
 │  FastAPI /metrics 端点                         │
 │  prometheus_client.make_asgi_app()             │
 │                                               │
 │  Prometheus Server (scrape :8000/metrics)     │
 │  → Grafana Dashboard                          │
 └───────────────────────────────────────────────┘
```
