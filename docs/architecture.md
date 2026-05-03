# AgentClaw 架构设计

> 版本：6.2 | 更新：2026-05

---

## 1. 设计目标

AgentClaw 是一个生产级 AI Agent 框架，核心设计目标：

| 目标 | 说明 |
|------|------|
| **可靠性** | LLM 调用容错、自动降级、熔断、重试 — 即使上游 API 不可用也能优雅降级 |
| **可演进** | 自主进化系统通过反馈信号自动优化工具选择、路由权重和提示词策略 |
| **可观测** | 全链路追踪 + Prometheus 指标 + LangFuse 集成，运行时状态透明 |
| **可扩展** | 装饰器驱动的工具注册体系，新增工具只需一行 `@registry.register` |
| **安全** | 命令白名单、路径过滤、SQL/XSS 注入检测、令牌桶限流多层防护 |

---

## 2. 六层架构

```
 ┌─────────────────────────────────────────────────────────┐
 │  Level 6: Monitoring Layer                              │
 │  tools/health.py → /health, /health/detailed           │
 │  core/metrics.py → /metrics (Prometheus)               │
 │  tools/langfuse_adapter.py → LangFuse Cloud            │
 ├─────────────────────────────────────────────────────────┤
 │  Level 5: Service Layer                                 │
 │  api/server.py → FastAPI (REST + SSE streaming)        │
 │  demo/ui.py → Gradio Web UI (8 个 Tab)                 │
 ├─────────────────────────────────────────────────────────┤
 │  Level 4: Orchestration Layer                           │
 │  agent/core.py → LangGraph ReAct Agent                 │
 │  ├─ init_all_tools() → tools/registry                  │
 │  ├─ get_react_agent() → create_react_agent()           │
 │  ├─ init_chains() → ErrorChain + TraceChain            │
 │  └─ init_evolution() → Learning System                 │
 │                                                         │
 │  aoi/workflow.py → LangGraph StateGraph (4 个 Specialist)│
 │  └─ defect_analyst → param_optimizer → config_executor  │
 │     → verifier                                          │
 ├─────────────────────────────────────────────────────────┤
 │  Level 3: Security & Reliability Layer                  │
 │  tools/security.py → TokenBucket 限流 / SQL/XSS 过滤   │
 │  core/llm_guard.py → 超时 / 重试 / 降级 / 熔断 / 缓存  │
 │  core/error_chain.py → 错误分类 / CircuitBreaker        │
 │  core/trace_chain.py → Span 嵌套 / JSONL 持久化         │
 │  core/rate_limiter.py → 令牌桶算法                      │
 │  core/retry.py → 指数退避 + 抖动                        │
 ├─────────────────────────────────────────────────────────┤
 │  Level 2: Tool Layer                                    │
 │  tools/registry.py → 单例注册中心 + @registry.register  │
 │  tools/registry_adapter.py → LangGraph StructuredTool   │
 │  tools/dispatcher.py → 传统 ReAct 调度器 (legacy)       │
 │  ├─ tools/builtin.py    (搜索/计算/文件/命令/代码/RAG)  │
 │  ├─ tools/vision.py     (analyze/ocr/compare)            │
 │  ├─ tools/image_gen.py  (CogView 图片生成)               │
 │  ├─ aoi/engine.py       (AOI 缺陷检测)                   │
 │  ├─ tools/xml_config.py (AOI 参数配置管理)               │
 │  └─ os_tools/           (文件/监控/进程/浏览器/调度)     │
 ├─────────────────────────────────────────────────────────┤
 │  Level 1: Foundation Layer                              │
 │  core/config.py → Frozen dataclass 配置验证             │
 │  core/logger.py → 彩色日志 + 文件轮转 (30 天保留)       │
 └─────────────────────────────────────────────────────────┘
```

### 分层原则

- **严格单向依赖**：上层依赖下层，下层对上层无感知
- **每层独立可测**：各层可独立 mock 下层依赖进行单元测试
- **跨层通信**：仅通过 `agent/core.py` Facade 进行

---

## 3. 模块依赖关系

```
                    ┌──────────────┐
                    │  scripts/    │  Docker / CLI 入口
                    │  main.py     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ agent/core   │  ⬅ 全局编排器 (Facade)
                    │ .py          │
                    └──┬──┬──┬──┬──┘
                       │  │  │  │
          ┌────────────┘  │  │  └──────────────────┐
          ▼               ▼  ▼                     ▼
   ┌────────────┐  ┌──────────────┐  ┌──────────────────────┐
   │ tools/     │  │ core/        │  │ learning/            │
   │ registry   │  │ llm_guard    │  │ evolution            │
   │     │      │  │      │       │  │    │                 │
   │     ▼      │  │      ▼       │  │    ▼                 │
   │ registry_  │  │ guarded_chat │  │ feedback             │
   │ adapter    │  │ _model       │  │    │                 │
   │     │      │  │              │  │    ▼                 │
   │     ▼      │  │ error_chain  │  │ learner              │
   │ builtin    │  │      │       │  │    │                 │
   │ vision     │  │      ▼       │  │    ▼                 │
   │ image_gen  │  │ trace_chain  │  │ optimizer            │
   │ searcher   │  │              │  │                      │
   │ xml_config │  │ metrics      │  │                      │
   │            │  │ rate_limiter │  │                      │
   │ os_tools/  │  │ retry        │  │                      │
   │ aoi/engine │  │ config       │  │                      │
   └────────────┘  └──────────────┘  └──────────────────────┘
```

---

## 4. 核心组件职责

### 4.1 三链联动系统

三条链以"插拔式"设计共存，通过 `agent/core.py` 统一初始化：

```
 ┌─────────────────────────────────────────────────────────────┐
 │                    用户请求                                    │
 │                        │                                      │
 │  ┌─────────────────────┼──────────────────────┐              │
 │  │                     │                      │              │
 │  ▼                     ▼                      ▼              │
 │  ┌──────────────┐ ┌──────────────┐  ┌──────────────┐        │
 │  │ TraceChain   │ │ ErrorChain   │  │ LLMGuard     │        │
 │  │              │ │              │  │              │        │
 │  │ 每个请求 →   │ │ 工具调用 →   │  │ LLM 调用 →   │        │
 │  │ 1 Trace      │ │ 分类 + 重试  │  │ 超时 + 降级  │        │
 │  │ N Spans      │ │ Circuit      │  │ 熔断 + 缓存  │        │
 │  │ JSONL 持久化 │ │ Breaker      │  │              │        │
 │  └──────────────┘ └──────────────┘  └──────────────┘        │
 │                        │                                      │
 │                        ▼                                      │
 │               ┌────────────────┐                              │
 │               │  core/metrics  │  Prometheus 采集              │
 │               └────────────────┘                              │
 └─────────────────────────────────────────────────────────────┘
```

### 4.2 工具注册体系

```python
# 第 1 步：模块导入时自动注册
@registry.register(
    name="web_search",
    description="Search the web...",
    parameters=[...],
    category=ToolCategory.SEARCH
)
def web_search(query: str) -> str:
    ...

# 第 2 步：RegistryAdapter 桥接到 LangGraph
adapter = RegistryAdapter(registry)
langgraph_tools = adapter.get_langgraph_tools()
# → 每个工具生成 Pydantic v2 schema + 包装函数

# 第 3 步：create_react_agent 使用
agent = create_react_agent(llm, langgraph_tools, checkpointer=sqlite_saver)
```

### 4.3 自主进化系统

```
 FeedbackCollector ←─── RegistryAdapter (每次工具执行)
       │
       │ 收集的信号: success, latency, error_type, user_rating
       │ (环形缓冲区, 最大 10,000 条)
       ▼
 ExperienceLearner
       │
       ├── 策略挖掘: TF-IDF 语义匹配 + Wilson 置信区间
       ├── 反模式检测: 从失败路径提取"不该做什么"
       └── 时间加权衰减 (7 天半衰期)
       │
       ▼
 AdaptiveOptimizer
       │
       ├── 路由权重调整: 自适应学习率 α = 1/(1+0.1n)
       ├── 提示词优化: 按失败类别生成改进建议
       └── 工具偏好矩阵: 上下文 → 最佳工具映射
       │
       ▼
 EvolutionManager (后台 daemon 线程, 默认周期 1 小时)
       │
       ├── 策略注入 → Agent Prompt (top-3, effectiveness > 0.5)
       ├── 反模式回写 → RegistryAdapter.check_anti_pattern()
       └── 生成进化报告 → get_evolution_report()
```

---

## 5. 设计模式使用

| 模式 | 位置 | 用途 |
|------|------|------|
| **Singleton** | `tools/registry.py`, `core/llm_guard.py`, `aoi/engine.py`, `core/config.py` | 全局唯一实例，避免状态分散 |
| **Adapter** | `tools/registry_adapter.py`, `core/guarded_chat_model.py` | 桥接不同接口（Registry→LangGraph, LLMGuard→BaseChatModel） |
| **Facade** | `agent/core.py` | 统一编排所有子系统 |
| **Chain of Responsibility** | `core/error_chain.py`, `core/llm_guard.py` (retry chain), `core/trace_chain.py` (span nesting) | 错误分级处理 / 多级重试 / Span 嵌套 |
| **Strategy** | `learning/learner.py`, `core/error_chain.py` (error classification), `aoi/engine.py` (detection mode) | 可互换的策略算法 |
| **Circuit Breaker** | `core/error_chain.py` | 防止级联故障 |
| **Decorator** | `@registry.register`, `@rate_limit`, `@retry_with_backoff`, `@chain.tool_guard` | 非侵入式功能增强 |
| **Observer** | `learning/feedback.py` → `learning/learner.py`, `trace_chain.py` TraceLogHandler | 事件驱动通知 |
| **Pipeline** | `aoi/workflow.py` (4-stage LangGraph StateGraph) | 多步骤顺序处理 |
| **Lazy Initialization** | RAG engine, AOI engine, LangFuse tracer, LLM instance | 按需加载资源 |
| **Immutable Object** | `_ConfigValidator` (frozen dataclass) | 配置不可变性保证 |
| **State Machine** | `CircuitBreaker` (closed/open/half_open), `TaskScheduler` (paused/alive) | 状态驱动的行为切换 |

---

## 6. 包的线数分布

| 包 | 行数 | 占比 | 职责 |
|----|------|------|------|
| `tools/` | 3,075 | 14% | 工具注册 + 22 个内置工具 |
| `core/` | 2,694 | 13% | 基础设施：配置/日志/容错/追踪/指标 |
| `test/` | 2,653 | 12% | 13 个测试模块 |
| `demo/` | 1,971 | 9% | Gradio UI + 学习演示 |
| `learning/` | 1,494 | 7% | 反馈/学习/优化/进化 |
| `eval/` | 1,512 | 7% | Agent 评估框架 |
| `aoi/` | 1,455 | 7% | AOI 检测引擎 + 工作流 |
| `os_tools/` | 720 | 3% | 操作系统工具 |
| `agent/` | 476 | 2% | Agent 核心编排 |
| `api/` | 247 | 1% | FastAPI 服务 |
| `scripts/` | 119 | 1% | Docker 入口 |
| Root `*.py` | 83 | <1% | 兼容包装层 |
| **Total** | **21,381** | 100% | |

---

## 7. 外部依赖图谱

```
                    AgentClaw
                   /    |    \
                  /     |     \
          LangGraph    FastAPI   Gradio
          LangChain    Uvicorn    |
              |           |     gradio
         langchain-    prometheus_
          openai       client
              |           |
          openai       httpx
              |
         ┌────┴────┐
         │         │
    chromadb    sentence-
    langchain-  transformers
    chroma      (bge-small-zh)
         │
    onnxruntime / opencv-python (AOI)
    playwright (browser, optional)
    autogen (multi-agent, optional)
```

---

## 8. 与外部系统集成

| 系统 | 集成方式 | 用途 |
|------|---------|------|
| DeepSeek API | OpenAI-compatible HTTP | 主 LLM 推理 |
| Zhipu GLM-4V | OpenAI-compatible HTTP | 视觉多模态分析 |
| Zhipu CogView | HTTP API | AI 图片生成 |
| SerpAPI | HTTP API | 网络搜索（主要） |
| DuckDuckGo | HTML scraping (fallback) | 网络搜索（降级） |
| LangFuse | LangFuse SDK | LLM 可观测性 |
| ChromaDB | langchain-chroma | 向量存储 / RAG |
| Prometheus | prometheus_client | 运行时指标 |
| SQLite | sqlite3 + langgraph-checkpoint-sqlite | Agent 状态持久化 |

---

## 9. 安全架构

```
请求进入
   │
   ▼
┌─────────────────────────────────────────────────────┐
│ Layer 1: API Key 验证 (api/server.py)                │
│  └─ X-API-Key header → 配置的 API_KEY 比对          │
├─────────────────────────────────────────────────────┤
│ Layer 2: Security 中间件 (tools/security.py)          │
│  ├─ TokenBucket 限流 (60 RPM LLM, 20 RPM search)    │
│  ├─ SQL 注入检测 (敏感关键词匹配)                    │
│  ├─ XSS 检测 (<script>, onerror= 等模式)             │
│  └─ 敏感数据过滤 (API Key 脱敏)                      │
├─────────────────────────────────────────────────────┤
│ Layer 3: 工具级安全                                  │
│  ├─ run_command: 30+ 命令白名单 + 危险模式检测       │
│  ├─ code_execute: AST 检查 + 关键词过滤 + 受限内建    │
│  ├─ file_read: 路径白名单/黑名单 + 扩展名过滤 + 大小限制 │
│  └─ file_write: 路径黑名单 + 危险内容检测 + 自动备份  │
├─────────────────────────────────────────────────────┤
│ Layer 4: LLM 安全 (core/llm_guard.py)                 │
│  └─ 超时控制 + 熔断 + 降级消息（防止无限等待）        │
└─────────────────────────────────────────────────────┘
```
