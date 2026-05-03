# Changelog

All notable changes to AgentClaw are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [6.3.0] - 2026-05-03

### Added
- **UI Tab8 自反馈学习**: 进化状态面板 + 学习策略展示 + 反模式预警 + 进化历史图表
- **UI Tab9 日志查询**: 日志浏览/搜索/过滤 + JSON 导入导出 + 一键清空
- **低内存模式**: `CHROMA_ENABLED=false` 跳过 HuggingFace 嵌入模型和 ChromaDB 加载，仅用轻量 TF-IDF 检索，适配 8GB 设备
- **JSON 容错修复** (`core/guarded_chat_model.py:_repair_tool_args`): 自动修复 LLM 返回的畸形 JSON（缺引号、缺括号、尾逗号）
- **工具执行统计** (`tools/text_stats.py`): 新增字符/行/词数统计工具
- `create_llm()` 工厂函数: 统一 LLM 实例创建入口，消除 4 个调用点的 ChatOpenAI 重复代码

### Changed
- **重依赖延迟加载**: `numpy` (~150MB)、`openai` (~50MB)、`psutil` (~5MB) 改为首次使用时导入，降低启动内存
- **UI 布局优化**: 参数区改用 Accordion 折叠，减少文本密度；AOI 独立检测 Tab 合并至检测分析
- **Gradio 升级**: 5.x → 6.14.0
- **LLM 工厂统一**: `agent/core.py`、`aoi/workflow.py` 全部改用 `create_llm()`
- **UI Tab 重编号**: 7个Tab → 9个Tab（合并 AOI 独立检测，新增自反馈学习/日志查询）

### Fixed
- **ReAct ChatInterface 死锁**: `threading.Lock()` + generator `yield` 导致锁永不释放，改为 `return` + 唯一 `thread_id`
- **LangGraph `pending_sends` 错误**: 固定 thread_id 导致从过期 checkpoint 恢复，改为时间戳唯一 ID
- **RAG 上传断连**: ChromaDB 嵌入阻塞主线程 → 改用 generator yield 保持连接
- **递归限制 25**: ReAct agent 25+ 工具陷入循环，AOI 分析改为直接 LLM 调用
- **`run_command` /dev/null 误拦截**: 危险模式正则 `>\s*/dev/` 误伤安全设备 `/dev/null` 等，改用负向先行断言
- **RAG 守护线程被杀**: ChromaDB 写入在线程被 Gradio 清理时丢失，改回同步写入

## [6.2.0] - 2026-04

### Added
- **Three-chain linkage system**: LLMGuard fault-tolerance layer, ErrorChain error
  handling chain, TraceChain distributed tracing — operating in coordinated fashion
- **AOI detection engine** (`aoi/engine.py`): Hybrid detection (traditional CV + YOLO
  ONNX inference), 8 defect types, CLAHE preprocessing, IoU deduplication
- **AOI closed-loop workflow** (`aoi/workflow.py`): LangGraph StateGraph with 4
  specialist agents (defect_analyst → param_optimizer → config_executor → verifier)
- **XML configuration tool** (`tools/xml_config.py`): Read/write/diff/backup/validate
  for AOI XML parameter files
- **Agent evaluation system** (`eval/`): Test case framework (6 task types),
  RAGAS-style metrics (Context Precision, Recall, Faithfulness, Answer Relevancy,
  Tool Selection Accuracy), JSON + Markdown reporting
- **Evaluation test cases**: 30+ cases spanning single tool, multi-tool chain,
  conditional branching, error recovery, multi-agent collaboration, RAG retrieval

### Changed
- Tool inventory expanded from 18 to 22 registered tools
- `aoi/engine.register_aoi_tools()` integrates AOI detection as a registrable tool
- README: Architecture description updated, tool inventory listing, eval system docs

## [6.1.0] - 2026-04

### Added
- TokenBucket rate limiter (`core/rate_limiter.py`) replacing sliding-window algorithm:
  global LLM limiter (60 RPM) and search limiter (20 RPM)
- `@rate_limit` decorator for per-function rate control
- Security middleware (`tools/security.py`): SQL injection detection, XSS filtering,
  sensitive data masking, request size limits
- Health check module (`tools/health.py`): ChromaDB heartbeat, LLM API reachability,
  system memory monitoring
- Detailed health endpoint `GET /health/detailed`

### Changed
- `tools/registry.execute()` integrated with `_llm_limiter.consume()` for rate limiting
- Retry decorator (`core/retry.py`) promoted to first-class infrastructure, integrated
  into main call chain
- Security upgraded from basic IP-based sliding window to TokenBucket algorithm

## [6.0.0] - 2026-04

### Added
- **Five-layer architecture**: Config → Registry → Security → Retry → RateLimiter → LLM
- **Singleton ToolRegistry** (`tools/registry.py`): `@registry.register` decorator,
  7 tool categories, parameter validation, OpenAI function-calling schema generation
- **18 built-in tools**: web_search, calculator, file_read, run_command, code_execute,
  file_write, sys_monitor, sys_process_list, sys_disk_info, process_start/stop/list,
  vision_analyze/ocr/compare, image_generate, knowledge_search, browser tools
- **RAG engine** (`tools/searcher.py`): TF-IDF vector store, document chunking,
  BM25 retrieval, shared knowledge base
- **RegistryAdapter** (`tools/registry_adapter.py`): LangGraph StructuredTool bridge
- **GuardedChatModel** (`core/guarded_chat_model.py`): BaseChatModel wrapper for
  LLMGuard integration
- **Autonomous evolution system** (`learning/`): FeedbackCollector, ExperienceLearner
  (Wilson confidence, TF-IDF strategy matching, time-weighted decay),
  AdaptiveOptimizer (adaptive learning rate, tool preference matrix),
  EvolutionManager (background daemon loop, strategy pruning)
- **FastAPI server** (`api/server.py`): POST /ask, POST /ask/stream, GET /health,
  CORS, API key middleware, SSE streaming
- **Gradio Web UI** (`demo/ui.py`): Multi-tab interface (Chat, Knowledge Base, Tools,
  Evolution, AOI Detection)
- **os_tools module**: Secure file write, system monitoring, process management,
  Playwright browser automation, task scheduler
- **Multi-modal routing** (`tools/multimodal_router.py`): Text-only ↔ multimodal
  provider switching
- **LLMGuard** (`core/llm_guard.py`): Timeout control, smart retry with exponential
  backoff, fallback chain, circuit breaker, LLM cache with TTL
- **ErrorChain** (`core/error_chain.py`): Error classification (9 categories),
  severity levels, auto-retry, circuit breaker, graceful degradation, `tool_guard`
  decorator
- **TraceChain** (`core/trace_chain.py`): OpenTelemetry-inspired spans/traces,
  JSONL persistence, nested spans, timeline generator, TraceLogHandler
- **Vision tools** (`tools/vision.py`): GLM-4V / GPT-4o integration, 3 tools,
  local file / URL / base64 input support
- **Docker support**: Dockerfile, docker-compose.yml, .env example
- **pyproject.toml**: Project metadata, Python 3.10+ requirement
- MIT License

[Unreleased]: https://github.com/JamesW-ang/AgentClaw/compare/v6.2.0...HEAD
[6.2.0]: https://github.com/JamesW-ang/AgentClaw/compare/v6.1.0...v6.2.0
[6.1.0]: https://github.com/JamesW-ang/AgentClaw/compare/v6.0.0...v6.1.0
[6.0.0]: https://github.com/JamesW-ang/AgentClaw/releases/tag/v6.0.0
