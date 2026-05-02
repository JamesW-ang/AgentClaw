# Changelog

All notable changes to AgentClaw are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Root-level backward-compatible entry point wrappers (`demo_ui.py`, `aoi_workflow.py`,
  `eval_runner.py`, `eval_cases.py`, `demo_self_learning.py`, `xml_config_tool.py`,
  `langfuse_adapter.py`) for module refactor migration
- Prometheus metrics system (`core/metrics.py`) with 9 metric families:
  request latency, LLM call count/errors/latency, tool call count/latency,
  span observability, system gauges
- `/metrics` endpoint mounted on FastAPI server
- LangFuse callback handler integration in API server and agent core
- 5 new test modules: `test_api.py`, `test_error_chain.py`, `test_learning.py`,
  `test_llm_guard.py`, `test_trace_chain.py` (214+ total test cases)
- `GuardedChatModel.bind_tools()` override for LangGraph 0.3.x compatibility
- ChromaDB + bge-small-zh semantic search as primary RAG backend with TF-IDF fallback
- Learning system closed-loop: strategy injection into agent prompt, anti-pattern
  detection in RegistryAdapter, tool preference matrix updates
- `.gitignore` entries for `*.db` and `data/traces/`
- `aoi_cases.json` / `aoi_workflow.py` AOI case database and standalone entry point

### Changed
- **Refactor**: 19 root-level modules reorganized into subpackages (`agent/`, `core/`,
  `tools/`, `learning/`, `aoi/`, `api/`, `demo/`, `eval/`, `scripts/`)
- **Persistence**: `MemorySaver` replaced with `SqliteSaver` (SQLite) for persistent
  conversation checkpoints
- **RAG**: Vector store path unified to `settings.CHROMA_DB_PATH` (`./data/chroma_db`),
  eliminating split-brain between `tools/builtin.py` and `tools/health.py`
- **LLMGuard**: Cache `hit_rate` property fix, `_generate()` streaming detection,
  fallback chain robustness improvements
- **TraceChain**: `end_trace()` no longer overwrites already-finished traces
- **ErrorChain**: Enhanced circuit breaker state management, error classification
  coverage expanded
- **RegistryAdapter**: Anti-pattern sequence tracking (sliding window of 20),
  dependency injection setters for learner/optimizer
- **README**: Project structure updated (root wrappers, full test listing),
  stats corrected to 21,000+ lines / 70+ files, MemorySaver→SqliteSaver
- **Eval cases**: Test cases expanded, module exports `main()` for CLI use

### Fixed
- `BaseChatModel.bind_tools()` `NotImplementedError` when creating
  `create_react_agent` with guarded model
- LLMCache content length check rejecting short-but-valid content (< 10 chars)
- `hit_rate` property/method mismatch between `LLMCache` and callers
- Frozen dataclass `_ConfigValidator` incompatibility with `patch.object`
  in tests (migrated to `patch.dict(settings._values, ...)`)
- Streaming mock coroutine/async-generator mismatch in tests
- `/metrics` 307 redirect not in `SKIP_PATHS` causing middleware interference
- `LLMRetryPolicy` API drift (`max_retries` → `max_attempts`, removed `get_delay`)

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
