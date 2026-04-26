# AgentClaw 🤖

> 生产级 AI Agent 开发框架 | Python · DeepSeek · LangGraph · FastAPI · Gradio

[![CI](https://github.com/JamesW-ang/AgentClaw/actions/workflows/ci.yml/badge.svg)](https://github.com/JamesW-ang/AgentClaw/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/JamesW-ang/AgentClaw?style=social)](https://github.com/JamesW-ang/AgentClaw/stargazers)

---

## 目录

1. [项目概览](#1-项目概览)
2. [六层架构](#2-六层架构)
3. [请求全链路（集成版）](#3-请求全链路集成版)
4. [工具清单（18个）](#4-工具清单18个)
5. [核心模块说明](#5-核心模块说明)
6. [安全与可靠性](#6-安全与可靠性)
7. [自主进化系统（Level 4）](#7-自主进化系统level-4)
8. [快速开始](#8-快速开始)
9. [部署方式](#9-部署方式)
10. [API 接口文档](#10-api-接口文档)
11. [项目结构](#11-项目结构)
12. [开发指南](#12-开发指南)

---

## 1. 项目概览

AgentClaw v6.1 是一个生产级 AI Agent 框架，核心特性包括：

- **六层架构设计**：Config → Registry → Security → Retry → RateLimiter → LLM，每一层独立可测试
- **18 个内置工具**：搜索、计算、文件操作、命令执行、代码沙箱、系统监控、进程管理、视觉分析、图片生成、知识库检索、浏览器控制
- **三重可靠性保障**：指数退避重试（retry）+ 令牌桶限流（rate_limiter）+ 健康检查（health）
- **自主进化系统**：FeedbackCollector → ExperienceLearner → AdaptiveOptimizer → EvolutionManager，后台自动学习优化
- **多入口部署**：FastAPI REST API（:8000）+ Gradio Web UI（:7860）+ Docker 容器化

### 技术栈

| 组件 | 技术选型 |
|------|---------|
| LLM 推理 | DeepSeek Chat（OpenAI 兼容接口） |
| 视觉理解 | 智谱 GLM-4V-Flash（OpenAI 兼容接口） |
| 图片生成 | 智谱 CogView-3-Flash |
| Agent 编排 | LangGraph（create_react_agent） |
| 工具注册 | 自研 ToolRegistry（装饰器 + 单例） |
| REST API | FastAPI + Uvicorn |
| Web UI | Gradio 5.x |
| 向量检索 | ChromaDB + bge-small-zh-v1.5 |
| 部署 | Docker + docker-compose |

---

## 2. 六层架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Level 6: 检测层                            │
│         health.py — ChromaDB/LLM API/内存三维健康检查          │
├─────────────────────────────────────────────────────────────┤
│                    Level 5: 服务层                            │
│     api_server.py (FastAPI)  +  demo_ui.py (Gradio)          │
├─────────────────────────────────────────────────────────────┤
│                    Level 4: 编排层                            │
│     react_agent.py → LangGraph create_react_agent            │
│     demo_ui.py    → LangGraph StateGraph (多Agent)           │
│     agent_core.py → 统一入口（懒加载单例）                     │
├─────────────────────────────────────────────────────────────┤
│                    Level 3: 安全与可靠性层                     │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │ security.py  │ │ retry.py     │ │ rate_limiter.py      │ │
│  │ SQL/XSS/敏感 │ │ 指数退避重试  │ │ 令牌桶限流(双层)     │ │
│  │ 信息过滤     │ │ 3次重试+抖动  │ │ LLM+Search独立限流   │ │
│  └──────────────┘ └──────────────┘ └──────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Level 2: 工具层                            │
│  tool_registry.py (单例注册中心) + registry_adapter.py       │
│  18 个工具: 搜索/计算/文件/命令/代码/监控/进程/视觉/生成/检索   │
├─────────────────────────────────────────────────────────────┤
│                    Level 1: 基础层                            │
│  core/config.py  — 环境变量验证器 (Frozen dataclass)         │
│  core/logger.py  — 彩色控制台 + 旋转文件日志                  │
│  core/retry.py   — 重试装饰器 (指数退避 + 抖动)              │
│  core/rate_limiter.py — TokenBucket 限流器                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 请求全链路（集成版）

一个用户请求从进入到返回的完整路径：

```
HTTP POST /ask
  │
  ▼
[1] SecurityMiddleware (Starlette)
  │  ├─ TokenBucket.consume() — 全局限流 (rate=0.5/s, capacity=30)
  │  ├─ SQL 注入检测 (8 种模式)
  │  ├─ XSS 攻击检测 (5 种模式)
  │  ├─ 敏感信息检测 (7 种关键词)
  │  └─ 请求体长度检查 (≤5000 字符)
  │
  ▼
[2] api_server.ask() 路由
  │  ├─ _llm_limiter.consume() — LLM 专用限流 (rate=1.0/s, capacity=5)
  │  ├─ get_react_agent() — 懒加载单例
  │  │    ├─ init_all_tools() — 触发工具注册 (18个)
  │  │    └─ create_react_agent(LLM, tools, MemorySaver)
  │  └─ agent.ainvoke(messages)
  │
  ▼
[3] LangGraph ReAct 循环
  │  ├─ LLM 推理 (DeepSeek Chat)
  │  ├─ 工具选择 → registry.execute()
  │  │    ├─ _llm_limiter.consume() — 第三层限流检查
  │  │    ├─ 工具函数执行
  │  │    └─ 反馈信号采集 (FeedbackSignal)
  │  └─ 结果汇总 → 下一轮推理
  │
  ▼
[4] Vision API 调用 (如涉及视觉工具)
  │  ├─ _call_vision_api() → _do_call()
  │  ├─ 失败 → 重试 (最多 3 次)
  │  │    ├─ 指数退避: delay = 1.0 × 2^attempt
  │  │    └─ 抖动: delay × random(0.75, 1.25)
  │  └─ 成功 → 返回 VisionResult
  │
  ▼
[5] 响应返回
     ├─ Answer(answer=..., usage={elapsed, thread_id})
     └─ 日志记录 (console + file)
```

---

## 4. 工具清单（18个）

### 核心工具（builtin_tools.py — 14个）

| # | 工具名 | 分类 | 说明 | 安全机制 |
|---|--------|------|------|---------|
| 1 | `web_search` | search | 三级降级搜索（SerpAPI → DuckDuckGo → 缓存） | API Key 可选 |
| 2 | `calculator` | calculator | AST 安全数学计算（支持函数） | 禁止 eval 注入 |
| 3 | `file_read` | file_io | 安全文件读取（白名单+黑名单+大小限制） | 路径白名单+10MB限制 |
| 4 | `file_write` | file_io | 安全文件写入（自动备份+黑名单） | 路径黑名单+沙箱目录 |
| 5 | `run_command` | system | 系统命令执行（白名单+危险模式拦截） | 命令白名单+9种危险模式 |
| 6 | `code_execute` | system | Python 沙箱执行（禁止import/IO/网络） | AST检查+线程超时 |
| 7 | `knowledge_search` | search | RAG 知识库检索（ChromaDB 优先/TF-IDF 回退） | 本地检索无外部调用 |
| 8 | `sys_monitor` | system | 系统资源概览（CPU/内存/平台） | 只读 |
| 9 | `sys_process_list` | system | 进程列表（按CPU/内存排序） | 只读 |
| 10 | `sys_disk_info` | system | 磁盘使用情况 | 只读 |
| 11 | `process_start` | system | 启动后台进程（命令白名单） | 命令白名单 |
| 12 | `process_stop` | system | 停止后台进程（SIGTERM→SIGKILL） | 仅管理已启动进程 |
| 13 | `process_list` | system | 列出已管理进程 | 只读 |
| 14 | `browser_navigate` | system | 浏览器导航（需 Playwright） | 可选依赖 |
| 15 | `browser_get_content` | system | 提取页面文本内容 | 可选依赖 |
| 16 | `browser_screenshot` | system | 页面截图 | 可选依赖 |

### 多模态工具（vision_tool.py — 3个）

| # | 工具名 | 说明 | 特性 |
|---|--------|------|------|
| 17 | `vision_analyze` | 图片分析（描述/对象检测/OCR） | 支持 GLM-4V / GPT-4o，自动重试 3 次 |
| 18 | `vision_ocr` | OCR 文字识别 | 多语言支持 |
| 19 | `vision_compare` | 多图对比分析 | 支持批量对比 |

### 图片生成（multimodal_image_gen.py — 1个）

| # | 工具名 | 说明 | 特性 |
|---|--------|------|------|
| 20 | `image_generate` | AI 文生图 | CogView-3-Flash，7种尺寸，内容安全审计 |

> 注：browser_tool 为可选依赖（需安装 Playwright），实际注册数量 17-20 个。

---

## 5. 核心模块说明

### 5.1 agent_core.py — 统一入口

项目的"枢纽模块"，解决各子系统孤立的问题。提供：

- `init_all_tools()` — 初始化并注册全部工具到 ToolRegistry
- `get_react_tools()` — 获取 LangGraph 兼容的 StructuredTool 列表
- `get_react_agent()` — 获取统一 ReAct Agent（懒加载单例 + rate_limiter 就绪）
- `init_evolution()` — 初始化自主学习系统
- `record_feedback()` — 记录工具执行反馈
- `get_tool_summary()` — 获取工具摘要（UI 展示用）

```python
# 使用方式
from agent_core import get_react_agent
agent = get_react_agent()
result = agent.invoke(
    {"messages": [("human", "帮我搜索Python最新版本")]},
    {"configurable": {"thread_id": "user-1"}}
)
```

### 5.2 tool_registry.py — 工具注册中心

生产级单例注册中心，核心特性：

- **装饰器注册** + **手动注册** 两种方式
- **同步执行**，兼容 `execute("tool", {"k":"v"})` 和 `execute("tool", k="v")` 两种调用风格
- **内置速率限制**：每次 `execute()` 调用前检查 `_llm_limiter` 令牌桶
- **完整统计**：调用次数、成功率、平均延迟
- **OpenAI Schema 输出**：`get_tools_for_llm()` 生成 function calling 格式

```python
# 注册工具
@registry.register(
    name="my_tool",
    description="我的工具",
    parameters=["query"],
    category=ToolCategory.CUSTOM,
)
def my_tool(query: str) -> dict:
    return {"result": query}

# 执行工具
result = registry.execute("my_tool", query="hello")
```

### 5.3 registry_adapter.py — LangGraph 桥接

将 ToolRegistry 中的工具转换为 LangGraph 的 `StructuredTool`，同时：

- 自动注入 **FeedbackCollector** 反馈采集
- 为每个工具调用生成唯一的 `task_id`
- 错误分类（`permission_error` / `execute_error` / 异常类型）

### 5.4 core/config.py — 配置验证器

Frozen dataclass 单例，启动时验证必需配置：

```python
settings = _ConfigValidator()  # 全局单例
settings.DEEPSEEK_API_KEY      # 属性访问（大小写不敏感）
settings.validate()            # 返回缺失配置列表
validate_on_startup()          # 缺失则 sys.exit(1)
```

支持的配置项：

| 配置 | 默认值 | 必需 | 说明 |
|------|--------|------|------|
| `DEEPSEEK_API_KEY` | — | ✅ | DeepSeek API Key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | — | DeepSeek 接口地址 |
| `ZHIPU_API_KEY` | — | — | 智谱 API Key（视觉/图片生成） |
| `ZHIPU_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4` | — | 智谱接口地址 |
| `LLM_MODEL` | `deepseek-chat` | — | 默认 LLM 模型 |
| `VISION_MODEL` | `glm-4v-flash` | — | 视觉模型 |
| `SERPAPI_KEY` | — | — | SerpAPI Key（可选，web_search 增强） |
| `CHROMA_DB_PATH` | `./data/chroma_db` | — | ChromaDB 存储路径 |
| `LOG_LEVEL` | `INFO` | — | 日志级别 |

### 5.5 core/logger.py — 日志系统

- 彩色控制台输出（DEBUG=蓝 / INFO=绿 / WARNING=黄 / ERROR=红）
- 文件日志自动轮转（每天午夜，保留 30 天）
- 防止日志重复（`propagate = False`）

### 5.6 core/retry.py — 重试机制

已集成到 `vision_tool._call_vision_api()`：

- **指数退避**：`delay = initial_delay × backoff_factor^attempt`
- **抖动**：`delay × random(0.75, 1.25)`
- **异常分类**：可重试（TimeoutError / ConnectionError / OSError）vs 不可重试（ValueError / TypeError / 等）
- **最大重试**：3 次

### 5.7 core/rate_limiter.py — 令牌桶限流

已集成到三个位置：

| 集成位置 | 限制器 | 配置 | 说明 |
|---------|--------|------|------|
| `security.py` (SecurityMiddleware) | `TokenBucket(rate=0.5, capacity=30)` | 每秒 0.5 请求，突发 30 | 全局 HTTP 请求限流 |
| `api_server.py` (`/ask` 路由) | `_llm_limiter` (rate=1.0, capacity=5) | 每秒 1 请求，突发 5 | LLM 调用专用限流 |
| `tool_registry.py` (`execute()`) | `_llm_limiter` (rate=1.0, capacity=5) | 同上 | 工具执行级限流 |

### 5.8 health.py — 健康检查

已挂载到 `api_server.py` 两个路由：

- `GET /health` — 基础检查（版本、运行时间、内存）
- `GET /health/detailed` — 详细检查（ChromaDB 心跳 + LLM API HEAD 请求 + 系统内存）

返回格式：

```json
{
  "status": "healthy" | "degraded",
  "timestamp": 1777087951.0,
  "checks": {
    "chromadb": {"status": "ok"},
    "llm_api": {"status": "ok", "status_code": 200},
    "memory": {"status": "ok", "percent": 42.3, "available_mb": 9216}
  }
}
```

### 5.9 security.py — 安全中间件

Starlette BaseHTTPMiddleware，五层防护：

1. **令牌桶限流**（TokenBucket，已替换旧的滑动窗口）
2. **请求体长度检查**（≤ 5000 字符）
3. **SQL 注入检测**（8 种正则模式）
4. **XSS 攻击检测**（5 种正则模式）
5. **敏感信息检测**（7 种关键词匹配）

---

## 6. 安全与可靠性

### 6.1 安全体系总览

```
请求入口
  ├── SecurityMiddleware (security.py)
  │    ├── 令牌桶限流 — rate=0.5/s, capacity=30
  │    ├── SQL 注入拦截 — 8 种正则模式
  │    ├── XSS 攻击拦截 — 5 种正则模式
  │    ├── 敏感信息过滤 — password/secret/api_key/token/...
  │    └── 请求体长度限制 — ≤ 5000 字符
  │
  ├── /ask 路由 (api_server.py)
  │    └── LLM 限流 — _llm_limiter (rate=1.0/s, capacity=5)
  │
  ├── 工具执行 (tool_registry.py)
  │    ├── 第三层限流 — _llm_limiter
  │    ├── 文件操作白名单 + 黑名单
  │    ├── 命令执行白名单 + 危险模式检测
  │    ├── 沙箱代码执行 (AST 检查)
  │    └── 图片生成内容安全审计
  │
  └── Vision API 调用 (vision_tool.py)
       └── 自动重试 3 次 (指数退避 + 抖动)
```

### 6.2 工具级安全

- **file_read**：路径白名单（/home, /tmp, /var/log, 当前目录）+ 敏感文件黑名单（.env, .ssh/, id_rsa 等）+ 扩展名白名单 + 10MB 大小限制
- **file_write**：路径黑名单 + 沙箱目录限制 + 自动备份
- **run_command**：命令白名单（30+ 基础命令）+ 9 种危险模式正则（rm -rf /, fork bomb 等）+ 30 秒超时
- **code_execute**：禁止 import / open / eval / exec / 网络 IO + AST 安全检查 + 线程超时
- **image_generate**：提示词内容审计（16 种禁止模式）+ 尺寸白名单（7 种）+ 输出目录白名单

---

## 7. 自主进化系统（Level 4）

```
FeedbackCollector ← 工具执行结果（成功/失败/延迟）
       │
       ▼
ExperienceLearner ← 从历史反馈中挖掘成功模式
       │
       ▼
AdaptiveOptimizer ← 动态调整路由权重和 Prompt 模板
       │
       ▼
EvolutionManager ← 协调子系统，后台定期进化（默认每小时）
```

### 集成方式

- **RegistryAdapter**：每个工具调用后自动采集 FeedbackSignal
- **agent_core.init_evolution()**：启动进化循环
- **agent_core.record_feedback()**：手动记录反馈

```python
from agent_core import init_evolution, record_feedback

# 启动进化（后台线程）
init_evolution(interval=3600)

# 手动记录反馈
record_feedback(
    task_id="task-001",
    tool_name="web_search",
    success=True,
    latency=1.23,
)
```

---

## 8. 快速开始

### 8.1 安装依赖

```bash
pip install -r requirements.txt
```

### 8.2 配置环境变量

```bash
cp AgentClaw_Docker_.env.example .env

# 编辑 .env，至少配置：
# DEEPSEEK_API_KEY=sk-xxx

# 可选配置：
# ZHIPU_API_KEY=xxx          # 视觉/图片生成
# SERPAPI_KEY=xxx            # 搜索增强
```

### 8.3 启动服务

```bash
# 方式 1：统一启动（API + Web UI）
python main.py

# 方式 2：单独启动 API
python api_server.py

# 方式 3：单独启动 Web UI
python demo_ui.py
```

### 8.4 验证

```bash
# 健康检查
curl http://localhost:8000/health
curl http://localhost:8000/health/detailed

# API 测试
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "帮我搜索Python最新版本", "session_id": "test-1"}'

# Web UI
open http://localhost:7860
```

---

## 9. 部署方式

### 9.1 Docker 部署

```bash
# 构建镜像
docker build -t agentclaw:v6.1 -f Dockerfile .

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f
```

### 9.2 端口映射

| 服务 | 端口 | 说明 |
|------|------|------|
| FastAPI | 8000 | REST API + Swagger 文档 |
| Gradio | 7860 | Web Demo UI |

---

## 10. API 接口文档

### POST /ask

主问答接口。

**请求体**：
```json
{
  "question": "帮我搜索2026年AI最新进展",
  "session_id": "user-001"
}
```

**响应**：
```json
{
  "answer": "根据搜索结果...",
  "usage": {
    "thread_id": "user-001",
    "elapsed": "3.45s"
  }
}
```

**错误响应**（速率限制）：
```json
{
  "answer": "请求过于频繁，请稍后再试",
  "usage": {
    "thread_id": "user-001",
    "elapsed": "0s",
    "error": "rate_limited"
  }
}
```

### GET /health

基础健康检查。

### GET /health/detailed

详细健康检查（ChromaDB + LLM API + 内存）。

---

## 11. 项目结构

```
AgentClaw_code/
├── main.py                    # 统一启动入口（API + Web UI）
├── agent_core.py              # 核心：工具注册 + Agent 创建 + 进化管理
├── api_server.py              # FastAPI 服务（/ask, /health）
├── demo_ui.py                 # Gradio Web UI（6 大场景）
├── react_agent.py             # ReAct Agent（兼容旧接口）
├── builtin_tools.py           # 14 个内置工具
├── vision_tool.py             # 3 个视觉工具（带重试）
├── multimodal_image_gen.py    # 1 个图片生成工具
├── tool_registry.py           # 工具注册中心（单例）
├── registry_adapter.py        # LangGraph 桥接适配器
├── tool_dispatcher.py         # 工具分发器
├── security.py                # 安全中间件（TokenBucket 限流）
├── health.py                  # 健康检查（三维检测）
├── rag_searcher.py            # RAG 检索引擎
├── prompts.py                 # Prompt 模板
├── query.py                   # 查询处理
├── feedback_collector.py      # 反馈采集器
├── experience_learner.py      # 经验学习器
├── adaptive_optimizer.py      # 自适应优化器
├── evolution_manage.py        # 进化管理器
├── load_docs.py               # 文档加载器
├── create_vector_db.py        # 向量库创建
├── requirements.txt           # Python 依赖
├── Dockerfile                 # Docker 镜像
├── docker-compose.yml         # Docker Compose
├── AgentClaw_Docker_.env.example  # 环境变量模板
│
├── core/
│   ├── config.py              # 配置验证器（Frozen dataclass）
│   ├── logger.py              # 彩色日志 + 文件轮转
│   ├── retry.py               # 重试装饰器（指数退避 + 抖动）
│   └── rate_limiter.py        # 令牌桶限流器
│
└── os_tools/
    ├── file_write.py          # 安全文件写入
    ├── sys_monitor.py         # 系统监控
    ├── process_mgr.py         # 进程管理
    ├── browser_tool.py        # 浏览器自动化（可选）
    └── scheduler.py           # 任务调度（预留）
```

**代码统计**：约 8,300 行 Python 代码，27 个源文件。

---

## 12. 开发指南

### 12.1 添加新工具

```python
# 方式 1：装饰器注册（推荐）
from tool_registry import registry, ToolCategory

@registry.register(
    name="my_tool",
    description="工具描述",
    parameters=[
        {"name": "input", "type": "string", "description": "输入参数", "required": True},
    ],
    category=ToolCategory.CUSTOM,
    timeout=10,
)
def my_tool(input: str) -> dict:
    return {"result": f"处理: {input}"}

# 方式 2：手动注册
registry.register_func(my_func, name="tool_name", description="...", category=ToolCategory.CUSTOM)
```

### 12.2 使用重试机制

```python
from core.retry import retry_with_backoff

@retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
def unstable_api_call():
    # 可能抛出 TimeoutError / ConnectionError 的操作
    return requests.get("https://api.example.com/data")
```

### 12.3 使用速率限制器

```python
from core.rate_limiter import TokenBucket, rate_limit

# 方式 1：直接使用预定义的限制器
from core.rate_limiter import _llm_limiter
if _llm_limiter.consume(tokens=1):
    call_llm_api()

# 方式 2：装饰器
@rate_limit(rpm=60)
def my_function():
    pass

# 方式 3：自定义限制器
my_limiter = TokenBucket(rate=10.0, capacity=20)
```

### 12.4 运行测试

```bash
# 工具注册中心自测
python tool_registry.py

# 视觉工具自测
python vision_tool.py

# 内置工具自测
python builtin_tools.py

# 核心初始化自测
python agent_core.py
```

### 12.5 自定义配置

在 `.env` 文件或环境变量中设置：

```bash
# 必需
DEEPSEEK_API_KEY=sk-your-key

# LLM 模型（默认 deepseek-chat）
LLM_MODEL=deepseek-reasoner

# 视觉模型（默认 glm-4v-flash）
VISION_MODEL=glm-4v-plus

# 日志级别
LOG_LEVEL=DEBUG

# API 端口
API_PORT=8080
WEB_PORT=8081
```

---

## 13. 数据流与层级依赖

### 13.1 完整请求数据流

```
1. 用户输入 (前端/API 客户端)
   │
   ├─ Input: {"question": "搜索Python最新版本", "session_id": "user-1"}
   │
   ▼
2. HTTP 请求处理 (api_server.py)
   │
   ├─ Route: POST /ask
   ├─ Query Params: ?stream=true (可选)
   │
   ▼
3. 安全层防护 (security.py)
   │
   ├─ TokenBucket.consume(1) — 全局限流
   ├─ SQL 注入正则检测 (8 patterns)
   ├─ XSS 攻击正则检测 (5 patterns)
   ├─ 敏感信息关键词扫描 (7 keywords)
   ├─ 请求体大小验证 (≤ 5000 chars)
   │
   ├─ ✓ 通过 → 继续
   └─ ✗ 失败 → 返回 400/413/429 错误
   │
   ▼
4. 工具初始化 (agent_core.py)
   │
   ├─ get_react_agent() — 获取或创建 Agent 单例
   ├─ init_all_tools() — 注册全部 18 个工具
   │  ├─ builtin_tools — 14 个工具
   │  ├─ vision_tool — 3 个工具
   │  ├─ multimodal_image_gen — 1 个工具
   │  └─ aoi_engine — 1 个工具（可选）
   ├─ get_react_tools() — 获取 LangGraph StructuredTool 列表
   │
   ▼
5. Agent 推理循环 (agent_core.py + LangGraph)
   │
   ├─ 循环最多 5 轮 (REACT_MAX_ROUNDS)
   │  │
   │  ├─ Round N:
   │  │  ├─ LLM 思考 (DeepSeek Chat API)
   │  │  │  ├─ Prompt: 系统角色 + 对话历史 + 可用工具列表
   │  │  │  ├─ Temperature: 0.0 (确定性)
   │  │  │  ├─ Stop: ["Observation:"]
   │  │  │  └─ Max Tokens: 2000
   │  │  │
   │  │  ├─ LLM 输出 (思考 + 工具选择)
   │  │  │  ├─ Thought: "..."
   │  │  │  ├─ Action: "web_search"
   │  │  │  └─ Action Input: {"query": "..."}
   │  │  │
   │  │  ├─ 工具执行 (tool_registry.execute())
   │  │  │  ├─ 工具选择验证 (工具名称有效性)
   │  │  │  ├─ 参数验证 (Pydantic schema)
   │  │  │  ├─ 限流检查 (_llm_limiter.consume())
   │  │  │  ├─ @retry_with_backoff 装饰
   │  │  │  │  ├─ Attempt 1: 直接执行
   │  │  │  │  ├─ Attempt 2: 延迟 1.0s * 2^0 * jitter 后重试
   │  │  │  │  ├─ Attempt 3: 延迟 1.0s * 2^1 * jitter 后重试
   │  │  │  │  └─ Attempt 4: 延迟 1.0s * 2^2 * jitter 后重试 (最后)
   │  │  │  │
   │  │  │  ├─ 工具返回
   │  │  │  │  ├─ Success: {"success": true, "result": "..."}
   │  │  │  │  └─ Failure: {"success": false, "error": "..."}
   │  │  │  │
   │  │  │  ├─ 性能指标记录
   │  │  │  │  ├─ call_count++
   │  │  │  │  ├─ success_count++ (if ok)
   │  │  │  │  ├─ error_count++ (if fail)
   │  │  │  │  └─ total_latency += elapsed
   │  │  │  │
   │  │  │  └─ 反馈采集 (FeedbackCollector)
   │  │  │     ├─ task_id = "task-{uuid}"
   │  │  │     ├─ signal = FeedbackSignal(...)
   │  │  │     └─ collector.collect(signal)
   │  │  │
   │  │  ├─ 观察 (LLM 阅读工具结果)
   │  │  │  ├─ Observation: "{工具返回结果}"
   │  │  │
   │  │  └─ 继续判断
   │  │     ├─ 是否有 final_answer? → 结束循环
   │  │     ├─ token 已达上限? → 强制结束
   │  │     ├─ 轮数达上限? → 强制结束
   │  │     └─ 否则 → 继续下一轮
   │  │
   │  └─ End Round N
   │
   ├─ 循环结束
   │
   ▼
6. 响应组装 (api_server.py)
   │
   ├─ 提取最终答案
   ├─ 计算执行统计
   │  ├─ 总耗时
   │  ├─ 轮数
   │  ├─ 工具调用次数
   │  └─ token 使用量
   │
   ▼
7. HTTP 响应
   │
   ├─ Status: 200
   ├─ Content-Type: application/json
   ├─ Body:
   │  {
   │    "answer": "最终答案...",
   │    "usage": {
   │      "session_id": "user-1",
   │      "elapsed": "3.45s",
   │      "rounds": 2,
   │      "tokens_used": 456
   │    }
   │  }
   │
   ▼
8. 用户接收 (前端/API 客户端)
```

### 13.2 工具执行细节流程

```
Agent 决策: 调用 web_search
  │
  ├─ action = "web_search"
  ├─ action_input = {"query": "Python 3.13", "num_results": 5}
  │
  ▼
RegistryAdapter.make_wrapper()
  │
  ├─ Pydantic Schema 验证参数
  │  ├─ query: str → ✓
  │  ├─ num_results: int → ✓
  │  └─ language: str (可选, 默认 "zh-CN")
  │
  ▼
ToolRegistry.execute("web_search", {...})
  │
  ├─ 获取 ToolInfo
  ├─ 检查工具状态 (READY / ERROR / DISABLED)
  ├─ 检查限流 (_llm_limiter.consume())
  │  └─ TokenBucket: 桶中剩余 3 个令牌
  │     ├─ 消耗 1 个令牌
  │     └─ 桶中剩余 2 个令牌
  │
  ▼
@retry_with_backoff(max_retries=3, initial_delay=1.0)
  │
  ├─ Attempt 1:
  │  ├─ web_search() 执行
  │  │  ├─ SerpAPI 首选 (需 SERPAPI_KEY)
  │  │  │  ├─ 发送 API 请求 (timeout=10s)
  │  │  │  ├─ 解析 JSON 结果
  │  │  │  └─ 返回 10 个结果中的前 5 个
  │  │  │
  │  │  ├─ DuckDuckGo 降级 (无需 API Key)
  │  │  │  ├─ 发送 HTML 请求 (timeout=10s)
  │  │  │  ├─ 正则提取结果块
  │  │  │  └─ 返回前 5 个 HTML 结果
  │  │  │
  │  │  └─ 本地缓存降级 (完全离线)
  │     └─ 返回缓存或错误信息
  │
  ├─ ✓ Success:
  │  └─ 返回 {"success": true, "result": [...]}
  │
  └─ ✗ Failure:
     ├─ Attempt 2:
     │  └─ sleep(1.0 * 2^0 * random(0.75,1.25)) ≈ 0.75-1.25s
     │
     ├─ Attempt 3:
     │  └─ sleep(1.0 * 2^1 * random(0.75,1.25)) ≈ 1.5-2.5s
     │
     └─ Attempt 4 (最后):
        └─ sleep(1.0 * 2^2 * random(0.75,1.25)) ≈ 3.0-5.0s
           若仍失败 → 抛出异常
  │
  ▼
工具返回结果
  │
  ├─ Result: {"success": true, "result": [...]}
  │
  ├─ 记录指标
  │  ├─ call_count: 14 → 15
  │  ├─ success_count: 13 → 14
  │  ├─ total_latency: 12.34s → 14.12s
  │  └─ avg_latency: 0.88s → 0.94s
  │
  ▼
FeedbackCollector.collect()
  │
  ├─ FeedbackSignal:
  │  ├─ task_id: "task-a1b2c3d4"
  │  ├─ tool_name: "web_search"
  │  ├─ success: true
  │  ├─ latency: 1.78s
  │  ├─ error_type: null
  │  ├─ timestamp: 1777087951.0
  │  └─ user_rating: null (用户未评分)
  │
  ├─ 加入缓冲区 (deque, maxlen=10000)
  │  └─ len(signals) = 456 → 457
  │
  └─ 每 50 条信号持久化一次
     └─ evolution_data/feedback.jsonl
  │
  ▼
返回到 Agent
   └─ Agent 继续推理或返回最终答案
```

### 13.3 层级依赖关系

```
┌─────────────────────────────────────────────────────┐
│ Level 6: 监控层 (health.py)                          │
│ ├─ 依赖: Level 1 (logger/config)                     │
│ └─ 被依赖: api_server (GET /health*)                │
├─────────────────────────────────────────────────────┤
│ Level 5: 服务层 (api_server.py, demo_ui.py)         │
│ ├─ 依赖: Level 4 (agent_core) + Level 3 (security)  │
│ └─ 被依赖: 用户 (HTTP/WebSocket)                    │
├─────────────────────────────────────────────────────┤
│ Level 4: 编排层 (agent_core.py)                      │
│ ├─ 依赖: Level 2 (registry_adapter) + Level 1       │
│ ├─ 调用: LangGraph create_react_agent()             │
│ └─ 被依赖: Level 5 (api_server/demo_ui)             │
├─────────────────────────────────────────────────────┤
│ Level 3: 安全与可靠性层                              │
│ ├─ security.py (中间件)                            │
│ │  ├─ 依赖: core/rate_limiter                       │
│ │  └─ 被依赖: FastAPI (middleware stack)            │
│ ├─ core/retry.py (装饰器)                           │
│ │  ├─ 依赖: core/logger                             │
│ │  └─ 被依赖: vision_tool (自动重试)                │
│ └─ core/rate_limiter.py (限流)                      │
│    ├─ 依赖: threading                               │
│    └─ 被依赖: security + tool_registry               │
├─────────────────────────────────────────────────────┤
│ Level 2: 工具层 (tool_registry.py, registry_adapter) │
│ ├─ tool_registry.py (单例注册中心)                  │
│ │  ├─ 被注册: builtin_tools + vision_tool + ...    │
│ │  ├─ 依赖: core/logger + core/rate_limiter         │
│ │  └─ 被依赖: registry_adapter + 各工具模块         │
│ └─ registry_adapter.py                              │
│    ├─ 依赖: tool_registry + LangGraph               │
│    ├─ 调用: 反馈采集 (feedback_collector)           │
│    └─ 被依赖: agent_core (get_react_tools)          │
├─────────────────────────────────────────────────────┤
│ Level 1: 基础层 (core/)                              │
│ ├─ config.py (配置验证器)                           │
│ │  ├─ 依赖: os, sys, dataclasses                    │
│ │  └─ 被依赖: 所有模块 (settings)                   │
│ ├─ logger.py (日志系统)                             │
│ │  ├─ 依赖: logging, sys                            │
│ │  └─ 被依赖: 所有模块 (get_logger)                 │
│ ├─ retry.py (重试装饰器)                            │
│ │  ├─ 依赖: functools, random, time, logger         │
│ │  └─ 被依赖: vision_tool (自动重试)                │
│ └─ rate_limiter.py (限流器)                         │
│    ├─ 依赖: time, threading                         │
│    └─ 被依赖: security + tool_registry              │
└─────────────────────────────────────────────────────┘

自主进化系统 (平行运行):
  FeedbackCollector → ExperienceLearner → AdaptiveOptimizer → EvolutionManager
  └─ 被 RegistryAdapter 触发，收集工具执行反馈
```

---

## 14. 技术选型详解

### 14.1 LLM 推理引擎对比

| 方案 | 成本 | 延迟 | 中文能力 | 推理能力 | 知识时效 | 推荐场景 |
|------|------|------|---------|---------|---------|---------|
| **DeepSeek Chat** | ⭐ 最便宜 | ⭐⭐⭐ 快 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中等 | **推荐** 日常助手 |
| DeepSeek Reasoner | ⭐⭐ 较贵 | ⭐ 较慢 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 中等 | 复杂推理 |
| OpenAI GPT-4 | ⭐⭐⭐⭐ 最贵 | ⭐⭐⭐ 快 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ 优秀 | 关键业务 |
| Zhipu GLM-4 | ⭐⭐ 中等 | ⭐⭐ 较快 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中等 | 中文专项 |
| Qwen (阿里) | ⭐ 便宜 | ⭐⭐⭐ 快 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 中等 | 高吞吐 |

**当前配置**：
- 主模型：`deepseek-chat` (生产推荐)
- 备用模型：`deepseek-reasoner` (复杂任务)
- 切换方法：修改 `core/config.py::LLM_MODEL`

### 14.2 视觉理解方案

| 方案 | 精度 | 速度 | 成本 | 特点 | 集成 |
|------|------|------|------|------|------|
| **Zhipu GLM-4V-Flash** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ 低 | **推荐** 均衡 | 已集成 |
| OpenAI GPT-4o | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ 高 | 精度最高 | 可替换 |
| Google Gemini | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ 低 | 多任务 | 可替换 |
| Claude 3.5 Vision | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 推理强 | 可替换 |

**当前配置**：`VISION_MODEL=glm-4v-flash`

### 14.3 图片生成方案

| 方案 | 质量 | 速度 | 成本 | 中文友好 | 推荐 |
|------|------|------|------|---------|------|
| **Zhipu CogView-3** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ 低 | ⭐⭐⭐⭐⭐ | **推荐** |
| OpenAI DALL-E 3 | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 质量优先 |
| Stable Diffusion 3 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 本地部署 |
| 阿里 Qwen VL | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ 低 | ⭐⭐⭐⭐ | 成本优化 |

**当前配置**：`IMAGE_GEN_MODEL=cogview-3-flash`

### 14.4 向量检索方案

| 方案 | 隐私 | 速度 | 成本 | 易用性 | 推荐 |
|------|------|------|------|--------|------|
| **ChromaDB** | ⭐⭐⭐⭐⭐ 本地 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ 免费 | ⭐⭐⭐⭐ | **推荐** |
| Milvus | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ 中等 | ⭐⭐ | 高性能 |
| Qdrant | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ 低 | ⭐⭐⭐ | 云部署 |
| Pinecone | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 无运维 |

**Embedding 模型**：`bge-small-zh-v1.5` (BAAI, 开源中文友好)

### 14.5 Agent 编排框架对比

| 框架 | 易用性 | 控制流 | 流式输出 | 多工具 | 推荐 |
|------|--------|--------|---------|--------|------|
| **LangGraph** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ 显式 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **推荐** |
| LangChain Agent | ⭐⭐⭐⭐ | ⭐⭐⭐ 隐式 | ⭐⭐⭐ | ⭐⭐⭐⭐ | 快速原型 |
| Dspy | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | 研究项目 |
| AutoGPT | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐ | 复杂系统 |

**当前配置**：LangGraph 原生 `create_react_agent()`

---

## 15. 常见问题与故障恢复

### Q1: 启动时报错 "Missing config: DEEPSEEK_API_KEY"

**答**：检查 `.env` 文件：
```bash
cat .env | grep DEEPSEEK_API_KEY
echo $DEEPSEEK_API_KEY  # 验证环境变量
```

若缺失，从 https://platform.deepseek.com 申请 API Key。

### Q2: 工具执行超时 ("Tool timeout after 30s")

**答**：
1. 检查网络连接（搜索工具）
2. 增大超时时间：
   ```python
   @registry.register(..., timeout=60)  # 从 30s → 60s
   ```
3. 查看工具日志：`data/logs/`

### Q3: 频繁触发速率限制 (HTTP 429)

**答**：
1. 检查 `core/rate_limiter.py` 中的限流参数：
   ```python
   _llm_limiter = TokenBucket(rate=1.0, capacity=5)
   # rate=1.0 表示每秒 1 个请求，调整为 2.0 允许每秒 2 个请求
   ```
2. 或修改中间件限流：
   ```python
   # security.py
   class SecurityMiddleware:
       rate_limit = 30  # 从 30 → 60 (每分钟 60 个请求)
   ```

### Q4: Vision API 返回错误 ("Invalid image format")

**答**：
1. 确保提供的图片是有效的 URL 或 base64
2. 检查图片大小 (< 10MB)
3. 查看 `vision_tool.py` 中的 retry 日志

### Q5: ChromaDB 连接失败

**答**：
1. 检查 `CHROMA_DB_PATH` 目录权限：
   ```bash
   ls -la ./data/chroma_db/
   chmod 755 ./data/chroma_db/
   ```
2. 删除损坏的数据库并重新初始化：
   ```bash
   rm -rf ./data/chroma_db/
   python create_vector_db.py
   ```

### Q6: 自主进化系统不工作

**答**：
1. 确保已调用 `init_evolution()`：
   ```python
   from agent_core import init_evolution
   init_evolution(interval=3600)  # 每小时一次
   ```
2. 检查 `evolution_data/` 目录是否有新文件生成
3. 查看日志：`data/logs/evolution_*.log`

---

## 16. 性能优化建议

### 16.1 LLM 推理加速

| 优化 | 方法 | 收益 | 实施难度 |
|------|------|------|---------|
| **缓存 KV** | 启用 Vision Cache (GLM-4V) | -40% 延迟 | 中 |
| **量化推理** | 模型量化 (int8/int4) | -50% 内存 | 高 |
| **本地部署** | Ollama / vLLM | -99% 成本 | 高 |
| **异步调用** | AsyncIO 并行请求 | +50% 吞吐 | 中 |

### 16.2 工具执行优化

| 优化 | 方法 | 收益 | 难度 |
|------|------|------|------|
| **工具并行** | `asyncio.gather()` 多工具 | -50% 总耗时 | 中 |
| **缓存结果** | Redis / 内存缓存 | -70% 重复查询 | 中 |
| **预热** | 启动时初始化常用工具 | -30% 首次延迟 | 低 |
| **限流优化** | 分层限流 (LLM/Search) | +30% 吞吐 | 低 |

### 16.3 数据库优化

| 优化 | 方法 | 收益 | 难度 |
|------|------|------|------|
| **索引优化** | ChromaDB 向量索引调优 | -60% 查询时间 | 中 |
| **批量操作** | 批量 insert/query | +40% 吞吐 | 低 |
| **本地缓存** | LRU 缓存常查向量 | -80% DB 查询 | 中 |

### 16.4 系统资源优化

| 优化 | 方法 | 收益 | 难度 |
|------|------|------|------|
| **内存管理** | 循环缓冲区 (maxlen=10000) | 内存稳定 | 低 |
| **异步 I/O** | FastAPI 异步路由 | +3x 并发 | 中 |
| **连接池** | HTTP 连接重用 | -50% TCP 开销 | 中 |
| **GPU 推理** | CUDA/TensorRT 加速 | -70% 延迟 | 高 |

---

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v6.1 | 2026-04 | 集成 retry/rate_limiter/health 到主调用链路；security 升级 TokenBucket；tool_registry.execute() 集成限流 |
| v6.0 | 2026-04 | 五步架构整合（Config→Registry→Security→Retry→RateLimiter→LLM）；18 个工具注册；自主进化系统 |
