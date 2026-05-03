# AgentClaw 配置参考

> 版本：6.2 | 更新：2026-05

---

## 1. 配置方式

AgentClaw 通过 `.env` 文件 + 环境变量配置。配置在 `core/config.py` 中集中管理，使用 frozen dataclass 确保不可变，启动时校验必填项。

```bash
# 复制模板
cp .env.example .env

# 编辑配置
vim .env
```

---

## 2. 配置项总览

### 2.1 API 密钥

| 变量名 | 默认值 | 必填 | 说明 |
|--------|--------|------|------|
| `DEEPSEEK_API_KEY` | — | **是** | DeepSeek API Key，核心 LLM 所需 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | 否 | DeepSeek API 端点（支持自定义代理） |
| `ZHIPU_API_KEY` | — | 否 | 智谱 API Key，用于视觉分析（GLM-4V）和图片生成（CogView） |
| `ZHIPU_BASE_URL` | `https://open.bigmodel.cn/api/paas/v4` | 否 | 智谱 API 端点 |
| `OPENAI_API_KEY` | — | 否 | OpenAI API Key（可选备用 Provider） |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | 否 | OpenAI API 端点或兼容代理 |
| `API_KEY` | — | 否 | API 服务认证密钥。**未设置时 API 无认证保护** |
| `SERPAPI_KEY` | — | 否 | SerpAPI Key，`web_search` 工具的三级降级后端 |

### 2.2 模型配置

| 变量名 | 默认值 | 必填 | 说明 |
|--------|--------|------|------|
| `LLM_MODEL` | `deepseek-chat` | 否 | 默认 LLM 模型。可选值：`deepseek-chat`、`deepseek-reasoner` |
| `VISION_MODEL` | `glm-4v-flash` | 否 | 视觉模型。可选值：`glm-4v-flash`、`gpt-4o`、`qwen-vl-plus` |

### 2.3 服务器配置

| 变量名 | 默认值 | 必填 | 说明 |
|--------|--------|------|------|
| `API_PORT` | `8000` | 否 | FastAPI REST API 监听端口 |
| `WEB_PORT` | `7860` | 否 | Gradio Web UI 监听端口 |

### 2.4 存储

| 变量名 | 默认值 | 必填 | 说明 |
|--------|--------|------|------|
| `CHROMA_DB_PATH` | `./data/chroma_db` | 否 | ChromaDB 向量数据库路径，RAG 知识库检索使用 |

### 2.5 日志

| 变量名 | 默认值 | 必填 | 说明 |
|--------|--------|------|------|
| `LOG_LEVEL` | `INFO` | 否 | 日志级别：`DEBUG`、`INFO`、`WARNING`、`ERROR` |
| `LOG_DIR` | `./data/logs` | 否 | 日志文件输出目录 |

### 2.6 工具执行

| 变量名 | 默认值 | 必填 | 说明 |
|--------|--------|------|------|
| `TOOL_TIMEOUT` | `30` | 否 | 单个工具执行超时（秒）。建议：网络工具 10-15s，图像处理 30-60s |
| `REACT_MAX_ROUNDS` | `5` | 否 | ReAct Agent 最大推理轮次，防止无限循环 |

---

## 3. 配置场景

### 3.1 最小配置（仅 LLM）

```bash
# .env — 仅需 DeepSeek API Key
DEEPSEEK_API_KEY=sk-your-key-here
```

只有核心对话功能可用。视觉分析、图片生成不可用（会自动降级）。

### 3.2 完整配置（所有功能）

```bash
# .env — 所有功能
# LLM
DEEPSEEK_API_KEY=sk-your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 视觉 & 图片生成
ZHIPU_API_KEY=your-zhipu-key
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# API 认证
API_KEY=your-api-auth-key

# Web 搜索（可选）
SERPAPI_KEY=your-serpapi-key

# 服务器
API_PORT=8000
WEB_PORT=7860

# 存储
CHROMA_DB_PATH=./data/chroma_db

# 日志
LOG_LEVEL=INFO
LOG_DIR=./data/logs

# 工具
TOOL_TIMEOUT=30
REACT_MAX_ROUNDS=5
```

### 3.3 生产环境建议

```bash
# 生产环境 .env
DEEPSEEK_API_KEY=sk-prod-key
ZHIPU_API_KEY=prod-zhipu-key
API_KEY=<生成强随机密钥>
LOG_LEVEL=WARNING           # 减少日志量
TOOL_TIMEOUT=60             # 提高超时容忍度（模型推理可能较慢）
REACT_MAX_ROUNDS=10         # 允许更多轮推理
```

**生产环境额外建议：**

- `API_KEY` 必须设置 — 避免未授权访问
- `LOG_LEVEL=WARNING` — 减少磁盘 I/O
- 使用反向代理（nginx）处理 HTTPS 和限流
- 在 `docker-compose.yml` 中通过 `env_file` 注入

---

## 4. 启动验证

启动时 `validate_on_startup()` 检查必填配置项。缺失必填项时进程会直接退出并打印错误：

```
Startup FAILED: missing config
X DEEPSEEK_API_KEY -- DeepSeek API Key
```

---

## 5. 配置不可变

`settings` 对象是 frozen dataclass，运行时无法修改。如需动态切换配置，需要重启进程并更新 `.env`。

---

## 6. 相关文档

- [快速入门教程](quick-start-tutorial.md) — 配置 .env 的实操步骤
- [架构设计](architecture.md) — 配置层在六层架构中的位置
- [API 参考](api-reference.md) — API_KEY 的实际使用方式
