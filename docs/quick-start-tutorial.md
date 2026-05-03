# AgentClaw 快速入门教程

> 版本：6.2 | 更新：2026-05

本教程将带你从零开始运行 AgentClaw，完成第一次对话、第一次 API 调用、以及 Web UI 的完整体验。

---

## 前置要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.10+ | 推荐 3.12+ |
| pip | 23.0+ | |
| Git | 2.0+ | 用于克隆仓库 |
| DeepSeek API Key | — | [申请地址](https://platform.deepseek.com/api_keys) |
| 智谱 API Key | — | 可选，视觉/图片生成需要。在 [bigmodel.cn](https://open.bigmodel.cn) 注册 |

---

## Step 1: 克隆并安装

```bash
git clone https://github.com/JamesW-ang/AgentClaw.git
cd AgentClaw

# 创建虚拟环境（推荐）
python -m venv agent_env
source agent_env/bin/activate  # Linux/macOS
# agent_env\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

依赖较多（LangChain、LangGraph、Gradio、ChromaDB 等），首次安装可能需要 2-5 分钟。

---

## Step 2: 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填入 DeepSeek API Key：

```bash
# 最小配置
DEEPSEEK_API_KEY=sk-your-key-here
```

如需完整功能，参考 [配置参考](configuration-reference.md)。

---

## Step 3: 验证安装

```bash
# 测试工具注册中心
python tools/registry.py
```

预期输出：

```
============================================================
ToolRegistry 生产级自测
============================================================
已注册工具: ['add', 'greet']
...
所有测试通过!
============================================================
```

```bash
# 测试 Agent 核心
python agent/core.py
```

预期输出：

```
============================================================
AgentClaw v6 — 核心 Agent 初始化测试
============================================================
已注册工具 (22 个):
  [search] web_search: ...
  [calculator] calculator: ...
...
全部测试通过!
============================================================
```

如果这两步都通过，说明核心模块工作正常。

---

## Step 4: 启动 API 服务

```bash
python api/server.py
```

预期输出：

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

验证：

```bash
# 另开一个终端
curl http://localhost:8000/health
```

```json
{"status": "ok", "service": "agent-api", "version": "6.1.0", "uptime_seconds": 5, "memory_mb": 120.5}
```

---

## Step 5: 第一次 API 调用

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "你好，请用一句话介绍你自己"}' | jq .
```

```json
{
  "answer": "我是 AgentClaw 智能助手，可以搜索网页、计算数学、读写文件...",
  "usage": {
    "thread_id": "default",
    "elapsed": "2.15s"
  }
}
```

**流式调用：**

```bash
curl -N -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "搜索今天的科技新闻"}'
```

你会看到逐字输出的效果。按 `Ctrl+C` 中断。

---

## Step 6: 启动 Web UI

```bash
python demo_ui.py
```

浏览器打开 `http://localhost:7860`。

---

## Step 7: 体验 8 个 Tab

### Tab 1: 检测分析
上传图片（如 PCB 板、电路图），使用 AOI 引擎进行缺陷检测。

### Tab 2: 对话助手
核心 ReAct Agent，支持调用 22 个工具完成复杂任务。尝试：

- *"搜索 Python 3.13 的新特性"*
- *"计算 (1024 + 2048) * 0.5"*
- *"读取 README.md 的前 50 行"*

### Tab 3: RAG 知识库
上传文档（txt/md/pdf），Agent 基于文档内容回答问题。

### Tab 4: 多模态视觉
上传图片，使用 GLM-4V 进行视觉理解、OCR 识别、图片对比分析。

### Tab 5: 图片生成
输入提示词，使用 CogView-3 生成图片。

### Tab 6: 多 Agent 协作
选择场景（代码审查 / 技术设计 / 问题诊断），3 个 Agent 角色协作完成。

### Tab 7: AOI 独立检测
上传 PCB 图片，直接进行 AOI 缺陷检测，查看检测结果和标注。

### Tab 8: AOI 智能闭环
上传 PCB 图片 + XML 配置文件，运行完整的 4 Agent 闭环调优流水线。

---

## Step 8: Docker 部署（可选）

```bash
docker-compose up -d
```

访问：
- API: `http://localhost:8000`
- API 文档: `http://localhost:8000/docs`
- Web UI: `http://localhost:7860`

```bash
# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

---

## 常见问题

### Q: 启动时报 `DEEPSEEK_API_KEY` 缺失

确保 `.env` 文件在项目根目录，且包含 `DEEPSEEK_API_KEY=sk-...`。

### Q: 工具执行失败 "API Key 未配置"

某些工具需要额外的 API Key：
- 视觉分析 / 图片生成 → `ZHIPU_API_KEY`
- Web 搜索（SerpAPI 降级）→ `SERPAPI_KEY`

这些是可选的。工具失败时 Agent 会自动尝试其他方式。

### Q: Gradio UI 启动失败

确保端口 7860 未被占用：

```bash
lsof -i :7860
```

### Q: ChromaDB 连接警告

ChromaDB 使用 SQLite 作为后端，确认 `data/chroma_db/` 目录可写。

### Q: macOS 上 `onnxruntime` 安装失败

AOI 深度学习模式需要 ONNX Runtime。安装 CPU 版本：

```bash
pip install onnxruntime
```

不需要深度学习模式时可以跳过。

---

## 下一步

- [工具开发指南](tool-development-guide.md) — 开发自己的工具
- [API 参考](api-reference.md) — 完整的 REST API 文档
- [配置参考](configuration-reference.md) — 所有配置项详解
- [自学习系统](self-learning-system.md) — 了解 Agent 如何自我进化
- [架构设计](architecture.md) — 深入理解六层架构
- [技术选型分析](trade-offs.md) — 了解技术决策的背后原因
