# 技术选型与 Trade-off 分析

> AgentClaw v6.2 | 2026-05

---

## 1. Agent 框架：LangGraph vs 自定义 ReAct 循环

| 维度 | LangGraph (`create_react_agent`) | 自定义 ReAct (`tools/dispatcher.py`) |
|------|------|------|
| **实现复杂度** | 低：~10 行核心配置 | 高：~160 行手动循环 + JSON 解析 |
| **状态管理** | 内置 MemorySaver / SqliteSaver | 需自建状态管理 |
| **流式支持** | 原生 astream_events | 需手动构建 SSE |
| **灵活性** | 受限于 create_react_agent 接口 | 完全可控 |
| **依赖风险** | langgraph 0.3.x API 不稳定（如 bind_tools 变更） | 零额外依赖 |
| **调试** | LangGraph 调用链透明 | 完全可见 |
| **社区** | 活跃，持续更新 | 仅有本项目维护 |

**结论**：主路径使用 LangGraph，保留 `tools/dispatcher.py` 作为低依赖备选。LangGraph 的 API 稳定性风险通过 GuardedChatModel Adapter 层隔离。

---

## 2. LLM 提供商：DeepSeek vs 多 Provider 方案

| 维度 | DeepSeek 主推 | 多 Provider 并行 |
|------|------|------|
| **成本** | 极低 (¥1/百万 token) | 较高 (GPT-4o ~$10/百万 token) |
| **延迟** | ~800ms (典型) | GPT-4o ~1.2s, GLM-4V ~1.5s |
| **中文能力** | 优秀 (中文预训练占比高) | GPT-4o 中文可, 但非专项优化 |
| **视觉能力** | 不支持 | GLM-4V / GPT-4o 支持 |
| **多 Provider 成本** | 代码中已保留 Zhipu/OpenAI 配置 | 需要为每个 provider 维护适配 |
| **降级策略** | DeepSeek → Zhipu → Cache → 降级消息 | 同左 |

**结论**：DeepSeek 为主 LLM，Zhipu GLM-4V 为视觉专用，OpenAI 作为可选备选。LLMGuard 的 fallback chain 天然支持多 Provider 降级。

---

## 3. 向量数据库：ChromaDB vs FAISS vs Milvus

| 维度 | ChromaDB | FAISS | Milvus |
|------|------|------|------|
| **部署** | 嵌入式 (pip install) | 嵌入式 | 需独立服务 (docker-compose) |
| **持久化** | 磁盘持久化 | 需额外实现 | 内置 |
| **规模上限** | ~1M 向量 | ~10M+ | ~1B+ |
| **中文 Embedding** | 支持 bge-small-zh | 支持 | 支持 |
| **学习成本** | 极低 (langchain-chroma) | 中 | 高 |
| **运维成本** | 零 | 零 | 需运维集群 |
| **查询速度 (10K)** | ~5ms | ~3ms | ~2ms |
| **本项目场景** | 代码文档 RAG (< 10K chunks) | N/A | 严重 overkill |

**结论**：ChromaDB 完全满足当前需求（文档规模 < 1 万块），零运维成本。TF-IDF 作为兜底，在 ChromaDB 不可用时自动降级。

---

## 4. Agent 状态持久化：SQLite vs Redis vs 内存

| 维度 | SQLite (SqliteSaver) | Redis | MemorySaver |
|------|------|------|------|
| **持久性** | 磁盘持久化 | 磁盘持久化 (RDB/AOF) | 进程级易失 |
| **延迟** | ~1-5ms | ~0.1ms | ~0.01ms |
| **部署复杂度** | 零 (Python 内置) | 需 Redis 服务 | 零 |
| **并发** | 单写多读 (check_same_thread=False) | 高并发 | 单进程 |
| **数据量上限** | ~TB 级 | ~GB 级 (内存受限) | 进程内存上限 |
| **适用场景** | 单实例持久化 | 分布式会话共享 | 开发调试 |
| **故障恢复** | 重启后恢复 | 重启后恢复 | 丢失 |

**结论**：SQLite 是单实例部署的最佳平衡点 — 零依赖 + 持久化保障。若未来需要水平扩展可切换到 Redis，但当前阶段 SQLite 提供了 90% 收益而无需任何运维成本。

---

## 5. RAG Embedding：bge-small-zh vs bge-large-zh vs text-embedding-3-small

| 维度 | bge-small-zh-v1.5 | bge-large-zh-v1.5 | text-embedding-3-small |
|------|------|------|------|
| **模型大小** | ~33MB | ~1.3GB | API 调用 |
| **向量维度** | 512 | 1024 | 1536 |
| **推理速度** | ~2ms (CPU) | ~20ms (CPU) | ~100ms (API 延迟) |
| **MTEB 中文** | 58.3 | 63.7 | 62.1 (英文优化) |
| **本地运行** | 是 | 是 (但内存需求大) | 否 (依赖 API) |
| **Transformer 签名** | 兼容 sentence-transformers | 同上 | — |

**结论**：bge-small-zh 在精度/速度/资源消耗上取得最佳平衡。33MB 的模型可在任何 CPU 上毫秒级推理，512 维向量在 ChromaDB 中查询速度优异。bge-large 精度高出 ~5 个百分点但资源消耗 ~40 倍，本项目非检索精度敏感场景下收益递减。

---

## 6. Web UI 框架：Gradio vs Streamlit vs 纯 React

| 维度 | Gradio | Streamlit | React + FastAPI |
|------|------|------|------|
| **开发速度** | 极快（内置组件） | 快 | 慢（前后端分离） |
| **布局灵活度** | 中 (Blocks API) | 中 | 高 |
| **LLM 交互** | 内置 Streaming/Chat | 需自建 | 需自建 SSE |
| **美观度** | 基础 | 基础 | 自定义高 |
| **维护成本** | 低 | 低 | 高（双代码库） |
| **适合场景** | 演示 / 内部工具 | 数据分析仪表盘 | 面向用户的产品 |

**结论**：Gradio 适合当前阶段 — 快速迭代验证，8 个 Tab 展示系统全貌。未来若面向外部用户可迁移到 React，但当前 1,118 行的 Gradio 代码实现了完整功能。

---

## 7. 异步框架：FastAPI vs Flask vs Django

| 维度 | FastAPI | Flask | Django |
|------|------|------|------|
| **性能** | 高 (async) | 中 | 中 |
| **Streaming** | 原生 SSE | 需额外库 | 需额外库 |
| **类型检查** | Pydantic 集成 | 无 | DRF 庞大 |
| **学习成本** | 低 | 极低 | 高 |
| **内置功能** | 最少 (轻量) | 最少 | 全功能 (ORM/Admin) |
| **Middleware** | ASGI 原生 | WSGI | WSGI |
| **Prometheus 集成** | make_asgi_app 直接 mount | 需适配 | 需适配 |

**结论**：FastAPI 是 AI Agent 服务的最佳选择 — 原生 async 支持 SSE 流式响应，Pydantic 集成减少重复定义，轻量架构与微服务理念一致。

---

## 8. 错误分类策略：基于码表 vs 基于 ML

| 维度 | 基于码表 (ErrorClassifier) | 基于 ML 分类 |
|------|------|------|
| **准确率** | ~95% (已知错误类型) | ~98% (需标注数据) |
| **冷启动** | 零数据即可工作 | 需大量标注样本 |
| **新增错误** | 手动添加规则 | 重新训练 |
| **可解释性** | 完全透明 | 黑盒 |
| **运行时开销** | ~0.01ms | ~10-50ms |
| **复杂度** | 200 行 if/else + 正则 | 需维护模型 + 推理管道 |

**结论**：基于码表的分类在 Agent 场景下效果出色 — LLM API 错误模式相对固定（超时/限流/鉴权/模型不可用），码表覆盖 95%+ 线上场景，零运行时开销。

---

## 9. 学习系统设计：在线学习 vs 离线训练

| 维度 | 在线学习 (本项目) | 离线训练 |
|------|------|------|
| **反馈闭环** | 秒级 (下轮迭代生效) | 天级 (批处理) |
| **样本效率** | 逐步优化，少量样本即可 | 需要批量积累 |
| **计算开销** | 极低 (单次 ~10ms) | 高 (全量重训练) |
| **稳定性** | 可能受单次异常扰动 | 批处理天然平滑 |
| **冷启动** | Wilson 下限处理小样本 | 需要初始训练集 |
| **实现复杂度** | 低 (增量更新公式) | 高 (特征工程 + 训练管道) |

**结论**：在线学习更适合 Agent 场景 — Agent 的决策质量需要在每次交互中持续改进，无法等待天级训练周期。Wilson 置信区间和自适应学习率有效缓解了小样本扰动问题。

---

## 10. 测试策略：单元测试 vs 集成测试 vs E2E

| 维度 | 单元测试 (12 模块) | 集成测试 (test_integration) | E2E (conftest mock) |
|------|------|------|------|
| **速度** | ~50ms/用例 | ~2s/用例 | ~5s/用例 |
| **数量** | 200+ | 10+ | 4+ |
| **覆盖** | 各模块内部逻辑 | 模块间交互 | 全链路 |
| **Mock 深度** | 全 mock | 部分 mock | external API mock |
| **发现缺陷** | 逻辑错误 | 接口不匹配 | 集成问题 |
| **CI 稳定性** | 稳定 | 中等 | 依赖 mock 质量 |

**结论**：采用测试金字塔策略 — 大量单元测试覆盖核心逻辑（LLMGuard/ErrorChain/TraceChain/Learning ~80% coverage），适量集成测试验证模块交互，少量 E2E 确保关键路径畅通。

---

## 11. 综合决策矩阵

| 决策 | 选择 | 备选 | 关键因素 |
|------|------|------|----------|
| Agent 框架 | LangGraph | 自定义 ReAct | 开发效率 > 控制粒度 |
| LLM 主推 | DeepSeek | GPT-4o | 成本 + 中文 > 多模态 |
| 向量数据库 | ChromaDB | FAISS / Milvus | 运维零成本 > 规模上限 |
| 状态持久化 | SQLite | Redis | 零依赖 > 延迟 |
| Embedding | bge-small-zh | bge-large | 速度 + 内存 > 精度 +5% |
| Web UI | Gradio | React | 开发速度 > 美观度 |
| API 框架 | FastAPI | Flask / Django | async > 内置功能 |
| 错误分类 | 码表 | ML 分类 | 可解释性 + 零开销 > 精度 |
| 学习策略 | 在线学习 | 离线训练 | 实时性 > 稳定性 |
| 测试策略 | 单元为主 | 集成为主 | 速度 + 稳定性 > 覆盖率 |
