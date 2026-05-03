# ADR-003: Three-Chain Reliability Architecture (LLMGuard + ErrorChain + TraceChain)

**状态**：已采纳 | **日期**：2026-04 | **决定者**：架构组

## 背景

生产级 Agent 需要同时解决：LLM API 不稳定性、工具执行异常、全链路可观测性。这三个关注点相互关联但职责不同。

## 方案

采用三条独立责任链，通过 `agent/core.py` Facade 统一初始化：

- **LLMGuard**：LLM 调用容错（超时/重试/降级/熔断/缓存）
- **ErrorChain**：全局错误处理（分类/CircuitBreaker/优雅降级）
- **TraceChain**：全链路追踪（Span 嵌套/JSONL 持久化）

三条链通过 `core/metrics.py` 统一暴露指标。

## 考虑过的选项

### 选项 A：单一体（一个类处理所有）
- 优点：简单，调用路径短
- 缺点：职责耦合，难以独立测试和替换

### 选项 B：三层嵌套（LLMGuard → ErrorChain → TraceChain）
- 优点：天然编排顺序
- 缺点：强耦合，任一链的变更影响整体

### 选项 C：三链独立 + 共享 Metrics（选定）
- 优点：每链可独立测试、独立替换、独立演进

## 决策理由

1. **独立可测试性**：每条链的测试不依赖其他链（200+ 测试覆盖）
2. **组合灵活**：AOI 工作流只用了 TraceChain，没有 LLMGuard
3. **共享观测**：三条链都向 `core/metrics.py` 报告，统一监控入口

## 后果

- 正：新增链路（如未来添加 CacheChain）只需实现接口并接入 metrics
- 正：LLMGuard v2 重写不影响 ErrorChain 和 TraceChain
- 负：三层调用链增加 ~5-10% 延迟（主要是 TraceChain 的 span 创建 + JSON 序列化）

## 关联

- ADR-004：SQLite 持久化（TraceChain 使用 JSONL，Checkpointer 使用 SQLite）
- ADR-002：GuardedChatModel 桥接 LLMGuard 到 LangGraph
