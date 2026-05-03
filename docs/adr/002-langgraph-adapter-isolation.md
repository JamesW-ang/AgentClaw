# ADR-002: LangGraph as Agent Framework with Adapter Isolation

**状态**：已采纳 | **日期**：2026-04 | **决定者**：架构组

## 背景

Agent 推理循环需要支持工具调用、多轮对话、状态持久化。选项包括：自定义 ReAct 循环、LangGraph、AutoGen、CrewAI。

## 方案

使用 **LangGraph `create_react_agent`** 作为主 Agent 框架，通过 **RegistryAdapter** 和 **GuardedChatModel** 两层 Adapter 隔离外部依赖。

## 考虑过的选项

### 选项 A：自定义 ReAct 循环
即 `tools/dispatcher.py` 中的手动实现：
- 优点：零外部依赖、完全控制
- 缺点：缺少流式支持、状态管理、社区生态

### 选项 B：LangGraph 直接依赖
- 优点：开箱即用的 agent loop、SqliteSaver、流式
- 缺点：API 不稳定（0.3.x 变更 `bind_tools` 接口）

### 选项 C：LangGraph + Adapter 层（选定）
- 优点：享受 LangGraph 生态 + 通过 Adapter 隔离变更影响
- 缺点：两层抽象增加 ~220 行适配代码

## 决策理由

1. `RegistryAdapter` 隔离了 ToolRegistry 和 LangGraph 的 `StructuredTool` 格式
2. `GuardedChatModel` 隔离了 LLMGuard（自定义实现）和 LangChain 的 `BaseChatModel`
3. 当 LangGraph API 变更时，只需修改 Adapter 层

## 后果

- 正：LangGraph 从 0.2.x 升级到 0.3.x 只改了 `GuardedChatModel.bind_tools()` 一个方法
- 正：保留 `tools/dispatcher.py` 作为低依赖备选（可用于无 LangGraph 环境）
- 负：两层 Adapter 增加调试链路深度

## 关联

- ADR-001：ToolRegistry 提供数据源
- ADR-003：LLMGuard 通过 GuardedChatModel 桥接
