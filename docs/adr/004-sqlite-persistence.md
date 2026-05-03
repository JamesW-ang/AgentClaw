# ADR-004: SQLite for Agent State Persistence

**状态**：已采纳 | **日期**：2026-05 | **决定者**：架构组

## 背景

Agent 需要在重启后恢复对话历史。LangGraph 提供 `MemorySaver`（内存）、`SqliteSaver`（SQLite）、`PostgresSaver`、`RedisSaver` 等 Checkpointer。

## 方案

使用 **SQLite (`langgraph-checkpoint-sqlite`)** 作为 Agent 状态持久层。

```python
conn = sqlite3.connect("data/agent_checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
```

## 考虑过的选项

### 选项 A：MemorySaver（默认）
- 优点：零配置、最快
- 缺点：进程重启后丢失所有对话历史

### 选项 B：RedisSaver
- 优点：高并发、低延迟
- 缺点：需部署 Redis 服务、增加运维复杂度

### 选项 C：SQLite（选定）
- 优点：零依赖（Python 内置）、持久化保证、足够单机场景

## 决策理由

1. 当前部署为单实例 Docker，SQLite 的并发模型（单写多读）完全满足
2. 零外部依赖 = 零运维成本
3. 与项目其他数据存储一致（JSONL traces 文件、chroma_db 目录）

## 后果

- 正：Docker 重启后对话恢复
- 正：数据文件可备份、迁移
- 负：未来水平扩展需迁移到 Redis（需预留接口）

## 关联

- ADR-003：TraceChain 使用 JSONL 而非 SQLite（避免写入竞争）
- 限制：`check_same_thread=False` 需注意线程安全
