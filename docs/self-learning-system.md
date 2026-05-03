# AgentClaw 自学习系统

> 版本：6.2 | 更新：2026-05

---

## 1. 概述

自学习系统是 AgentClaw 的差异化核心 —— 它不仅让 Agent 执行工具，还会**从每次执行中学习**，逐步优化工具选择、参数偏好和回答策略。系统在后台以 daemon 线程运行，对主流程零阻塞。

```
FeedbackCollector → ExperienceLearner → AdaptiveOptimizer → EvolutionManager
       ↑                                                         |
       └─── tools/registry_adapter.py 记录每次工具执行 ←─────────┘
```

**四个核心组件：**

| 组件 | 职责 | 文件 |
|------|------|------|
| FeedbackCollector | 收集原始反馈信号，缓冲 + JSONL 持久化 | `learning/feedback.py` |
| ExperienceLearner | 挖掘成功策略和失败反模式 | `learning/learner.py` |
| AdaptiveOptimizer | 调优路由权重、工具偏好、提示词 | `learning/optimizer.py` |
| EvolutionManager | 协调循环，产出进化报告 | `learning/evolution.py` |

---

## 2. FeedbackCollector — 反馈采集

### 2.1 信号结构

```python
from learning.feedback import FeedbackSignal

signal = FeedbackSignal(
    task_id="task-001",       # 任务唯一标识（通常为 thread_id）
    tool_name="web_search",   # 工具名称
    success=True,             # 是否成功
    latency=0.523,            # 执行耗时（秒）
    error_type="timeout",     # 失败时的错误类型（可选）
    context="用户询问天气",    # 上下文描述（可选）
    user_rating=None,         # 用户评分 0-1（可选）
)
```

### 2.2 采集流程

```
工具执行 → registry_adapter.py wrapper finally 块
         → collector.collect(signal)
         → deque 缓冲 (maxlen=10000)
         → 每 50 条自动 flush → feedback.jsonl
```

### 2.3 持久化

- 格式：JSON Lines（每行一个 JSON 对象）
- 路径：`evolution_data/feedback.jsonl`（可通过 `persist_dir` 参数修改）
- 容错：加载时跳过损坏行，写入时使用原子重命名

### 2.4 查询接口

```python
collector = FeedbackCollector(persist_dir="./evolution_data")
recent = collector.get_recent(200)           # 最近 200 条
by_tool = collector.get_by_tool("web_search") # 按工具筛选
```

---

## 3. ExperienceLearner — 经验学习

### 3.1 核心算法

**时间衰减加权 N-gram 挖掘：**

每条反馈信号有一个时间戳，其权重随"年龄"指数衰减：

```
weight = exp(-λ × age)
λ = ln(2) / half_life = ln(2) / 604800 ≈ 1.15 × 10⁻⁶
```

| 年龄 | 权重 |
|------|------|
| 1 小时 | ~0.999 |
| 1 天 | ~0.906 |
| 7 天 | ~0.500 |
| 30 天 | ~0.052 |

算法对所有成功任务路径提取长度为 2-3 的 N-gram，累加每个 N-gram 的加权支持度。支持度 ≥ 2.0 的模式成为候选策略。

**Wilson 置信区间下限（95%）：**

替代简单的成功率，在样本量少时更保守：

```
center = (p + z²/(2n)) / (1 + z²/n)
spread = z × √((p(1-p) + z²/(4n)) / n) / (1 + z²/n)
confidence = max(0, center - spread)
```

| 样本数 | 成功次数 | 简单成功率 | Wilson 下限 |
|--------|---------|-----------|------------|
| 1 | 1 | 1.00 | 0.21 |
| 5 | 4 | 0.80 | 0.38 |
| 20 | 18 | 0.90 | 0.70 |
| 100 | 90 | 0.90 | 0.83 |

**Lift（提升度）：**

```
lift = strategy_success_rate / overall_success_rate
```

- `lift > 1`：策略正相关于成功（使用该策略成功率高于平均水平）
- `lift < 1`：策略与成功无正相关
- `lift = 1`：策略成功率等于随机水平

### 3.2 策略结构

```python
class Strategy:
    name: str                  # 策略名（如 "web_search->calculator"）
    trigger_pattern: str       # 触发模式（工具序列第一个工具名）
    tool_sequence: list        # 工具序列（如 ["web_search", "calculator"]）
    success_rate: float        # 成功比例
    confidence: float          # Wilson 置信下限
    support: float             # 时间衰减加权支持度
    lift: float                # 提升度
    anti_pattern: bool         # 是否为反模式
    usage_count: int           # 被推荐次数
    success_after_recommend: int  # 推荐后实际成功次数

    @property
    def effectiveness(self) -> float:
        # usage_count < 3: 先验估计
        # usage_count >= 3: 后验修正
```

### 3.3 策略推荐（双层匹配）

`get_strategy(task_type)` 按以下优先级匹配：

1. **精确匹配**：`task_type` 字符串包含在策略名称或触发模式中
2. **TF-IDF 语义匹配**：纯 Python 实现的稀疏向量余弦相似度，阈值 0.3

```python
learner = ExperienceLearner(collector)
strategy = learner.get_strategy("搜索并计算")
# 返回 effectiveness 最高的匹配策略，自动排除反模式
```

### 3.4 反模式检测

与正向策略完全相同的 N-gram 算法，但输入为**失败路径**。

```python
# 离线检测（进化循环中）
learner.learn_from_history(feedback)  # 自动发现反模式

# 在线检测（工具执行时）
learner.check_anti_pattern(["web_search", "calculator", "web_search"])
# 检查当前序列是否以某个反模式的前缀开头
```

反模式存储为 `Strategy(name="ANTI:xxx", anti_pattern=True)`，且不参与策略推荐。

### 3.5 策略淘汰

超过 30 天未更新且 `effectiveness < 0.3` 的策略自动删除。

---

## 4. AdaptiveOptimizer — 自适应优化

### 4.1 路由权重（自适应 EMA）

每个工具维护一个成功率权重，使用自适应学习率：

```python
alpha = 0.5 / (1 + 0.1 × sample_count)
alpha = max(alpha, 0.01)
new_weight = alpha × feedback + (1 - alpha) × old_weight
```

| 样本数 | alpha | 行为 |
|--------|-------|------|
| 1 | 0.45 | 快速学习 |
| 10 | 0.25 | |
| 50 | 0.08 | |
| 100 | 0.05 | 趋于稳定 |
| 500 | 0.01 | 几乎不变 |

`get_best_route(candidates)` 选择权重最高的工具，对样本数 < 5 的工具给予 1.1× 探索奖励。

### 4.2 工具偏好矩阵

维护 `(tool_name × context_keyword) → preference_score` 矩阵：

```python
# 更新偏好
optimizer.update_tool_preference("web_search", "搜索", success=True)
optimizer.update_tool_preference("calculator", "计算", success=True)

# 查询偏好
best_tool = optimizer.get_tool_preference("搜索")  # → "web_search"
```

使用在线均值更新（Welford 算法），无需存储历史值。

### 4.3 提示词优化

当整体成功率 < 50% 时触发：

1. 对失败反馈分类：`timeout` / `permission` / `format` / `api_error` / `unknown`
2. 选取最频繁的失败类别
3. 生成针对性的优化建议追加到 prompt 模板
4. 保留最近 5 个历史版本供对比

### 4.4 失败模式分析

```python
report = optimizer.analyze_failure_patterns()
# {
#     "total_failures": 42,
#     "categories": {
#         "timeout": {"count": 15, "percentage": 0.357},
#         "api_error": {"count": 12, "percentage": 0.286},
#         ...
#     },
#     "suggestions": ["增加超时时间", "检查 API Key 配置"],
# }
```

---

## 5. EvolutionManager — 进化管理器

### 5.1 7 步进化循环

每轮循环（默认 1 小时间隔）执行以下步骤：

```
步骤 1: 采集反馈      → 获取最近 200 条信号，统计成功率
步骤 2: 学习策略      → N-gram 挖掘 + 反模式检测
步骤 3: 更新路由权重   → 最近 50 条信号更新 EMA 权重
步骤 4: 更新工具偏好   → 最近 100 条信号更新偏好矩阵
步骤 5: 优化 Prompt   → 成功率 < 50% 时触发
步骤 6: 评估策略效果   → 将实际执行结果反馈给策略的 record_outcome
步骤 7: 生成进化报告   → 记录周期指标到 metrics_history
```

### 5.2 启动方式

```python
from agent.core import init_evolution

# 在应用启动时调用
evo = init_evolution(interval=3600)  # 每小时一轮

# 可选：停止进化
evo.stop()
```

### 5.3 策略注入 Agent Prompt

进化学习到的策略会被注入到 ReAct Agent 的系统提示词中（`agent/core.py:_build_agent_prompt`）：

```
[学习策略提示]
- 模式「web_search→calculator」: web_search → calculator (效果评分 0.85)
- 模式「file_read→code_execute」: file_read → code_execute (效果评分 0.72)

[应避免的模式]
- 反模式「web_search→web_search→web_search」: web_search → ... (置信度 0.65)
```

缓存 10 分钟，避免每次调用都重新计算。

### 5.4 进化报告

```python
report = evo.get_evolution_report()
# 完整报告包含：
#   - total_cycles, total_feedback_processed
#   - strategies / anti_patterns 数量
#   - overall_success_rate
#   - top_strategies (Top 5 by effectiveness)
#   - route_weights (Top 10)
#   - failure_analysis (按类别)
#   - recent_trend ("improving" / "stable" / "declining")
```

```python
metrics = evo.get_metrics()
# 轻量指标，适合监控面板：
#   - cycles_completed, feedback_count
#   - success_rate, strategy_count, anti_pattern_count
#   - route_count, last_cycle_time, trend
```

### 5.5 趋势判断

比较最近 3 轮进化的成功率：
- 持续上升 → `"improving"`
- 持续下降 → `"declining"`
- 其他 → `"stable"`

---

## 6. 调优参数

当前所有学习参数均为代码常量（类属性），需要修改源码调整：

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `EvolutionManager.interval` | `init_evolution()` | 3600s | 进化循环间隔 |
| `learn_from_history` 采样量 | `_run_evolution_cycle()` | 200 | 每轮处理的反馈数 |
| `HALF_LIFE` | `ExperienceLearner` | 7 天 | 时间衰减半衰期 |
| `SIMILARITY_THRESHOLD` | `ExperienceLearner` | 0.3 | TF-IDF 匹配最低阈值 |
| `INITIAL_ALPHA` | `AdaptiveOptimizer` | 0.5 | 路由权重初始学习率 |
| `DECAY_RATE` | `AdaptiveOptimizer` | 0.1 | 学习率衰减速度 |
| `MIN_ALPHA` | `AdaptiveOptimizer` | 0.01 | 学习率下限 |
| `DEFAULT_WEIGHT` | `AdaptiveOptimizer` | 0.5 | 未见路由的默认权重 |

---

## 7. 数据流集成图

```
┌─────────────────────────────────────────────────────────────┐
│                    agent/core.py                             │
│  init_evolution()    init_all_tools()    get_react_agent()  │
│       │                    │                    │            │
│       ▼                    ▼                    ▼            │
│  EvolutionManager    tools/registry     RegistryAdapter     │
│       │                    │                    │            │
│       │  ┌─────────────────┘                    │            │
│       │  │                                      │            │
│       ▼  ▼                                      ▼            │
│  ┌──────────────┐    ┌──────────────────────────────────┐   │
│  │ collector    │◄───│ registry_adapter.py wrapper      │   │
│  │ .collect()   │    │ (每次工具执行后记录 FeedbackSignal)│   │
│  └──────┬───────┘    └──────────────────────────────────┘   │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │ learner      │  N-gram 挖掘 + 反模式检测                  │
│  │ .learn_from  │                                           │
│  │ _history()   │                                           │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │ optimizer    │  路由权重 + 偏好矩阵 + Prompt 优化         │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐                                           │
│  │ Agent Prompt │  策略注入 + 反模式警告                     │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 已知限制

1. **冷启动**：系统需要积累足够反馈（建议 ≥ 50 条）才能产出有意义的策略。新部署的实例在前几个小时内策略库较空。
2. **N-gram 长度**：当前只挖掘长度 2-3 的工具序列，无法发现更长的复杂模式。
3. **参数硬编码**：half-life、阈值、alpha 等参数是类常量，不支持通过环境变量或配置文件动态调整。
4. **无模型持久化**：策略库仅存于内存中，进程重启后丢失。反馈数据在 JSONL 中持久化，但学习结果（策略、权重、偏好）需重新学习。
5. **单进程**：进化循环运行在 daemon 线程中，不支持多进程共享状态。
6. **中文分词简陋**：TF-IDF 使用空格分词，对中文的分词效果有限。RAG 场景的精确匹配更可靠。

---

## 9. 相关文档

- [架构设计](architecture.md) — 进化系统在六层架构中的位置
- [工具开发指南](tool-development-guide.md) — 工具反馈采集的集成方式
- [配置参考](configuration-reference.md) — 日志和数据目录配置
