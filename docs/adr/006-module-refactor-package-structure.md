# ADR-006: Module Refactor — Flat to Package Structure

**状态**：已采纳 | **日期**：2026-05 | **决定者**：架构组

## 背景

初始代码将所有模块平铺在项目根目录（19 个 `.py` 文件），随代码增长（21,000+ 行）导致导航困难、命名空间污染、包管理混乱。

## 方案

将 19 个根级模块重组为 9 个功能子包，每个包有明确的职责边界：

```
重构前 (19 files in root)    →    重构后 (9 packages)
agent_core.py                →  agent/core.py
config.py, logger.py, ...     →  core/config.py, core/logger.py, ...
tools/registry.py, ...        →  tools/registry.py, ...
api_server.py                 →  api/server.py
demo_ui.py                    →  demo/ui.py
learning_*.py                 →  learning/{evolution,feedback,learner,optimizer}.py
eval_*.py                     →  eval/{runner,cases,...}.py
aoi_engine.py                 →  aoi/engine.py
```

## 考虑过的选项

### 选项 A：维持扁平结构
- 优点：导入路径短、重构成本零
- 缺点：19 个文件在根目录，代码无组织

### 选项 B：单包 monorepo
- 优点：极简
- 缺点：包内线数太大（21K+），无模块边界

### 选项 C：子包拆分（选定）
- 优点：按功能域组织、清晰的依赖边界、独立测试

## 向后兼容策略

在根目录保留 shim/wrapper 文件，透明委托到新位置：
```python
# demo_ui.py (proxy)
from demo.ui import *
if __name__ == "__main__":
    demo.ui.main()
```

所有现有入口点（docker-compose、systemd service、CI 脚本）无需修改。

## 后果

- 正：代码组织清晰，新开发者快速定位
- 正：每个子包可独立查看复杂度（最大 ~3000 行）
- 正：7 个 shim 文件确保零中断迁移
- 负：额外维护 7 个 shim 文件

## 关联

- CHANGELOG：v6.2 重构记录
- docs/architecture.md：六层架构与包的对应关系
