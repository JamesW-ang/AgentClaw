# ADR-001: Singleton Tool Registry with Decorator Registration

**状态**：已采纳 | **日期**：2026-04 | **决定者**：架构组

## 背景

Agent 框架需要注册 20+ 工具，每个工具需要提供名称、描述、参数模式、分类信息，并能够生成 OpenAI function-calling schema。

## 方案

采用 **Singleton 模式 + 装饰器注册**：

```python
@registry.register(name="web_search", description="...", parameters=[...], category=SEARCH)
def web_search(query: str) -> str: ...
```

注册中心在模块导入时由 `@registry.register` 装饰器自动填充。

## 考虑过的选项

### 选项 A：显式注册表（dict 手动添加）
- 优点：显式、易于追踪
- 缺点：容易遗漏、维护成本高、与函数定义分离

### 选项 B：配置文件注册（YAML/JSON 声明）
- 优点：声明式、非开发者友好
- 缺点：参数模式与实现分离，容易不同步

### 选项 C：类继承注册（BaseTool 子类）
- 优点：类型安全、面向对象
- 缺点：样板代码多（每个工具一个类）

## 决策理由

1. **位置邻近**：工具定义和注册信息在同一位置，降低不一致风险
2. **最小样板**：一行装饰器 vs 整个类定义
3. **模块化导入**：简单的 `import tools.builtin` 即完成注册
4. **Schema 自动生成**：从参数列表自动生成 OpenAI function-calling schema

## 后果

- 正：工具添加极简，新工具只需 `@registry.register` + 函数实现
- 正：LangGraph、dispatcher、demo UI 共享同一注册数据源
- 负：模块导入顺序隐含依赖（需保证 registry 先初始化）
- 负：装饰器元数据在函数编译时静态确定，不支持运行时动态参数

## 关联

- ADR-002：RegistryAdapter 桥接 LangGraph
- ADR-005：装饰器驱动的横切关注点
