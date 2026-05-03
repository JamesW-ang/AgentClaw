# AgentClaw 工具开发指南

> 版本：6.2 | 更新：2026-05

---

## 1. 概述

AgentClaw 的工具系统围绕 **单例注册中心** (`tools/registry.py`) 构建。所有工具通过统一的注册接口接入，由注册中心负责参数校验、速率限制、执行追踪和错误处理。LangGraph 通过 `tools/registry_adapter.py` 桥接层将注册工具转换为 LLM 可调用的 `StructuredTool`。

开发一个新工具只需 3 步：**写函数 → 加装饰器 → import 模块**。

---

## 2. 快速开始：第一个工具

```python
# my_tools/weather.py
from tools.registry import registry, ToolCategory

@registry.register(
    name="get_weather",
    description="查询指定城市的实时天气信息",
    parameters=[
        {"name": "city", "type": "string", "description": "城市名称", "required": True},
        {"name": "unit", "type": "string", "description": "温度单位(celsius/fahrenheit)", "required": False, "default": "celsius"},
    ],
    category=ToolCategory.SEARCH,
    examples=["查询北京的天气", "get_weather city=Tokyo unit=celsius"],
    timeout=10,
)
def get_weather(city: str, unit: str = "celsius") -> dict:
    """查询天气并返回结构化结果"""
    # 业务逻辑...
    return {
        "city": city,
        "temperature": 22,
        "humidity": 65,
        "condition": "晴",
        "unit": unit,
    }
```

然后在 `agent/core.py` 的 `init_all_tools()` 中添加一行 `import my_tools.weather`，工具即自动注册。

---

## 3. 注册方式

### 3.1 装饰器注册（推荐）

```python
@registry.register(
    name="tool_name",          # 工具名称，默认取函数名
    description="工具描述",     # 供 LLM 理解工具用途
    parameters=[...],          # 参数定义列表
    category=ToolCategory.XXX, # 工具分类
    examples=["示例1"],        # 使用示例（可选）
    timeout=30,                # 超时秒数（默认 30）
)
def my_tool(param1: str, param2: int = 10) -> dict:
    ...
```

### 3.2 手动注册（用于包装外部函数）

```python
from tools.registry import registry

registry.register_func(
    external_function,
    name="my_tool",
    description="...",
    parameters=[...],
    category=ToolCategory.CUSTOM,
    timeout=10,
)
```

适用于：包装已有类方法、第三方库函数、或需要延迟绑定的场景。参见 `tools/builtin.py` 中 `file_write` 等 os_tools 的注册方式。

---

## 4. 参数定义

### 4.1 两种格式

**格式 A：字典列表（推荐，信息完整）**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 参数名 |
| `type` | string | 是 | `string` / `number` / `integer` / `boolean` / `array` / `object` |
| `description` | string | 是 | 参数说明 （供 LLM 理解） |
| `required` | bool | 否 | 是否必填，默认 `True` |
| `default` | any | 否 | 默认值（`required=False` 时生效） |
| `enum` | list | 否 | 允许的枚举值列表 |

```python
parameters=[
    {"name": "query", "type": "string", "description": "搜索关键词", "required": True},
    {"name": "limit", "type": "number", "description": "返回条数", "required": False, "default": 5},
    {"name": "sort", "type": "string", "description": "排序方式",
     "required": False, "default": "relevance", "enum": ["relevance", "date", "popularity"]},
]
```

**格式 B：字符串列表（简洁，无类型信息）**

```python
parameters=["query", "num_results", "language"]
```

所有参数默认为 `type=string, required=True`。适合简单工具快速注册。参见 `tools/builtin.py` 的 `web_search` 工具。

### 4.2 参数校验

注册中心在 `execute()` 时自动校验：
- 必填参数是否提供
- 参数类型是否匹配（`_check_type`）
- 枚举值是否合法
- 可选参数未提供时填入默认值

校验失败会返回 `{"success": False, "error": "..."}` 且不消耗速率令牌。

---

## 5. 返回值规范

### 5.1 标准返回格式

```python
def my_tool(...) -> dict:
    return {
        "field1": value1,
        "field2": value2,
        # ... 任意 LLM 需要理解的结构化字段
    }
```

注册中心会将其包装为：
```json
{"success": true, "result": {"field1": ..., "field2": ...}, "error": null, "latency": 0.123}
```

LangGraph 适配器会进一步将 `result` 转为字符串传给 LLM。

### 5.2 推荐返回约定

- **数据字段用名词**：`results`、`count`、`summary`、`items`
- **包含元信息**：`source`、`timestamp` 帮助 LLM 判断数据时效性
- **错误信息用 `error` 字段**：`{"error": "文件不存在"}` 作为正常返回值（不抛异常）
- **避免过大的返回值**：LLM 上下文有限，超过 2000 字符的内容应截断

---

## 6. 错误处理策略

### 6.1 抛出异常 vs 返回错误字典

| 场景 | 策略 | 示例 |
|------|------|------|
| **安全违规** | 抛出异常 | `raise PermissionError("路径不在白名单中")` |
| **参数验证失败** | 抛出异常 | `raise ValueError(f"不支持的尺寸: {size}")` |
| **依赖缺失** | 抛出异常 | `raise EnvironmentError("API Key 未配置")` |
| **业务逻辑失败** | 返回错误字典 | `return {"error": "网络请求超时，请稍后重试"}` |
| **可恢复错误** | 返回错误字典 | `return {"error": "文件读取失败", "detail": str(e)}` |

注册中心会捕获所有异常并包装为 `{"success": False, "error": "ExceptionType: message"}`。

### 6.2 异常类型参考

```python
# 安全相关
raise PermissionError("不允许的操作")    # → 适配器输出 "permission denied"
raise ValueError("参数不合法")           # → 通用错误

# 环境相关
raise EnvironmentError("API Key 未设置")  # → 配置缺失

# 运行时错误
raise RuntimeError("下游服务异常")        # → 通用错误
```

工具函数内部也可以用 `try/except` 捕获异常后返回错误字典，适合需要精细错误分类的场景。

### 6.3 超时

- 注册中心的 `timeout` 参数控制 `asyncio.wait_for` 超时
- 超时后工具状态自动标记为 `TIMEOUT`
- 建议：简单工具 5-10s，网络请求 15s，图像处理 30s，AI 模型推理 60s

---

## 7. 异步函数支持

注册中心自动检测 `async def` 函数并适配：

```python
@registry.register(name="async_search", ...)
async def async_search(query: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/search?q={query}")
    return response.json()
```

注册中心执行逻辑：
- 若事件循环未运行 → `loop.run_until_complete(func(**kwargs))`
- 若事件循环已在运行 → 新线程中 `asyncio.run(func(**kwargs))`

**建议**：无特殊需求时优先使用同步函数，避免线程调度开销。

---

## 8. 安全开发规范

### 8.1 文件操作

```python
# 路径白名单
ALLOWED_DIRS = ["./generated_images", "./data", "./output"]

def _validate_path(path: str) -> Path:
    resolved = Path(path).resolve()
    for allowed in ALLOWED_DIRS:
        if str(resolved).startswith(str(Path(allowed).resolve())):
            return resolved
    raise PermissionError(f"路径不在白名单中: {path}")

# 扩展名检查
ALLOWED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".log", ".py"}
if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
    raise PermissionError(f"不允许的文件类型: {resolved.suffix}")
```

### 8.2 命令执行

```python
# 命令白名单
ALLOWED_COMMANDS = {"ls", "cat", "head", "tail", "wc", "grep", "find", "du", "df", "echo"}

# 危险模式黑名单
DANGEROUS_PATTERNS = [
    r"rm\s+(-rf?|--recursive)",  # 删除命令
    r">\s*/dev/",                 # 设备文件
    r"\|\s*sh\b",                # 管道到 shell
    r"`[^`]*`",                  # 命令替换
    r"\$\([^)]*\)",             # 命令替换
    r";\s*\w+",                  # 命令链
    r"sudo\b", r"chmod\b",       # 权限操作
]

def _validate_command(command: str):
    cmd_base = command.strip().split()[0]
    if cmd_base not in ALLOWED_COMMANDS:
        raise PermissionError(f"不允许的命令: {cmd_base}")
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command):
            raise PermissionError(f"检测到危险模式: {pattern}")
```

### 8.3 代码执行

```python
# AST 安全检查
import ast

FORBIDDEN_NODES = (ast.Import, ast.ImportFrom)
FORBIDDEN_NAMES = {"__import__", "eval", "exec", "compile", "open", "os", "sys", "subprocess"}

def _validate_code(code: str):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise PermissionError("不允许 import 语句")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise PermissionError(f"禁用的内置函数: {node.id}")

# 受限执行环境
safe_builtins = {
    "print": print, "len": len, "range": range, "int": int, "float": float,
    "str": str, "list": list, "dict": dict, "set": set, "tuple": tuple,
    "bool": bool, "abs": abs, "min": min, "max": max, "sum": sum,
    "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "round": round, "isinstance": isinstance, "type": type, "True": True, "False": False,
}
exec(code, {"__builtins__": safe_builtins}, {})
```

### 8.4 安全检查清单

- [ ] 文件操作有路径白名单 + 扩展名白名单
- [ ] 命令执行有命令白名单 + 危险模式正则
- [ ] 代码执行有 AST 检查 + 受限 builtins
- [ ] 网络请求有超时设置
- [ ] API Key / 敏感信息不硬编码，从环境变量读取
- [ ] 输入内容有长度限制
- [ ] 返回值不泄露系统路径或内部状态

---

## 9. 高级模式

### 9.1 类工具包装

适用于需要全局状态的工具（如模型加载、连接池）：

```python
# tools/my_tool.py

_my_tool_instance = None
_lock = threading.Lock()

class MyTool:
    def __init__(self):
        self.client = SomeClient()  # 重量初始化（仅一次）

    def do_something(self, input: str) -> dict:
        return self.client.process(input)

def get_my_tool() -> MyTool:
    global _my_tool_instance
    if _my_tool_instance is None:
        with _lock:
            if _my_tool_instance is None:
                _my_tool_instance = MyTool()
    return _my_tool_instance

# 包装函数供注册中心调用
def _my_tool_wrapper(input: str) -> dict:
    tool = get_my_tool()
    return tool.do_something(input)

# 注册包装函数
from tools.registry import registry, ToolCategory

registry.register_func(
    _my_tool_wrapper,
    name="my_tool",
    description="...",
    parameters=[
        {"name": "input", "type": "string", "description": "输入内容", "required": True},
    ],
    category=ToolCategory.CUSTOM,
    timeout=30,
)
```

### 9.2 可选依赖保护

```python
try:
    import optional_dependency
    HAS_OPTIONAL = True
except ImportError:
    HAS_OPTIONAL = False

if HAS_OPTIONAL:
    @registry.register(name="advanced_tool", ...)
    def advanced_tool(...) -> dict:
        return optional_dependency.process(...)
else:
    logger.info("optional_dependency 未安装，advanced_tool 不可用")
```

### 9.3 多后端降级

```python
def my_tool(query: str) -> dict:
    # 主后端
    try:
        result = primary_backend.search(query)
        return {"results": result, "backend": "primary"}
    except Exception:
        logger.warning("主后端失败，降级到备用后端")

    # 备用后端
    try:
        result = fallback_backend.search(query)
        return {"results": result, "backend": "fallback"}
    except Exception:
        return {"error": "所有后端均不可用"}
```

---

## 10. 完整示例：weather 工具

```python
"""
天气查询工具
文件: my_tools/weather.py
"""
import httpx
from tools.registry import registry, ToolCategory
from core.logger import get_logger

logger = get_logger("WeatherTool")

# 安全验证
def _validate_city(city: str):
    if not city or len(city) > 100:
        raise ValueError(f"无效的城市名: {city[:50]}...")
    if any(c in city for c in ("../", "\\", "\x00")):
        raise PermissionError("城市名包含不安全字符")

ALLOWED_UNITS = {"celsius", "fahrenheit"}

def _validate_unit(unit: str):
    if unit not in ALLOWED_UNITS:
        raise ValueError(f"不支持的温度单位: {unit}，允许: {ALLOWED_UNITS}")

@registry.register(
    name="get_weather",
    description="查询指定城市的实时天气信息，返回温度、湿度、天气状况等",
    parameters=[
        {"name": "city", "type": "string", "description": "城市名称（中文或英文）", "required": True},
        {"name": "unit", "type": "string", "description": "温度单位", "required": False,
         "default": "celsius", "enum": list(ALLOWED_UNITS)},
    ],
    category=ToolCategory.SEARCH,
    examples=["get_weather city=北京", "get_weather city=Tokyo unit=fahrenheit"],
    timeout=10,
)
def get_weather(city: str, unit: str = "celsius") -> dict:
    # 1. 参数验证
    _validate_city(city)
    _validate_unit(unit)

    # 2. 调用天气 API
    api_key = os.getenv("WEATHER_API_KEY", "")
    if not api_key:
        return {"error": "天气 API Key 未配置，请在 .env 中设置 WEATHER_API_KEY"}

    try:
        response = httpx.get(
            "https://api.weatherapi.com/v1/current.json",
            params={"key": api_key, "q": city},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as e:
        logger.warning(f"天气 API 返回错误: {e.response.status_code}")
        return {"error": f"查询失败: {e.response.status_code}"}
    except httpx.TimeoutException:
        return {"error": "天气 API 请求超时，请稍后重试"}
    except Exception as e:
        logger.error(f"天气 API 异常: {e}")
        return {"error": f"查询异常: {str(e)[:100]}"}

    # 3. 提取关键数据
    current = data.get("current", {})
    return {
        "city": data.get("location", {}).get("name", city),
        "country": data.get("location", {}).get("country", ""),
        "temperature": current.get("temp_c" if unit == "celsius" else "temp_f", 0),
        "condition": current.get("condition", {}).get("text", "未知"),
        "humidity": current.get("humidity", 0),
        "wind_kph": current.get("wind_kph", 0),
        "unit": unit,
        "local_time": data.get("location", {}).get("localtime", ""),
    }
```

**注册此工具：** 在 `agent/core.py` 的 `init_all_tools()` 中添加：

```python
def init_all_tools():
    ...
    # 6. 天气查询工具
    import my_tools.weather  # noqa: F401
    ...
```

---

## 11. 工具开发清单

开发新工具时逐项检查：

- [ ] 函数签名使用类型注解（`param: str`）
- [ ] 返回值是 `dict`（非字符串、非直接打印）
- [ ] 参数验证在函数开头完成（不合法时抛异常）
- [ ] 安全敏感操作有白名单检查
- [ ] 网络请求有超时设置
- [ ] 外部依赖用 try/except 保护
- [ ] 选择合适的 `ToolCategory`
- [ ] `timeout` 设置合理
- [ ] `description` 清晰描述工具能做什么
- [ ] 每个参数有 `description` 说明
- [ ] 在 `agent/core.py` 中 import 模块触发注册
- [ ] 运行 `python agent/core.py` 验证工具出现在注册列表中

---

## 12. 相关文档

- [架构设计](architecture.md) — 六层架构中工具层的位置
- [配置参考](configuration-reference.md) — `TOOL_TIMEOUT` 等工具相关配置
- [API 参考](api-reference.md) — 通过 REST API 调用工具
