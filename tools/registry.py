"""
AgentClaw 生产级工具注册中心 - 修复版
修复: execute() 改为同步方法，支持 **kwargs 传参
"""

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from core.logger import get_logger
from core.metrics import observe_tool_call

logger = get_logger("ToolRegistry")

# Phase 1: ErrorChain 集成支持 (lazy import)
_error_chain_module = None


# ============================================================
# 枚举 & 数据模型
# ============================================================

class ToolCategory(Enum):
    """工具分类"""
    SEARCH = "search"
    CALCULATOR = "calculator"
    FILE_IO = "file_io"
    SYSTEM = "system"
    CODE = "code"
    COMMUNICATION = "communication"
    CUSTOM = "custom"


class ToolStatus(Enum):
    """工具状态"""
    READY = "ready"
    ERROR = "error"
    DISABLED = "disabled"
    TIMEOUT = "timeout"


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str  # string, number, boolean, array, object
    description: str
    required: bool = True
    default: Any = None
    enum: list[str] | None = None


@dataclass
class ToolInfo:
    """工具完整信息"""
    name: str
    description: str
    category: ToolCategory
    func: Callable
    parameters: list[ToolParameter] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    timeout: int = 30
    status: ToolStatus = ToolStatus.READY
    call_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def avg_latency(self) -> float:
        return self.total_latency / max(self.call_count, 1)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.call_count, 1)


# ============================================================
# 工具注册中心（单例）
# ============================================================

class ToolRegistry:
    """
    AgentClaw 工具注册中心

    用法:
        registry = ToolRegistry()

        # 方式1: 装饰器注册
        @registry.register(
            name="my_tool",
            description="我的工具",
            parameters=[{"name": "query", "type": "string", "description": "查询内容"}],
            category=ToolCategory.CUSTOM
        )
        def my_tool(query: str) -> dict:
            return {"result": query}

        # 方式2: 手动注册
        registry.register_func(my_func, name="tool_name", ...)

        # 执行
        result = registry.execute("my_tool", query="hello")
    """

    _instance: Optional['ToolRegistry'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._tools: dict[str, ToolInfo] = {}
        self._categories: dict[ToolCategory, list[str]] = {}
        self._error_chain = None  # Phase 1: ErrorChain 实例
        self._initialized = True
        logger.info("ToolRegistry 初始化完成")

    def set_error_chain(self, chain) -> None:
        """Phase 1: 注入 ErrorChain 实例，用于工具调用的统一错误处理"""
        self._error_chain = chain
        logger.info("ErrorChain 已注入 ToolRegistry")

    # ----------------------------------------------------------
    # 注册方法
    # ----------------------------------------------------------

    def register(
        self,
        name: str | None = None,
        description: str = "",
        parameters: list | None = None,
        category: ToolCategory = ToolCategory.CUSTOM,
        examples: list[str] | None = None,
        timeout: int = 30
    ):
        """
        装饰器: 注册工具函数

        Args:
            name: 工具名称（默认使用函数名）
            description: 工具描述
            parameters: OpenAI function calling 格式的参数列表
            category: 工具分类
            examples: 使用示例
            timeout: 超时时间（秒）
        """
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__

            # 解析参数列表（兼容字符串列表和字典列表两种格式）
            tool_params = []
            if parameters:
                for p in parameters:
                    if isinstance(p, str):
                        # 简单格式: ["query", "expression"]
                        tool_params.append(ToolParameter(
                            name=p,
                            type="string",
                            description=p,
                            required=True,
                        ))
                    elif isinstance(p, dict):
                        # 完整格式: [{"name": "query", "type": "string", ...}]
                        tool_params.append(ToolParameter(
                            name=p.get("name", ""),
                            type=p.get("type", "string"),
                            description=p.get("description", ""),
                            required=p.get("required", True),
                            default=p.get("default", None),
                            enum=p.get("enum", None),
                        ))
                    else:
                        logger.warning(f"忽略未知格式的参数定义: {p}")

            tool_info = ToolInfo(
                name=tool_name,
                description=description,
                category=category,
                func=func,
                parameters=tool_params,
                examples=examples or [],
                timeout=timeout,
            )

            self._tools[tool_name] = tool_info

            # 分类索引
            if category not in self._categories:
                self._categories[category] = []
            if tool_name not in self._categories[category]:
                self._categories[category].append(tool_name)

            logger.info(f"注册工具: [{category.value}] {tool_name}")
            return func

        return decorator

    def register_func(
        self,
        func: Callable,
        name: str | None = None,
        description: str = "",
        parameters: list | None = None,
        category: ToolCategory = ToolCategory.CUSTOM,
        examples: list[str] | None = None,
        timeout: int = 30
    ) -> None:
        """手动注册工具函数（非装饰器方式）"""
        tool_name = name or func.__name__
        self.register(
            name=tool_name,
            description=description,
            parameters=parameters,
            category=category,
            examples=examples,
            timeout=timeout,
        )(func)

    def unregister(self, tool_name: str) -> bool:
        """取消注册工具"""
        if tool_name in self._tools:
            info = self._tools.pop(tool_name)
            cat = info.category
            if cat in self._categories and tool_name in self._categories[cat]:
                self._categories[cat].remove(tool_name)
            logger.info(f"取消注册工具: {tool_name}")
            return True
        return False

    # ----------------------------------------------------------
    # 参数校验
    # ----------------------------------------------------------

    def _validate_args(self, tool_name: str, kwargs: dict) -> str | None:
        """校验工具参数是否合法

        检查埋藏在 ToolParameter 元数据中的约束:
          - 必填参数是否提供
          - 参数类型是否匹配
          - 枚举值是否合法
          - 未提供可选参数时填入默认值

        Returns:
            error message (str) 或 None (校验通过)
        """
        tool_info = self._tools.get(tool_name)
        if not tool_info:
            return None  # 不在这里报错, execute() 会处理

        for param in tool_info.parameters:
            # 只校验 ToolParameter dataclass 格式的参数
            if not isinstance(param, ToolParameter):
                continue

            value = kwargs.get(param.name)

            # 1. 必填检查
            if param.required and value is None:
                return f"缺少必填参数 '{param.name}'"

            # 2. 可选参数填入默认值
            if value is None and param.default is not None:
                kwargs[param.name] = param.default
                continue
            if value is None:
                continue  # 未提供的可选参数, 跳过类型检查

            # 3. 类型检查
            if not self._check_type(value, param.type):
                return (
                    f"参数 '{param.name}' 类型错误: "
                    f"期望 {param.type}, 实际 {type(value).__name__}"
                )

            # 4. 枚举值检查
            if param.enum and str(value) not in param.enum:
                return (
                    f"参数 '{param.name}' 取值不在允许范围内: "
                    f"{param.enum}, 实际 '{value}'"
                )

        return None  # 校验通过

    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """检查值是否匹配期望类型"""
        type_map = {
            "string": (str,),
            "number": (int, float),
            "integer": (int,),
            "boolean": (bool,),
            "array": (list, tuple),
            "object": (dict,),
        }
        allowed = type_map.get(expected_type)
        if allowed is None:
            return True  # 未知类型, 跳过检查
        return isinstance(value, allowed)

    # ----------------------------------------------------------
    # 执行方法（同步）
    # ----------------------------------------------------------

    def execute(self, tool_name: str, args: dict = None, **kwargs) -> dict:
        """
        同步执行工具，兼容两种调用格式：

        格式1（字典）: registry.execute("web_search", {"query": "xxx"})
        格式2（关键字）: registry.execute("web_search", query="xxx")

        集成速率限制：通过 core.rate_limiter.TokenBucket 对所有工具调用限流。

        Args:
            tool_name: 工具名称
            args: 参数字典（位置参数方式传入时使用）
            **kwargs: 传递给工具函数的关键字参数

        Returns:
            dict: {"success": bool, "result": Any, "error": str|None}
        """
        # 合并参数：如果 args 是字典，合并到 kwargs
        if isinstance(args, dict):
            kwargs.update(args)
        # 如果 args 不是字典也不是 None，忽略（防御性处理）

        if tool_name not in self._tools:
            logger.warning(f"工具 '{tool_name}' 未注册，可用: {list(self._tools.keys())}")
            return {
                "success": False,
                "result": None,
                "error": f"工具 '{tool_name}' 未注册。可用工具: {list(self._tools.keys())}"
            }

        tool_info = self._tools[tool_name]

        if tool_info.status == ToolStatus.DISABLED:
            logger.warning(f"工具 '{tool_name}' 已被禁用")
            return {
                "success": False,
                "result": None,
                "error": f"工具 '{tool_name}' 已被禁用"
            }

        # 参数校验（在 rate_limiter 之前，避免浪费限流令牌）
        validation_error = self._validate_args(tool_name, kwargs)
        if validation_error:
            tool_info.call_count += 1
            tool_info.error_count += 1
            return {
                "success": False,
                "result": None,
                "error": validation_error,
            }

        # 速率限制检查（集成 core.rate_limiter）
        try:
            from core.rate_limiter import _llm_limiter
            if not _llm_limiter.consume(tokens=1):
                logger.warning(f"工具 {tool_name} 触发速率限制，请求被拒绝")
                return {
                    "success": False,
                    "result": None,
                    "error": f"工具 '{tool_name}' 触发速率限制，请稍后重试"
                }
        except Exception:
            pass  # rate_limiter 不可用时跳过限流

        tool_info.call_count += 1
        start_time = time.time()

        # Phase 1: Use ErrorChain if available
        if self._error_chain is not None:
            try:
                func = tool_info.func
                result = self._error_chain.execute(
                    tool_name=tool_name,
                    func=func,
                    kwargs=kwargs,
                    caller_context="tool_registry",
                )
                latency = time.time() - start_time
                tool_info.total_latency += latency
                # ErrorChain returns the raw result or a fallback dict
                # Check if it's an error response from ErrorChain
                if isinstance(result, dict) and result.get('_error'):
                    tool_info.error_count += 1
                    observe_tool_call(tool=tool_name, status="error", duration=latency)
                    return {"success": False, "result": None, "error": result.get('_message', 'Unknown error')}
                tool_info.success_count += 1
                logger.info(f"工具执行成功 (ErrorChain): {tool_name} ({latency:.3f}s)")
                observe_tool_call(tool=tool_name, status="success", duration=latency)
                return {"success": True, "result": result, "error": None, "latency": round(latency, 3)}
            except Exception as e:
                latency = time.time() - start_time
                tool_info.error_count += 1
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"工具执行失败 (ErrorChain): {tool_name} - {error_msg}")
                observe_tool_call(tool=tool_name, status="error", duration=latency)
                return {"success": False, "result": None, "error": error_msg}

        # Original execution path (without ErrorChain)
        try:
            func = tool_info.func

            # 如果是异步函数，用 asyncio 运行
            if asyncio.iscoroutinefunction(func):
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 在已有事件循环中，创建新线程运行
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, func(**kwargs))
                        result = future.result(timeout=tool_info.timeout)
                else:
                    result = loop.run_until_complete(func(**kwargs))
            else:
                # 同步函数，直接调用
                result = func(**kwargs)

            latency = time.time() - start_time
            tool_info.total_latency += latency
            tool_info.success_count += 1

            logger.info(f"工具执行成功: {tool_name} ({latency:.3f}s)")
            observe_tool_call(tool=tool_name, status="success", duration=latency)

            return {
                "success": True,
                "result": result,
                "error": None,
                "latency": round(latency, 3)
            }

        except asyncio.TimeoutError:
            latency = time.time() - start_time
            tool_info.error_count += 1
            tool_info.status = ToolStatus.TIMEOUT
            observe_tool_call(tool=tool_name, status="timeout", duration=latency)
            return {
                "success": False,
                "result": None,
                "error": f"工具 '{tool_name}' 执行超时（{tool_info.timeout}s）"
            }
        except Exception as e:
            latency = time.time() - start_time
            tool_info.error_count += 1
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"工具执行失败: {tool_name} - {error_msg}")
            observe_tool_call(tool=tool_name, status="error", duration=latency)
            return {
                "success": False,
                "result": None,
                "error": error_msg
            }

    # ----------------------------------------------------------
    # 查询方法
    # ----------------------------------------------------------

    def get_tool(self, tool_name: str) -> ToolInfo | None:
        """获取工具信息"""
        return self._tools.get(tool_name)

    def list_tools(self) -> list[str]:
        """列出所有已注册工具名称"""
        return list(self._tools.keys())

    def list_tools_by_category(self, category: ToolCategory) -> list[str]:
        """按分类列出工具"""
        return self._categories.get(category, [])

    def get_tools_for_llm(self) -> list[dict]:
        """
        获取 OpenAI function calling 格式的工具列表

        Returns:
            [
                {
                    "type": "function",
                    "function": {
                        "name": "tool_name",
                        "description": "...",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "param1": {"type": "string", "description": "..."},
                                ...
                            },
                            "required": ["param1", ...]
                        }
                    }
                },
                ...
            ]
        """
        tools_schema = []
        for name, info in self._tools.items():
            properties = {}
            required = []
            for param in info.parameters:
                prop = {"type": param.type, "description": param.description}
                if param.enum:
                    prop["enum"] = param.enum
                if param.default is not None:
                    prop["default"] = param.default
                properties[param.name] = prop
                if param.required:
                    required.append(param.name)

            func_schema = {
                "name": name,
                "description": info.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }

            tools_schema.append({
                "type": "function",
                "function": func_schema
            })

        return tools_schema

    def get_tool_stats(self) -> dict[str, dict]:
        """获取所有工具的统计信息"""
        stats = {}
        for name, info in self._tools.items():
            stats[name] = {
                "status": info.status.value,
                "call_count": info.call_count,
                "success_count": info.success_count,
                "error_count": info.error_count,
                "success_rate": f"{info.success_rate:.1%}",
                "avg_latency": f"{info.avg_latency:.3f}s",
            }
        return stats

    # ----------------------------------------------------------
    # 管理方法
    # ----------------------------------------------------------

    def enable(self, tool_name: str) -> bool:
        """启用工具"""
        if tool_name in self._tools:
            self._tools[tool_name].status = ToolStatus.READY
            return True
        return False

    def disable(self, tool_name: str) -> bool:
        """禁用工具"""
        if tool_name in self._tools:
            self._tools[tool_name].status = ToolStatus.DISABLED
            return True
        return False

    def clear(self) -> None:
        """清空所有已注册工具"""
        self._tools.clear()
        self._categories.clear()
        logger.info("已清空所有注册工具")

    @classmethod
    def reset(cls):
        """重置单例（用于测试）"""
        cls._instance = None


# ============================================================
# 全局单例
# ============================================================

registry = ToolRegistry()


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ToolRegistry 生产级自测")
    print("=" * 60)

    # 重置单例
    ToolRegistry.reset()
    reg = ToolRegistry()

    # 1. 测试装饰器注册
    @reg.register(
        name="add",
        description="两数相加",
        parameters=[
            {"name": "a", "type": "number", "description": "第一个数", "required": True},
            {"name": "b", "type": "number", "description": "第二个数", "required": True},
        ],
        category=ToolCategory.CALCULATOR
    )
    def add(a: float, b: float) -> dict:
        return {"result": a + b}

    # 2. 测试手动注册
    reg.register_func(
        lambda name="World": {"message": f"Hello, {name}!"},
        name="greet",
        description="打招呼",
        category=ToolCategory.CUSTOM
    )

    # 3. 列出工具
    print(f"\n已注册工具: {reg.list_tools()}")

    # 4. 测试同步执行 + **kwargs 传参
    print("\n--- 测试 execute() 同步调用 ---")
    result = reg.execute("add", a=3, b=5)
    print(f"add(a=3, b=5) => {result}")
    assert result["success"] is True
    assert result["result"]["result"] == 8.0

    result = reg.execute("greet", name="AgentClaw")
    print(f"greet(name='AgentClaw') => {result}")
    assert result["success"] is True

    # 5. 测试错误处理
    result = reg.execute("nonexistent")
    print(f"nonexistent => {result}")
    assert result["success"] is False

    # 6. 测试 LLM schema 输出
    print("\n--- LLM Tools Schema ---")
    schema = reg.get_tools_for_llm()
    print(json.dumps(schema, indent=2, ensure_ascii=False))

    # 7. 测试统计信息
    print("\n--- 工具统计 ---")
    stats = reg.get_tool_stats()
    for name, stat in stats.items():
        print(f"  {name}: {stat}")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
