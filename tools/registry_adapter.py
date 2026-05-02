# 将 tool_registry 的工具适配为 LangGraph 兼容格式
import time
import uuid
import traceback
from tools.registry import ToolInfo, ToolParameter, registry
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model
import inspect
import logging

from core.logger import get_logger
logger = get_logger("registry_adapter")

# 延迟导入，避免循环依赖
_collector = None

def set_feedback_collector(collector):
    """注入 FeedbackCollector 实例（由 main/demo_ui 启动时调用）"""
    global _collector
    _collector = collector
    logger.info("FeedbackCollector 已注入 RegistryAdapter")


def _make_pydantic_schema(params):
    """从 ToolParameter 列表生成 Pydantic v2 兼容 schema

    使用 create_model() 而非 type() 动态创建模型，
    确保 Pydantic v2 的类型注解正确传递。
    """
    fields = {}
    for p in params:
        # 兼容三种参数格式: ToolParameter dataclass / str / dict
        if isinstance(p, ToolParameter):
            name = p.name
            ptype = getattr(p, 'type', None) or 'str'
            pdesc = getattr(p, 'description', name) or name
            pdefault = getattr(p, 'default', None)

            field_type = _type_map(ptype)
            if pdefault is not None:
                fields[name] = (field_type, Field(default=pdefault, description=pdesc))
            else:
                fields[name] = (field_type, Field(description=pdesc))

        elif isinstance(p, str):
            fields[p] = (str, Field(description=p))

        elif isinstance(p, dict):
            name = p.get("name", "arg")
            ftype = _type_map(p.get("type", "str"))
            fdesc = p.get("description", name)
            fdefault = p.get("default", None)
            if fdefault is not None:
                fields[name] = (ftype, Field(default=fdefault, description=fdesc))
            else:
                fields[name] = (ftype, Field(description=fdesc))

    return create_model("DynamicArgs", **fields) if fields else None


def _type_map(t: str):
    return {"str": str, "string": str, "int": int, "float": float, "bool": bool, "number": float}.get(t, str)


class RegistryAdapter:
    """适配器: tool_registry -> LangGraph tools

    确保使用全局 registry 单例，而不是新建实例。
    调用 get_langchain_tools() 前需确保 builtin_tools 等模块已被 import（触发注册）。
    """

    def __init__(self):
        self._cached_tools = None

    def get_langchain_tools(self):
        """获取所有已注册工具的 LangGraph StructuredTool 列表"""
        tools = []
        for name, info in registry._tools.items():
            # info 是 ToolInfo dataclass，不是 dict
            if not isinstance(info, ToolInfo):
                logger.warning(f"跳过非标准工具: {name}")
                continue

            schema = _make_pydantic_schema(info.parameters)

            def make_wrapper(tool_name):
                def wrapper(**kwargs):
                    t0 = time.perf_counter()
                    task_id = f"task-{uuid.uuid4().hex[:8]}"
                    success = False
                    error_type = None
                    latency = 0.0

                    try:
                        result = registry.execute(tool_name, kwargs)
                        latency = time.perf_counter() - t0

                        if result.get("success"):
                            success = True
                            # 把 dict 结果序列化为字符串，给 LLM 阅读
                            return str(result.get("result", ""))
                        else:
                            error_type = "execute_error"
                            return f"工具执行失败: {result.get('error', '未知错误')}"
                    except PermissionError as e:
                        latency = time.perf_counter() - t0
                        error_type = "permission_error"
                        return f"权限拒绝: {e}"
                    except Exception as e:
                        latency = time.perf_counter() - t0
                        error_type = type(e).__name__
                        logger.error(f"工具 {tool_name} 异常: {e}", exc_info=True)
                        return f"工具异常: {e}"
                    finally:
                        # 收集反馈信号（如果 collector 已注入）
                        if _collector is not None:
                            try:
                                from learning.feedback import FeedbackSignal
                                signal = FeedbackSignal(
                                    task_id=task_id,
                                    tool_name=tool_name,
                                    success=success,
                                    latency=round(latency, 4),
                                    error_type=error_type,
                                )
                                _collector.collect(signal)
                            except Exception:
                                pass  # 反馈采集失败不影响主流程

                return wrapper

            tool = StructuredTool.from_function(
                func=make_wrapper(name),
                name=name,
                description=info.description,
                args_schema=schema
            )
            tools.append(tool)
            logger.info(f"适配工具: {name}")

        self._cached_tools = tools
        return tools

    def get_tool_names(self):
        """获取所有已注册工具名称"""
        return registry.list_tools()

    def get_tool_count(self):
        """获取已注册工具数量"""
        return len(registry._tools)
