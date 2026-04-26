"""
AgentClaw v6 — ReAct Agent（统一工具版）

v6 架构变更:
    - 不再使用 tools.py（旧的 LangChain @tool 定义）
    - 通过 agent_core 从 tool_registry 获取全部 15 个工具
    - RegistryAdapter 自动转换为 LangGraph StructuredTool

兼容性:
    - 保留 app 属性（延迟加载），兼容 api_server.py 的导入方式
    - from react_agent import app  -> 延迟初始化
    - from react_agent import get_app  -> 显式获取

工具清单:
    web_search, calculator, file_read, file_write,
    run_command, code_execute, vision_analyze, vision_ocr,
    image_generate, sys_overview, sys_processes, sys_disk,
    browser_navigate, browser_screenshot
"""

from core.config import settings
from core.logger import get_logger
logger = get_logger("ReActAgent")

# 延迟导入，避免循环依赖
_app = None


def get_app():
    """
    获取 ReAct Agent 实例（懒加载单例）。
    首次调用时会初始化所有工具并创建 Agent。
    """
    global _app
    if _app is None:
        from agent_core import get_react_agent
        _app = get_react_agent()
        logger.info("ReAct Agent 初始化完成（统一工具版）")
    return _app


# 兼容旧接口: api_server.py 引用 react_agent.app
# 使用 property 或延迟赋值避免 import 时初始化
def _get_app_property():
    """延迟属性访问，首次使用时才初始化"""
    return get_app()


# 注意: 这里不能用 @property 因为是模块级别
# 旧代码中 api_server.py 可能引用 react_agent.app
# 因此提供 app 变量，但设为 None，由 get_app() 触发实际初始化
app = None


def ensure_app():
    """确保 app 已初始化（供外部调用）"""
    global app
    if app is None:
        app = get_app()
    return app


if __name__ == "__main__":
    agent = get_app()
    config = {"configurable": {"thread_id": "user-1"}}
    result = agent.invoke(
        {"messages": [("human", "帮我搜索Python最新版本是什么")]},
        config,
    )
    print(result["messages"][-1].content)
