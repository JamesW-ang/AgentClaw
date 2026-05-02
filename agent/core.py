"""
AgentClaw v6 — 统一 Agent 核心
整合所有工具注册中心、LLM、反馈系统，提供统一的 Agent 入口。

这是整个项目的"枢纽模块"，解决之前各子系统孤立的问题。

架构:
    agent_core.py (本文件)
    |
    +-- init_all_tools()
    |   |-- builtin_tools      -> web_search, calculator, file_read, run_command, code_execute (5个)
    |   |   +-- os_tools (内置注册) -> file_write, sys_monitor, process_mgr, browser_tool (8个)
    |   |   +-- rag_searcher      -> knowledge_search (1个)
    |   |-- vision_tool        -> vision_analyze, vision_ocr, vision_compare (3个)
    |   +-- multimodal_image_gen -> image_generate (1个)
    |   总计: 18 个工具，全部挂载到 tool_registry
    |
    +-- get_react_tools() -> RegistryAdapter -> LangGraph StructuredTool[]
    |
    +-- get_react_agent() -> create_react_agent(LLM, 全部工具, MemorySaver)
    |
    +-- init_evolution()  -> FeedbackCollector + ExperienceLearner
    |                       + AdaptiveOptimizer + EvolutionManager
    |
    +-- record_feedback() -> 写入 FeedbackCollector，供 EvolutionManager 后台学习

使用方式:
    # 在 demo_ui.py / api_server.py 中:
    from agent.core import get_react_agent

    agent = get_react_agent()
    result = agent.invoke({"messages": [("human", "帮我搜索Python最新版本")]},
                          {"configurable": {"thread_id": "user-1"}})
"""

import sys
import time
from pathlib import Path

# 确保工作目录正确
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from core.config import settings
from core.logger import get_logger

logger = get_logger("AgentCore")

# 全局状态
_initialized = False
_evolution_manager = None
_feedback_collector = None

# LangFuse 可观测性
_langfuse_tracer = None

def _get_langfuse_tracer():
    """延迟初始化 LangFuse tracer"""
    global _langfuse_tracer
    if _langfuse_tracer is None:
        try:
            from tools.langfuse_adapter import LangFuseTracer
            _langfuse_tracer = LangFuseTracer()
            if _langfuse_tracer.enabled:
                logger.info("LangFuse 可观测性已启用")
            else:
                logger.info("LangFuse 未配置 (LANGFUSE_PUBLIC_KEY/SECRET_KEY 未设置)")
        except Exception as e:
            logger.warning(f"LangFuse 初始化失败: {e}")
            _langfuse_tracer = None
    return _langfuse_tracer


def get_langfuse_handler(session_id: str = None, user_id: str = None):
    """
    获取 LangFuse 回调处理器（供 invoke/astream 时传入 callbacks）。
    如果不返回 handler=LangFuse 未启用

    Usage:
        handler = get_langfuse_handler(session_id="user-1")
        agent.invoke(input, config={"callbacks": [handler]})
    """
    tracer = _get_langfuse_tracer()
    if tracer and tracer.enabled:
        return tracer.get_callback_handler(
            session_id=session_id or "default",
            user_id=user_id,
            tags=["agentclaw"],
        )
    return None


# ============================================================
# 工具初始化
# ============================================================

def init_all_tools():
    """
    初始化并注册所有工具到 tool_registry。
    每个工具模块内部通过 @registry.register 装饰器自动注册，
    这里只需要 import 即可触发注册。

    注意: builtin_tools 内部已经 import 并注册了 os_tools 的全部工具，
    因此不需要再次手动注册，避免重复。

    注册顺序:
        1. builtin_tools  -> 5个核心工具 + 8个os_tools + 1个knowledge_search (14个)
        2. vision_tool    -> vision_analyze, vision_ocr, vision_compare (3个)
        3. multimodal_image_gen -> image_generate (1个)
        4. aoi_engine     -> aoi_detect (1个)
        5. xml_config_tool -> xml_config_read/write/diff (3个)
        总计: 22 个工具，全部挂载到 tool_registry
    """
    global _initialized
    if _initialized:
        return

    logger.info("开始初始化工具注册中心...")

    # 1. 内置工具（装饰器自动注册，内部已包含 os_tools 和 rag_searcher 的注册）
    import tools.builtin

    # 2. 视觉工具（底部 try/except 自动注册 3 个工具）
    import tools.vision

    # 3. 图片生成工具（@registry.register 自动注册 1 个工具）
    import tools.image_gen

    # 4. AOI 检测工具
    import aoi.engine
    aoi.engine.register_aoi_tools()

    # 5. XML 配置管理工具（AOI 闭环调参支持）
    import tools.xml_config
    tools.xml_config.register_xml_config_tools()

    # 打印注册结果
    from tools.registry import registry
    tools = registry.list_tools()
    logger.info(f"工具注册完成: {len(tools)} 个 -> {tools}")

    _initialized = True


# ============================================================
# LangGraph 工具桥接
# ============================================================

def get_react_tools():
    """
    获取 LangGraph 兼容的完整工具列表。
    通过 RegistryAdapter 将 tool_registry 中的所有工具转换为 StructuredTool。
    """
    init_all_tools()
    from tools.registry_adapter import RegistryAdapter
    adapter = RegistryAdapter()
    tools = adapter.get_langchain_tools()
    logger.info(f"LangGraph 工具列表: {len(tools)} 个 -> {[t.name for t in tools]}")
    return tools


# ============================================================
# 统一 ReAct Agent
# ============================================================

_react_agent_instance = None
_learned_prompt_cache = {"prompt": None, "timestamp": 0}
_sqlite_checkpointer = None

def _build_agent_prompt() -> str:
    """构建 Agent 系统提示词，含学习策略注入（缓存 10 分钟）"""
    prompt = (
        "你是 AgentClaw 智能助手，拥有以下能力：\n"
        "1. web_search — 搜索互联网获取最新信息\n"
        "2. calculator — 安全数学计算（支持函数）\n"
        "3. file_read / file_write — 安全文件读写\n"
        "4. run_command — 安全执行系统命令\n"
        "5. code_execute — 沙箱中执行 Python 代码\n"
        "6. vision_analyze / vision_ocr / vision_compare — 图片分析和OCR\n"
        "7. image_generate — AI 文生图\n"
        "8. knowledge_search — RAG 知识库检索\n"
        "9. sys_monitor / sys_process_list / sys_disk_info — 系统监控\n"
        "10. process_start / process_stop / process_list — 进程管理\n"
        "11. browser_navigate / browser_screenshot — 浏览器控制\n"
        "12. aoi_detect — AOI 电路板缺陷检测\n"
        "13. xml_config_read — 读取 AOI XML 配置文件（视觉参数/运动坐标）\n"
        "14. xml_config_write — 修改 AOI XML 配置参数（含备份、校验、原子写入）\n"
        "15. xml_config_diff — 对比当前配置与默认值的差异\n\n"
        "根据用户需求自动选择合适的工具。回答要准确、专业、清晰。\n"
        "如果工具执行失败，尝试用其他方式解决问题，并向用户解释。"
    )

    # Phase 3: 注入学习策略（如果可用且缓存未过期）
    global _evolution_manager
    now = time.time()
    if (_learned_prompt_cache["prompt"] is not None
            and now - _learned_prompt_cache["timestamp"] < 600):
        return _learned_prompt_cache["prompt"]

    try:
        if _evolution_manager is not None:
            learner = getattr(_evolution_manager, 'learner', None)
            if learner and learner.strategies:
                strategies = learner.strategies
                # 正向策略: effectiveness > 0.5
                good = sorted(
                    [s for s in strategies.values()
                     if not s.anti_pattern and s.effectiveness > 0.5],
                    key=lambda s: s.effectiveness, reverse=True
                )[:3]
                # 反模式: confidence > 0.5
                bad = sorted(
                    [s for s in strategies.values()
                     if s.anti_pattern and s.confidence > 0.5],
                    key=lambda s: s.confidence, reverse=True
                )[:3]

                parts = [prompt]
                if good:
                    parts.append("\n\n[学习策略提示]")
                    for s in good:
                        seq = " → ".join(s.tool_sequence[:3])
                        parts.append(
                            f"- 模式「{s.name}」: {seq} "
                            f"(效果评分 {s.effectiveness:.2f})"
                        )
                if bad:
                    parts.append("\n[应避免的模式]")
                    for s in bad:
                        seq = " → ".join(s.tool_sequence[:3])
                        parts.append(
                            f"- 反模式「{s.name}」: {seq} "
                            f"(置信度 {s.confidence:.2f})"
                        )
                prompt = "\n".join(parts)
    except Exception as e:
        logger.debug(f"策略注入未生效: {e}")

    _learned_prompt_cache["prompt"] = prompt
    _learned_prompt_cache["timestamp"] = time.time()
    return prompt


def get_react_agent():
    """
    获取统一 ReAct Agent（懒加载单例）。
    该 Agent 拥有全部已注册工具，可以自动选择合适的工具完成任务。
    Phase 1: 使用 LLMGuard 容错模式包装 LLM 调用。
    """
    global _react_agent_instance, _sqlite_checkpointer
    if _react_agent_instance is not None:
        return _react_agent_instance

    import sqlite3
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langgraph.checkpoint.sqlite import SqliteSaver

    # Phase 1: Use LLMGuard for fault-tolerant LLM calls
    try:
        from core.llm_guard import LLMGuard
        from core.guarded_chat_model import GuardedChatModel

        _llm_guard = LLMGuard(
            default_model=settings.LLM_MODEL,
            backup_models=["deepseek-reasoner"],
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
        llm = GuardedChatModel(guard=_llm_guard, temperature=0)
        logger.info("ReAct Agent 使用 LLMGuard 容错模式")
    except Exception as e:
        logger.warning(f"LLMGuard 初始化失败，回退到裸 ChatOpenAI: {e}")
        llm = ChatOpenAI(
            model=settings.LLM_MODEL, temperature=0,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )

    tools = get_react_tools()

    # Phase 3: 注入反馈采集器到 RegistryAdapter（统一 Agent 全局生效）
    try:
        from learning.feedback import FeedbackCollector
        from tools.registry_adapter import set_feedback_collector
        collector = FeedbackCollector(persist_dir=str(SCRIPT_DIR / "evolution_data"))
        set_feedback_collector(collector)
        logger.info("统一 Agent 已接入反馈采集")
    except Exception as e:
        logger.warning(f"反馈采集器注入失败: {e}")

    # SQLite checkpointer for persistent conversation history
    if _sqlite_checkpointer is None:
        db_path = SCRIPT_DIR / "data" / "agent_checkpoints.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        _sqlite_checkpointer = SqliteSaver(conn)
        logger.info(f"对话持久化已启用 (SQLite: {db_path})")

    _react_agent_instance = create_react_agent(
        model=llm,
        tools=tools,
        prompt=_build_agent_prompt(),
        checkpointer=_sqlite_checkpointer,
    )

    logger.info("统一 ReAct Agent 创建完成（含全部注册工具）")
    return _react_agent_instance


# ============================================================
# 三链联动初始化（Phase 1）
# ============================================================

def init_chains():
    """
    初始化三链联动系统 (Phase 1)
    将 ErrorChain、TraceChain、LLMGuard 联动起来
    """
    try:
        from core.error_chain import ErrorChain
        from core.trace_chain import TraceChain

        error_chain = ErrorChain()
        trace_chain = TraceChain(
            persist_dir=str(SCRIPT_DIR / "data" / "traces"),
            console_enabled=False,
        )

        # Inject ErrorChain into tool_registry
        from tools.registry import registry
        registry.set_error_chain(error_chain)

        logger.info("三链联动系统已初始化 (ErrorChain + TraceChain)")
        return error_chain, trace_chain
    except Exception as e:
        logger.warning(f"三链初始化失败 (不影响主功能): {e}")
        return None, None


# ============================================================
# 自主学习系统（Level 4）
# ============================================================

def init_evolution(interval: int = 3600):
    """
    初始化自主学习系统（后台进化循环）。

    工作原理:
        1. FeedbackCollector 收集每次工具执行的反馈（成功/失败/延迟）
        2. ExperienceLearner 从历史反馈中挖掘成功的工具调用模式
        3. AdaptiveOptimizer 根据反馈动态调整路由权重和 Prompt 模板
        4. EvolutionManager 协调以上三个子系统，后台定期运行进化循环

    Args:
        interval: 进化循环间隔时间（秒），默认3600（1小时）
    Returns:
        EvolutionManager 实例（如果初始化成功），否则 None
    """
    global _evolution_manager, _feedback_collector

    if _evolution_manager is not None:
        return _evolution_manager

    try:
        from learning.feedback import FeedbackCollector
        from learning.learner import ExperienceLearner
        from learning.optimizer import AdaptiveOptimizer
        from learning.evolution import EvolutionManager

        _feedback_collector = FeedbackCollector()
        learner = ExperienceLearner(_feedback_collector)
        optimizer = AdaptiveOptimizer()
        _evolution_manager = EvolutionManager(_feedback_collector, learner, optimizer)
        _evolution_manager.start_evolution_loop(interval=interval)
        # 注入到 RegistryAdapter（反模式检测 + 策略优化）
        from tools.registry_adapter import set_learner, set_optimizer
        set_learner(learner)
        set_optimizer(optimizer)
        logger.info(f"自主学习系统已启动 (间隔 {interval}s)")
        return _evolution_manager
    except Exception as e:
        logger.warning(f"自主学习初始化失败: {e}")
        return None


def record_feedback(task_id: str, tool_name: str, success: bool,
                    latency: float, error_type: str = None, context: str = ""):
    """
    记录工具执行反馈（供外部调用）。

    使用场景: 在 tool_registry.execute() 完成后调用此函数，
    将执行结果记录到 FeedbackCollector，供 EvolutionManager 后台学习。

    Args:
        task_id: 任务唯一标识
        tool_name: 工具名称
        success: 是否成功
        latency: 执行耗时（秒）
        error_type: 错误类型（可选）
        context: 上下文信息（可选）
    """
    global _feedback_collector
    if _feedback_collector is None:
        init_evolution()
    if _feedback_collector is not None:
        from learning.feedback import FeedbackSignal
        _feedback_collector.collect(FeedbackSignal(
            task_id=task_id,
            tool_name=tool_name,
            success=success,
            latency=latency,
            error_type=error_type,
            context=context,
        ))


# ============================================================
# 工具统计（调试/展示用）
# ============================================================

def get_tool_summary() -> dict:
    """获取所有工具的摘要信息，用于 UI 展示"""
    init_all_tools()
    from tools.registry import registry
    return {
        "total": len(registry.list_tools()),
        "tools": registry.list_tools(),
        "stats": registry.get_tool_stats(),
        "categories": {
            cat.value: registry.list_tools_by_category(cat)
            for cat in registry._categories
        },
    }


# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AgentClaw v6 — 核心 Agent 初始化测试")
    print("=" * 60)

    # 1. 初始化所有工具
    init_all_tools()

    # 2. 显示注册结果
    from tools.registry import registry
    tools = registry.list_tools()
    print(f"\n已注册工具 ({len(tools)} 个):")
    for name in tools:
        info = registry.get_tool(name)
        desc = info.description[:50] + "..." if len(info.description) > 50 else info.description
        print(f"  [{info.category.value}] {name}: {desc}")

    # 3. 显示 LangGraph 工具
    lc_tools = get_react_tools()
    print(f"\nLangGraph 兼容工具: {len(lc_tools)} 个")
    for t in lc_tools:
        print(f"  - {t.name}")

    # 4. 显示 LLM Schema
    print(f"\nOpenAI Function Calling Schema: {len(registry.get_tools_for_llm())} 个")

    # 5. 初始化自主学习（可选）
    evo = init_evolution(interval=3600)
    if evo:
        print("\n自主学习系统: 已启动")
    else:
        print("\n自主学习系统: 未启动")

    print("\n" + "=" * 60)
    print("全部测试通过!")
    print("=" * 60)
