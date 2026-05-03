"""
AgentClaw v6 — 统一 Demo UI
集成8大场景: ReAct Agent / RAG知识库 / 多模态视觉 / 图片生成 / 多Agent协作 / AOI检测 / AOI智能闭环

v6 迁移变更:
  - dotenv 加载移至 core/config.py
  - os.getenv 全部替换为 settings.X
  - 新增 logger 统一日志
  - 新增 AOI 上位机检测模块（Tab 6）
  - Step2: 接入 EvolutionManager 自主进化

用法:
    python demo_ui.py

依赖:
    pip install gradio langchain-openai langchain-chroma langchain-huggingface langgraph python-dotenv openai opencv-python numpy Pillow
    可选: pip install onnxruntime  (AOI 深度学习模式)
"""
import os
import sys
import json
import threading
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict

# 确保工作目录正确（以脚本所在目录为基准）
SCRIPT_DIR = Path(__file__).resolve().parent.parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

# v6: dotenv 加载已移至 core/config.py，不再在此处调用

from core.config import settings
from core.logger import get_logger
logger = get_logger("demo_ui")

import gradio as gr


# ============================================================
# AOI 检测引擎（已提取到独立模块 aoi_engine.py）
# ============================================================
from aoi.engine import (
    get_aoi_engine, aoi_visualize, AOIEngine,
    _aoi_lock,  # 复用 aoi_engine 的线程锁
)


# ============================================================
# 延迟初始化各模块（避免 import 时跑 __main__ 逻辑）
# ============================================================

_lock = threading.Lock()

# --- LLM ---
_llm = None
def get_llm():
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            model=settings.LLM_MODEL, temperature=0,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
    return _llm

# --- 1. ReAct Agent + Step2 FeedbackCollector ---
_react_app = None
_feedback_collector = None

def get_feedback_collector():
    """获取/初始化全局 FeedbackCollector"""
    global _feedback_collector
    if _feedback_collector is None:
        from learning.feedback import FeedbackCollector
        _feedback_collector = FeedbackCollector(
            persist_dir=str(SCRIPT_DIR / "evolution_data")
        )
        logger.info("FeedbackCollector 已初始化")
    return _feedback_collector

def get_react_app():
    """获取 ReAct Agent（复用 agent_core 统一实例，消除双实例问题）"""
    global _react_app
    if _react_app is None:
        from agent.core import get_react_agent
        # 确保工具已注册
        import tools.builtin as builtin_tools  # noqa: F401
        try:
            import tools.vision  # noqa: F401
        except ImportError:
            pass
        try:
            import tools.image_gen  # noqa: F401
        except ImportError:
            pass

        # 注入反馈采集器到 RegistryAdapter
        collector = get_feedback_collector()
        from tools.registry_adapter import set_feedback_collector
        set_feedback_collector(collector)

        _react_app = get_react_agent()
        logger.info("ReAct Agent 复用 agent_core 统一实例")
    return _react_app

# --- 2. RAG Engine ---
_rag = None
def get_rag():
    """获取 RAG 引擎（复用 builtin_tools 的共享实例）"""
    import tools.builtin as builtin_tools
    return builtin_tools._get_rag_engine()

# --- 3. Multi-Agent (多场景工具增强) ---
_multi_apps = {}

# 多 Agent 场景配置：角色、System Prompt
MULTI_AGENT_SCENES = {
    "code_review": {
        "title": "代码审查",
        "desc": "技术分析师 → 重构工程师 → 审核员，多维度深度审查代码质量",
        "roles": [
            ("analyst", "技术分析师",
             "你是高级技术分析师。对用户提交的代码或技术问题进行多维度深度分析：\n\n"
             "## 分析维度\n"
             "1. **代码质量**：可读性、命名规范、复杂度\n"
             "2. **潜在缺陷**：空指针、边界条件、竞态条件、内存泄漏\n"
             "3. **安全风险**：注入漏洞、权限问题、敏感数据暴露\n"
             "4. **性能瓶颈**：算法复杂度、IO阻塞、不必要的计算\n"
             "5. **架构合理性**：模块耦合度、职责划分、扩展性\n\n"
             "## 输出格式\n"
             "- 每个问题标注等级：致命/严重/建议/优化\n"
             "- 给出具体代码位置和修改建议\n"
             "- 如需查阅资料，使用搜索工具"),
            ("programmer", "重构工程师",
             "你是高级程序员。基于分析师的报告，提供具体修改方案：\n\n"
             "1. 每个问题给出可直接使用的代码修改（Diff格式）\n"
             "2. 评估改动影响范围和风险\n"
             "3. 提供单元测试建议\n"
             "4. 必要时使用工具验证方案\n\n"
             "输出标注优先级：P0紧急 / P1重要 / P2建议，并给出回滚方案"),
            ("reviewer", "审核员",
             "你是技术审核员（终审）。全面审核分析和修改方案：\n\n"
             "1. 分析是否遗漏关键问题\n"
             "2. 修改方案是否引入新风险\n"
             "3. 整体方案的一致性\n\n"
             "给出最终结论：通过 / 条件通过 / 需重做\n"
             "附优先级排序的行动清单"),
        ],
    },
    "tech_design": {
        "title": "技术方案设计",
        "desc": "系统架构师 → 方案师 → 评审员，从需求到技术方案的完整设计链路",
        "roles": [
            ("architect", "系统架构师",
             "你是资深系统架构师（10年+经验）。从需求出发设计系统方案：\n\n"
             "1. **需求拆解**：功能/非功能需求/约束条件\n"
             "2. **技术选型**：对比2-3种方案，说明选择理由\n"
             "3. **架构设计**：模块划分、数据流、接口定义\n"
             "4. **容错设计**：异常处理、降级策略\n\n"
             "输出架构描述、接口定义和技术风险。必要时使用工具调研。"),
            ("engineer", "技术方案师",
             "你是高级技术方案工程师。将架构设计细化为实现方案：\n\n"
             "1. 数据模型和核心逻辑\n"
             "2. 接口协议和错误码\n"
             "3. 性能预估（QPS/延迟/资源）\n"
             "4. 开发排期和里程碑\n\n"
             "每个模块给出实现要点和技术难点应对策略。"),
            ("reviewer", "方案评审员",
             "你是技术评审专家（CTO视角）。评审方案：\n\n"
             "1. 架构合理性：是否过度/不足设计\n"
             "2. 技术风险：关键技术是否成熟\n"
             "3. 成本效益：开发成本 vs 业务价值\n"
             "4. 扩展性：未来6-12个月演进空间\n\n"
             "给出评审结论和改进建议。"),
        ],
    },
    "problem_diagnosis": {
        "title": "问题诊断",
        "desc": "诊断专家 → 方案师 → 验证员，系统化定位和解决技术问题",
        "roles": [
            ("diagnostician", "诊断专家",
             "你是技术诊断专家。系统化分析技术问题：\n\n"
             "1. **现象分类**：功能性/性能/安全/兼容性\n"
             "2. **5-Why分析**：连续追问根因\n"
             "3. **排除法**：逐步缩小范围\n"
             "4. **模式匹配**：对比已知问题\n\n"
             "输出问题分类、根因假设（Top 3）、验证方法。\n"
             "必要时使用搜索工具查找解决方案，或执行命令诊断。"),
            ("solver", "解决方案师",
             "你是解决方案专家。设计修复方案：\n\n"
             "1. 修复步骤（按优先级排序）\n"
             "2. 风险评估和副作用\n"
             "3. 预防措施\n"
             "4. 回滚方案\n\n"
             "给出可执行的命令或代码，标注预计修复时间。\n"
             "必要时使用工具验证命令。"),
            ("validator", "验证员",
             "你是方案验证专家。检验方案有效性：\n\n"
             "1. 方案是否覆盖所有根因\n"
             "2. 是否引入新问题\n"
             "3. 预防措施是否有效\n\n"
             "输出最终验证清单和后续监控建议。"),
        ],
    },
}


def get_multi_app(scene="code_review"):
    """获取/创建多Agent应用（按场景缓存，每个节点具备工具调用能力）"""
    global _multi_apps
    if scene in _multi_apps:
        return _multi_apps[scene]

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
    from langgraph.graph import StateGraph, START, END, MessagesState
    from langchain_core.tools import tool as lc_tool
    from learning.feedback import FeedbackSignal

    scene_cfg = MULTI_AGENT_SCENES.get(scene, MULTI_AGENT_SCENES["code_review"])
    collector = get_feedback_collector()

    # --- Agent 工具（通过 tool_registry 统一调度，复用已注册工具）---
    from tools.registry import registry
    # 确保工具已注册
    import tools.builtin as builtin_tools  # 触发 @registry.register 装饰器注册
    import os_tools.file_write  # 触发 os_tools 注册

    # registry → lc_tool 的桥接函数
    def _registry_tool_wrapper(registry_name: str, tool_desc: str = "",
                                param_name: str = "input", param_desc: str = "输入内容"):
        """将 registry 工具包装为 LangChain @lc_tool"""
        from langchain_core.tools import StructuredTool
        from pydantic import create_model

        _input_schema = create_model(
            f"{registry_name}_input",
            **{param_name: (str, param_desc)},
        )

        _valid_params = set(_input_schema.model_fields.keys())

        def _run(**kwargs) -> str:
            try:
                # 动态补齐 registry 工具声明了但 LLM 没传的参数
                tool_info = registry._tools.get(registry_name)
                if tool_info:
                    for param in tool_info.parameters:
                        if param.name not in kwargs:
                            if param.default is not None:
                                kwargs[param.name] = param.default
                            else:
                                # 无默认值的必填参数：按类型填占位符
                                if param.type in ("int", "integer", "number"):
                                    kwargs[param.name] = 0
                                elif param.type == "boolean":
                                    kwargs[param.name] = False
                                else:
                                    kwargs[param.name] = ""
                    # 过滤掉既不在 schema 也不在 registry 参数列表中的幻觉参数
                    registry_params = {p.name for p in tool_info.parameters}
                    allowed = _valid_params | registry_params
                    kwargs = {k: v for k, v in kwargs.items() if k in allowed}
                else:
                    kwargs = {k: v for k, v in kwargs.items() if k in _valid_params}
                result = registry.execute(registry_name, kwargs)
                if result.get("success"):
                    return str(result.get("result", ""))[:3000]
                else:
                    return f"工具执行失败: {result.get('error', '未知错误')}"
            except Exception as e:
                return f"工具异常: {e}"

        return StructuredTool.from_function(
            func=_run,
            name=f"agent_{registry_name}",
            description=tool_desc or f"调用 {registry_name} 工具",
            args_schema=_input_schema,
        )

    agent_search = _registry_tool_wrapper("web_search",
        tool_desc="搜索互联网获取最新技术资料和解决方案",
        param_name="query", param_desc="搜索关键词")

    agent_calculate = _registry_tool_wrapper("calculator",
        tool_desc="计算数学表达式（四则运算、对数、幂运算等）",
        param_name="expression", param_desc="数学表达式，如 2+3*4")

    agent_read_file = _registry_tool_wrapper("file_read",
        tool_desc="读取文件内容（代码、配置、日志等）",
        param_name="file_path", param_desc="文件路径")

    agent_run_command = _registry_tool_wrapper("run_command",
        tool_desc="执行Shell命令（用于诊断验证，如 ls/cat/grep/ping）",
        param_name="command", param_desc="Shell命令")

    tools = [agent_search, agent_calculate, agent_read_file, agent_run_command]
    tool_map = {t.name: t for t in tools}

    # --- LLM 绑定工具 ---
    llm = ChatOpenAI(
        model=settings.LLM_MODEL, temperature=0,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
    )
    llm_with_tools = llm.bind_tools(tools)

    # --- 创建带工具调用循环的 Agent 节点 ---
    def make_node(role_id, role_name, system_prompt):
        """创建带工具调用循环的Agent节点（最多5轮工具调用）"""
        def node(state):
            t0 = time.perf_counter()
            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            logger.info(f"[多Agent] [{role_name}] 开始处理，输入消息数={len(state['messages'])}")

            response = None
            for round_i in range(5):
                try:
                    logger.info(f"[多Agent] [{role_name}] 第{round_i+1}轮LLM调用...")
                    # 清理历史消息中的 DSML 标签，防止污染
                    import re as _re
                    from langchain_core.messages import ToolMessage as _TM, AIMessage as _AM
                    clean_msgs = []
                    for m in messages:
                        if isinstance(m, _TM):
                            # ToolMessage 必须保留 tool_call_id，否则序列化失败
                            _c = _re.sub(r'<|｜DSML|｜[^>]*>', '', m.content) if isinstance(m.content, str) else m.content
                            clean_msgs.append(_TM(content=_c, tool_call_id=m.tool_call_id))
                        elif isinstance(m, _AM) and m.tool_calls:
                            # AIMessage 带工具调用时必须保留 tool_calls，只清理 content
                            _c = _re.sub(r'<|｜DSML|｜[^>]*>', '', m.content) if isinstance(m.content, str) else m.content
                            clean_msgs.append(_AM(content=_c, tool_calls=m.tool_calls))
                        elif hasattr(m, 'content') and isinstance(m.content, str):
                            _cleaned = _re.sub(r'<|｜DSML|｜[^>]*>', '', m.content)
                            clean_msgs.append(type(m)(content=_cleaned))
                        else:
                            clean_msgs.append(m)
                    response = llm_with_tools.invoke(clean_msgs)
                except Exception as e:
                    logger.error(f"[多Agent] [{role_name}] LLM调用失败: {e}")
                    response = None
                    break

                # 检查 response 本身是否包含 DSML 标签（模型异常输出）
                # 注意：DeepSeek 使用全角字符 ＜｜｜，需同时匹配半角和全角
                _dsml_pattern = r'<｜｜DSML｜｜|<\|\|DSML<\|\|'
                if hasattr(response, 'content') and response.content and _re.search(_dsml_pattern, str(response.content)):
                    logger.warning(f"[多Agent] [{role_name}] 模型返回了 DSML 标签而非标准 tool_calls，尝试重新生成")
                    messages.append(response)
                    messages.append(SystemMessage(content="请使用标准工具调用格式，不要使用任何特殊标签格式。直接调用工具即可。"))
                    continue

                if not response.tool_calls:
                    logger.info(f"[多Agent] [{role_name}] 第{round_i+1}轮无工具调用，直接输出 (content长度={len(response.content or '')})")
                    break

                # 有工具调用
                logger.info(f"[多Agent] [{role_name}] 第{round_i+1}轮请求{len(response.tool_calls)}个工具调用: {[tc['name'] for tc in response.tool_calls]}")
                messages.append(response)
                for tc in response.tool_calls:
                    logger.info(f"[多Agent] [{role_name}] 调用工具: {tc['name']}({tc['args']})")
                    try:
                        tool_fn = tool_map.get(tc["name"])
                        if tool_fn is None:
                            result = f"未知工具: {tc['name']}，可用工具: {list(tool_map.keys())}"
                            logger.warning(f"[多Agent] [{role_name}] 工具映射缺失: {tc['name']}")
                        else:
                            result = tool_fn.invoke(tc["args"])
                        logger.info(f"[多Agent] [{role_name}] 工具返回({tc['name']}): {str(result)[:200]}")
                    except Exception as e:
                        result = f"工具执行失败: {e}"
                        logger.error(f"[多Agent] [{role_name}] 工具执行异常: {tc['name']} -> {e}")
                    messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

            # 如果最后仍是工具调用（content为空），不带工具再调一次逼它输出文字总结
            if response and (not response.content or response.content.strip() == ""):
                logger.info(f"[多Agent] [{role_name}] 工具调用完成但无文字输出，触发总结...")
                try:
                    messages.append(SystemMessage(content="请基于以上工具调用结果，直接输出你的分析总结，不要再调用工具。"))
                    summary_resp = llm.invoke(messages)
                    if summary_resp.content and summary_resp.content.strip():
                        response = summary_resp
                        logger.info(f"[多Agent] [{role_name}] 总结生成完成 (长度={len(response.content)})")
                except Exception as e:
                    logger.error(f"[多Agent] [{role_name}] 总结生成失败: {e}")

            if response is None:
                final_text = f"Agent 响应失败（请检查API Key和网络连接）"
            else:
                final_text = response.content if isinstance(response.content, str) else str(response.content)
                # 最终清洗：移除任何残留的 DSML 标签
                if final_text:
                    final_text = _re.sub(r'<｜｜DSML｜｜[^>]*>', '', final_text)
                if not final_text.strip():
                    final_text = "（工具调用完成，但未生成文字总结）"

            logger.info(f"[多Agent] [{role_name}] 处理完成，耗时={time.perf_counter()-t0:.2f}s，输出长度={len(final_text)}")

            collector.collect(FeedbackSignal(
                task_id=f"ma-{role_id}-{time.time()}",
                tool_name=f"multi_agent.{role_id}",
                success=response is not None,
                latency=round(time.perf_counter() - t0, 4),
            ))
            return {"messages": [AIMessage(content=f"## [{role_name}]\n\n{final_text}")]}
        return node

    # --- 构建工作流 ---
    builder = StateGraph(MessagesState)
    role_ids = []
    for role_id, role_name, prompt in scene_cfg["roles"]:
        builder.add_node(role_id, make_node(role_id, role_name, prompt))
        role_ids.append(role_id)

    builder.add_edge(START, role_ids[0])
    for i in range(len(role_ids) - 1):
        builder.add_edge(role_ids[i], role_ids[i + 1])
    builder.add_edge(role_ids[-1], END)

    app = builder.compile()
    _multi_apps[scene] = app
    logger.info(f"多Agent应用已创建: 场景={scene}, 角色={role_ids}")
    return app


# ============================================================
# 场景处理函数
# ============================================================

# ---- 1. ReAct Agent ----
def react_respond(message, history):
    with _lock:
        try:
            from agent.core import get_react_agent
            app = get_react_agent()
            config = {"configurable": {"thread_id": "demo-react-1"}}
            result = app.invoke({"messages": [("human", message)]}, config)
            yield result["messages"][-1].content
        except Exception as e:
            yield f"ReAct Agent 错误: {e}"


# ---- 2. RAG 知识库 ----
RAG_MAX_FILE_SIZE = 5 * 1024 * 1024

def rag_upload(files):
    if not files:
        return "未选择文件", rag_get_stats()
    import tools.builtin as builtin_tools
    added = 0; msgs = []
    for f in files:
        try:
            if isinstance(f, str):
                file_path = f; name = Path(f).name
            else:
                file_path = f.name; name = Path(f.name).name
            file_size = os.path.getsize(file_path)
            if file_size > RAG_MAX_FILE_SIZE:
                msgs.append(f"[{name}] 文件过大 ({file_size/1024/1024:.1f}MB)")
                continue
            if file_size == 0:
                msgs.append(f"[{name}] 文件为空"); continue
            count = builtin_tools.rag_add_documents(file_path)
            added += count; msgs.append(f"[{name}] {count} 个文档块")
        except Exception as e:
            fname = name if 'name' in dir() else str(f)
            msgs.append(f"[{fname}] 失败: {e}")
    return f"共添加 {added} 个文档块\n" + "\n".join(msgs), rag_get_stats()

def rag_add_text(text):
    if not text or not text.strip():
        return "请输入文本内容", rag_get_stats()
    import tools.builtin as builtin_tools
    count = builtin_tools.rag_add_text_to_shared(text.strip(), source="手动输入")
    return f"已添加 {count} 个文档块", rag_get_stats()

def rag_search(query, top_k):
    if not query or not query.strip():
        return "请输入查询内容"
    import tools.builtin as builtin_tools
    rag = get_rag()
    if rag.doc_count == 0:
        return "知识库为空，请先上传文档或添加文本。"
    results = builtin_tools.rag_search_shared(query.strip(), top_k=int(top_k))
    if not results:
        return "未找到相关内容。"
    parts = []
    for item in results:
        parts.append(f"--- 结果 {item['rank']} (相似度: {item['score']:.4f} | 来源: {item['source']}) ---\n{item['content']}")
    return "\n\n".join(parts)

def rag_clear():
    import tools.builtin as builtin_tools
    builtin_tools.rag_clear_shared()
    return "知识库已清空", rag_get_stats()

def rag_get_stats():
    import tools.builtin as builtin_tools
    stats = builtin_tools.rag_get_shared_stats()
    return f"文档块数: {stats['doc_count']} | 分块大小: {stats['chunk_size']} | 来源: {len(stats['sources'])} 个"


# ---- 3. 多模态视觉（通过 tool_registry 调用） ----
def vision_analyze(image, question, output_type):
    if image is None:
        return "请上传一张图片"
    try:
        import tools.builtin as builtin_tools
        import tools.vision     # noqa: F401  注册 vision_analyze 到 registry
        from tools.registry import registry

        # Phase 4: 使用 MultimodalRouter 进行自动路由
        if output_type == "auto":
            from tools.multimodal_router import get_multimodal_router
            router = get_multimodal_router()
            route = router.route(question or "")
            output_type = route.mode.value
            question = route.prompt
            auto_info = f"[自动路由: {route.mode.value}, 置信度={route.confidence:.0%}]\n"
        else:
            auto_info = ""

        # 构造提示词
        type_prompts = {
            "text": question or "请分析这张图片。",
            "ocr": "请对这张图片进行 OCR 文字识别，提取所有可见文字内容。" + (question or ""),
            "describe": "请详细描述这张图片的内容、构图和主要元素。" + (question or ""),
            "analyze": "请对这张图片进行深度分析，包括内容、风格、潜在含义等。" + (question or ""),
        }
        prompt = type_prompts.get(output_type, question or "请分析这张图片。")

        result = registry.execute("vision_analyze", image_path=image, prompt=prompt)
        if result.get("success"):
            data = result["result"]
            if isinstance(data, dict):
                return auto_info + data.get("description", str(data))
            return auto_info + str(data)
        return f"视觉分析失败: {result.get('error', '未知错误')}"
    except Exception as e:
        return f"视觉分析错误: {e}"


# ---- 4. 图片生成（通过 tool_registry 调用） ----
def image_generate(prompt, size):
    if not prompt or not prompt.strip():
        return None, "请输入提示词"
    try:
        import tools.builtin as builtin_tools  # noqa: F401
        import tools.image_gen  # noqa: F401  注册 image_generate
        from tools.registry import registry

        result = registry.execute("image_generate", prompt=prompt.strip(), size=size, output_dir=str(SCRIPT_DIR / "generated_images"))
        if result.get("success"):
            data = result["result"]
            if isinstance(data, dict):
                file_path = data.get("file_path", "")
                return file_path, f"生成成功! 尺寸: {size} | 模型: {data.get('model_used', '')} | 保存: {Path(file_path).name}"
            return None, str(data)
        return None, f"图片生成失败: {result.get('error', '未知错误')}"
    except Exception as e:
        return None, f"图片生成错误: {e}"


# ---- 5. 多 Agent 协作（多场景 + 工具增强）----
def multi_agent_respond(message, history, scene):
    with _lock:
        try:
            app = get_multi_app(scene)
            result = app.invoke({"messages": [("human", message)]})
            outputs = [msg.content for msg in result["messages"] if msg.type == "ai" and msg.content]
            yield "\n\n---\n\n".join(outputs) if outputs else "无输出"
        except Exception as e:
            yield f"多Agent错误: {e}"


# ---- 6. AOI 检测 (Gradio 接口) ----

def aoi_load_model(model_file):
    """加载 ONNX 模型"""
    if model_file is None:
        return "请选择 .onnx 模型文件"
    path = model_file if isinstance(model_file, str) else model_file.name
    engine = get_aoi_engine()
    with _aoi_lock:
        ok = engine.onnx_detector.load_model(path)
    if ok:
        engine.mode = "deeplearning"
        return f"模型加载成功: {Path(path).name}\n尺寸: {engine.onnx_detector.input_size}\n已切换到深度学习模式"
    return f"模型加载失败，请确认文件为 .onnx 格式且已安装 onnxruntime"

def aoi_unload_model():
    """卸载模型"""
    engine = get_aoi_engine()
    with _aoi_lock:
        engine.onnx_detector.session = None
        engine.onnx_detector.model_path = None
        engine.mode = "traditional"
    return "模型已卸载，已切换到传统算法模式"

def aoi_inspect(image, mode, canny_low, canny_high, clahe_clip, min_area, conf_thresh, iou_thresh):
    """执行 AOI 检测"""
    import cv2
    import numpy as np  # 延迟加载
    if image is None:
        return None, None, "请上传图片"
    if isinstance(image, str):
        img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if img is None:
        return None, None, "图片读取失败"

    engine = get_aoi_engine()
    needs_model = mode in ("deeplearning", "hybrid")
    if needs_model and not engine.onnx_detector.is_loaded:
        return None, None, "当前模式需要加载 ONNX 模型，请先加载模型或切换到传统算法模式"

    with _aoi_lock:
        engine.mode = mode
        processed, result = engine.inspect(
            img, canny_low=canny_low, canny_high=canny_high,
            clahe_clip=clahe_clip, min_area=min_area,
            conf_thresh=conf_thresh, iou_thresh=iou_thresh)
        vis = aoi_visualize(img, result)

    # 生成报告文本
    status = "合格 PASS" if result.pass_flag else "不合格 FAIL"
    report_lines = [
        f"任务编号: {result.task_id}",
        f"检测时间: {result.timestamp}",
        f"图像尺寸: {result.image_size[0]}x{result.image_size[1]}",
        f"处理耗时: {result.total_time_ms:.1f}ms",
        f"检测模式: {result.product_model}",
        f"判定结果: {status}",
        f"缺陷数量: {result.defect_count} (严重: {result.critical_count})",
        "",
    ]
    if result.defects:
        report_lines.append(f"{'ID':<12} {'类型':<10} {'等级':<6} {'置信度':<8} {'方法':<10} {'位置'}")
        report_lines.append("-" * 72)
        for d in result.defects:
            report_lines.append(
                f"{d.defect_id:<12} {d.defect_type.value:<10} {d.severity.value:<6} "
                f"{d.confidence:<8.3f} {d.method.value:<10} ({d.bbox.x},{d.bbox.y},{d.bbox.width},{d.bbox.height})")
    else:
        report_lines.append("未检测到缺陷")
    report = "\n".join(report_lines)

    # 预处理图 BGR→RGB
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)  # type: ignore
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    return processed_rgb, vis_rgb, report


# ---- 检测分析 Tab: Agent 智能分析回调 ----
def aoi_agent_analyze(detect_report, history):
    """将检测结果发送给 ReAct Agent 进行智能分析"""
    if not detect_report or not detect_report.strip():
        yield "请先完成 Step 1 的 PCB 检测"
        return
    prompt = (
        f"你是 PCB 电路板缺陷分析专家。请对以下 AOI 检测报告进行深度分析：\n\n"
        f"=== 检测报告 ===\n{detect_report}\n\n"
        f"请提供以下内容：\n"
        f"1. 缺陷严重性评估（总体判定）\n"
        f"2. 各缺陷可能原因分析\n"
        f"3. 工艺改进建议\n"
        f"4. 检测参数优化建议（如 Canny 阈值、CLAHE、最小面积等）\n"
        f"5. 是否需要重新检测以及推荐的新参数值"
    )
    with _lock:
        try:
            app = get_react_app()
            config = {"configurable": {"thread_id": "demo-aoi-analyze-1"}}
            result = app.invoke({"messages": [("human", prompt)]}, config)
            yield result["messages"][-1].content
        except Exception as e:
            yield f"Agent 分析错误: {e}"


# ---- 7. AOI 智能闭环（多Agent工作流） ----
def aoi_closed_loop_run(image, config_path):
    """
    AOI 智能闭环主入口。
    调用 aoi_workflow.run_aoi_closed_loop() 执行完整的四阶段闭环流程。
    使用 yield 实时输出各阶段日志到 UI。

    注意: 闭环仅支持传统算法模式，因为上位机 XML 参数（CannyLow/High、Clahe 等）
    都是传统视觉算法的阈值，深度学习调优是换模型/调权重，不在 XML 参数范围内。
    """
    if image is None:
        return "请先上传 PCB 图片", "未上传图片，无法启动闭环流程。"

    log_lines = []

    def _log(msg):
        log_lines.append(msg)
        logger.info(f"[AOI闭环] {msg}")

    _log("正在启动 AOI 智能闭环工作流...")
    _log("检测模式: 传统算法（闭环仅支持传统算法，XML 参数为传统视觉阈值）")

    # 闭环固定使用传统算法
    det_mode = "traditional"

    # 配置路径处理
    cfg_path = config_path.strip() if config_path and config_path.strip() else None
    if cfg_path:
        _log(f"XML 配置路径: {cfg_path}")
    else:
        _log("未指定 XML 配置路径，将仅调整检测参数（不写入文件）")

    _log("---")

    with _lock:
        try:
            from aoi.workflow import run_aoi_closed_loop

            _log("Stage 1: 缺陷识别（defect_analyst）...")
            yield "\n".join(log_lines), "执行中..."

            final_state = run_aoi_closed_loop(
                image_path=image,
                detection_mode=det_mode,
                config_path=cfg_path,
            )

            # 解析最终状态，生成报告
            _log("---")
            _log("闭环流程执行完成!")

            initial = final_state.get("initial_result") or {}
            reverify = final_state.get("reverify_result") or {}
            params = final_state.get("recommended_params") or {}
            verdict = final_state.get("final_verdict") or {}
            stage = final_state.get("stage", "completed")

            # 构建详细报告
            report_parts = []
            report_parts.append("=" * 60)
            report_parts.append("AOI 智能闭环 — 最终报告")
            report_parts.append("=" * 60)

            # 初始检测结果
            report_parts.append("\n[1] 初始检测结果")
            if initial:
                report_parts.append(f"  判定: {'合格 PASS' if initial.get('pass') else '不合格 FAIL'}")
                report_parts.append(f"  缺陷总数: {initial.get('total_defects', 'N/A')}")
                report_parts.append(f"  严重缺陷: {initial.get('critical_defects', 'N/A')}")
                report_parts.append(f"  检测耗时: {initial.get('detection_time_ms', 'N/A')}ms")
                if initial.get("defects"):
                    report_parts.append("  缺陷列表:")
                    for d in initial["defects"][:5]:
                        report_parts.append(f"    - {d.get('type')}: {d.get('description', '')} (置信度: {d.get('confidence', 'N/A')})")
                    if len(initial["defects"]) > 5:
                        report_parts.append(f"    ... 共 {len(initial['defects'])} 个")
            else:
                report_parts.append("  无检测结果")

            # RAG案例检索
            rag_cases = final_state.get("rag_cases") or []
            if rag_cases:
                report_parts.append(f"\n[2] RAG 相似案例检索 (命中 {len(rag_cases)} 条)")
                for i, case in enumerate(rag_cases[:3], 1):
                    report_parts.append(f"  案例{i}: {case.get('case_id', 'N/A')} — {case.get('defect_type', '')} | {case.get('product_type', '')}")
                    report_parts.append(f"    调参: {case.get('tuning_rationale', '')[:60]}...")
                    report_parts.append(f"    效果: {case.get('result', '')[:60]}...")
            else:
                report_parts.append("\n[2] RAG 相似案例检索 (未命中)")

            # 参数调整
            if params:
                report_parts.append(f"\n[3] 参数优化方案 (Agent 推荐)")
                for k, v in params.items():
                    report_parts.append(f"  {k}: {v}")
            else:
                report_parts.append(f"\n[3] 参数优化方案 (无需调参)")

            # XML写入结果
            xml_result = final_state.get("xml_write_result")
            if xml_result:
                report_parts.append(f"\n[4] XML 配置改写")
                report_parts.append(f"  备份路径: {xml_result.get('backup_path', 'N/A')}")
                report_parts.append(f"  校验警告: {xml_result.get('validation_warnings', []) or '无'}")

            # 复检验证
            if reverify:
                report_parts.append(f"\n[5] 复检验证结果")
                report_parts.append(f"  判定: {'合格 PASS' if reverify.get('pass') else '不合格 FAIL'}")
                report_parts.append(f"  缺陷总数: {reverify.get('total_defects', 'N/A')}")
                report_parts.append(f"  严重缺陷: {reverify.get('critical_defects', 'N/A')}")
                report_parts.append(f"  检测耗时: {reverify.get('detection_time_ms', 'N/A')}ms")

            # 最终判定
            if verdict:
                report_parts.append(f"\n[6] 最终评估")
                report_parts.append(f"  改善判定: {verdict.get('improvement', 'N/A')}")
                report_parts.append(f"  改善幅度: {verdict.get('improvement_detail', 'N/A')}")
                report_parts.append(f"  风险提示: {verdict.get('risk_warning', 'N/A')}")
                report_parts.append(f"  最终建议: {verdict.get('suggestion', 'N/A')}")

            report_parts.append("\n" + "=" * 60)
            final_report = "\n".join(report_parts)

            _log(f"最终判定: {verdict.get('improvement', '未知')}")
            yield "\n".join(log_lines), final_report

        except Exception as e:
            _log(f"闭环流程异常: {e}")
            import traceback
            _log(traceback.format_exc()[-500:])
            yield "\n".join(log_lines), f"闭环流程异常: {e}"



# ============================================================
# 构建 UI
# ============================================================

def build_ui():
    # Step2: 启动自主进化系统（零侵入，不阻塞 UI）
    try:
        collector = get_feedback_collector()
        from learning.learner import ExperienceLearner
        from learning.optimizer import AdaptiveOptimizer
        from learning.evolution import EvolutionManager

        learner = ExperienceLearner(collector)
        optimizer = AdaptiveOptimizer()
        evo_manager = EvolutionManager(collector, learner, optimizer)
        evo_manager.start_evolution_loop(interval=3600)  # 每小时进化一次
        logger.info("自主进化系统已启动 (间隔: 3600秒)")
    except Exception as e:
        logger.warning(f"自主进化系统启动失败 (不影响主功能): {e}")

    with gr.Blocks(title="AgentClaw v6 Demo") as demo:

        gr.Markdown("""
            # AgentClaw v6 — 统一 Demo
            **八层架构全链路演示** | 检测分析 · 对话助手 · RAG 知识库 · 多模态视觉 · 图片生成 · 多Agent协作 · AOI 独立检测 · AOI 智能闭环
        """)

        with gr.Tabs():

            # ===================== Tab 1: 检测分析（默认首页） =====================
            with gr.Tab("检测分析"):
                # 共享状态：保存检测结果
                detect_state = gr.State({"report": "", "image": None})

                # ---- Step 1: PCB 检测 ----
                gr.Markdown("### Step 1: PCB 检测")
                with gr.Row():
                    with gr.Column(scale=1):
                        detect1_image = gr.Image(type="filepath", label="上传 PCB 图片")
                        with gr.Row():
                            detect1_mode = gr.Radio(
                                ["传统算法", "深度学习", "混合检测"],
                                value="传统算法", label="检测模式",
                                info="深度学习和混合模式需要先加载 ONNX 模型")
                        # 模型管理
                        with gr.Group():
                            gr.Markdown("**ONNX 模型管理**")
                            detect1_model_file = gr.File(label="选择 .onnx 模型", file_types=[".onnx"])
                            with gr.Row():
                                detect1_load_btn = gr.Button("加载模型", variant="primary", size="sm")
                                detect1_unload_btn = gr.Button("卸载模型", variant="stop", size="sm")
                            detect1_model_info = gr.Textbox(label="模型状态", lines=2, interactive=False, value="未加载模型，使用传统算法")
                        with gr.Row():
                            detect1_canny_low = gr.Number(label="Canny 低", value=50, minimum=10, maximum=200)
                            detect1_canny_high = gr.Number(label="Canny 高", value=150, minimum=50, maximum=300)
                        with gr.Row():
                            detect1_clahe = gr.Number(label="CLAHE", value=2.0, minimum=0.5, maximum=5.0, step=0.5)
                            detect1_min_area = gr.Number(label="最小面积", value=50, minimum=10, maximum=500)
                        with gr.Row():
                            detect1_conf = gr.Number(label="置信度", value=0.5, minimum=0.1, maximum=0.99, step=0.05)
                            detect1_iou = gr.Number(label="NMS IoU", value=0.45, minimum=0.1, maximum=0.9, step=0.05)
                        detect1_btn = gr.Button("开始检测", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        with gr.Row():
                            detect1_processed = gr.Image(label="预处理结果", type="numpy")
                            detect1_result = gr.Image(label="检测结果标注", type="numpy")
                        detect1_report = gr.Textbox(label="检测报告", lines=12, interactive=False)

                gr.Markdown("---")
                gr.Markdown("### Step 2: Agent 智能分析")
                gr.Markdown("检测完成后，点击下方按钮让 Agent 对检测报告进行智能分析")
                detect2_btn = gr.Button("Agent 智能分析", variant="primary")
                detect2_result = gr.Textbox(label="Agent 分析结果", lines=15, interactive=False)

                gr.Markdown("---")
                gr.Markdown("### Step 3: 调参验证")
                gr.Markdown("根据 Agent 分析建议，调整参数后重新检测")
                with gr.Row():
                    detect3_canny_low = gr.Number(label="Canny 低", value=50, minimum=10, maximum=200)
                    detect3_canny_high = gr.Number(label="Canny 高", value=150, minimum=50, maximum=300)
                    detect3_clahe = gr.Number(label="CLAHE", value=2.0, minimum=0.5, maximum=5.0, step=0.5)
                    detect3_min_area = gr.Number(label="最小面积", value=50, minimum=10, maximum=500)
                detect3_btn = gr.Button("重新检测", variant="secondary")
                detect3_report = gr.Textbox(label="重新检测结果", lines=10, interactive=False)

                # Step1 事件绑定
                def detect_step1(image, mode, canny_low, canny_high, clahe_clip, min_area, conf_thresh, iou_thresh):
                    """Step1: 执行检测并保存状态"""
                    processed, vis, report = aoi_inspect(
                        image, mode, canny_low, canny_high, clahe_clip, min_area, conf_thresh, iou_thresh)
                    if processed is None:
                        return processed, vis, report, {"report": "", "image": None}
                    return processed, vis, report, {"report": report, "image": image}

                def detect_step3(state, canny_low, canny_high, clahe_clip, min_area):
                    """Step3: 使用调整后的参数重新检测"""
                    image = state.get("image") if state else None
                    if not image:
                        return "请先完成 Step 1 检测"
                    _, _, report = aoi_inspect(
                        image, "传统算法",
                        int(canny_low), int(canny_high), float(clahe_clip), int(min_area),
                        0.5, 0.45)
                    return report or "检测失败"

                detect1_btn.click(
                    detect_step1,
                    [detect1_image, detect1_mode, detect1_canny_low, detect1_canny_high,
                     detect1_clahe, detect1_min_area, detect1_conf, detect1_iou],
                    [detect1_processed, detect1_result, detect1_report, detect_state],
                )
                detect1_load_btn.click(aoi_load_model, [detect1_model_file], [detect1_model_info])
                detect1_unload_btn.click(aoi_unload_model, [], [detect1_model_info])
                detect2_btn.click(
                    aoi_agent_analyze,
                    [detect1_report, gr.State([])],
                    [detect2_result],
                )
                detect3_btn.click(
                    detect_step3,
                    [detect_state, detect3_canny_low, detect3_canny_high,
                     detect3_clahe, detect3_min_area],
                    [detect3_report],
                )

            # ===================== Tab 2: 对话助手（原 ReAct Agent） =====================
            with gr.Tab("对话助手"):
                gr.Markdown("### ReAct 推理 Agent（统一工具入口）\n自动选择 10+ 工具：搜索/计算/文件/命令/代码/系统监控/进程管理/图片分析/图片生成/浏览器")
                gr.ChatInterface(
                    react_respond,
                    examples=[
                        "帮我搜索 2026 年 AI Agent 最新进展",
                        "计算 (35 + 47) * 2 - 100 的结果",
                        "查看当前系统 CPU 和内存使用情况",
                        "读取当前目录下的文件列表",
                    ],
                )

            # ===================== Tab 3: RAG 知识库 =====================
            with gr.Tab("RAG 知识库"):
                gr.Markdown("### RAG 知识库检索\n支持 TXT/Markdown/JSON/CSV，自动切换 ChromaDB / TF-IDF")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**知识库管理**")
                        rag_file = gr.File(label="上传文档 (< 5MB)", file_count="multiple",
                                          file_types=[".txt", ".md", ".json", ".csv"])
                        rag_upload_btn = gr.Button("加载文件", variant="primary")
                        rag_upload_out = gr.Textbox(label="加载结果", lines=3, interactive=False)
                        rag_text = gr.Textbox(label="或直接输入文本", placeholder="粘贴文本内容...", lines=4)
                        rag_text_btn = gr.Button("添加文本", variant="secondary")
                        rag_clear_btn = gr.Button("清空知识库", variant="stop")
                        rag_stats = gr.Textbox(label="知识库状态", value=rag_get_stats(), lines=2, interactive=False)
                    with gr.Column(scale=2):
                        gr.Markdown("**知识检索**")
                        with gr.Row():
                            rag_query = gr.Textbox(label="查询内容", placeholder="输入要检索的内容...", scale=4)
                            rag_topk = gr.Number(label="返回数量", value=3, minimum=1, maximum=20, step=1, scale=1)
                        rag_search_btn = gr.Button("检索", variant="primary")
                        rag_search_out = gr.Textbox(label="检索结果", lines=10, interactive=False)
                rag_upload_btn.click(rag_upload, [rag_file], [rag_upload_out, rag_stats])
                rag_text_btn.click(rag_add_text, [rag_text], [rag_upload_out, rag_stats])
                rag_search_btn.click(rag_search, [rag_query, rag_topk], [rag_search_out])
                rag_clear_btn.click(rag_clear, [], [rag_upload_out, rag_stats])

            # ===================== Tab 4: 多模态视觉 =====================
            with gr.Tab("多模态视觉"):
                gr.Markdown("### VLM 视觉理解\n上传图片，选择分析模式，调用 GLM-4V 视觉模型")
                with gr.Row():
                    with gr.Column(scale=1):
                        vision_image = gr.Image(type="filepath", label="上传图片")
                        vision_type = gr.Radio(
                            ["auto", "text", "ocr", "describe", "analyze"], value="auto", label="分析模式",
                            info="auto=智能路由 | text=自定义问题 | ocr=文字识别 | describe=描述 | analyze=深度分析")
                        vision_q = gr.Textbox(label="自定义问题", placeholder="针对图片提问...", lines=2)
                        vision_btn = gr.Button("开始分析", variant="primary")
                    with gr.Column(scale=2):
                        vision_out = gr.Textbox(label="分析结果", lines=15, interactive=False)
                vision_btn.click(vision_analyze, [vision_image, vision_q, vision_type], [vision_out])

            # ===================== Tab 5: 图片生成 =====================
            with gr.Tab("图片生成"):
                gr.Markdown("### 文生图 — CogView-3-Flash\n输入描述文字，AI 生成图片")
                with gr.Row():
                    with gr.Column(scale=1):
                        gen_prompt = gr.Textbox(label="提示词", placeholder="描述你想要的图片...", lines=4)
                        gen_size = gr.Dropdown(
                            ["1024x1024", "768x1344", "864x1152", "1344x768", "1152x864"],
                            value="1024x1024", label="图片尺寸")
                        gen_btn = gr.Button("生成图片", variant="primary")
                    with gr.Column(scale=2):
                        gen_output = gr.Image(label="生成结果", type="filepath")
                        gen_info = gr.Textbox(label="生成信息", interactive=False)
                gen_btn.click(image_generate, [gen_prompt, gen_size], [gen_output, gen_info])

            # ===================== Tab 6: 多Agent协作 =====================
            with gr.Tab("多Agent协作"):
                gr.Markdown("### 多Agent协作系统\n基于 LangGraph StateGraph，每个Agent角色具备独立工具调用能力（搜索/计算/文件读写/命令执行）")

                ma_scene = gr.Dropdown(
                    choices=[(cfg["title"], key) for key, cfg in MULTI_AGENT_SCENES.items()],
                    value="code_review",
                    label="协作场景",
                    info="不同场景使用不同的Agent角色组合",
                )
                ma_desc = gr.Markdown(MULTI_AGENT_SCENES["code_review"]["desc"])

                def _update_scene_desc(scene):
                    return MULTI_AGENT_SCENES.get(scene, MULTI_AGENT_SCENES["code_review"])["desc"]
                ma_scene.change(_update_scene_desc, [ma_scene], [ma_desc])

                gr.ChatInterface(
                    multi_agent_respond,
                    additional_inputs=[ma_scene],
                    examples=[
                        ["帮我审查这段代码的并发安全性：\n```python\nimport threading\ncounter = 0\ndef increment():\n    global counter\n    for _ in range(100000):\n        counter += 1\nthreads = [threading.Thread(target=increment) for _ in range(10)]\nfor t in threads: t.start()\nfor t in threads: t.join()\nprint(counter)```", "code_review"],
                        ["设计一个支持百万级并发的消息队列系统架构", "tech_design"],
                        ["我的Python服务内存占用持续增长（每24小时约增加500MB），可能是什么原因？", "problem_diagnosis"],
                    ],
                )

            # ===================== Tab 7: AOI 独立检测 =====================
            with gr.Tab("AOI 独立检测"):
                gr.Markdown("### AOI 上位机检测系统\n支持传统算法 / 深度学习(ONNX) / 混合检测，OpenCV 图像处理 + YOLO 推理")

                with gr.Row():
                    # 左栏：参数控制
                    with gr.Column(scale=1):
                        # 检测模式
                        aoi_mode = gr.Radio(
                            ["传统算法", "深度学习", "混合检测"],
                            value="传统算法", label="检测模式",
                            info="深度学习和混合模式需要先加载 ONNX 模型")

                        # 模型管理
                        with gr.Group():
                            gr.Markdown("**ONNX 模型管理**")
                            aoi_model_file = gr.File(label="选择 .onnx 模型", file_types=[".onnx"])
                            with gr.Row():
                                aoi_load_btn = gr.Button("加载模型", variant="primary", size="sm")
                                aoi_unload_btn = gr.Button("卸载模型", variant="stop", size="sm")
                            aoi_model_info = gr.Textbox(label="模型状态", lines=2, interactive=False, value="未加载模型，使用传统算法")

                        # 传统算法参数
                        with gr.Group():
                            gr.Markdown("**传统算法参数**")
                            with gr.Row():
                                aoi_canny_low = gr.Number(label="Canny 低", value=50, minimum=10, maximum=200)
                                aoi_canny_high = gr.Number(label="Canny 高", value=150, minimum=50, maximum=300)
                            with gr.Row():
                                aoi_clahe = gr.Number(label="CLAHE", value=2.0, minimum=0.5, maximum=5.0, step=0.5)
                                aoi_min_area = gr.Number(label="最小面积", value=50, minimum=10, maximum=500)

                        # 深度学习参数
                        with gr.Group():
                            gr.Markdown("**深度学习参数**")
                            with gr.Row():
                                aoi_conf = gr.Number(label="置信度", value=0.5, minimum=0.1, maximum=0.99, step=0.05)
                                aoi_iou = gr.Number(label="NMS IoU", value=0.45, minimum=0.1, maximum=0.9, step=0.05)

                        aoi_btn = gr.Button("开始检测", variant="primary", size="lg")

                    # 右栏：结果展示
                    with gr.Column(scale=2):
                        aoi_input = gr.Image(type="filepath", label="上传检测图片")

                        with gr.Row():
                            aoi_processed = gr.Image(label="预处理结果", type="numpy")
                            aoi_result = gr.Image(label="检测结果标注", type="numpy")

                        aoi_report = gr.Textbox(label="检测报告", lines=12, interactive=False)

                # 事件绑定
                aoi_load_btn.click(aoi_load_model, [aoi_model_file], [aoi_model_info])
                aoi_unload_btn.click(aoi_unload_model, [], [aoi_model_info])
                aoi_btn.click(
                    aoi_inspect,
                    [aoi_input, aoi_mode, aoi_canny_low, aoi_canny_high,
                     aoi_clahe, aoi_min_area, aoi_conf, aoi_iou],
                    [aoi_processed, aoi_result, aoi_report],
                )

            # ===================== Tab 8: AOI 智能闭环 =====================
            with gr.Tab("AOI 智能闭环"):
                gr.Markdown("""
                ### AOI 智能闭环 — 多 Agent 自主调参系统
                **一键触发完整闭环**: 缺陷识别 → RAG案例检索 → 参数推理 → XML改写 → 复检验证
                基于 LangGraph StateGraph 条件分支，4 个专业 Agent 角色协同工作。

                > **仅限传统算法模式**: 上位机 XML 参数（CannyLow/High、Clahe、MinArea 等）
                > 均为传统视觉算法阈值，深度学习调优不在 XML 参数范围内。
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**输入配置**")
                        aoi_cl_image = gr.Image(type="filepath", label="上传 PCB 图片")
                        aoi_cl_config = gr.Textbox(
                            label="XML配置路径（可选）",
                            placeholder="留空则不写入XML，仅调整检测参数",
                            value="",
                            lines=1)
                        aoi_cl_btn = gr.Button(
                            "启动智能闭环", variant="primary", size="lg")
                        gr.Markdown("*检测模式固定为传统算法（传统视觉阈值闭环调优）*")
                        gr.Markdown("*运行过程约 30-60 秒，取决于 LLM 响应速度*")

                    with gr.Column(scale=2):
                        gr.Markdown("**执行日志**")
                        aoi_cl_log = gr.Textbox(
                            label="闭环执行日志", lines=8, interactive=False,
                            placeholder="点击「启动智能闭环」后，这里会实时显示各阶段执行进度...")
                        gr.Markdown("**最终结果**")
                        aoi_cl_result = gr.Textbox(
                            label="闭环结果报告", lines=20, interactive=False)

                # 事件绑定
                aoi_cl_btn.click(
                    aoi_closed_loop_run,
                    [aoi_cl_image, aoi_cl_config],
                    [aoi_cl_log, aoi_cl_result],
                )

        # 底部信息
        gr.Markdown("""
            ---
            **AgentClaw v6** | Python + DeepSeek + LangGraph + Gradio + OpenCV
            | 六层架构: 基础层 -> 工具层 -> 核心层 -> 编排层 -> 服务层 -> 检测层
        """)

    return demo


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=False,
        theme=gr.themes.Soft(),
        css="""
            .tab-nav button { font-size: 15px; padding: 10px 20px; }
            .main-title { text-align: center; margin-bottom: 10px; }
        """,
    )
