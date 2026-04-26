"""
AgentClaw v6 — 统一 Demo UI
集成6大场景: ReAct Agent / RAG知识库 / 多模态视觉 / 图片生成 / 多Agent协作 / AOI检测

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
import re
import json
import threading
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict

# 确保工作目录正确（以脚本所在目录为基准）
SCRIPT_DIR = Path(__file__).parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

# v6: dotenv 加载已移至 core/config.py，不再在此处调用

from core.config import settings
from core.logger import get_logger
logger = get_logger("demo_ui")

import gradio as gr

# Phase 4: TraceChain 状态常量（延迟导入，避免循环依赖）
try:
    from core.trace_chain import SpanStatus as TraceStatus_OK, SpanStatus
    TraceStatus_ERROR = SpanStatus.ERROR
    TraceStatus_OK = SpanStatus.OK
except Exception:
    TraceStatus_OK = None
    TraceStatus_ERROR = None


# ============================================================
# AOI 检测引擎（已提取到独立模块 aoi_engine.py）
# ============================================================
from aoi_engine import (
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
        from feedback_collector import FeedbackCollector
        _feedback_collector = FeedbackCollector(
            persist_dir=str(SCRIPT_DIR / "evolution_data")
        )
        logger.info("FeedbackCollector 已初始化")
    return _feedback_collector

def get_react_app():
    """获取 ReAct Agent（复用 agent_core 统一实例，消除双实例问题）"""
    global _react_app
    if _react_app is None:
        from agent_core import get_react_agent
        # 确保工具已注册
        import builtin_tools  # noqa: F401
        try:
            import vision_tool  # noqa: F401
        except ImportError:
            pass
        try:
            import multimodal_image_gen  # noqa: F401
        except ImportError:
            pass

        # 注入反馈采集器到 RegistryAdapter
        collector = get_feedback_collector()
        from registry_adapter import set_feedback_collector
        set_feedback_collector(collector)

        _react_app = get_react_agent()
        logger.info("ReAct Agent 复用 agent_core 统一实例")
    return _react_app

# --- 2. RAG Engine ---
_rag = None
def get_rag():
    """获取 RAG 引擎（复用 builtin_tools 的共享实例）"""
    import builtin_tools
    return builtin_tools._get_rag_engine()

# --- 3. Multi-Agent (代码审查) ---
_multi_app = None
def get_multi_app():
    global _multi_app
    if _multi_app is None:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage
        from langgraph.graph import StateGraph, START, END, MessagesState
        from feedback_collector import FeedbackSignal

        # Phase 3: 使用共享 LLM 创建（统一与 agent_core）
        llm_ma = ChatOpenAI(
            model=settings.LLM_MODEL, temperature=0,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )

        collector = get_feedback_collector()

        def analyst(state):
            msgs = [SystemMessage(content="你是技术分析师。分析用户问题，给出详细的技术分析和解决方案建议。输出要结构化。")] + state["messages"]
            t0 = time.perf_counter()
            result = llm_ma.invoke(msgs)
            # Phase 4: 收集反馈
            collector.collect(FeedbackSignal(
                task_id=f"ma-analyst-{time.time()}",
                tool_name="multi_agent.analyst",
                success=True,
                latency=round(time.perf_counter() - t0, 4),
            ))
            return {"messages": [result]}

        def programmer(state):
            msgs = [SystemMessage(content="你是高级程序员。根据分析师的建议，编写具体的代码实现或详细技术方案。")] + state["messages"]
            t0 = time.perf_counter()
            result = llm_ma.invoke(msgs)
            collector.collect(FeedbackSignal(
                task_id=f"ma-programmer-{time.time()}",
                tool_name="multi_agent.programmer",
                success=True,
                latency=round(time.perf_counter() - t0, 4),
            ))
            return {"messages": [result]}

        def reviewer(state):
            msgs = [SystemMessage(content="你是技术审核员。审核前面的分析和方案，指出问题并给出最终结论。")] + state["messages"]
            t0 = time.perf_counter()
            result = llm_ma.invoke(msgs)
            collector.collect(FeedbackSignal(
                task_id=f"ma-reviewer-{time.time()}",
                tool_name="multi_agent.reviewer",
                success=True,
                latency=round(time.perf_counter() - t0, 4),
            ))
            return {"messages": [result]}

        builder = StateGraph(MessagesState)
        builder.add_node("analyst", analyst)
        builder.add_node("programmer", programmer)
        builder.add_node("reviewer", reviewer)
        builder.add_edge(START, "analyst")
        builder.add_edge("analyst", "programmer")
        builder.add_edge("programmer", "reviewer")
        builder.add_edge("reviewer", END)
        _multi_app = builder.compile()
        logger.info("多Agent协作已接入反馈采集")
    return _multi_app


# ============================================================
# Phase 4: 全局三链实例（延迟初始化）
# ============================================================

_error_chain = None
_trace_chain = None


def _get_error_chain():
    """获取全局 ErrorChain 实例"""
    global _error_chain
    if _error_chain is None:
        try:
            from core.error_chain import ErrorChain
            _error_chain = ErrorChain()
            # 配置工具降级策略
            _error_chain.configure_tool("web_search",    fallback={"count": 0, "results": [], "source": "fallback", "message": "搜索暂不可用"})
            _error_chain.configure_tool("code_execute",  fallback={"output": "", "error": "执行环境异常"})
            _error_chain.configure_tool("calculator",   fallback={"expression": "0", "result": 0, "type": "int", "error": "计算服务异常"})
            _error_chain.configure_tool("vision_analyze", fallback="视觉分析服务暂不可用，请稍后重试")
            _error_chain.configure_tool("image_generate", fallback="图片生成服务暂不可用，请稍后重试")
            _error_chain.configure_tool("aoi_detect",    fallback={"error": "AOI 检测服务暂不可用"})
            # 注入到 tool_registry
            from tool_registry import registry
            registry.set_error_chain(_error_chain)
            logger.info("ErrorChain 已初始化并注入 tool_registry")
        except Exception as e:
            logger.warning(f"ErrorChain 初始化失败: {e}")
    return _error_chain


def _get_trace_chain():
    """获取全局 TraceChain 实例"""
    global _trace_chain
    if _trace_chain is None:
        try:
            from core.trace_chain import TraceChain
            _trace_chain = TraceChain(
                persist_dir=str(SCRIPT_DIR / "data" / "traces"),
                max_memory=200,
                persist_enabled=True,
                console_enabled=False,
            )
            logger.info("TraceChain 已初始化")
        except Exception as e:
            logger.warning(f"TraceChain 初始化失败: {e}")
    return _trace_chain


# ============================================================
# 场景处理函数
# ============================================================

# ---- 1. ReAct Agent（Phase 4: 接入 TraceChain + ErrorChain） ----
def react_respond(message, history):
    with _lock:
        trace = None
        tc = _get_trace_chain()
        if tc is not None:
            trace = tc.start_trace(request_text=message, session_id="demo-react")

        try:
            from agent_core import get_react_agent
            app = get_react_agent()
            config = {"configurable": {"thread_id": "demo-react-1"}}

            if trace is not None:
                from core.trace_chain import span_context, SpanKind
                with span_context(trace, "agent.react", SpanKind.AGENT) as ctx:
                    result = app.invoke({"messages": [("human", message)]}, config)
                    last_msg = result["messages"][-1]
                    ctx.set_output(last_msg.content[:500])
            else:
                result = app.invoke({"messages": [("human", message)]}, config)
                last_msg = result["messages"][-1]

            yield last_msg.content

            if trace is not None:
                trace.finish(status=TraceStatus_OK, response_text=last_msg.content)
                tc.end_trace(trace)

        except Exception as e:
            if trace is not None:
                trace.finish(status=TraceStatus_ERROR, error=str(e))
                tc.end_trace(trace)

            # Phase 4: ErrorChain 全局兜底
            ec = _get_error_chain()
            if ec is not None:
                safe = ec.handle_global(e, context="react_respond")
                if isinstance(safe, dict) and safe.get("_error"):
                    yield f"处理遇到问题: {safe.get('_message', '请稍后重试')}"
                else:
                    yield str(safe) if safe else f"ReAct Agent 错误: {e}"
            else:
                yield f"ReAct Agent 错误: {e}"


# ---- 2. RAG 知识库 ----
RAG_MAX_FILE_SIZE = 5 * 1024 * 1024

def rag_upload(files):
    if not files:
        return "未选择文件", rag_get_stats()
    import builtin_tools
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
    import builtin_tools
    count = builtin_tools.rag_add_text_to_shared(text.strip(), source="手动输入")
    return f"已添加 {count} 个文档块", rag_get_stats()

def rag_search(query, top_k):
    """RAG 检索（直接返回原始结果，供 Agent 推理模式使用）"""
    if not query or not query.strip():
        return "请输入查询内容"
    import builtin_tools
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


def rag_agent_respond(query, history):
    """Phase 4: RAG + Agent 推理模式

    工作流程:
      1. 从知识库检索相关文档片段
      2. 将检索结果作为上下文发送给 Agent
      3. Agent 基于检索结果进行推理、综合、回答

    相比直接返回检索结果，Agent 推理模式能:
      - 综合多个片段的信息
      - 回答需要推理的复杂问题
      - 处理检索结果与用户问题不完全匹配的情况
    """
    if not query or not query.strip():
        yield "请输入查询内容"
        return

    import builtin_tools
    rag = get_rag()
    if rag.doc_count == 0:
        yield "知识库为空，请先上传文档或添加文本。"
        return

    # Step 1: RAG 检索
    raw_results = builtin_tools.rag_search_shared(query.strip(), top_k=5)
    if not raw_results:
        yield "未找到相关内容。"
        return

    # Step 2: 构建 Agent 提示词（检索结果 + 用户问题）
    context_parts = []
    for item in raw_results:
        context_parts.append(f"[来源: {item['source']}, 相似度: {item['score']:.4f}]\n{item['content']}")
    rag_context = "\n\n".join(context_parts)

    agent_prompt = (
        f"你是知识库助手。请基于以下从知识库中检索到的内容来回答用户问题。\n"
        f"如果检索内容不足以完整回答，请说明缺少什么信息，并基于你的知识给出最佳回答。\n\n"
        f"=== 知识库检索结果 ===\n{rag_context}\n\n"
        f"=== 用户问题 ===\n{query.strip()}\n\n"
        f"请给出准确、专业的回答，并在适当的地方引用来源。"
    )

    # Step 3: Agent 推理
    tc = _get_trace_chain()
    trace = None
    if tc is not None:
        trace = tc.start_trace(request_text=f"[RAG+Agent] {query.strip()}", session_id="demo-rag-agent")

    try:
        app = get_react_app()
        config = {"configurable": {"thread_id": "demo-rag-agent-1"}}

        if trace is not None:
            from core.trace_chain import span_context, SpanKind
            with span_context(trace, "rag.retrieve", SpanKind.RETRIEVE) as ctx:
                ctx.set_output({"matches": len(raw_results)})
            with span_context(trace, "agent.rag_reasoning", SpanKind.AGENT) as ctx:
                result = app.invoke({"messages": [("human", agent_prompt)]}, config)
                last_msg = result["messages"][-1]
                ctx.set_output(last_msg.content[:500])
        else:
            result = app.invoke({"messages": [("human", agent_prompt)]}, config)
            last_msg = result["messages"][-1]

        answer = last_msg.content
        # 追加检索来源信息
        source_info = f"\n\n---\n参考来源 ({len(raw_results)} 条): "
        source_info += ", ".join(set(item['source'] for item in raw_results))
        answer += source_info

        yield answer

        if trace is not None:
            trace.finish(status=TraceStatus_OK, response_text=answer[:500])
            tc.end_trace(trace)

    except Exception as e:
        if trace is not None:
            trace.finish(status=TraceStatus_ERROR, error=str(e))
            tc.end_trace(trace)

        ec = _get_error_chain()
        if ec is not None:
            safe = ec.handle_global(e, context="rag_agent_respond")
            if isinstance(safe, dict) and safe.get("_error"):
                yield f"Agent 推理失败: {safe.get('_message', '请稍后重试')}"
            else:
                yield str(safe) if safe else f"Agent 推理错误: {e}"
        else:
            yield f"Agent 推理错误: {e}"

def rag_clear():
    import builtin_tools
    builtin_tools.rag_clear_shared()
    return "知识库已清空", rag_get_stats()

def rag_get_stats():
    import builtin_tools
    stats = builtin_tools.rag_get_shared_stats()
    return f"文档块数: {stats['doc_count']} | 分块大小: {stats['chunk_size']} | 来源: {len(stats['sources'])} 个"


# ---- 3. 多模态视觉（通过 tool_registry 调用） ----
def vision_analyze(image, question, output_type):
    if image is None:
        return "请上传一张图片"
    try:
        import builtin_tools   # noqa: F401  确保内置工具已注册
        import vision_tool     # noqa: F401  注册 vision_analyze 到 registry
        from tool_registry import registry

        # Phase 4: 使用 MultimodalRouter 进行自动路由
        if output_type == "auto":
            from multimodal_router import get_multimodal_router
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
        import builtin_tools  # noqa: F401
        import multimodal_image_gen  # noqa: F401  注册 image_generate
        from tool_registry import registry

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


# ---- 5. 多 Agent 协作 ----
def multi_agent_respond(message, history):
    """Phase 4: Multi-Agent 接入 TraceChain + ErrorChain + 反馈采集"""
    with _lock:
        tc = _get_trace_chain()
        trace = None
        if tc is not None:
            trace = tc.start_trace(request_text=f"[MultiAgent] {message}", session_id="demo-multi-agent")

        try:
            app = get_multi_app()

            if trace is not None:
                from core.trace_chain import span_context, SpanKind
                with span_context(trace, "multi_agent.pipeline", SpanKind.AGENT, attributes={"agents": "analyst,programmer,reviewer"}) as ctx:
                    result = app.invoke({"messages": [("human", message)]})
                    outputs = [msg.content for msg in result["messages"] if msg.type == "ai" and msg.content]
                    ctx.set_output({"agent_outputs": len(outputs)})
            else:
                result = app.invoke({"messages": [("human", message)]})
                outputs = [msg.content for msg in result["messages"] if msg.type == "ai" and msg.content]

            answer = "\n\n---\n\n".join(outputs) if outputs else "无输出"
            yield answer

            if trace is not None:
                trace.finish(status=TraceStatus_OK, response_text=answer[:500])
                tc.end_trace(trace)

        except Exception as e:
            if trace is not None:
                trace.finish(status=TraceStatus_ERROR, error=str(e))
                tc.end_trace(trace)

            # Phase 4: ErrorChain 兜底
            ec = _get_error_chain()
            if ec is not None:
                safe = ec.handle_global(e, context="multi_agent_respond")
                if isinstance(safe, dict) and safe.get("_error"):
                    yield f"多Agent协作失败: {safe.get('_message', '请稍后重试')}"
                else:
                    yield str(safe) if safe else f"多Agent错误: {e}"
            else:
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

def _parse_agent_params(text: str) -> dict:
    """从 Agent 回复中解析推荐的检测参数"""
    params = {}
    # 匹配 canny_low=80, canny_high=200 等格式（支持中文冒号、等号）
    patterns = {
        'canny_low':    [r'canny[_\s]*低\s*[=:：]\s*(\d+)', r'canny[_\s]*low\s*[=:：]\s*(\d+)'],
        'canny_high':   [r'canny[_\s]*高\s*[=:：]\s*(\d+)', r'canny[_\s]*high\s*[=:：]\s*(\d+)'],
        'clahe':        [r'clahe\s*[=:：]\s*([\d.]+)'],
        'min_area':     [r'最小面积\s*[=:：]\s*(\d+)', r'min[_\s]*area\s*[=:：]\s*(\d+)'],
        'conf_thresh':  [r'conf[_\s]*thresh\s*[=:：]\s*([\d.]+)', r'置信度\s*[=:：]\s*([\d.]+)'],
        'iou_thresh':   [r'iou[_\s]*thresh\s*[=:：]\s*([\d.]+)'],
    }
    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                params[key] = float(m.group(1))
                break
    return params


def aoi_agent_analyze(detect_report, history):
    """将检测结果发送给 ReAct Agent 进行智能分析，自动解析推荐参数回填 Step 3"""
    if not detect_report or not detect_report.strip():
        yield "请先完成 Step 1 的 PCB 检测", 50, 150, 2.0, 50
        return
    prompt = (
        f"你是 PCB 电路板缺陷分析专家。请对以下 AOI 检测报告进行深度分析：\n\n"
        f"=== 检测报告 ===\n{detect_report}\n\n"
        f"请提供以下内容：\n"
        f"1. 缺陷严重性评估（总体判定）\n"
        f"2. 各缺陷可能原因分析\n"
        f"3. 工艺改进建议\n"
        f"4. 检测参数优化建议（如 Canny 阈值、CLAHE、最小面积等）\n"
        f"5. 是否需要重新检测以及推荐的新参数值\n\n"
        f"重要：如果建议调整参数，请以 canny_low=值, canny_high=值, clahe=值, min_area=值 的格式明确写出推荐值。"
    )
    with _lock:
        try:
            app = get_react_app()
            config = {"configurable": {"thread_id": "demo-aoi-analyze-1"}}
            result = app.invoke({"messages": [("human", prompt)]}, config)
            analysis = result["messages"][-1].content

            # 解析 Agent 建议的参数
            parsed = _parse_agent_params(analysis)
            cl = int(parsed.get('canny_low', 50))
            ch = int(parsed.get('canny_high', 150))
            ca = round(parsed.get('clahe', 2.0), 1)
            ma = int(parsed.get('min_area', 50))

            # 如果 Agent 给了新参数，在分析结果末尾追加提示
            if parsed:
                hint = f"\n\n---\n[参数已自动填入 Step 3] canny_low={cl}, canny_high={ch}, clahe={ca}, min_area={ma}"
                yield analysis + hint, cl, ch, ca, ma
            else:
                yield analysis, 50, 150, 2.0, 50
        except Exception as e:
            yield f"Agent 分析错误: {e}", 50, 150, 2.0, 50



# ============================================================
# 构建 UI
# ============================================================

def build_ui():
    # Step2: 启动自主进化系统（零侵入，不阻塞 UI）
    try:
        collector = get_feedback_collector()
        from experience_learner import ExperienceLearner
        from adaptive_optimizer import AdaptiveOptimizer
        from evolution_manage import EvolutionManager

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
            **七层架构全链路演示** | 检测分析 · 对话助手 · RAG 知识库 · 多模态视觉 · 图片生成 · 多Agent协作 · AOI 独立检测
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
                    [detect2_result, detect3_canny_low, detect3_canny_high,
                     detect3_clahe, detect3_min_area],
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
                gr.Markdown("### RAG 知识库检索 + Agent 智能推理\n支持 TXT/Markdown/JSON/CSV，自动切换 ChromaDB / TF-IDF")
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
                        gr.Markdown("**知识检索与 Agent 推理**")
                        with gr.Row():
                            rag_query = gr.Textbox(label="查询内容", placeholder="输入要检索的内容...", scale=4)
                            rag_topk = gr.Number(label="返回数量", value=3, minimum=1, maximum=20, step=1, scale=1)
                        rag_mode = gr.Radio(
                            ["直接检索", "Agent 推理"],
                            value="直接检索", label="检索模式",
                            info="直接检索=返回原始结果 | Agent 推理=Agent 基于检索结果智能回答")
                        with gr.Row():
                            rag_search_btn = gr.Button("检索", variant="primary")
                            rag_agent_btn = gr.Button("Agent 推理", variant="primary")
                        rag_search_out = gr.Textbox(label="结果", lines=15, interactive=False)
                rag_upload_btn.click(rag_upload, [rag_file], [rag_upload_out, rag_stats])
                rag_text_btn.click(rag_add_text, [rag_text], [rag_upload_out, rag_stats])
                rag_search_btn.click(rag_search, [rag_query, rag_topk], [rag_search_out])
                rag_agent_btn.click(rag_agent_respond, [rag_query, gr.State([])], [rag_search_out])
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
                gr.Markdown("### 三角色代码审查工作流\n分析师 -> 程序员 -> 审核员，基于 LangGraph StateGraph")
                gr.ChatInterface(
                    multi_agent_respond,
                    examples=[
                        "帮我设计一个线程安全的连接池",
                        "如何实现一个带超时的HTTP请求重试机制？",
                        "设计一个插件热更新系统",
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

        # 底部信息
        gr.Markdown("""
            ---
            **AgentClaw v6.1.3** | Python + DeepSeek + LangGraph + Gradio + OpenCV
            | 七层架构: 基础层 -> 工具层 -> 核心层 -> 编排层 -> 服务层 -> 检测层 -> 容错层
            | 三链联动: ErrorChain + TraceChain + LLMGuard | 全局统一
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
