"""
LangFuse 适配器 — 与现有 TraceChain 并存的 LLM 可观测性方案

集成方式:
    1. pip install langfuse
    2. 在 config.py 中配置:
       LANGFUSE_PUBLIC_KEY="pk-..."
       LANGFUSE_SECRET_KEY="sk-..."
       LANGFUSE_HOST="https://cloud.langfuse.com"  # 或自建
    3. 在 agent_core.py 中导入并初始化:
       from tools.langfuse_adapter import LangFuseTracer
       tracer = LangFuseTracer()
       handler = tracer.get_callback_handler()  # 传给 langgraph

核心能力:
    - 全链路 Trace: 每次对话生成一个 trace，包含所有 LLM 调用和工具执行
    - 与 LangGraph 原生集成: 通过 CallbackHandler 自动捕获
    - 评分体系: 支持 LLM-as-judge 自动评分
    - 与 TraceChain 对比: 本模块侧重可观测性，TraceChain 侧重内部日志
"""

import os
import time
import uuid
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager


class LangFuseConfig:
    """LangFuse 配置管理，从环境变量读取"""

    def __init__(self):
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.enabled = bool(self.public_key and self.secret_key)

    def validate(self) -> bool:
        """检查配置是否可用"""
        if not self.enabled:
            return False
        # 尝试 import langfuse
        try:
            import langfuse  # noqa: F401
            return True
        except ImportError:
            return False


class LangFuseTracer:
    """
    LangFuse 集成适配器

    提供两种使用模式:
    1. LangGraph Callback 模式 (推荐): 自动捕获所有 LLM 调用和工具执行
    2. 手动 Span 模式: 在自定义代码中手动创建 span 记录

    用法示例:
        # 模式1: LangGraph 自动集成
        tracer = LangFuseTracer()
        handler = tracer.get_callback_handler()
        # 传给 langgraph 的 invoke/stream 方法

        # 模式2: 手动追踪
        with tracer.trace("aoi_workflow", user_id="operator_1") as trace:
            with tracer.span(trace, "detect_defects", input={"image": "sample.jpg"}) as span:
                # ... 执行检测逻辑
                span.update(output={"defects": [...]})
    """

    def __init__(self):
        self._config = LangFuseConfig()
        self._langfuse = None
        self._initialized = False

    def _ensure_initialized(self):
        """延迟初始化，避免 import 时就要求 langfuse 已安装"""
        if self._initialized:
            return
        if not self._config.enabled:
            return
        try:
            from langfuse import Langfuse
            from langfuse.callback import CallbackHandler
            self._langfuse = Langfuse(
                public_key=self._config.public_key,
                secret_key=self._config.secret_key,
                host=self._config.host,
            )
            self._CallbackHandler = CallbackHandler
            self._initialized = True
        except ImportError:
            print("[LangFuse] langfuse 未安装，请运行: pip install langfuse")
        except Exception as e:
            print(f"[LangFuse] 初始化失败: {e}")

    @property
    def enabled(self) -> bool:
        return self._config.enabled and self._initialized

    def get_callback_handler(self, **kwargs):
        """
        获取 LangGraph/LLM 框架的回调处理器

        支持的参数:
            session_id: 会话 ID (用于关联同一会话的多次对话)
            user_id: 用户 ID
            tags: 标签列表 (用于过滤和分组)

        返回:
            CallbackHandler 实例，可直接传给:
            - langgraph 的 invoke() / stream() 的 callbacks 参数
            - ChatOpenAI / ChatAnthropic 的 callbacks 参数
        """
        self._ensure_initialized()
        if not self.enabled:
            return None

        handler = self._CallbackHandler(
            public_key=self._config.public_key,
            secret_key=self._config.secret_key,
            host=self._config.host,
            **kwargs,
        )
        return handler

    @contextmanager
    def trace(self, name: str, user_id: str = "", tags: Optional[List[str]] = None,
              metadata: Optional[Dict[str, Any]] = None):
        """
        手动创建一个 Trace 上下文

        用法:
            with tracer.trace("aoi_workflow", user_id="op1") as trace:
                trace.update(input={...})
                # ... 执行逻辑
                trace.update(output={...})
        """
        self._ensure_initialized()
        if not self.enabled:
            yield _DummyTrace()
            return

        trace = self._langfuse.trace(
            id=str(uuid.uuid4()),
            name=name,
            user_id=user_id,
            tags=tags or [],
            metadata={
                **(metadata or {}),
                "timestamp": datetime.now().isoformat(),
            },
        )
        yield trace
        self._langfuse.flush()

    @contextmanager
    def span(self, trace, name: str, input: Optional[Dict] = None,
             metadata: Optional[Dict] = None):
        """
        在 Trace 内创建一个 Span

        用法:
            with tracer.trace("workflow") as t:
                with tracer.span(t, "step1", input={"x": 1}) as s:
                    # ... 执行逻辑
                    s.update(output={"y": 2})
        """
        self._ensure_initialized()
        if not self.enabled:
            yield _DummyTrace()
            return

        span = trace.span(
            id=str(uuid.uuid4()),
            name=name,
            input=input,
            metadata=metadata,
        )
        start = time.time()
        try:
            yield span
            span.update(
                output={"status": "success"},
                metadata={**(metadata or {}), "duration_ms": round((time.time() - start) * 1000)},
            )
        except Exception as e:
            span.update(
                output={"status": "error", "error": str(e)},
                metadata={**(metadata or {}), "duration_ms": round((time.time() - start) * 1000)},
            )
            raise

    def score_trace(self, trace_id: str, name: str, value: float,
                    comment: str = ""):
        """
        为 Trace 打分 (LLM-as-judge 或人工评分)

        用法:
            tracer.score_trace(trace_id, "task_completion", 1.0, "成功完成调参闭环")
        """
        self._ensure_initialized()
        if not self.enabled:
            return

        self._langfuse.score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
        )

    def generate_langfuse_gen_observation(self, trace, name: str,
                                          model: str, input_data: Any,
                                          output_data: Any, usage: Dict):
        """
        手动记录一个 LLM Generation (用于非 LangGraph 集成的场景)

        用法:
            tracer.generate_observation(
                trace, "deepseek-reasoning",
                model="deepseek-chat",
                input_data={"messages": [...]},
                output_data={"content": "..."},
                usage={"prompt_tokens": 100, "completion_tokens": 50}
            )
        """
        self._ensure_initialized()
        if not self.enabled:
            return

        trace.generation(
            id=str(uuid.uuid4()),
            name=name,
            model=model,
            input=input_data,
            output=output_data,
            usage=usage,
        )


class _DummyTrace:
    """LangFuse 未启用时的空实现，保证代码兼容"""

    def update(self, **kwargs):
        pass

    def span(self, **kwargs):
        return _DummyTrace()

    def generation(self, **kwargs):
        pass


# ========================================
# 与 agent_core.py 的集成示例
# ========================================
"""
在 agent_core.py 中的集成方式:

    from tools.langfuse_adapter import LangFuseTracer

    # 全局初始化
    _langfuse_tracer = LangFuseTracer()

    def build_agent(tools, config):
        # ... 现有的 agent 构建逻辑 ...

        # 获取 LangFuse 回调 (如果启用)
        callbacks = []
        handler = _langfuse_tracer.get_callback_handler(
            session_id=config.session_id,
            user_id=config.user_id,
            tags=["aoi", "agent"],
        )
        if handler:
            callbacks.append(handler)

        # 传给 LangGraph
        result = agent.invoke(
            {"messages": messages},
            config={"callbacks": callbacks},
        )

        return result


# 与 TraceChain 的关系说明:
# - TraceChain: 内部轻量级日志，记录执行链路，用于本地调试和错误排查
# - LangFuse: 外部可观测性平台，提供 UI 仪表盘、团队协作、趋势分析
# - 两者可以并存，互不冲突
# - 推荐: 开发阶段用 TraceChain，部署阶段启用 LangFuse
"""
