"""
═══════════════════════════════════════════════════════════════════════════
  TraceChain - 统一请求追踪/日志链
  AgentClaw v6.1.3
  放置: core/trace_chain.py

  解决的问题:
    日志零散在各个模块，没法回答"这个请求经历了什么"。
    排查问题时要在几十个文件里 grep，拼不出完整链路。

  核心设计:
    一个请求一个 Trace，Trace 下挂多个 Span，每个 Span 是一个阶段。
    所有日志自动关联 trace_id，一条时间线看清全貌。

    Request → Trace (id, meta)
                ├─ Span: agent.think     (LLM 推理)
                ├─ Span: tool.web_search (工具调用)
                ├─ Span: tool.code_exec  (工具调用)
                ├─ Span: agent.respond   (生成回复)
                └─ Result (耗时, token, 状态)

  核心能力:
    1. 自动 trace_id: 每个请求唯一 ID, 所有日志自动携带
    2. Span 嵌套: 父子关系, 看清调用层次
    3. 全链路计时: 每个阶段耗时, 一眼定位瓶颈
    4. 结构化事件: 不再是散乱的 print, 而是结构化记录
    5. 上下文透传: 谁调的、用什么参数、返回了什么
    6. 持久化: JSONL 按日期存储, 可回溯任意历史请求
    7. 实时汇总: 当前请求的实时 timeline, 可嵌入回复
    8. 零依赖: 纯标准库
═══════════════════════════════════════════════════════════════════════════
"""

import time
import json
import os
import uuid
import logging
import threading
import traceback
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══ 1. 事件类型 ═══

class SpanStatus(Enum):
    OK       = "ok"
    ERROR    = "error"
    TIMEOUT  = "timeout"
    SKIP     = "skip"       # 跳过 (如降级)
    FALLBACK = "fallback"   # 降级执行

class SpanKind(Enum):
    AGENT    = "agent"       # Agent 推理/决策
    TOOL     = "tool"        # 工具调用
    LLM      = "llm"         # LLM 调用
    RETRIEVE = "retrieve"    # 检索
    STORE    = "store"       # 存储
    SYSTEM   = "system"      # 系统级 (启动/配置/健康检查)
    EXTERNAL = "external"    # 外部 API
    CUSTOM   = "custom"      # 自定义


# ═══ 2. Span: 单个追踪段 ═══

@dataclass
class Span:
    """
    一个追踪段: 代表请求处理中的一个阶段
    
    示例:
        Span(id="s1", name="tool.web_search", kind=SpanKind.TOOL,
             parent_id="root", start=1714000000, end=1714000001.5,
             status=SpanStatus.OK, input={"query":"..."}, output={"results":[...]})
    """
    id: str = ""
    name: str = ""
    kind: SpanKind = SpanKind.CUSTOM
    status: SpanStatus = SpanStatus.OK
    parent_id: str = "root"

    # 时间
    start_time: float = 0.0
    end_time: float = 0.0

    # 数据
    input_data: Any = None
    output_data: Any = None
    error: str = ""
    error_category: str = ""

    # 补充
    tags: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)  # 阶段内事件

    # token / 计费 (LLM 调用时)
    token_input: int = 0
    token_output: int = 0
    cost_estimate: float = 0.0

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:8]

    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def add_event(self, name: str, data: Any = None, timestamp: float = None):
        """在 Span 内记录事件"""
        self.events.append({
            'name': name,
            'time': timestamp or time.time(),
            'data': self._safe_serialize(data),
        })

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'kind': self.kind.value,
            'status': self.status.value,
            'parent': self.parent_id,
            'start': round(self.start_time, 3),
            'end': round(self.end_time, 3),
            'duration_ms': round(self.duration_ms, 1),
            'input': self._safe_serialize(self.input_data),
            'output': self._safe_serialize(self.output_data),
            'error': self.error[:200] if self.error else '',
            'error_category': self.error_category,
            'tags': self.tags,
            'attrs': {k: self._safe_serialize(v) for k, v in self.attributes.items()},
            'events': self.events,
            'tokens': {'in': self.token_input, 'out': self.token_output},
            'cost': round(self.cost_estimate, 6),
        }

    @staticmethod
    def _safe_serialize(val, max_len=500):
        """安全序列化, 截断大内容"""
        if val is None:
            return None
        if isinstance(val, (str, int, float, bool)):
            return str(val)[:max_len] if isinstance(val, str) else val
        if isinstance(val, (dict, list)):
            s = json.dumps(val, ensure_ascii=False, default=str)
            return s[:max_len] if len(s) > max_len else s
        return str(val)[:max_len]


# ═══ 3. Trace: 完整请求追踪 ═══

class Trace:
    """
    一个请求的完整追踪记录
    
    包含: 请求元信息 + 所有 Span + 最终结果
    """

    def __init__(self, trace_id: str = None, name: str = "request",
                 user_id: str = "", session_id: str = ""):
        self.id = trace_id or uuid.uuid4().hex[:16]
        self.name = name
        self.user_id = user_id
        self.session_id = session_id

        self.spans: List[Span] = []
        self.span_index: Dict[str, Span] = {}  # id → Span

        self.start_time = time.time()
        self.end_time = 0.0

        # 请求/响应
        self.request_text = ""
        self.response_text = ""
        self.final_status = SpanStatus.OK
        self.final_error = ""

        # 汇总
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_cost = 0.0

        # 线程安全
        self._lock = threading.Lock()

    # ── 创建 Span ──

    def start_span(self, name: str, kind: SpanKind = SpanKind.CUSTOM,
                   parent_id: str = "root", tags: List[str] = None,
                   attributes: Dict = None) -> Span:
        """开始一个新 Span"""
        span = Span(
            name=name,
            kind=kind,
            parent_id=parent_id,
            tags=tags or [],
            attributes=attributes or {},
            start_time=time.time(),
        )
        with self._lock:
            self.spans.append(span)
            self.span_index[span.id] = span
        return span

    def end_span(self, span_id: str, status: SpanStatus = SpanStatus.OK,
                 output_data: Any = None, error: str = "",
                 error_category: str = "", tokens_in: int = 0,
                 tokens_out: int = 0, cost: float = 0.0):
        """结束一个 Span"""
        span = self.span_index.get(span_id)
        if not span:
            return
        span.end_time = time.time()
        span.status = status
        span.output_data = output_data
        span.error = error
        span.error_category = error_category
        span.token_input = tokens_in
        span.token_output = tokens_out
        span.cost_estimate = cost
        with self._lock:
            self.total_tokens_in += tokens_in
            self.total_tokens_out += tokens_out
            self.total_cost += cost

    # ── 结束 Trace ──

    def finish(self, status: SpanStatus = SpanStatus.OK,
               response_text: str = "", error: str = ""):
        """结束整个请求追踪"""
        self.end_time = time.time()
        self.final_status = status
        self.response_text = response_text
        self.final_error = error
        # 关闭所有未结束的 Span
        for span in self.spans:
            if not span.end_time:
                span.end_time = self.end_time
                span.status = SpanStatus.ERROR

    # ── 查询 ──

    def get_span(self, span_id: str) -> Optional[Span]:
        return self.span_index.get(span_id)

    def get_spans_by_kind(self, kind: SpanKind) -> List[Span]:
        return [s for s in self.spans if s.kind == kind]

    def get_errors(self) -> List[Span]:
        return [s for s in self.spans if s.status in (SpanStatus.ERROR, SpanStatus.TIMEOUT)]

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    @property
    def span_count(self) -> int:
        return len(self.spans)

    # ── 输出 ──

    def to_dict(self) -> dict:
        return {
            'trace_id': self.id,
            'name': self.name,
            'user': self.user_id,
            'session': self.session_id,
            'status': self.final_status.value,
            'error': self.final_error[:200] if self.final_error else '',
            'start': round(self.start_time, 3),
            'end': round(self.end_time, 3),
            'duration_ms': round(self.duration_ms, 1),
            'request': self.request_text[:500],
            'response': self.response_text[:500],
            'spans': [s.to_dict() for s in self.spans],
            'summary': {
                'span_count': len(self.spans),
                'error_count': len(self.get_errors()),
                'tokens': {'in': self.total_tokens_in, 'out': self.total_tokens_out},
                'total_cost': round(self.total_cost, 6),
            },
        }

    def timeline(self) -> str:
        """
        生成人类可读的时间线
        
        示例输出:
            [00ms] ⏱ START  request "帮我搜索xxx"
            [02ms] → agent.think       (推理中)
            [850ms] ✓ agent.think       848ms  tokens:1200/350
            [852ms] → tool.web_search   query="xxx"
            [1230ms] ✓ tool.web_search  378ms  results:5
            [1235ms] → agent.respond    (生成回复)
            [2100ms] ✓ agent.respond    865ms  tokens:800/220
            [2100ms] ⏱ END    2100ms  OK  total_tokens:2000/570
        """
        lines = []
        t0 = self.start_time
        lines.append(f"[{'00':>6}ms] START  {self.name} \"{self.request_text[:60]}\"")
        lines.append(f"         trace_id={self.id}")

        for span in self.spans:
            offset = (span.start_time - t0) * 1000
            dur = span.duration_ms

            # 状态图标
            icon = {'ok': '✓', 'error': '✗', 'timeout': '⏰',
                    'skip': '⊘', 'fallback': '↩'}.get(span.status.value, '?')

            # 信息行
            info = f"{span.name}"
            if span.status.value == 'ok':
                info += f"  {dur:.0f}ms"
            elif span.status.value == 'error':
                info += f"  {dur:.0f}ms  ERROR: {span.error[:60]}"
            elif span.status.value == 'fallback':
                info += f"  {dur:.0f}ms  FALLBACK"

            # tokens
            if span.token_input or span.token_output:
                info += f"  tokens:{span.token_input}/{span.token_output}"

            lines.append(f"[{offset:>6}ms] {icon} {info}")

        total = self.duration_ms
        status_icon = '✓' if self.final_status == SpanStatus.OK else '✗'
        lines.append(
            f"[{total:>6}ms] {status_icon} END    {total:.0f}ms  "
            f"{self.final_status.value}  "
            f"tokens:{self.total_tokens_in}/{self.total_tokens_out}  "
            f"spans:{self.span_count}  errors:{len(self.get_errors())}"
        )

        return '\n'.join(lines)


# ═══ 4. TraceChain: 全局追踪管理器 ═══

class TraceChain:
    """
    全局追踪管理器
    
    用法:
        chain = TraceChain()
        
        # 开始追踪一个请求
        trace = chain.start_trace("帮我搜索xxx", user_id="u123")
        
        # 记录各阶段
        with trace.span("agent.think", SpanKind.AGENT):
            response = llm.chat(...)
            trace.current_span.tokens = 1200
        
        with trace.span("tool.web_search", SpanKind.TOOL):
            results = search(...)
        
        # 结束
        trace.finish(response_text="搜索结果是...", status=SpanStatus.OK)
        chain.end_trace(trace)
        
        # 查看
        print(trace.timeline())
    """

    def __init__(self, persist_dir: str = 'data/traces', max_memory: int = 200,
                 persist_enabled: bool = True, console_enabled: bool = True):
        """
        Args:
            persist_dir:     追踪文件存储目录
            max_memory:      内存中保留的最大 trace 数
            persist_enabled: 是否持久化到文件
            console_enabled: 是否输出到控制台
        """
        self._persist_dir = persist_dir
        self._max_memory = max_memory
        self._persist_enabled = persist_enabled
        self._console_enabled = console_enabled

        self._traces: Dict[str, Trace] = {}
        self._all_traces: List[Trace] = []
        self._lock = threading.Lock()

        # 当前线程的 active trace (用于跨函数传递)
        self._active = threading.local()

        # 统计
        self._stats = {
            'total_requests': 0,
            'total_errors': 0,
            'total_tokens_in': 0,
            'total_tokens_out': 0,
            'avg_duration_ms': 0,
        }

        if persist_enabled:
            os.makedirs(persist_dir, exist_ok=True)

    # ── 生命周期 ──

    def start_trace(self, request_text: str = "", name: str = "request",
                    user_id: str = "", session_id: str = "") -> Trace:
        """开始一个新的请求追踪"""
        trace = Trace(name=name, user_id=user_id, session_id=session_id)
        trace.request_text = request_text

        with self._lock:
            self._traces[trace.id] = trace
            self._all_traces.append(trace)
            self._stats['total_requests'] += 1
            # 内存淘汰
            if len(self._all_traces) > self._max_memory:
                removed = self._all_traces[:len(self._all_traces) - self._max_memory]
                self._all_traces = self._all_traces[-self._max_memory:]
                for r in removed:
                    self._traces.pop(r.id, None)

        # 设为当前线程的 active trace
        self._active.current = trace

        logger.info(f"[Trace] START trace={trace.id} request=\"{request_text[:80]}\"")
        return trace

    def end_trace(self, trace: Trace):
        """结束追踪并持久化"""
        trace.finish()
        with self._lock:
            if trace.final_status != SpanStatus.OK:
                self._stats['total_errors'] += 1
            self._stats['total_tokens_in'] += trace.total_tokens_in
            self._stats['total_tokens_out'] += trace.total_tokens_out
            n = self._stats['total_requests'] or 1
            self._stats['avg_duration_ms'] = (
                self._stats['avg_duration_ms'] * (n - 1) + trace.duration_ms
            ) / n

        # 输出时间线
        if self._console_enabled:
            tl = trace.timeline()
            if trace.final_status != SpanStatus.OK:
                logger.warning(f"\n{tl}")
            else:
                logger.info(f"\n{tl}")

        # 持久化
        if self._persist_enabled:
            self._persist(trace)

        # 清理 active
        if getattr(self._active, 'current', None) is trace:
            self._active.current = None

        logger.info(
            f"[Trace] END trace={trace.id} status={trace.final_status.value} "
            f"duration={trace.duration_ms:.0f}ms spans={trace.span_count} "
            f"errors={len(trace.get_errors())}"
        )

    def get_active_trace(self) -> Optional[Trace]:
        """获取当前线程的活跃 trace"""
        return getattr(self._active, 'current', None)

    # ── 查询 ──

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """按 ID 查找 trace"""
        return self._traces.get(trace_id)

    def get_recent_traces(self, limit: int = 10) -> List[Dict]:
        """获取最近的 trace 列表"""
        with self._lock:
            recent = self._all_traces[-limit:]
        return [t.to_dict() for t in recent]

    def get_error_traces(self, limit: int = 10) -> List[Dict]:
        """获取最近的错误 trace"""
        with self._lock:
            errors = [t for t in self._all_traces
                      if t.final_status != SpanStatus.OK][-limit:]
        return [t.to_dict() for t in errors]

    def get_stats(self) -> dict:
        return {**self._stats, 'in_memory': len(self._all_traces)}

    def search_traces(self, keyword: str, limit: int = 20) -> List[Dict]:
        """按关键词搜索历史 trace"""
        keyword = keyword.lower()
        results = []
        with self._lock:
            for trace in reversed(self._all_traces):
                if keyword in trace.request_text.lower() or keyword in trace.response_text.lower():
                    results.append(trace.to_dict())
                    if len(results) >= limit:
                        break
        return results

    # ── 持久化 ──

    def _persist(self, trace: Trace):
        """追加写入 JSONL (按日期分文件)"""
        try:
            date_str = datetime.now().strftime('%Y-%m-%d')
            filepath = os.path.join(self._persist_dir, f"traces_{date_str}.jsonl")
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + '\n')
        except IOError as e:
            logger.error(f"[Trace] 持久化失败: {e}")


# ═══ 5. Trace Span 上下文管理器 ═══

class span_context:
    """
    Span 上下文管理器 (配合 Trace 使用)
    
    用法:
        trace = chain.start_trace("搜索xxx")
        
        with span_context(trace, "agent.think", SpanKind.AGENT) as ctx:
            response = llm.chat(...)
            ctx.set_tokens(1200, 350)
        
        with span_context(trace, "tool.search", SpanKind.TOOL) as ctx:
            ctx.set_input({"query": "xxx"})
            results = search(...)
            ctx.set_output(results)
    
    自动处理: start_span / end_span / 错误捕获
    """

    def __init__(self, trace: Trace, name: str, kind: SpanKind = SpanKind.CUSTOM,
                 parent_id: str = "root", tags: List[str] = None,
                 attributes: Dict = None):
        self._trace = trace
        self._name = name
        self._kind = kind
        self._parent_id = parent_id
        self._tags = tags
        self._attributes = attributes
        self._span: Optional[Span] = None

    def __enter__(self):
        self._span = self._trace.start_span(
            name=self._name, kind=self._kind,
            parent_id=self._parent_id, tags=self._tags,
            attributes=self._attributes,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._trace.end_span(
                self._span.id,
                status=SpanStatus.ERROR,
                error=str(exc_val) or exc_type.__name__,
                error_category=getattr(exc_val, 'category', 'unknown') if exc_val else '',
            )
        else:
            self._trace.end_span(self._span.id)
        return False  # 不吞异常

    @property
    def span(self) -> Span:
        return self._span

    def set_input(self, data):
        if self._span:
            self._span.input_data = data

    def set_output(self, data):
        if self._span:
            self._span.output_data = data

    def set_tokens(self, inp: int, out: int, cost: float = 0.0):
        if self._span:
            self._span.token_input = inp
            self._span.token_output = out
            self._span.cost_estimate = cost

    def set_tags(self, *tags):
        if self._span:
            self._span.tags.extend(tags)

    def add_event(self, name: str, data=None):
        if self._span:
            self._span.add_event(name, data)


# ═══ 6. 自动追踪装饰器 ═══

class TraceDecorator:
    """
    为函数/方法自动添加追踪
    
    用法:
        td = TraceDecorator(chain)
        
        # 自动追踪 LLM 调用
        @td.trace("llm.chat", SpanKind.LLM)
        def chat_completion(messages):
            ...
        
        # 自动追踪工具调用
        @td.trace("tool.search", SpanKind.TOOL)
        def web_search(query):
            ...
        
        # 追踪 + 自动捕获异常
        @td.trace_safe("tool.db", SpanKind.TOOL, fallback=[])
        def query_db(sql):
            ...
    """

    def __init__(self, chain: TraceChain):
        self._chain = chain

    def trace(self, span_name: str, kind: SpanKind = SpanKind.CUSTOM,
              tags: List[str] = None):
        """普通追踪装饰器"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                trace = self._chain.get_active_trace()
                if not trace:
                    return func(*args, **kwargs)

                with span_context(trace, span_name, kind, tags=tags) as ctx:
                    ctx.set_input(kwargs if kwargs else (str(args)[:200] if args else None))
                    result = func(*args, **kwargs)
                    ctx.set_output(result)
                    return result
            return wrapper
        return decorator

    def trace_safe(self, span_name: str, kind: SpanKind = SpanKind.TOOL,
                   fallback=None):
        """追踪 + 异常安全装饰器 (不抛异常)"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                trace = self._chain.get_active_trace()
                if not trace:
                    try:
                        return func(*args, **kwargs)
                    except Exception:
                        return fallback

                with span_context(trace, span_name, kind) as ctx:
                    ctx.set_input(kwargs if kwargs else (str(args)[:200] if args else None))
                    try:
                        result = func(*args, **kwargs)
                        ctx.set_output(result)
                        return result
                    except Exception as e:
                        ctx.set_output(fallback)
                        return fallback
            return wrapper
        return decorator


import functools


# ═══ 7. 兼容 logging 的 TraceHandler ═══

class TraceLogHandler(logging.Handler):
    """
    将 logging 日志自动关联到当前 trace
    
    所有 logger.info/warning/error 的日志都会带上 trace_id，
    输出格式: [trace_id] [level] message
    
    用法:
        handler = TraceLogHandler(chain)
        handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(handler)
    """

    def __init__(self, chain: TraceChain):
        super().__init__()
        self._chain = chain

    def emit(self, record):
        trace = self._chain.get_active_trace()
        trace_id = trace.id if trace else "--------"
        record.trace_id = trace_id  # type: ignore
        self._write(record)

    def _write(self, record):
        ts = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        tid = getattr(record, 'trace_id', '--------')
        msg = self.format(record)
        line = f"[{tid}] [{ts}] [{record.levelname}] {msg}"

        # 写入当前 trace 的 span 事件 (如果有)
        trace = self._chain.get_active_trace()
        if trace and trace.spans:
            current_span = trace.spans[-1]
            if not current_span.end_time:
                current_span.add_event(f"log.{record.levelname}", msg[:200])


# ═══ 8. 主动式日志 API ═══

class TracedLogger:
    """
    主动式追踪日志 (比 print/logging 更适合 Agent 场景)
    
    自动关联当前 trace，所有日志都是结构化事件。
    
    用法:
        tlog = TracedLogger(chain)
        tlog.info("开始处理用户输入")           # 自动关联 trace
        tlog.tool("web_search", input={"query":"xxx"}, output={"results":5})
        tlog.error("搜索失败", error=e, category="network")
        tlog.metric("tokens_used", value=1200)
    """

    def __init__(self, chain: TraceChain):
        self._chain = chain

    def _current_span(self) -> Optional[Span]:
        trace = self._chain.get_active_trace()
        if trace and trace.spans:
            # 找最后一个未关闭的 span
            for s in reversed(trace.spans):
                if not s.end_time:
                    return s
        return None

    def info(self, message: str, **attrs):
        span = self._current_span()
        if span:
            span.add_event("info", {"msg": message, **attrs})
        logger.info(f"[TraceLog] {message}")

    def warning(self, message: str, **attrs):
        span = self._current_span()
        if span:
            span.add_event("warn", {"msg": message, **attrs})
        logger.warning(f"[TraceLog] {message}")

    def error(self, message: str, error: Exception = None, category: str = ""):
        span = self._current_span()
        data = {'msg': message}
        if error:
            data['error'] = str(error)[:200]
            data['type'] = type(error).__name__
        if category:
            data['category'] = category
        if span:
            span.add_event("error", data)
        logger.error(f"[TraceLog] {message}")

    def tool(self, tool_name: str, input_data=None, output_data=None,
             duration_ms: float = 0, status: str = "ok"):
        """记录工具调用"""
        trace = self._chain.get_active_trace()
        if not trace:
            return
        trace.start_span(
            name=f"tool.{tool_name}",
            kind=SpanKind.TOOL,
            input_data=input_data,
        )
        # 立即结束 (快速记录模式)
        span = trace.spans[-1]
        span.end_time = span.start_time + (duration_ms / 1000)
        span.output_data = output_data
        span.status = SpanStatus.ERROR if status != "ok" else SpanStatus.OK
        if status == "fallback":
            span.status = SpanStatus.FALLBACK
        trace.total_tokens_in += 0  # 占位, 不影响计数

    def metric(self, name: str, value, unit: str = ""):
        """记录指标"""
        span = self._current_span()
        if span:
            span.attributes[f"metric.{name}"] = f"{value}{unit}"


# ═══ 9. 请求汇总 (给用户看的) ═══

def format_trace_summary(trace: Trace) -> str:
    """
    生成用户友好的请求处理摘要
    
    示例:
        处理完成 (2.1秒)
        → 推理: 0.8秒 (使用 1550 tokens)
        → 搜索: 0.4秒 (返回 5 条结果)
        → 生成回复: 0.9秒 (使用 1020 tokens)
    """
    lines = [f"处理完成 ({trace.duration_ms/1000:.1f}秒)"]

    kind_names = {
        SpanKind.AGENT: "推理",
        SpanKind.LLM: "模型调用",
        SpanKind.TOOL: "工具",
        SpanKind.RETRIEVE: "检索",
        SpanKind.STORE: "存储",
    }

    grouped = defaultdict(list)
    for span in trace.spans:
        grouped[span.kind].append(span)

    for kind, spans in grouped.items():
        label = kind_names.get(kind, kind.value)
        total_dur = sum(s.duration_ms for s in spans)
        total_tokens = sum(s.token_input + s.token_output for s in spans)

        parts = [f"{label}: {total_dur/1000:.1f}秒"]
        if total_tokens:
            parts.append(f"使用 {total_tokens} tokens")
        if len(spans) > 1:
            parts.append(f"{len(spans)}次调用")

        icon = "✓" if all(s.status == SpanStatus.OK for s in spans) else "⚠"
        lines.append(f"  {icon} {' | '.join(parts)}")

    return '\n'.join(lines)


# ═══ 10. 主程序集成 ═══
"""
═══════════════════════════════════════════════════════════════════════
  集成到 AgentClaw 主程序
═══════════════════════════════════════════════════════════════════════

─── 方案A: Agent 主循环集成 (推荐) ───────────────────────────────

  from core.trace_chain import TraceChain, span_context, SpanKind, SpanStatus
  from core.trace_chain import format_trace_summary, TracedLogger, TraceLogHandler

  class AgentClaw:
      def __init__(self):
          # 初始化追踪链
          self.trace_chain = TraceChain(persist_dir='data/traces')
          self.tlog = TracedLogger(self.trace_chain)
          
          # 让 logging 也带上 trace_id
          handler = TraceLogHandler(self.trace_chain)
          logging.getLogger().addHandler(handler)
      
      async def process(self, user_input, user_id=""):
          # 1. 开始追踪
          trace = self.trace_chain.start_trace(
              request_text=user_input, user_id=user_id
          )
          
          try:
              # 2. Agent 推理阶段
              with span_context(trace, "agent.think", SpanKind.AGENT) as ctx:
                  plan = await self.plan(user_input)
                  ctx.set_output(plan)
                  self.tlog.info(f"生成执行计划: {len(plan)} 步")
              
              # 3. 工具调用阶段 (每个工具自动追踪)
              for step in plan:
                  with span_context(trace, f"tool.{step.tool}", SpanKind.TOOL) as ctx:
                      ctx.set_input(step.args)
                      try:
                          result = await self.execute_tool(step)
                          ctx.set_output(result)
                      except Exception as e:
                          ctx.set_output({"error": str(e)})
                          self.tlog.error(f"{step.tool} 失败", error=e)
              
              # 4. 生成回复
              with span_context(trace, "agent.respond", SpanKind.LLM) as ctx:
                  response = await self.generate_response(user_input)
                  ctx.set_output(response)
              
              # 5. 结束追踪
              trace.finish(
                  status=SpanStatus.OK,
                  response_text=response,
              )
              self.trace_chain.end_trace(trace)
              
              return response
          
          except Exception as e:
              trace.finish(status=SpanStatus.ERROR, error=str(e))
              self.trace_chain.end_trace(trace)
              raise


─── 方案B: 装饰器模式 (最小改动) ─────────────────────────────────

  chain = TraceChain()
  td = TraceDecorator(chain)
  
  @td.trace("llm.chat", SpanKind.LLM)
  async def call_llm(messages):
      return await client.chat(messages)
  
  @td.trace_safe("tool.search", SpanKind.TOOL, fallback={"results": []})
  async def web_search(query):
      return await requests.get(...)


─── 方案C: 事后排查 ──────────────────────────────────────────────

  # 查看最近 10 个请求
  recent = chain.get_recent_traces(10)
  for t in recent:
      print(f"[{t['trace_id']}] {t['status']} {t['duration_ms']}ms {t['request'][:40]}")
  
  # 查看某个请求的完整时间线
  trace = chain.get_trace("abc123def456")
  print(trace.timeline())
  
  # 搜索历史请求
  results = chain.search_traces("搜索失败")
  
  # 查看统计数据
  stats = chain.get_stats()
  print(f"总请求: {stats['total_requests']}, 错误率: {stats['total_errors']/stats['total_requests']:.1%}")
  
  # 持久化的 trace 文件在 data/traces/traces_2025-01-15.jsonl
  # 可用 jq / grep 等工具分析
"""
