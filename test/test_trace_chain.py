# ============================================================
# AgentClaw — TraceChain 单元测试
# ============================================================

import os
import sys
import time
import json
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSpan:
    """Span 基本功能"""

    def test_span_creation(self):
        from core.trace_chain import Span, SpanKind, SpanStatus
        span = Span(name="test_span", kind=SpanKind.TOOL, start_time=time.time())
        assert span.name == "test_span"
        assert span.kind == SpanKind.TOOL
        assert span.status == SpanStatus.OK
        assert span.id is not None

    def test_span_duration(self):
        from core.trace_chain import Span
        t = time.time()
        span = Span(name="s", start_time=t, end_time=t + 0.01)
        assert span.duration_ms >= 9.0

    def test_span_add_event(self):
        from core.trace_chain import Span
        span = Span(name="s")
        span.add_event("test_event", {"key": "value"})
        assert len(span.events) == 1
        assert span.events[0]["name"] == "test_event"

    def test_span_to_dict(self):
        from core.trace_chain import Span, SpanKind
        span = Span(name="dict_test", kind=SpanKind.LLM,
                    start_time=time.time(), end_time=time.time())
        d = span.to_dict()
        assert d["name"] == "dict_test"
        assert d["kind"] == SpanKind.LLM.value
        assert "duration_ms" in d


class TestTrace:
    """Trace 生命周期"""

    def test_trace_creation(self):
        from core.trace_chain import Trace, SpanStatus
        trace = Trace(trace_id="test-123", name="test_request")
        assert trace.id == "test-123"
        assert trace.name == "test_request"
        assert trace.final_status == SpanStatus.OK
        assert len(trace.spans) == 0

    def test_start_end_span(self):
        from core.trace_chain import Trace, SpanKind, SpanStatus
        trace = Trace(trace_id="t1")
        span = trace.start_span("step1", kind=SpanKind.AGENT)
        assert span.id in trace.span_index

        trace.end_span(span.id, status=SpanStatus.OK)
        assert span.status == SpanStatus.OK
        assert span.end_time > 0

    def test_finish(self):
        from core.trace_chain import Trace, SpanStatus
        trace = Trace(trace_id="t1")
        trace.start_span("s1")
        time.sleep(0.01)
        trace.finish(status=SpanStatus.OK)
        assert trace.final_status == SpanStatus.OK
        assert trace.duration_ms > 0

    def test_get_errors(self):
        from core.trace_chain import Trace, SpanKind, SpanStatus
        trace = Trace(trace_id="t1")
        ok = trace.start_span("ok", kind=SpanKind.TOOL)
        err = trace.start_span("err", kind=SpanKind.LLM)
        trace.end_span(ok.id, status=SpanStatus.OK)
        trace.end_span(err.id, status=SpanStatus.ERROR)
        errors = trace.get_errors()
        assert len(errors) == 1
        assert errors[0].name == "err"

    def test_timeline(self):
        from core.trace_chain import Trace, SpanKind
        trace = Trace(trace_id="t1")
        s1 = trace.start_span("step1", kind=SpanKind.AGENT)
        trace.end_span(s1.id)
        tl = trace.timeline()
        assert "step1" in tl

    def test_to_dict(self):
        from core.trace_chain import Trace, SpanKind
        trace = Trace(trace_id="t1", name="test")
        s1 = trace.start_span("s1", kind=SpanKind.TOOL)
        trace.end_span(s1.id)
        trace.finish()
        d = trace.to_dict()
        assert d["trace_id"] == "t1"
        assert d["name"] == "test"
        assert len(d["spans"]) == 1


class TestTraceChain:
    """TraceChain 完整功能"""

    def test_init_defaults(self):
        from core.trace_chain import TraceChain
        chain = TraceChain(persist_enabled=False)
        assert chain._max_memory == 200
        assert chain._persist_enabled is False

    def test_start_trace(self):
        from core.trace_chain import TraceChain
        chain = TraceChain(persist_enabled=False)
        trace = chain.start_trace(request_text="hello", session_id="s1")
        assert trace is not None
        assert trace.name == "request"
        assert trace.request_text == "hello"
        assert chain.get_active_trace() is trace

    def test_end_trace(self):
        from core.trace_chain import TraceChain, SpanStatus
        chain = TraceChain(persist_enabled=False)
        trace = chain.start_trace(request_text="test")
        trace.start_span("step1")
        chain.end_trace(trace)
        assert trace.final_status is not None
        assert chain.get_active_trace() is None

    def test_end_trace_updates_stats(self):
        from core.trace_chain import TraceChain, SpanStatus
        chain = TraceChain(persist_enabled=False)
        trace = chain.start_trace(request_text="stats_test")
        trace.total_tokens_in = 50
        trace.total_tokens_out = 100
        chain.end_trace(trace)
        stats = chain.get_stats()
        assert stats["total_requests"] >= 1

    def test_persist(self):
        from core.trace_chain import TraceChain
        with tempfile.TemporaryDirectory() as tmp:
            chain = TraceChain(persist_dir=tmp, persist_enabled=True, console_enabled=False)
            trace = chain.start_trace(request_text="persist_test")
            chain.end_trace(trace)
            files = list(Path(tmp).glob("*.jsonl"))
            assert len(files) >= 1

    def test_get_recent_traces(self):
        from core.trace_chain import TraceChain
        chain = TraceChain(persist_enabled=False)
        t1 = chain.start_trace(request_text="req1")
        chain.end_trace(t1)
        t2 = chain.start_trace(request_text="req2")
        chain.end_trace(t2)
        recent = chain.get_recent_traces(limit=2)
        assert len(recent) == 2

    def test_get_error_traces(self):
        from core.trace_chain import TraceChain, SpanStatus
        chain = TraceChain(persist_enabled=False)
        ok = chain.start_trace(request_text="ok")
        chain.end_trace(ok)
        err = chain.start_trace(request_text="err")
        # finish with ERROR status to mark it as errored
        err.finish(status=SpanStatus.ERROR)
        chain.end_trace(err)
        errors = chain.get_error_traces(limit=10)
        assert len(errors) >= 1

    def test_search_traces(self):
        from core.trace_chain import TraceChain
        chain = TraceChain(persist_enabled=False)
        t = chain.start_trace(request_text="unique_search_term_x1k9")
        chain.end_trace(t)
        results = chain.search_traces("unique_search_term_x1k9")
        assert len(results) >= 1

    def test_stats(self):
        from core.trace_chain import TraceChain
        chain = TraceChain(persist_enabled=False)
        stats = chain.get_stats()
        assert "total_requests" in stats
        assert "total_errors" in stats
        assert "avg_duration_ms" in stats

    def test_trace_id_unique(self):
        from core.trace_chain import TraceChain
        chain = TraceChain(persist_enabled=False)
        t1 = chain.start_trace(request_text="a")
        t2 = chain.start_trace(request_text="b")
        assert t1.id != t2.id

    def test_get_trace_by_id(self):
        from core.trace_chain import TraceChain
        chain = TraceChain(persist_enabled=False)
        t = chain.start_trace(request_text="find_me")
        chain.end_trace(t)
        found = chain.get_trace(t.id)
        assert found is not None
        assert found.id == t.id


class TestSpanContext:
    """span_context 上下文管理器"""

    def test_context_manager(self):
        from core.trace_chain import Trace, SpanKind, SpanStatus, span_context
        trace = Trace(trace_id="ctx-test")
        with span_context(trace, "ctx_span", SpanKind.TOOL) as ctx:
            ctx.set_input({"query": "test"})
            ctx.set_output("result")
            ctx.set_tokens(inp=10, out=20)
            assert ctx.span is not None
        assert ctx.span.end_time > 0

    def test_context_exception_records_error(self):
        from core.trace_chain import Trace, SpanKind, SpanStatus, span_context
        trace = Trace(trace_id="exc-test")
        try:
            with span_context(trace, "fail_span", SpanKind.LLM) as ctx:
                ctx.set_input({"x": 1})
                raise ValueError("test error")
        except ValueError:
            pass
        assert ctx.span.status == SpanStatus.ERROR

    def test_add_event(self):
        from core.trace_chain import Trace, SpanKind, span_context
        trace = Trace(trace_id="evt-test")
        with span_context(trace, "evt_span", SpanKind.CUSTOM) as ctx:
            ctx.add_event("milestone", {"step": 1})
        assert any(e["name"] == "milestone" for e in ctx.span.events)
