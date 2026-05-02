# ============================================================
# AgentClaw — ErrorChain 单元测试
# ============================================================

import os
import sys
import time
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestErrorClassifier:
    """ErrorClassifier 实例方法 classify(exc, tool_name='') → (ErrorCategory, Severity)"""

    def test_classify_timeout(self):
        from core.error_chain import ErrorClassifier, ErrorCategory
        clf = ErrorClassifier()
        cat, sev = clf.classify(TimeoutError("Connection timed out"))
        assert cat == ErrorCategory.TIMEOUT
        assert cat.retryable

    def test_classify_auth(self):
        from core.error_chain import ErrorClassifier, ErrorCategory
        clf = ErrorClassifier()
        cat, sev = clf.classify(Exception("HTTP 401 Unauthorized"))
        assert cat == ErrorCategory.AUTH
        assert not cat.retryable

    def test_classify_rate_limit(self):
        from core.error_chain import ErrorClassifier, ErrorCategory
        clf = ErrorClassifier()
        cat, sev = clf.classify(Exception("Rate limit exceeded 429"))
        assert cat == ErrorCategory.RATE_LIMIT
        assert cat.retryable

    def test_classify_data_error(self):
        from core.error_chain import ErrorClassifier, ErrorCategory
        clf = ErrorClassifier()
        cat, sev = clf.classify(ValueError("invalid input"))
        assert cat == ErrorCategory.DATA
        assert not cat.retryable

    def test_classify_unknown(self):
        from core.error_chain import ErrorClassifier, ErrorCategory
        clf = ErrorClassifier()
        cat, sev = clf.classify(Exception("some weird error"))
        assert cat == ErrorCategory.UNKNOWN
        # UNKNOWN defaults to non-retryable (not in TIMEOUT/NETWORK/RATE_LIMIT/LLM/TOOL)
        assert not cat.retryable

    def test_add_rule(self):
        from core.error_chain import ErrorClassifier, ErrorCategory, Severity
        clf = ErrorClassifier()
        clf.add_rule(lambda e: "CUSTOM" in e, ErrorCategory.FATAL, Severity.CRITICAL)
        cat, sev = clf.classify(Exception("CUSTOM ERROR"))
        assert cat == ErrorCategory.FATAL


class TestRetryPolicy:
    """RetryPolicy: max_attempts=3, base_delay=1.0"""

    def make_context(self, attempt=1, category=None):
        from core.error_chain import ErrorContext, ErrorCategory
        return ErrorContext(
            category=category or ErrorCategory.UNKNOWN,
            attempt=attempt,
            max_attempts=3,
        )

    def test_should_retry_on_first_failure(self):
        from core.error_chain import RetryPolicy, ErrorCategory
        policy = RetryPolicy(max_attempts=3)
        ctx = self.make_context(attempt=1, category=ErrorCategory.TIMEOUT)
        assert policy.should_retry(ctx)

    def test_should_not_retry_on_last_attempt(self):
        from core.error_chain import RetryPolicy, ErrorCategory
        policy = RetryPolicy(max_attempts=3)
        ctx = self.make_context(attempt=3, category=ErrorCategory.TIMEOUT)
        assert not policy.should_retry(ctx)

    def test_should_not_retry_non_retryable(self):
        from core.error_chain import RetryPolicy, ErrorCategory
        policy = RetryPolicy(max_attempts=3)
        ctx = self.make_context(attempt=1, category=ErrorCategory.DATA)
        assert not policy.should_retry(ctx)

    def test_get_delay_increases_with_attempts(self):
        from core.error_chain import RetryPolicy
        policy = RetryPolicy(base_delay=1.0, backoff_factor=2.0, jitter=False)
        d1 = policy.get_delay(1)
        d2 = policy.get_delay(2)
        d3 = policy.get_delay(3)
        assert d1 == 1.0
        assert d2 == 2.0
        assert d3 == 4.0

    def test_get_delay_capped(self):
        from core.error_chain import RetryPolicy
        policy = RetryPolicy(base_delay=10.0, max_delay=15.0, backoff_factor=2.0, jitter=False)
        d = policy.get_delay(3)  # 10 * 2^2 = 40, capped at 15
        assert d == 15.0


class TestCircuitBreaker:
    """CircuitBreaker: per-tool, allow/record_failure/record_success/get_state"""

    def test_initial_allow(self):
        from core.error_chain import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.allow("tool_a")

    def test_open_after_failures(self):
        from core.error_chain import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        cb.record_failure("tool_a")
        cb.record_failure("tool_a")
        cb.record_failure("tool_a")
        assert not cb.allow("tool_a")

    def test_different_tools_independent(self):
        from core.error_chain import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("tool_a")
        cb.record_failure("tool_a")
        assert not cb.allow("tool_a")
        assert cb.allow("tool_b")

    def test_success_resets(self):
        from core.error_chain import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure("tool_a")
        cb.record_failure("tool_a")
        cb.record_success("tool_a")
        assert cb.allow("tool_a")  # reset after success

    def test_half_open_recovers(self):
        from core.error_chain import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
        cb.record_failure("tool_a")
        cb.record_failure("tool_a")
        assert not cb.allow("tool_a")
        time.sleep(0.06)
        assert cb.allow("tool_a")  # half-open
        cb.record_success("tool_a")
        assert cb.allow("tool_a")  # fully closed

    def test_get_state(self):
        from core.error_chain import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=2)
        assert cb.get_state("tool_a") == "closed"
        cb.record_failure("tool_a")
        assert cb.get_state("tool_a") == "closed"
        cb.record_failure("tool_a")
        assert cb.get_state("tool_a") == "open"


class TestErrorReporter:
    """ErrorReporter 报告收集"""

    def test_report(self):
        from core.error_chain import ErrorReporter, ErrorContext, ErrorCategory, Severity
        reporter = ErrorReporter(max_history=10)
        ctx = ErrorContext(
            category=ErrorCategory.TIMEOUT, severity=Severity.HIGH,
            message="test error",
        )
        reporter.report(ctx)
        recent = reporter.get_recent()
        assert len(recent) == 1
        # get_recent returns dicts (ctx.to_dict())
        assert recent[0]["message"] == "test error"

    def test_summary(self):
        from core.error_chain import ErrorReporter, ErrorContext, ErrorCategory, Severity
        reporter = ErrorReporter()
        reporter.report(ErrorContext(category=ErrorCategory.TIMEOUT, severity=Severity.HIGH, message="err1"))
        reporter.report(ErrorContext(category=ErrorCategory.AUTH, severity=Severity.HIGH, message="err2"))
        summary = reporter.get_summary()
        assert summary["total_errors"] == 2
        assert summary["by_category"].get("timeout", 0) >= 1


class TestErrorChain:
    """ErrorChain 完整功能"""

    def test_execute_success(self):
        from core.error_chain import ErrorChain
        chain = ErrorChain()

        def add(x, y):
            return {"result": x + y}

        # ErrorChain.execute uses kwargs= parameter for function kwargs
        result = chain.execute("add_tool", add, kwargs={"x": 21, "y": 21})
        assert result == {"result": 42}

    def test_execute_with_retry(self):
        from core.error_chain import ErrorChain
        chain = ErrorChain()

        call_count = [0]

        def flaky(x):
            call_count[0] += 1
            if call_count[0] < 2:
                raise TimeoutError("timeout")
            return {"result": x}

        result = chain.execute("flaky", flaky, kwargs={"x": 42})
        assert result == {"result": 42}
        assert call_count[0] >= 2

    def test_execute_failure_fallback(self):
        from core.error_chain import ErrorChain
        chain = ErrorChain()

        def always_fail(x):
            raise ValueError("broken")

        # configure a fallback
        chain.configure_tool("failing", fallback={"fallback": True})

        result = chain.execute("failing", always_fail, kwargs={"x": 1})
        assert result == {"fallback": True}

    def test_circuit_breaker_opens(self):
        from core.error_chain import ErrorChain
        chain = ErrorChain()
        chain.circuit.failure_threshold = 2

        def fail(x):
            raise RuntimeError("fail")

        chain.execute("bad", fail, kwargs={"x": 1})
        chain.execute("bad", fail, kwargs={"x": 1})
        assert not chain.circuit.allow("bad")

    def test_circuit_blocks_fast(self):
        """熔断后直接返回降级，不再调用函数"""
        from core.error_chain import ErrorChain
        chain = ErrorChain()
        chain.circuit.failure_threshold = 2
        chain.circuit.recovery_timeout = 60

        def fail(_):
            raise RuntimeError("always fail")

        chain.execute("bad", fail, kwargs={"x": 1})
        chain.execute("bad", fail, kwargs={"x": 1})

        # configure fallback so we get a clean result instead of graceful degrade
        chain.configure_tool("bad", fallback={"blocked": True})

        # circuit is open, should return fallback without calling func
        def should_not_run(x):
            raise AssertionError("must not call")
        result = chain.execute("bad", should_not_run, kwargs={"x": 1})
        assert result == {"blocked": True}

    def test_configure_tool(self):
        from core.error_chain import ErrorChain
        chain = ErrorChain()
        chain.configure_tool("web_search", fallback={"results": []})
        assert "web_search" in chain._tool_configs

    def test_set_global_fallback(self):
        from core.error_chain import ErrorChain
        chain = ErrorChain()
        chain.set_global_fallback(lambda ctx: {"global": "fallback"})
        assert chain._global_fallback is not None


class TestSafeCall:
    """safe_call 使用 **kwargs 传递参数"""

    def test_safe_call_success(self):
        from core.error_chain import safe_call
        result = safe_call(lambda x: x + 1, tool_name="add", x=2)
        assert result == 3

    def test_safe_call_fallback(self):
        from core.error_chain import safe_call
        result = safe_call(
            lambda x: 1 // 0,  # will raise ZeroDivisionError
            tool_name="divide",
            fallback="fallback_val",
            x=1,
        )
        assert result == "fallback_val"
