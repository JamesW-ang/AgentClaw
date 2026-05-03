"""
═══════════════════════════════════════════════════════════════════════════
  ErrorChain - 统一错误处理链
  AgentClaw v6.1.3
  放置: core/error_chain.py

  解决的问题:
    现在每个工具各自抛异常，没有全局兜底。
    一个工具挂了 → 整个agent崩 → 返回裸栈信息给用户。

  架构:
    Tool Call → ErrorHandlerChain → [工具级捕获] → [分类] → [重试/降级/兜底] → 安全返回

  核心能力:
    1. 全局兜底: 任何异常都不会泄漏裸栈
    2. 错误分类: timeout / network / auth / rate_limit / data / fatal / unknown
    3. 自动重试: 可配置退避策略 (指数退避 + 抖动)
    4. 优雅降级: 工具挂了返回降级结果, 不中断流程
    5. 熔断保护: 连续失败自动熔断, 避免雪崩
    6. 错误上报: 统一收集, 支持回调通知
    7. 上下文透传: 错误携带调用上下文, 方便排查
═══════════════════════════════════════════════════════════════════════════
"""

import functools
import logging
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ═══ 1. 错误分类 ═══

class ErrorCategory(Enum):
    """错误分类 - 决定处理策略"""
    TIMEOUT     = "timeout"       # 超时, 可重试
    NETWORK     = "network"       # 网络故障, 可重试
    AUTH        = "auth"          # 认证失败, 不可自动重试
    RATE_LIMIT  = "rate_limit"    # 限流, 需等待后重试
    DATA        = "data"          # 数据错误 (格式/缺失), 不可重试
    TOOL        = "tool"          # 工具自身错误, 视情况重试
    LLM         = "llm"           # LLM 调用错误, 可重试
    FATAL       = "fatal"         # 致命错误, 立即终止
    UNKNOWN     = "unknown"       # 未分类

    @property
    def retryable(self) -> bool:
        """是否可重试"""
        return self in (
            ErrorCategory.TIMEOUT, ErrorCategory.NETWORK,
            ErrorCategory.RATE_LIMIT, ErrorCategory.LLM, ErrorCategory.TOOL,
        )

    @property
    def user_visible(self) -> bool:
        """是否应该让用户看到"""
        return self in (
            ErrorCategory.AUTH, ErrorCategory.DATA, ErrorCategory.FATAL,
        )


class Severity(Enum):
    """严重程度"""
    LOW      = 1   # 工具降级即可
    MEDIUM   = 2   # 影响当前任务
    HIGH     = 3   # 影响整体流程
    CRITICAL = 4   # 系统级故障


# ═══ 2. 结构化错误信息 ═══

@dataclass
class ErrorContext:
    """
    错误上下文 - 每个错误携带完整调用信息

    Agent 拿到这个就知道: 什么工具、什么参数、什么阶段、第几次尝试、怎么恢复
    """
    # 错误本体
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: Severity = Severity.MEDIUM
    message: str = ""
    detail: str = ""

    # 调用信息
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    call_id: str = ""
    timestamp: float = field(default_factory=time.time)
    attempt: int = 1          # 第几次尝试
    max_attempts: int = 3     # 最大尝试次数

    # 原始异常
    original_exception: Exception | None = None
    original_traceback: str = ""

    # 恢复信息
    recovery_hint: str = ""    # 给 Agent 的恢复建议
    fallback_result: Any = None  # 降级结果

    # 调用链 (嵌套工具调用时)
    parent_call_id: str = ""

    def to_dict(self) -> dict:
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'detail': self.detail[:200] if self.detail else '',
            'tool_name': self.tool_name,
            'tool_args': {k: str(v)[:100] for k, v in self.tool_args.items()},
            'call_id': self.call_id,
            'attempt': self.attempt,
            'max_attempts': self.max_attempts,
            'recovery_hint': self.recovery_hint,
            'has_fallback': self.fallback_result is not None,
            'parent_call_id': self.parent_call_id,
        }

    def to_agent_message(self) -> str:
        """生成给 Agent 看的错误消息 (不带裸栈)"""
        parts = [f"[{self.category.value.upper()}] {self.message}"]
        if self.tool_name:
            parts.append(f"工具: {self.tool_name}")
        if self.recovery_hint:
            parts.append(f"建议: {self.recovery_hint}")
        if self.attempt < self.max_attempts:
            parts.append(f"将自动重试 ({self.attempt}/{self.max_attempts})")
        elif self.fallback_result is not None:
            parts.append("已使用降级结果继续执行")
        return " | ".join(parts)


# ═══ 3. 错误分类器 ═══

class ErrorClassifier:
    """
    根据异常类型自动分类

    支持自定义规则: classifier.add_rule(match_condition, category, severity)
    """

    # 内置分类规则 (异常类型前缀/关键字 → 分类)
    BUILTIN_RULES = [
        # 超时
        (lambda e: _match_exc(e, 'Timeout', 'timed out', '超时'),
         ErrorCategory.TIMEOUT, Severity.MEDIUM),
        (lambda e: _match_exc(e, 'asyncio.TimeoutError', 'CancelledError'),
         ErrorCategory.TIMEOUT, Severity.MEDIUM),

        # 网络
        (lambda e: _match_exc(e, 'ConnectionError', 'ConnectionRefused', 'ConnectionReset',
                              'NetworkError', '网络', '连接'),
         ErrorCategory.NETWORK, Severity.HIGH),
        (lambda e: _match_exc(e, 'requests.Connection', 'urllib.error', 'httpx'),
         ErrorCategory.NETWORK, Severity.HIGH),

        # 认证
        (lambda e: _match_exc(e, 'AuthenticationError', 'Unauthorized', '401', '认证',
                              '权限', 'auth', 'token'),
         ErrorCategory.AUTH, Severity.HIGH),

        # 限流
        (lambda e: _match_exc(e, 'RateLimit', '429', 'rate_limit', '限流', '频率',
                              'too many'),
         ErrorCategory.RATE_LIMIT, Severity.MEDIUM),

        # 数据
        (lambda e: _match_exc(e, 'ValueError', 'KeyError', 'IndexError', 'TypeError',
                              'json.JSONDecodeError', 'ValidationError', '格式', '解析'),
         ErrorCategory.DATA, Severity.LOW),
        (lambda e: _match_exc(e, 'FileNotFoundError', 'FileExistsError', 'PermissionError'),
         ErrorCategory.DATA, Severity.MEDIUM),

        # LLM
        (lambda e: _match_exc(e, 'APIError', 'APIConnectionError', 'LLMError',
                              'CompletionError', '模型', 'completion'),
         ErrorCategory.LLM, Severity.HIGH),

        # 致命
        (lambda e: _match_exc(e, 'MemoryError', 'SystemError', 'RuntimeError'),
         ErrorCategory.FATAL, Severity.CRITICAL),
    ]

    def __init__(self):
        self._rules = list(self.BUILTIN_RULES)

    def classify(self, exc: Exception, tool_name: str = '') -> tuple:
        """
        分类异常

        Returns:
            (ErrorCategory, Severity)
        """
        exc_str = str(exc)
        exc_type = type(exc).__name__
        full = f"{exc_type}: {exc_str}"

        for match_fn, category, severity in self._rules:
            if match_fn(full):
                return category, severity

        # 特殊: KeyboardInterrupt 立即终止
        if isinstance(exc, KeyboardInterrupt):
            return ErrorCategory.FATAL, Severity.CRITICAL

        return ErrorCategory.UNKNOWN, Severity.MEDIUM

    def add_rule(self, match_fn, category: ErrorCategory, severity: Severity):
        """添加自定义分类规则 (优先于内置规则)"""
        self._rules.insert(0, (match_fn, category, severity))

    def add_type_rule(self, exc_type, category: ErrorCategory, severity: Severity):
        """按异常类型添加规则"""
        self.add_rule(
            lambda e: isinstance(e, exc_type),
            category, severity
        )


def _match_exc(text: str, *keywords) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


# ═══ 4. 重试策略 ═══

class RetryPolicy:
    """可配置的重试策略"""

    def __init__(self, max_attempts=3, base_delay=1.0, max_delay=30.0,
                 backoff_factor=2.0, jitter=True, retryable_categories=None):
        """
        Args:
            max_attempts:         最大尝试次数 (1=不重试)
            base_delay:           首次重试等待秒数
            max_delay:            最大等待秒数
            backoff_factor:       退避倍数 (指数退避)
            jitter:               是否加随机抖动 (防惊群)
            retryable_categories: 可重试的错误类别 (None=用默认的 category.retryable)
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retryable_categories = retryable_categories

    def should_retry(self, ctx: ErrorContext) -> bool:
        """判断是否应该重试"""
        if ctx.attempt >= ctx.max_attempts:
            return False
        if self.retryable_categories is not None:
            return ctx.category in self.retryable_categories
        return ctx.category.retryable

    def get_delay(self, attempt: int) -> float:
        """计算第 N 次重试的等待时间"""
        import random
        delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
        delay = min(delay, self.max_delay)
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        return delay


# ═══ 5. 熔断器 ═══

class CircuitBreaker:
    """
    熔断器 - 连续失败超过阈值后自动熔断

    状态: CLOSED(正常) → OPEN(熔断, 直接拒绝) → HALF_OPEN(试探恢复) → CLOSED
    """

    def __init__(self, failure_threshold=5, recovery_timeout=60, half_open_max=1):
        """
        Args:
            failure_threshold:  连续失败多少次触发熔断
            recovery_timeout:  熔断后多久进入半开 (秒)
            half_open_max:     半开状态允许通过的最大请求数
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self._failures = {}      # tool_name → 连续失败次数
        self._open_at = {}       # tool_name → 熔断开启时间
        self._half_open_count = {}  # tool_name → 半开状态通过数
        self._lock = threading.Lock()

    def allow(self, tool_name: str) -> bool:
        """判断是否允许调用"""
        with self._lock:
            self._failures.get(tool_name, 0)

            # 半开状态
            if tool_name in self._open_at:
                elapsed = time.time() - self._open_at[tool_name]
                if elapsed >= self.recovery_timeout:
                    # 进入半开
                    half_count = self._half_open_count.get(tool_name, 0)
                    if half_count < self.half_open_max:
                        self._half_open_count[tool_name] = half_count + 1
                        return True
                    return False
                return False  # 仍在熔断

            return True

    def record_success(self, tool_name: str):
        """记录成功, 重置计数"""
        with self._lock:
            self._failures[tool_name] = 0
            self._open_at.pop(tool_name, None)
            self._half_open_count.pop(tool_name, None)

    def record_failure(self, tool_name: str):
        """记录失败, 可能触发熔断"""
        with self._lock:
            self._failures[tool_name] = self._failures.get(tool_name, 0) + 1
            if self._failures[tool_name] >= self.failure_threshold:
                self._open_at[tool_name] = time.time()
                logger.warning(
                    f"[CircuitBreaker] 熔断触发: {tool_name}, "
                    f"连续失败={self._failures[tool_name]}"
                )

    def get_state(self, tool_name: str) -> str:
        """获取状态: closed / open / half_open"""
        with self._lock:
            if tool_name in self._open_at:
                elapsed = time.time() - self._open_at[tool_name]
                if elapsed >= self.recovery_timeout:
                    return "half_open"
                return "open"
            return "closed"


# ═══ 6. 错误上报器 ═══

class ErrorReporter:
    """统一错误收集与上报"""

    def __init__(self, max_history=1000, callback=None):
        """
        Args:
            max_history: 内存中保留的最大错误条数
            callback:    错误回调 fn(ErrorContext) - 可接入通知系统
        """
        self._history = []
        self._max_history = max_history
        self._callback = callback
        self._counts = {}  # category → count

    def report(self, ctx: ErrorContext):
        """记录错误"""
        self._history.append(ctx)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        cat = ctx.category.value
        self._counts[cat] = self._counts.get(cat, 0) + 1

        # 日志
        log_fn = logger.error if ctx.severity.value >= Severity.HIGH.value else logger.warning
        log_fn(f"[ErrorChain] {ctx.to_agent_message()}")

        # 回调
        if self._callback:
            try:
                self._callback(ctx)
            except Exception as e:
                logger.error(f"[ErrorReporter] 回调失败: {e}")

        # 写入自学习 (如果有 ExperienceLearner)
        self._report_to_learner(ctx)

    def get_recent(self, limit=20) -> list:
        """获取最近的错误"""
        return [ctx.to_dict() for ctx in self._history[-limit:]]

    def get_summary(self) -> dict:
        """获取错误统计"""
        return {
            'total_errors': len(self._history),
            'by_category': dict(self._counts),
            'recent_5': [ctx.to_dict() for ctx in self._history[-5:]],
        }

    def _report_to_learner(self, ctx: ErrorContext):
        """将错误经验写入自学习模块"""
        try:
            # 向上查找 learner (在 Agent 主类上)
            import inspect
            frame = inspect.currentframe()
            for _ in range(15):  # 往上查15层
                if frame is None:
                    break
                agent = frame.f_locals.get('self')
                if agent is not None:
                    learner = getattr(agent, 'experience_learner', None)
                    if learner is None:
                        learner = getattr(agent, '_learner', None)
                    if learner is not None and hasattr(learner, 'learn_from_failure'):
                        learner.learn_from_failure(
                            query=f"工具调用失败: {ctx.tool_name}",
                            error_description=ctx.message,
                            correct_approach=ctx.recovery_hint,
                            task_type='error_handling',
                        )
                        break
                frame = frame.f_back
        except Exception:
            pass  # 上报失败不影响主流程


# ═══ 7. 核心错误处理链 ═══

class ErrorChain:
    """
    统一错误处理链 - 主入口

    用法:
        chain = ErrorChain()

        # 方式1: 手动包装调用
        result = chain.execute(tool_name="search", func=my_search, args={"query": "..."})

        # 方式2: 装饰器
        @chain.tool_guard("web_search", fallback="搜索暂不可用")
        def web_search(query):
            ...

        # 方式3: 全局兜底 (在 Agent 主循环中)
        try:
            result = agent.run(user_input)
        except Exception as e:
            safe_result = chain.handle_global(e, context="agent_main_loop")
    """

    def __init__(self, retry_policy=None, classifier=None, reporter=None, circuit_breaker=None):
        self.retry = retry_policy or RetryPolicy(max_attempts=3)
        self.classifier = classifier or ErrorClassifier()
        self.reporter = reporter or ErrorReporter()
        self.circuit = circuit_breaker or CircuitBreaker()

        # 工具级配置: tool_name → {fallback, retry_override, is_critical}
        self._tool_configs = {}

        # 全局兜底回调
        self._global_fallback = None

    def configure_tool(self, tool_name, fallback=None, retry_policy=None,
                       is_critical=False, skip_circuit=False):
        """
        配置单个工具的错误处理策略

        Args:
            tool_name:     工具名
            fallback:      降级结果 (值或 callable(ctx) → value)
            retry_policy:  覆盖默认重试策略
            is_critical:   是否关键工具 (失败终止整个流程)
            skip_circuit:  是否跳过熔断检查
        """
        self._tool_configs[tool_name] = {
            'fallback': fallback,
            'retry_policy': retry_policy,
            'is_critical': is_critical,
            'skip_circuit': skip_circuit,
        }

    def set_global_fallback(self, fallback_fn):
        """设置全局兜底: fallback_fn(ctx) → result"""
        self._global_fallback = fallback_fn

    def execute(self, tool_name: str, func: Callable, args: dict = None,
                kwargs: dict = None, caller_context: str = '') -> Any:
        """
        执行工具调用，经过完整错误处理链

        Args:
            tool_name:      工具名称
            func:           工具函数
            args/kwargs:    调用参数
            caller_context: 调用方描述 (用于错误追踪)

        Returns:
            工具返回值 或 降级结果

        永远不会抛异常到调用方 (除非 is_critical=True)
        """
        import uuid
        args = args or {}
        kwargs = kwargs or {}
        config = self._tool_configs.get(tool_name, {})
        retry = config.get('retry_policy') or self.retry
        call_id = uuid.uuid4().hex[:8]

        # 熔断检查
        if not config.get('skip_circuit') and not self.circuit.allow(tool_name):
            ctx = ErrorContext(
                category=ErrorCategory.TOOL,
                severity=Severity.HIGH,
                message=f"工具 {tool_name} 已熔断，暂时不可用",
                tool_name=tool_name,
                tool_args=args,
                call_id=call_id,
                recovery_hint="等待熔断恢复后重试，或使用替代方案",
            )
            self.reporter.report(ctx)
            fallback = self._resolve_fallback(ctx, config)
            if fallback is not None:
                return fallback
            return self._global_catch(ctx)

        # 执行 + 重试循环
        last_ctx = None
        for attempt in range(1, retry.max_attempts + 1):
            try:
                result = func(*args, **kwargs) if not kwargs else func(**{**args, **kwargs})
                # 成功
                self.circuit.record_success(tool_name)
                return result

            except Exception as exc:
                category, severity = self.classifier.classify(exc, tool_name)

                last_ctx = ErrorContext(
                    category=category,
                    severity=severity,
                    message=str(exc) or type(exc).__name__,
                    detail=traceback.format_exc()[:500],
                    tool_name=tool_name,
                    tool_args=args,
                    call_id=call_id,
                    attempt=attempt,
                    max_attempts=retry.max_attempts,
                    original_exception=exc,
                    original_traceback=traceback.format_exc(),
                    parent_call_id=caller_context,
                )

                # 自动恢复建议
                last_ctx.recovery_hint = self._auto_hint(category, tool_name)

                self.reporter.report(last_ctx)
                self.circuit.record_failure(tool_name)

                # 不重试 → 立即降级
                if not retry.should_retry(last_ctx):
                    break

                # 等待退避
                delay = retry.get_delay(attempt)
                logger.info(f"[ErrorChain] {tool_name} 第{attempt}次失败, {delay:.1f}s后重试 ({category.value})")
                time.sleep(delay)

        # 所有重试耗尽 → 降级/兜底
        return self._graceful_degrade(last_ctx, config)

    def handle_global(self, exc: Exception, context: str = '',
                      tool_name: str = '') -> Any:
        """
        全局兜底 - Agent 主循环最外层调用

        捕获任何未被工具级处理的异常。
        返回安全的降级结果，永远不会抛异常。
        """
        category, severity = self.classifier.classify(exc, tool_name)

        ctx = ErrorContext(
            category=category,
            severity=Severity.CRITICAL,
            message=str(exc) or type(exc).__name__,
            detail=traceback.format_exc()[:500],
            tool_name=tool_name or context,
            call_id='global',
            original_exception=exc,
            original_traceback=traceback.format_exc(),
            recovery_hint="系统遇到问题，请稍后重试或换一种方式描述需求",
        )

        self.reporter.report(ctx)
        return self._global_catch(ctx)

    def tool_guard(self, tool_name, fallback=None, **config):
        """
        装饰器: 为工具函数加上错误保护

        @chain.tool_guard("web_search", fallback={"results": []})
        def web_search(query):
            return requests.get(...)
        """
        self.configure_tool(tool_name, fallback=fallback, **config)

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute(tool_name, func, kwargs=kwargs)
            return wrapper
        return decorator

    # ── 内部方法 ──

    def _resolve_fallback(self, ctx, config):
        """解析降级结果"""
        fallback = config.get('fallback')
        if fallback is None:
            return None
        if callable(fallback):
            try:
                return fallback(ctx)
            except Exception as e:
                logger.error(f"[ErrorChain] 降级函数失败: {e}")
                return None
        return fallback

    def _graceful_degrade(self, ctx: ErrorContext, config: dict) -> Any:
        """优雅降级"""
        # 关键工具失败 → 抛出安全的 AgentError
        if config.get('is_critical'):
            raise AgentError(ctx)

        # 工具级降级
        fallback = self._resolve_fallback(ctx, config)
        if fallback is not None:
            ctx.fallback_result = fallback
            return fallback

        # 全局兜底
        return self._global_catch(ctx)

    def _global_catch(self, ctx: ErrorContext) -> Any:
        """全局兜底"""
        if self._global_fallback:
            try:
                return self._global_fallback(ctx)
            except Exception:
                pass

        # 默认: 返回结构化错误 (给 Agent 消费, 不是给用户看裸栈)
        return {
            '_error': True,
            '_category': ctx.category.value,
            '_message': ctx.to_agent_message(),
            '_tool': ctx.tool_name,
            '_hint': ctx.recovery_hint,
            '_call_id': ctx.call_id,
        }

    def _auto_hint(self, category: ErrorCategory, tool_name: str) -> str:
        """自动生成恢复建议"""
        hints = {
            ErrorCategory.TIMEOUT:    f"{tool_name} 超时, 可尝试简化输入或稍后重试",
            ErrorCategory.NETWORK:    f"{tool_name} 网络故障, 请检查网络连接后重试",
            ErrorCategory.AUTH:       f"{tool_name} 认证失败, 请检查 API Key 或权限配置",
            ErrorCategory.RATE_LIMIT: f"{tool_name} 触发限流, 请等待片刻后重试",
            ErrorCategory.DATA:       f"{tool_name} 数据异常, 请检查输入格式是否正确",
            ErrorCategory.LLM:        f"{tool_name} 模型调用失败, 将自动重试或使用备用方案",
            ErrorCategory.FATAL:      "系统遇到严重问题, 建议重启或联系管理员",
        }
        return hints.get(category, "未知错误, 请稍后重试")


# ═══ 8. AgentError - 安全异常 (可抛给上层) ═══

class AgentError(Exception):
    """
    安全异常 - 携带结构化信息, 不会泄漏内部细节

    只在 is_critical=True 的工具失败时抛出,
    上层捕获后可以直接返回给用户。
    """
    def __init__(self, ctx: ErrorContext):
        self.ctx = ctx
        super().__init__(ctx.to_agent_message())

    def to_user_message(self) -> str:
        """生成用户可见的消息 (不含内部信息)"""
        if self.ctx.category == ErrorCategory.AUTH:
            return "权限不足，请检查配置后重试。"
        if self.ctx.category == ErrorCategory.FATAL:
            return "系统遇到了一个问题，请稍后重试。"
        return self.ctx.message


# ═══ 9. 便捷函数 ═══

def safe_call(func, *args, tool_name="", fallback=None, retries=3, **kwargs):
    """
    快速安全调用 (无需实例化 ErrorChain)

    result = safe_call(requests.get, "https://api.example.com",
                       tool_name="http_get", fallback={"error": "unavailable"})
    """
    chain = ErrorChain(retry_policy=RetryPolicy(max_attempts=retries))
    if fallback:
        chain.configure_tool(tool_name or func.__name__, fallback=fallback)
    return chain.execute(tool_name or func.__name__, func, kwargs=kwargs)


# ═══ 10. 主程序集成 ═══
#
# === 方案A: 最简集成 (Agent 主循环加两行) ===
#
#   from core.error_chain import ErrorChain, AgentError
#
#   class AgentClaw:
#       def __init__(self):
#           self.error_chain = ErrorChain()
#           self.error_chain.configure_tool("web_search",    fallback={"results": [], "summary": "搜索暂不可用"})
#           self.error_chain.configure_tool("code_execute",  fallback={"output": "", "error": "执行环境异常"})
#           self.error_chain.configure_tool("database",      is_critical=True)
#           self.error_chain.configure_tool("file_read",     retry_policy=RetryPolicy(max_attempts=2))
#
#       async def run(self, user_input):
#           try:
#               return await self._execute(user_input)
#           except AgentError as e:
#               return e.to_user_message()
#           except Exception as e:
#               return self.error_chain.handle_global(e, context="agent_run")
#
#       async def _execute(self, user_input):
#           results = self.error_chain.execute(
#               "web_search", self._search, kwargs={"query": user_input}
#           )
#           if results.get('_error'):
#               return "抱歉，搜索暂时不可用，请稍后重试。"
#           return self._process(results)
#
#
# === 方案B: 装饰器模式 (最小改动) ===
#
#   chain = ErrorChain()
#
#   @chain.tool_guard("web_search", fallback={"results": []})
#   def web_search(query):
#       return requests.get(...)
#
#   @chain.tool_guard("llm_call", retry_policy=RetryPolicy(max_attempts=3, base_delay=2))
#   def call_llm(prompt):
#       return client.chat(...)
#
#   @chain.tool_guard("database", is_critical=True)
#   def query_db(sql):
#       return db.execute(sql)
#
#
# === 方案C: 全局中间件 ===
#
#   class AgentClaw:
#       def __init__(self):
#           self.error_chain = ErrorChain()
#           self.error_chain.set_global_fallback(
#               lambda ctx: "处理遇到了一些问题，我会尝试其他方式来完成您的请求。"
#           )
#
#       def call_tool(self, tool_name, tool_func, **kwargs):
#           return self.error_chain.execute(tool_name, tool_func, kwargs=kwargs)
#
#       def shutdown(self):
#           summary = self.error_chain.reporter.get_summary()
#           print(f"错误统计: {json.dumps(summary, ensure_ascii=False, indent=2)}")
#
#
# === 自定义: 添加新的错误分类规则 ===
#
#   chain.classifier.add_type_rule(MyCustomException, ErrorCategory.TOOL, Severity.MEDIUM)
#   chain.classifier.add_rule(lambda e: "余额不足" in str(e), ErrorCategory.AUTH, Severity.HIGH)
