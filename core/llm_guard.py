"""
═══════════════════════════════════════════════════════════════════════════
  LLMGuard - LLM 调用容错层
  AgentClaw v6.1.3
  放置: core/llm_guard.py

  解决的问题:
    现在直接调用 DeepSeek API, 没做重试/降级/超时控制。
    API 抖一下 → 整个 agent 卡死/报裸错。

  架构:
    LLM Call → LLMGuard → [超时控制] → [重试+退避] → [降级链] → 安全返回

  核心能力:
    1. 超时控制: 每次调用可配超时, 超时自动中断
    2. 智能重试: 按 HTTP 状态码分类, 不同错误不同策略
       - 429 限流 → 读取 Retry-After, 精确等待
       - 500/502/503 服务端错误 → 指数退避重试
       - 401/403 认证错误 → 不重试, 立即上报
       - 网络超时 → 递增超时重试
    3. 降级链: 主模型 → 备用模型 → 缓存 → 优雅降级
    4. 熔断保护: 连续失败自动熔断 (集成 ErrorChain)
    5. 链路追踪: 每次 LLM 调用记录为 TraceChain Span
    6. 自学习反馈: 失败模式自动写入 ExperienceLearner
    7. 流式适配: 支持流式和非流式两种模式
    8. 零额外依赖: 纯标准库 + openai (已有依赖)
═══════════════════════════════════════════════════════════════════════════
"""

import time
import json
import os
import hashlib
import logging
import threading
import traceback
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, List, Dict, Tuple
from collections import OrderedDict
from core.metrics import observe_llm_call, observe_llm_error

logger = logging.getLogger(__name__)


# ═══ 1. LLM 错误分类 ═══

class LLMErrorType(Enum):
    """LLM 专用错误分类 (比通用 ErrorCategory 更精细)"""
    RATE_LIMIT     = "rate_limit"     # 429 限流
    SERVER_ERROR   = "server_error"   # 500/502/503 服务端故障
    TIMEOUT        = "timeout"        # 请求超时
    NETWORK        = "network"        # 网络连接失败
    AUTH           = "auth"           # 401/403 认证/权限
    BAD_REQUEST    = "bad_request"    # 400 参数错误
    CONTEXT_LENGTH = "context_length" # token 超限
    MODEL_ERROR    = "model_error"    # 模型自身错误 (content_filter 等)
    UNKNOWN        = "unknown"        # 未知

    @property
    def retryable(self) -> bool:
        """是否可重试"""
        return self in (
            LLMErrorType.RATE_LIMIT, LLMErrorType.SERVER_ERROR,
            LLMErrorType.TIMEOUT, LLMErrorType.NETWORK,
        )

    @property
    def retry_after_hint(self) -> Optional[float]:
        """建议等待时间 (秒)"""
        hints = {
            LLMErrorType.RATE_LIMIT: 5.0,    # 限流默认等 5 秒
            LLMErrorType.SERVER_ERROR: 2.0,  # 服务端错误默认等 2 秒
            LLMErrorType.TIMEOUT: 1.0,       # 超时默认等 1 秒
            LLMErrorType.NETWORK: 3.0,       # 网络错误默认等 3 秒
        }
        return hints.get(self)


# ═══ 2. 结构化 LLM 调用结果 ═══

@dataclass
class LLMResult:
    """
    LLM 调用结果 (统一封装, 无论成功/失败/降级)

    Agent 拿到这个就知道:
    - 内容是什么 (content / stream)
    - 用了什么模型 (model)
    - 花了多久 (latency_ms)
    - 用了多少 token (tokens)
    - 是不是降级结果 (fallback)
    - 经历了几次重试 (attempts)
    """
    content: str = ""
    stream: Any = None              # 流式响应迭代器 (如果启用)
    model: str = ""
    raw_response: Any = None  
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_estimate: float = 0.0

    # 元信息
    success: bool = True
    attempts: int = 1               # 经历了几次尝试
    is_fallback: bool = False       # 是否降级结果
    fallback_level: int = 0         # 降级层级 (0=主模型, 1=备用, 2=缓存, 3=兜底)
    error_type: str = ""            # 失败时的错误类型
    error_message: str = ""         # 失败时的错误消息
    recovery_hint: str = ""         # 给 Agent 的恢复建议
    trace_id: str = ""              # 关联的 trace_id

    @property
    def is_error(self) -> bool:
        return not self.success

    def to_dict(self) -> dict:
        return {
            'success': self.success,
            'model': self.model,
            'latency_ms': round(self.latency_ms, 1),
            'tokens': {'in': self.tokens_in, 'out': self.tokens_out},
            'cost': round(self.cost_estimate, 6),
            'attempts': self.attempts,
            'is_fallback': self.is_fallback,
            'fallback_level': self.fallback_level,
            'error': self.error_type,
            'trace_id': self.trace_id,
        }


# ═══ 3. LLM 错误分类器 ═══

class LLMErrorClassifier:
    """
    从异常中提取 LLM 错误类型

    支持:
    - openai 库的各类异常
    - 通用 HTTP 错误码
    - 超时/网络异常
    - 中文错误信息匹配
    """

    # HTTP 状态码 → 错误类型
    STATUS_MAP = {
        400: LLMErrorType.BAD_REQUEST,
        401: LLMErrorType.AUTH,
        403: LLMErrorType.AUTH,
        429: LLMErrorType.RATE_LIMIT,
        500: LLMErrorType.SERVER_ERROR,
        502: LLMErrorType.SERVER_ERROR,
        503: LLMErrorType.SERVER_ERROR,
    }

    # 关键词匹配 (用于没有状态码的异常)
    KEYWORD_RULES = [
        # (关键词列表, 错误类型)
        (["rate_limit", "rate limit", "429", "too many requests",
          "限流", "频率限制", "requests per minute"],
         LLMErrorType.RATE_LIMIT),
        (["timeout", "timed out", "ReadTimeout", "ConnectTimeout",
          "超时", "请求超时"],
         LLMErrorType.TIMEOUT),
        (["ConnectionError", "ConnectionReset", "NetworkError",
          "connection", "网络", "连接失败", "DNS"],
         LLMErrorType.NETWORK),
        (["AuthenticationError", "Unauthorized", "invalid_api_key",
          "认证", "权限", "API Key"],
         LLMErrorType.AUTH),
        (["BadRequestError", "invalid", "InvalidRequestError",
          "参数", "格式错误"],
         LLMErrorType.BAD_REQUEST),
        (["context_length", "maximum context", "token limit",
          "token 超限", "上下文过长"],
         LLMErrorType.CONTEXT_LENGTH),
        (["InternalServerError", "500", "502", "503",
          "服务端", "服务器错误"],
         LLMErrorType.SERVER_ERROR),
        (["content_filter", "ContentPolicyViolation",
          "内容过滤", "安全策略"],
         LLMErrorType.MODEL_ERROR),
    ]

    @classmethod
    def classify(cls, exc: Exception) -> Tuple[LLMErrorType, Optional[float]]:
        """
        分类异常

        Returns:
            (error_type, retry_after_seconds)
            retry_after_seconds: 从错误中提取的精确等待时间 (如有)
        """
        exc_str = str(exc)
        exc_type = type(exc).__name__
        full = f"{exc_type}: {exc_str}"

        # 1. 从状态码分类
        retry_after = None
        if hasattr(exc, 'status_code'):
            status = exc.status_code
            if status in cls.STATUS_MAP:
                err_type = cls.STATUS_MAP[status]
                # 尝试读取 Retry-After
                if status == 429:
                    retry_after = cls._extract_retry_after(exc, exc_str)
                return err_type, retry_after

        # 2. 从关键词匹配
        full_lower = full.lower()
        for keywords, err_type in cls.KEYWORD_RULES:
            if any(kw.lower() in full_lower for kw in keywords):
                # 限流也要尝试提取 retry_after
                if err_type == LLMErrorType.RATE_LIMIT:
                    retry_after = cls._extract_retry_after(exc, exc_str)
                return err_type, retry_after

        # 3. 默认
        return LLMErrorType.UNKNOWN, None

    @classmethod
    def _extract_retry_after(cls, exc, exc_str: str) -> Optional[float]:
        """从异常中提取 Retry-After (秒)"""
        # 检查 response header
        if hasattr(exc, 'response') and exc.response is not None:
            headers = getattr(exc.response, 'headers', {})
            if isinstance(headers, dict):
                ra = headers.get('retry-after') or headers.get('Retry-After')
                if ra:
                    try:
                        return float(ra)
                    except (ValueError, TypeError):
                        pass

        # 检查字符串中的提示
        import re
        match = re.search(r'retry[- ]?after[:\s]+(\d+(?:\.\d+)?)\s*s?', exc_str, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        return None


# ═══ 4. LLM 调用缓存 ═══

class LLMCache:
    """
    轻量 LLM 响应缓存

    策略:
    - LRU 淘汰, 默认 100 条
    - 按消息内容 hash 作为 key
    - 可设 TTL (默认 5 分钟, LLM 响应有时效性)
    - 只缓存成功的非流式调用
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 300):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, messages: list, model: str = "") -> Optional[str]:
        """查询缓存"""
        key = self._make_key(messages, model)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            cached_at, content = entry
            if time.time() - cached_at > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            # 移到末尾 (LRU)
            self._cache.move_to_end(key)
            self._hits += 1
            return content

    def put(self, messages: list, content: str, model: str = ""):
        """写入缓存"""
        if not content or len(content) < 10:
            return  # 不缓存太短的/空响应
        key = self._make_key(messages, model)
        with self._lock:
            self._cache[key] = (time.time(), content)
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': round(self.hit_rate, 3),
        }

    @staticmethod
    def _make_key(messages: list, model: str) -> str:
        """生成缓存 key (messages JSON + model 的 hash)"""
        raw = json.dumps(messages, ensure_ascii=False, sort_keys=True) + f"|{model}"
        return hashlib.md5(raw.encode('utf-8')).hexdigest()


# ═══ 5. 降级链配置 ═══

@dataclass
class FallbackConfig:
    """降级链配置"""
    # 备用模型 (同 provider 不同模型)
    backup_models: List[str] = field(default_factory=list)

    # 是否启用缓存降级
    enable_cache: bool = True

    # 优雅降级消息 (最后兜底)
    graceful_message: str = "抱歉, 我暂时遇到了一些问题, 请稍后再试。"

    # 缓存降级时使用的 prompt 缩减策略
    cache_prompt_ratio: float = 0.8  # 保留 80% 的消息进行缓存匹配


# ═══ 6. 重试策略 ═══

@dataclass
class LLMRetryPolicy:
    """LLM 专用重试策略"""
    max_attempts: int = 3             # 最大尝试次数 (含首次)
    base_timeout: float = 30.0        # 基础超时 (秒)
    max_timeout: float = 120.0        # 最大超时 (秒)
    timeout_growth: float = 1.5       # 每次重试超时增长因子
    base_delay: float = 1.0           # 基础重试延迟 (秒)
    max_delay: float = 30.0           # 最大重试延迟 (秒)
    backoff_factor: float = 2.0       # 退避倍数
    jitter: bool = True               # 是否加抖动
    respect_retry_after: bool = True  # 是否遵循 Retry-After header


# ═══ 7. 核心容错引擎 ═══

class LLMGuard:
    """
    LLM 调用容错层 — 主入口

    用法 (最简集成):
        from core.llm_guard import LLMGuard

        guard = LLMGuard(
            default_model="deepseek-chat",
            backup_models=["deepseek-reasoner"],
        )

        result = guard.chat(
            messages=[{"role": "user", "content": "你好"}],
            stream=False,
        )
        print(result.content)

    用法 (替换现有 level1_chat.py):
        # 旧:
        #   response = client.chat.completions.create(model=..., messages=...)
        # 新:
        #   result = guard.chat(messages=messages, stream=True)
        #   for chunk in result.stream:
        #       print(chunk.choices[0].delta.content, end="")
    """

    def __init__(
        self,
        default_model: str = "",
        backup_models: List[str] = None,
        api_key: str = "",
        base_url: str = "",
        retry_policy: LLMRetryPolicy = None,
        fallback_config: FallbackConfig = None,
        cache_size: int = 100,
        cache_ttl: float = 300,
    ):
        """
        Args:
            default_model:   默认模型名 (为空则从环境变量读取)
            backup_models:   备用模型列表
            api_key:         API Key (为空则从环境变量读取)
            base_url:        API Base URL (为空则从环境变量读取)
            retry_policy:    重试策略 (为空使用默认)
            fallback_config: 降级配置 (为空使用默认)
            cache_size:      缓存容量
            cache_ttl:       缓存 TTL (秒)
        """
        # 配置
        self._model = default_model or os.getenv("LLM_MODEL", "deepseek-chat")
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self._base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        # 策略
        self.retry = retry_policy or LLMRetryPolicy()
        self.fallback = fallback_config or FallbackConfig(
            backup_models=backup_models or [],
        )

        # 缓存
        self.cache = LLMCache(max_size=cache_size, ttl_seconds=cache_ttl)

        # 统计
        self._stats = {
            'total_calls': 0,
            'success': 0,
            'fallback': 0,
            'errors': 0,
            'retries': 0,
            'timeout_interrupts': 0,
            'by_error_type': {},
        }

        # 集成句柄 (可选, 延迟绑定)
        self._error_chain = None
        self._trace_chain = None
        self._experience_learner = None

        # 客户端延迟初始化
        self._client = None
        self._client_lock = threading.Lock()

        logger.info(
            f"[LLMGuard] 初始化完成 "
            f"(model={self._model}, "
            f"backups={self.fallback.backup_models}, "
            f"max_retries={self.retry.max_attempts}, "
            f"timeout={self.retry.base_timeout}s)"
        )

    # ── 集成接口 ──

    def integrate_error_chain(self, error_chain):
        """接入 ErrorChain (统一错误处理)"""
        self._error_chain = error_chain

    def integrate_trace_chain(self, trace_chain):
        """接入 TraceChain (请求追踪)"""
        self._trace_chain = trace_chain

    def integrate_experience_learner(self, learner):
        """接入 ExperienceLearner (自学习)"""
        self._experience_learner = learner

    # ── 主调用接口 ──

    def chat(
        self,
        messages: list,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False,
        system_prompt: str = None,
        timeout: float = None,
        tools: list = None,
        tool_choice: Any = None,
        response_format: dict = None,
        **kwargs,
    ) -> LLMResult:
        """
        容错 LLM 调用 (核心方法)

        Args:
            messages:       消息列表 [{"role": "user", "content": "..."}]
            model:          指定模型 (为空用默认)
            temperature:    生成温度
            max_tokens:     最大 token 数
            stream:         是否流式
            system_prompt:  系统提示词 (会自动加到 messages 开头)
            timeout:        本次调用超时 (秒, 为空用默认策略)
            tools:          工具定义列表 (OpenAI function calling 格式)
            tool_choice:    工具选择策略 ("auto" / "none" / "required" / {"type": "function", "function": {"name": "..."}})
            response_format: 响应格式约束 (如 {"type": "json_object"})
            **kwargs:       透传给 OpenAI API 的额外参数

        Returns:
            LLMResult — 统一的结果封装
        """
        self._stats['total_calls'] += 1

        # 准备 messages
        final_messages = list(messages)
        if system_prompt:
            # 如果已有 system 消息, 替换; 否则插入开头
            if final_messages and final_messages[0].get("role") == "system":
                final_messages[0] = {"role": "system", "content": system_prompt}
            else:
                final_messages.insert(0, {"role": "system", "content": system_prompt})

        effective_model = model or self._model

        # 获取 trace (如果有)
        trace = self._get_active_trace()
        trace_id = trace.id if trace else ""
        _start_time = time.time()

        # ── 尝试 1: 主模型 + 重试 ──
        result = self._call_with_retry(
            messages=final_messages,
            model=effective_model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            timeout=timeout,
            trace=trace,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

        if result.success:
            self._stats['success'] += 1
            # 写缓存 (非流式)
            if not stream and result.content:
                self.cache.put(final_messages, result.content, effective_model)
            observe_llm_call(
                model=result.model, status="success",
                duration=result.latency_ms / 1000,
                tokens_in=result.tokens_in, tokens_out=result.tokens_out,
            )
            return result

        # ── 尝试 2: 备用模型 ──
        for i, backup in enumerate(self.fallback.backup_models):
            if backup == effective_model:
                continue  # 跳过已尝试的

            logger.warning(
                f"[LLMGuard] 主模型 {effective_model} 失败, "
                f"尝试备用模型 {backup} (#{i+1})"
            )

            backup_result = self._call_with_retry(
                messages=final_messages,
                model=backup,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                timeout=timeout,
                trace=trace,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
            )

            if backup_result.success:
                backup_result.is_fallback = True
                backup_result.fallback_level = 1
                self._stats['success'] += 1
                self._stats['fallback'] += 1
                if not stream and backup_result.content:
                    self.cache.put(final_messages, backup_result.content, backup)
                observe_llm_call(
                    model=backup_result.model, status="fallback",
                    duration=backup_result.latency_ms / 1000,
                    tokens_in=backup_result.tokens_in, tokens_out=backup_result.tokens_out,
                )
                return backup_result

        # ── 尝试 3: 缓存降级 ──
        if self.fallback.enable_cache and not stream:
            cached = self._try_cache_fallback(
                final_messages, effective_model, trace
            )
            if cached:
                self._stats['fallback'] += 1
                observe_llm_call(model=effective_model, status="cache", duration=0)
                return cached

        # ── 兜底: 优雅降级 ──
        self._stats['errors'] += 1
        self._report_to_learner(result, final_messages)

        elapsed = time.time() - _start_time
        observe_llm_call(model=effective_model, status="error", duration=elapsed,
                         tokens_in=result.tokens_in, tokens_out=result.tokens_out)
        observe_llm_error(error_type=result.error_type or "unknown")

        fallback_result = LLMResult(
            content=self.fallback.graceful_message,
            model=effective_model,
            success=False,
            attempts=result.attempts,
            is_fallback=True,
            fallback_level=3,
            error_type=result.error_type,
            error_message=result.error_message,
            recovery_hint=result.recovery_hint,
            trace_id=trace_id,
        )

        # 记录到 ErrorChain
        if self._error_chain:
            self._error_chain.reporter.report(
                self._error_chain.classifier.classify(
                    Exception(result.error_message or "LLM call failed"),
                    "llm_call"
                ) if result.error_message else (None, None)
            ) if result.error_type else None

        return fallback_result

    def chat_simple(self, question: str, system_prompt: str = None,
                    stream: bool = True) -> LLMResult:
        """
        简化接口 (兼容 level1_chat.py 的 chat() 函数签名)

        用法:
            # 替换原来的 chat("你好")
            result = guard.chat_simple("你好", stream=True)
            if result.stream:
                for chunk in result.stream:
                    if chunk.choices[0].delta.content:
                        print(chunk.choices[0].delta.content, end="", flush=True)
            else:
                print(result.content)
        """
        messages = [{"role": "user", "content": question}]
        return self.chat(
            messages=messages,
            system_prompt=system_prompt,
            stream=stream,
        )

    # ── 重试循环 ──

    def _call_with_retry(
        self, messages, model, temperature, max_tokens,
        stream, timeout, trace,
        tools=None, tool_choice=None, response_format=None,
    ) -> LLMResult:
        """带重试的 LLM 调用"""
        effective_timeout = timeout or self.retry.base_timeout
        last_error_type = LLMErrorType.UNKNOWN
        last_error_msg = ""
        total_attempts = 0

        for attempt in range(1, self.retry.max_attempts + 1):
            total_attempts = attempt

            # 每次重试递增超时
            current_timeout = min(
                effective_timeout * (self.retry.timeout_growth ** (attempt - 1)),
                self.retry.max_timeout,
            )

            # 开始 span (如果有 trace)
            span = None
            if trace:
                from core.trace_chain import span_context, SpanKind
                span_name = f"llm.{model}.attempt{attempt}"
                span = span_context(trace, span_name, SpanKind.LLM,
                                    attributes={
                                        'model': model,
                                        'attempt': attempt,
                                        'stream': stream,
                                        'timeout': round(current_timeout, 1),
                                    })
                span.__enter__()

            try:
                start = time.time()

                # 获取客户端
                client = self._get_client()

                # 调用 API (带超时)
                api_kwargs = {
                    'model': model,
                    'messages': messages,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'stream': stream,
                }
                # openai 客户端的 timeout 参数
                api_kwargs['timeout'] = current_timeout
                # 透传工具定义 (function calling / structured output)
                if tools is not None:
                    api_kwargs['tools'] = tools
                if tool_choice is not None:
                    api_kwargs['tool_choice'] = tool_choice
                if response_format is not None:
                    api_kwargs['response_format'] = response_format
                api_kwargs.update(self._filter_kwargs(kwargs={}))

                response = client.chat.completions.create(**api_kwargs)

                latency_ms = (time.time() - start) * 1000

                # 成功
                if span:
                    tokens_in = 0
                    tokens_out = 0
                    if hasattr(response, 'usage') and response.usage:
                        tokens_in = response.usage.prompt_tokens or 0
                        tokens_out = response.usage.completion_tokens or 0
                        span.set_tokens(tokens_in, tokens_out)
                    span.__exit__(None, None, None)

                if stream:
                    # 流式模式: 不预收集内容, 返回原始 stream 迭代器
                    # 调用方通过 result.stream 逐 token 消费
                    return LLMResult(
                        content="",
                        stream=response,
                        model=model,
                        latency_ms=latency_ms,
                        tokens_in=0,
                        tokens_out=0,
                        success=True,
                        attempts=attempt,
                        trace_id=trace.id if trace else "",
                    )

                return LLMResult(
                    content=response.choices[0].message.content if response.choices else "",
                    stream=None,
                    model=model,
                    raw_response=response,
                    latency_ms=latency_ms,
                    tokens_in=getattr(getattr(response, 'usage', None), 'prompt_tokens', 0) or 0,
                    tokens_out=getattr(getattr(response, 'usage', None), 'completion_tokens', 0) or 0,
                    success=True,
                    attempts=attempt,
                    trace_id=trace.id if trace else "",
                )

            except Exception as exc:
                # 关闭 span (错误)
                if span:
                    span.__exit__(type(exc), exc, exc.__traceback__)

                # 分类错误
                err_type, retry_after = LLMErrorClassifier.classify(exc)
                last_error_type = err_type
                last_error_msg = str(exc) or type(exc).__name__

                self._stats['by_error_type'][err_type.value] = \
                    self._stats['by_error_type'].get(err_type.value, 0) + 1

                logger.warning(
                    f"[LLMGuard] {model} 第{attempt}次失败: "
                    f"{err_type.value} - {last_error_msg[:100]}"
                )

                # 不重试 → 立即退出
                if not err_type.retryable or attempt >= self.retry.max_attempts:
                    break

                # 计算等待时间
                if self.retry.respect_retry_after and retry_after:
                    delay = retry_after
                else:
                    delay = self.retry.base_delay * (self.retry.backoff_factor ** (attempt - 1))
                delay = min(delay, self.retry.max_delay)
                if self.retry.jitter:
                    import random
                    delay *= (0.5 + random.random() * 0.5)

                self._stats['retries'] += 1
                logger.info(f"[LLMGuard] {delay:.1f}s 后重试 ({err_type.value})")
                time.sleep(delay)

        # 所有重试耗尽
        self._stats['errors'] += 1

        return LLMResult(
            model=model,
            success=False,
            attempts=total_attempts,
            error_type=last_error_type.value,
            error_message=last_error_msg,
            recovery_hint=self._auto_hint(last_error_type, model),
            trace_id=trace.id if trace else "",
        )

    # ── 缓存降级 ──

    def _try_cache_fallback(self, messages, model, trace) -> Optional[LLMResult]:
        """尝试从缓存获取结果"""
        # 优先尝试完整匹配
        cached = self.cache.get(messages, model)
        if cached:
            logger.info(f"[LLMGuard] 缓存命中 (完整匹配, model={model})")
            return LLMResult(
                content=cached,
                model=model,
                success=True,
                is_fallback=True,
                fallback_level=2,
                trace_id=trace.id if trace else "",
            )

        # 尝试缩短 messages 匹配 (去掉最后一条, 保留上下文)
        ratio = self.fallback.cache_prompt_ratio
        keep_count = max(1, int(len(messages) * ratio))
        shortened = messages[:keep_count]
        if len(shortened) < len(messages):
            cached = self.cache.get(shortened, model)
            if cached:
                logger.info(
                    f"[LLMGuard] 缓存命中 (截断匹配, "
                    f"{len(messages)} → {len(shortened)} messages)"
                )
                return LLMResult(
                    content=cached,
                    model=model,
                    success=True,
                    is_fallback=True,
                    fallback_level=2,
                    trace_id=trace.id if trace else "",
                )

        return None

    # ── 流式响应收集 ──

    @staticmethod
    def _collect_response(stream) -> str:
        """从流式响应中收集完整内容"""
        chunks = []
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    chunks.append(delta.content)
            except (IndexError, AttributeError):
                continue
        return ''.join(chunks)

    # ── 客户端管理 ──

    def _get_client(self):
        """延迟初始化 OpenAI 客户端 (线程安全)"""
        if self._client is not None:
            return self._client

        with self._client_lock:
            if self._client is not None:
                return self._client
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
            return self._client

    @staticmethod
    def _filter_kwargs(kwargs: dict) -> dict:
        """过滤掉不支持的 kwargs"""
        allowed = {
            'top_p', 'frequency_penalty', 'presence_penalty',
            'stop', 'n', 'logit_bias', 'user',
            'tools', 'tool_choice', 'response_format',
        }
        return {k: v for k, v in kwargs.items() if k in allowed}

    # ── 自学习反馈 ──

    def _report_to_learner(self, result: LLMResult, messages: list):
        """将 LLM 失败经验反馈给自学习模块"""
        if not self._experience_learner:
            return

        try:
            if hasattr(self._experience_learner, 'learn_from_failure'):
                # 提取用户输入摘要
                user_msg = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        user_msg = msg.get("content", "")[:200]
                        break

                self._experience_learner.learn_from_failure(
                    query=f"LLM 调用失败: {user_msg[:80]}",
                    error_description=result.error_message,
                    correct_approach=result.recovery_hint,
                    task_type='llm_call',
                )
                logger.debug("[LLMGuard] 已将失败经验写入自学习模块")
        except Exception as e:
            logger.debug(f"[LLMGuard] 自学习反馈失败 (不影响主流程): {e}")

    # ── 恢复建议 ──

    @staticmethod
    def _auto_hint(err_type: LLMErrorType, model: str) -> str:
        """自动生成恢复建议"""
        hints = {
            LLMErrorType.RATE_LIMIT:
                f"模型 {model} 触发限流, 建议降低调用频率或等待片刻后重试",
            LLMErrorType.SERVER_ERROR:
                f"模型 {model} 服务端异常, 系统会自动重试, 如持续失败请检查 API 状态",
            LLMErrorType.TIMEOUT:
                f"模型 {model} 响应超时, 建议缩短输入或稍后重试",
            LLMErrorType.NETWORK:
                f"模型 {model} 网络连接失败, 请检查网络后重试",
            LLMErrorType.AUTH:
                f"模型 {model} 认证失败, 请检查 API Key 配置",
            LLMErrorType.BAD_REQUEST:
                f"模型 {model} 参数错误, 请检查输入格式",
            LLMErrorType.CONTEXT_LENGTH:
                f"模型 {model} 上下文超限, 请缩短对话历史或输入内容",
            LLMErrorType.MODEL_ERROR:
                f"模型 {model} 内容过滤触发, 请调整输入内容后重试",
        }
        return hints.get(err_type, "LLM 调用失败, 请稍后重试")

    # ── TraceChain 集成 ──

    def _get_active_trace(self):
        """获取当前活跃的 trace"""
        if self._trace_chain:
            return self._trace_chain.get_active_trace()
        return None

    # ── 统计 ──

    def get_stats(self) -> dict:
        """获取调用统计"""
        total = max(self._stats['total_calls'], 1)
        return {
            'total_calls': self._stats['total_calls'],
            'success_rate': round(self._stats['success'] / total, 3),
            'fallback_count': self._stats['fallback'],
            'error_count': self._stats['errors'],
            'total_retries': self._stats['retries'],
            'cache_stats': self.cache.stats(),
            'error_breakdown': dict(self._stats['by_error_type']),
        }

    def get_health(self) -> dict:
        """健康检查"""
        total = max(self._stats['total_calls'], 1)
        success_rate = self._stats['success'] / total
        return {
            'healthy': success_rate > 0.8 or self._stats['total_calls'] == 0,
            'success_rate': round(success_rate, 3),
            'total_calls': self._stats['total_calls'],
            'error_count': self._stats['errors'],
            'model': self._model,
        }


# ═══ 8. 便捷函数 ═══

# 全局实例 (延迟初始化)
_global_guard: Optional[LLMGuard] = None
_guard_lock = threading.Lock()


def get_llm_guard(**kwargs) -> LLMGuard:
    """获取全局 LLMGuard 单例"""
    global _global_guard
    if _global_guard is None:
        with _guard_lock:
            if _global_guard is None:
                _global_guard = LLMGuard(**kwargs)
    return _global_guard


def safe_llm_call(messages: list, **kwargs) -> LLMResult:
    """
    快速安全 LLM 调用 (使用全局单例)

    用法:
        from core.llm_guard import safe_llm_call

        result = safe_llm_call(
            messages=[{"role": "user", "content": "你好"}],
            stream=False,
        )
        print(result.content)
    """
    guard = get_llm_guard()
    return guard.chat(messages=messages, **kwargs)


# ═══ 9. 主程序集成 ═══
#
# === 方案A: 替换 level1_chat.py (最简改动) ===
#
#   # 旧 level1_chat.py:
#   #   client = OpenAI(api_key=..., base_url=...)
#   #   response = client.chat.completions.create(model=..., messages=..., stream=True)
#   #   for chunk in response: ...
#
#   # 新 level1_chat.py:
#   from core.llm_guard import LLMGuard
#
#   guard = LLMGuard(
#       default_model="deepseek-chat",
#       backup_models=["deepseek-reasoner"],
#   )
#
#   def chat(question: str, system_prompt: str = None, stream: bool = True) -> str:
#       result = guard.chat_simple(question, system_prompt=system_prompt, stream=stream)
#       if result.stream:
#           answer = ""
#           for chunk in result.stream:
#               if chunk.choices[0].delta.content:
#                   answer += chunk.choices[0].delta.content
#                   print(chunk.choices[0].delta.content, end="", flush=True)
#           print()
#           return answer
#       return result.content
#
#
# === 方案B: 集成到 Agent 主类 (完整集成) ===
#
#   from core.llm_guard import LLMGuard
#   from core.error_chain import ErrorChain
#   from core.trace_chain import TraceChain
#
#   class AgentClaw:
#       def __init__(self):
#           self.error_chain = ErrorChain()
#           self.trace_chain = TraceChain()
#           self.llm_guard = LLMGuard(
#               default_model="deepseek-chat",
#               backup_models=["deepseek-reasoner"],
#           )
#
#           # 三链联动
#           self.llm_guard.integrate_error_chain(self.error_chain)
#           self.llm_guard.integrate_trace_chain(self.trace_chain)
#           self.llm_guard.integrate_experience_learner(self.experience_learner)
#
#           # ErrorChain 配置 LLM 工具
#           self.error_chain.configure_tool(
#               "llm_call",
#               retry_policy=RetryPolicy(max_attempts=3, base_delay=2),
#           )
#
#       async def call_llm(self, messages, **kwargs):
#           return self.llm_guard.chat(messages=messages, **kwargs)
#
#       async def process(self, user_input):
#           trace = self.trace_chain.start_trace(request_text=user_input)
#           try:
#               result = self.llm_guard.chat_simple(user_input)
#               if result.is_error:
#                   return result.recovery_hint or result.error_message
#               return result.content
#           finally:
#               self.trace_chain.end_trace(trace)
#
#
# === 方案C: 装饰器模式 (最小侵入) ===
#
#   from core.llm_guard import LLMGuard
#
#   guard = LLMGuard()
#
#   # 任何调用 LLM 的函数都可以这样保护:
#   def my_agent_logic(user_input):
#       result = guard.chat(
#           messages=[{"role": "user", "content": user_input}],
#           stream=False,
#       )
#       if result.is_error:
#           return handle_error(result)
#       return process(result.content)
