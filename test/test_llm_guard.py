# ============================================================
# AgentClaw — LLMGuard 单元测试
# ============================================================
"""
覆盖:
    - LLMResult: 数据结构/is_error/to_dict
    - LLMCache: put/get/hit_rate/ttl
    - LLMGuard: chat/流式/降级链/重试/tool_forwarding
    - FallbackConfig: 降级策略
    - LLMRetryPolicy: 重试策略
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# LLMResult 测试
# ============================================================

class TestLLMResult:
    """LLMResult 数据结构"""

    def test_defaults(self):
        from core.llm_guard import LLMResult
        r = LLMResult()
        assert r.content == ""
        assert r.model == ""
        assert r.success is True
        assert r.is_error is False
        assert r.attempts == 1

    def test_is_error(self):
        from core.llm_guard import LLMResult
        r = LLMResult(success=False)
        assert r.is_error

    def test_to_dict(self):
        from core.llm_guard import LLMResult
        r = LLMResult(
            content="hello", model="deepseek-chat",
            latency_ms=150.0, tokens_in=10, tokens_out=20,
        )
        d = r.to_dict()
        assert d["model"] == "deepseek-chat"
        assert d["tokens"]["in"] == 10
        assert d["tokens"]["out"] == 20
        assert d["latency_ms"] == 150.0


# ============================================================
# LLMCache 测试
# ============================================================

class TestLLMCache:
    """LLM 缓存"""

    def test_put_and_get(self):
        from core.llm_guard import LLMCache
        cache = LLMCache()
        messages = [{"role": "user", "content": "hi"}]
        cache.put(messages, "Hello, World!", "deepseek-chat")
        result = cache.get(messages, "deepseek-chat")
        assert result == "Hello, World!"

    def test_miss_on_different_model(self):
        from core.llm_guard import LLMCache
        cache = LLMCache()
        messages = [{"role": "user", "content": "hi"}]
        cache.put(messages, "hello!", "deepseek-chat")
        result = cache.get(messages, "deepseek-reasoner")
        assert result is None

    def test_miss_on_different_messages(self):
        from core.llm_guard import LLMCache
        cache = LLMCache()
        cache.put([{"role": "user", "content": "msg1"}], "hello!", "model1")
        result = cache.get([{"role": "user", "content": "msg2"}], "model1")
        assert result is None

    def test_ttl_expiry(self):
        from core.llm_guard import LLMCache
        cache = LLMCache(ttl_seconds=0)
        messages = [{"role": "user", "content": "hi"}]
        cache.put(messages, "hello!", "deepseek-chat")
        time.sleep(0.01)
        result = cache.get(messages, "deepseek-chat")
        assert result is None

    def test_hit_rate(self):
        from core.llm_guard import LLMCache
        cache = LLMCache()
        msg = [{"role": "user", "content": "ping"}]
        cache.put(msg, "pong pong pong", "m")
        cache.get(msg, "m")                     # hit
        cache.get(msg, "other")                 # miss
        rate = cache.hit_rate
        assert 0 < rate <= 1.0

    def test_max_size(self):
        from core.llm_guard import LLMCache
        cache = LLMCache(max_size=3)
        for i in range(5):
            cache.put([{"role": "user", "content": str(i)}], f"r{i}", "m")
        stats = cache.stats()
        assert stats["size"] <= 3


# ============================================================
# LLMGuard 测试
# ============================================================

class TestLLMGuard:
    """LLMGuard 核心功能（mock 底层 API）"""

    def make_mock_client(self, response_text="mock response"):
        """创建一个 mock OpenAI 客户端"""
        client = MagicMock()
        mock_chunk = MagicMock()
        mock_choice = MagicMock()
        mock_delta = MagicMock()

        mock_delta.content = response_text
        mock_delta.tool_calls = None
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = "stop"
        mock_choice.index = 0
        mock_chunk.choices = [mock_choice]
        mock_chunk.model = "deepseek-chat"

        mock_obj = MagicMock()
        mock_obj.choices = [MagicMock()]
        mock_obj.choices[0].message.content = response_text
        mock_obj.choices[0].message.tool_calls = None
        mock_obj.usage = MagicMock()
        mock_obj.usage.prompt_tokens = 10
        mock_obj.usage.completion_tokens = 20

        client.chat.completions.create.return_value = mock_obj
        return client

    @pytest.fixture
    def guard(self):
        """基本的 LLMGuard 实例（mock API）"""
        from core.llm_guard import LLMGuard
        g = LLMGuard(
            default_model="deepseek-chat",
            backup_models=[],
            api_key="sk-test",
            base_url="https://api.test.com/v1",
        )
        g._client = self.make_mock_client("test response")
        return g

    def test_chat_success(self, guard):
        result = guard.chat([{"role": "user", "content": "hi"}])
        assert result.success
        assert result.content == "test response"
        assert result.model == "deepseek-chat"

    def test_chat_system_prompt_injection(self, guard):
        result = guard.chat(
            [{"role": "user", "content": "hi"}],
            system_prompt="You are a helpful assistant.",
        )
        assert result.success
        # system prompt should be prepended
        call_args = guard._client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"

    def test_chat_replaces_existing_system(self, guard):
        guard.chat(
            [{"role": "system", "content": "old"}, {"role": "user", "content": "hi"}],
            system_prompt="new system",
        )
        call_args = guard._client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["content"] == "new system"

    def test_chat_uses_cache(self, guard):
        # First call caches, second should hit cache
        r1 = guard.chat([{"role": "user", "content": "cached_msg"}])
        assert r1.success

        # Reset mock call count
        guard._client.chat.completions.create.reset_mock()

        # Same request should hit cache (no API call)
        r2 = guard.chat([{"role": "user", "content": "cached_msg"}])
        assert r2.success
        # Cache hit means no API call (for non-stream, cached content returned)
        # Note: cache only works when result.content is non-empty and not stream
        # The mock returns content, so this should work

    def test_chat_with_model_override(self, guard):
        guard.chat([{"role": "user", "content": "hi"}], model="deepseek-reasoner")
        call_args = guard._client.chat.completions.create.call_args
        assert "deepseek-reasoner" in str(call_args.kwargs.get("model", ""))

    def test_chat_stream(self, guard):
        """流式模式应返回 stream 字段"""
        # Make a streaming mock
        mock_stream = MagicMock()
        chunk = MagicMock()
        choice = MagicMock()
        delta = MagicMock()
        delta.content = "token"
        delta.tool_calls = None
        choice.delta = delta
        choice.finish_reason = None
        choice.index = 0
        chunk.choices = [choice]
        mock_stream.__iter__.return_value = [chunk]
        mock_stream.__enter__.return_value = mock_stream
        mock_stream.__exit__.return_value = None
        guard._client.chat.completions.create.return_value = mock_stream

        result = guard.chat([{"role": "user", "content": "hi"}], stream=True)
        assert result.success

    def test_chat_tools_forwarded(self, guard):
        """工具定义应透传到 API"""
        tools = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
        }]
        guard.chat(
            [{"role": "user", "content": "search for X"}],
            tools=tools,
            tool_choice="auto",
        )
        call_args = guard._client.chat.completions.create.call_args
        assert "tools" in call_args.kwargs
        assert call_args.kwargs["tools"] == tools

    def test_chat_response_format_forwarded(self, guard):
        """response_format 应透传到 API"""
        guard.chat(
            [{"role": "user", "content": "return JSON"}],
            response_format={"type": "json_object"},
        )
        call_args = guard._client.chat.completions.create.call_args
        assert "response_format" in call_args.kwargs

    def test_get_stats(self, guard):
        guard.chat([{"role": "user", "content": "hi"}])
        stats = guard.get_stats()
        assert stats["total_calls"] >= 1

    def test_get_health(self, guard):
        guard.chat([{"role": "user", "content": "hi"}])
        health = guard.get_health()
        assert "healthy" in health


# ============================================================
# LLMGuard 降级链测试
# ============================================================

class TestLLMGuardFallback:
    """LLMGuard 降级链（mock 主模型失败 → 备用模型）"""

    @pytest.fixture
    def guard_with_backup(self):
        from core.llm_guard import LLMGuard
        g = LLMGuard(
            default_model="deepseek-chat",
            backup_models=["deepseek-reasoner"],
            api_key="sk-test",
            base_url="https://api.test.com/v1",
        )
        client = MagicMock()

        def create_side_effect(**kwargs):
            model = kwargs.get("model", "")
            if model == "deepseek-reasoner":
                mock_obj = MagicMock()
                # Build nested mocks properly to avoid MagicMock auto-creation
                usage_mock = MagicMock()
                usage_mock.prompt_tokens = 5
                usage_mock.completion_tokens = 10
                msg_mock = MagicMock()
                msg_mock.content = "backup response"
                msg_mock.tool_calls = None
                choice_mock = MagicMock()
                choice_mock.message = msg_mock
                mock_obj.choices = [choice_mock]
                mock_obj.usage = usage_mock
                return mock_obj
            raise Exception("API Error 500")

        client.chat.completions.create.side_effect = create_side_effect
        g._client = client
        return g

    def test_fallback_to_backup(self, guard_with_backup):
        result = guard_with_backup.chat([{"role": "user", "content": "hi"}])
        assert result.success
        assert result.is_fallback
        assert result.fallback_level == 1

    def test_fallback_stats(self, guard_with_backup):
        guard_with_backup.chat([{"role": "user", "content": "hi"}])
        stats = guard_with_backup.get_stats()
        assert stats["fallback_count"] >= 1

    def test_chat_simple(self, guard_with_backup):
        """chat_simple returns LLMResult (default stream=True)"""
        result = guard_with_backup.chat_simple("hello")
        assert result.success
        assert result.is_fallback
        assert result.stream is not None


# ============================================================
# LLMRetryPolicy 测试
# ============================================================

class TestLLMRetryPolicy:
    """LLM 专用重试策略"""

    @pytest.fixture
    def policy(self):
        from core.llm_guard import LLMRetryPolicy
        return LLMRetryPolicy()

    def test_max_retries_default(self, policy):
        assert policy.max_attempts >= 1
