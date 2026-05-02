# ============================================================
# AgentClaw — API 集成测试
# ============================================================

import os
import sys
import json
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


class AsyncIteratorMock:
    """模拟 LangGraph astream 返回的异步可迭代对象"""

    def __init__(self, items=None):
        if items is None:
            from langchain_core.messages import AIMessageChunk
            items = [
                (AIMessageChunk(content="Hello "), {"langgraph_node": "agent"}),
                (AIMessageChunk(content="World"), {"langgraph_node": "agent"}),
            ]
        self._items = items

    def __aiter__(self):
        return self._async_gen()

    async def _async_gen(self):
        for item in self._items:
            yield item


# ============================================================
# Health 端点测试
# ============================================================

class TestHealthEndpoint:

    @pytest.fixture(autouse=True)
    def _setup(self):
        """每个测试前确保 API_KEY 为空（不阻塞请求）"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key", "API_KEY": ""}):
            # Force reimport by clearing cached modules
            for mod in list(sys.modules.keys()):
                if mod.startswith("api.") or mod in ("api", "api.server"):
                    del sys.modules[mod]
            yield

    @pytest.fixture
    def client(self):
        from api.server import app
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "agent-api"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_detailed_returns_json(self, client):
        resp = client.get("/health/detailed")
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert "status" in data

    def test_health_response_time(self, client):
        start = time.time()
        client.get("/health")
        assert time.time() - start < 2.0

    def test_health_content_type(self, client):
        resp = client.get("/health")
        assert "application/json" in resp.headers["content-type"]


class TestMetricsEndpoint:

    @pytest.fixture(autouse=True)
    def _setup(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key", "API_KEY": ""}):
            for mod in list(sys.modules.keys()):
                if mod.startswith("api.") or mod in ("api", "api.server"):
                    del sys.modules[mod]
            yield

    @pytest.fixture
    def client(self):
        from api.server import app
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c

    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_metrics_contains_agent_metrics(self, client):
        resp = client.get("/metrics")
        assert "agent_requests_total" in resp.text
        assert "agent_request_duration_seconds" in resp.text

    def test_metrics_llm_metrics_present(self, client):
        resp = client.get("/metrics")
        assert "agent_llm_calls_total" in resp.text
        assert "agent_tool_calls_total" in resp.text


class TestAPIKeyMiddleware:

    @pytest.fixture
    def client(self):
        from core.config import settings
        from api.server import app
        from fastapi.testclient import TestClient
        # Patch settings._values dict in-place (frozen dataclass, can't setattr)
        with patch.dict(settings._values, {'API_KEY': 'my-secret-key'}):
            with TestClient(app) as c:
                yield c

    def test_health_skips_auth(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_metrics_accessible(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_ask_without_key_returns_401(self, client):
        resp = client.post("/ask", json={"question": "hi", "session_id": "test"})
        assert resp.status_code == 401

    def test_ask_with_invalid_key_returns_401(self, client):
        resp = client.post(
            "/ask", json={"question": "hi", "session_id": "test"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_ask_with_valid_key_passes_middleware(self, client):
        """有效 API Key 应通过中间件"""
        resp = client.post(
            "/ask", json={"question": "hi", "session_id": "test"},
            headers={"X-API-Key": "my-secret-key"},
        )
        assert resp.status_code != 401


class TestAskEndpoint:

    @pytest.fixture(autouse=True)
    def _setup(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key", "API_KEY": ""}):
            for mod in list(sys.modules.keys()):
                if mod.startswith("api.") or mod in ("api", "api.server"):
                    del sys.modules[mod]
            yield

    @pytest.fixture
    def mock_agent(self):
        agent = AsyncMock()
        agent.ainvoke.return_value = {
            "messages": [MagicMock(content="test answer")],
        }
        return agent

    @pytest.fixture
    def client(self, mock_agent):
        with patch("agent.core.get_react_agent", return_value=mock_agent):
            from api.server import app
            from fastapi.testclient import TestClient
            with TestClient(app) as c:
                yield c

    def test_ask_success(self, client):
        resp = client.post(
            "/ask", json={"question": "what is AgentClaw?", "session_id": "test-session"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "usage" in data
        assert data["usage"]["thread_id"] == "test-session"

    def test_ask_default_session_id(self, client):
        resp = client.post("/ask", json={"question": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["usage"]["thread_id"] == "default"

    def test_ask_returns_elapsed_time(self, client):
        resp = client.post("/ask", json={"question": "hello", "session_id": "s1"})
        data = resp.json()
        assert "elapsed" in data["usage"]


class TestAskStreamEndpoint:

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent where astream is an async generator yielding (chunk, metadata) tuples."""
        from langchain_core.messages import AIMessageChunk
        class MockAgent:
            async def astream(self, input_data, config, stream_mode="messages"):
                yield AIMessageChunk(content="Hello "), {"langgraph_node": "agent"}
                yield AIMessageChunk(content="World"), {"langgraph_node": "agent"}

        return MockAgent()

    @pytest.fixture
    def client(self, mock_agent):
        with patch("agent.core.get_react_agent", return_value=mock_agent), \
             patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-key", "API_KEY": ""}):
            # Clear api module cache for clean import
            for mod in list(sys.modules.keys()):
                if mod.startswith("api.") or mod in ("api", "api.server"):
                    del sys.modules[mod]
            from api.server import app
            from fastapi.testclient import TestClient
            with TestClient(app) as c:
                yield c

    def test_ask_stream_returns_sse(self, client):
        resp = client.post(
            "/ask/stream", json={"question": "hello", "session_id": "test"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("content-type") == "text/event-stream; charset=utf-8"

    def test_ask_stream_contains_tokens(self, client):
        resp = client.post(
            "/ask/stream", json={"question": "hello", "session_id": "test"},
        )
        body = resp.text
        assert "Hello" in body
        assert "World" in body
