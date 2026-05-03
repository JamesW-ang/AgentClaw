"""健康检查端点测试"""
from unittest.mock import MagicMock, patch


class TestHealthCheck:

    def test_all_healthy_returns_200(self):
        """所有检查通过时应返回 200"""
        mock_chromadb = MagicMock()
        mock_httpx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_httpx.head.return_value = mock_resp

        mock_mem = MagicMock()
        mock_mem.percent = 45
        mock_mem.available = 8 * 1024**3

        with patch.dict("sys.modules", {"chromadb": mock_chromadb, "httpx": mock_httpx}), \
             patch("tools.health.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_mem
            from tools.health import health_check
            result = health_check()
            assert result.status_code == 200
            body = result.body.decode()
            assert "healthy" in body

    def test_chromadb_down_returns_503(self):
        """ChromaDB 不可达时应返回 503"""
        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.side_effect = Exception("connection refused")
        mock_httpx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_httpx.head.return_value = mock_resp

        mock_mem = MagicMock()
        mock_mem.percent = 45
        mock_mem.available = 8 * 1024**3

        with patch.dict("sys.modules", {"chromadb": mock_chromadb, "httpx": mock_httpx}), \
             patch("tools.health.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_mem
            from tools.health import health_check
            result = health_check()
            assert result.status_code == 503
            body = result.body.decode()
            assert "degraded" in body

    def test_high_memory_returns_warning(self):
        """内存超过 85% 应标记 warning"""
        mock_chromadb = MagicMock()
        mock_httpx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_httpx.head.return_value = mock_resp

        mock_mem = MagicMock()
        mock_mem.percent = 90
        mock_mem.available = 1 * 1024**3

        with patch.dict("sys.modules", {"chromadb": mock_chromadb, "httpx": mock_httpx}), \
             patch("tools.health.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = mock_mem
            from tools.health import health_check
            result = health_check()
            body = result.body.decode()
            assert "warning" in body
