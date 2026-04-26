# ============================================================
# AgentClaw v6.1 — 共享 conftest.py
# ============================================================
"""
全局 fixtures，供所有测试文件共享：
    - _reset_env: 环境变量隔离与恢复
    - _reset_registry: ToolRegistry 单例重置
    - tmp_project: 临时项目目录
    - mock_tool_registry: 模拟工具注册表（给 test_react_agent.py 用）
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# 确保 project root 在 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def _reset_env():
    """每个测试前保存环境变量，测试后恢复，避免测试间污染。

    注意：只 setdefault DEEPSEEK_API_KEY（CI 环境必须）。
    不 setdefault LOG_LEVEL 等其他变量，否则前一个测试的值
    会被 clear + update(saved) 还原，导致后续测试拿到脏值。
    """
    saved = dict(os.environ)
    os.environ.setdefault("DEEPSEEK_API_KEY", "sk-test-key-for-ci")
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def _reset_registry():
    """每个测试后重置 ToolRegistry 单例，避免跨测试状态泄露"""
    yield
    try:
        from tool_registry import ToolRegistry
        ToolRegistry._instance = None
    except ImportError:
        pass


@pytest.fixture
def tmp_project(tmp_path):
    """提供一个模拟的项目根目录结构"""
    (tmp_path / "core").mkdir()
    (tmp_path / "knowledge_base").mkdir()
    (tmp_path / "data" / "logs").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_tool_registry():
    """模拟工具注册表（供 test_react_agent.py 使用）"""
    mock_registry = MagicMock()
    mock_registry.execute.return_value = "Search result: test"
    yield mock_registry
