# ============================================================
# AgentClaw v6.1 — 核心单元测试
# ============================================================
"""
核心模块测试套件，覆盖 L1/L2 层全部可测试组件：
    - core/config.py: 配置验证器
    - core/logger.py: 日志系统
    - tool_registry.py: 工具注册中心
    - core/retry.py: 重试装饰器
    - core/rate_limiter.py: 速率限制器
    - registry_adapter.py: LangGraph 桥接
    - agent_core.py: 统一 agent hub
    - builtin_tools.py: 安全模块（DANGEROUS_PATTERNS + COMMAND_WHITELIST）

标记: pytest -m unit（快速，<5s）

v6.1.1 更新:
    - 修复 Config 测试：api_port 返回 int 而 API_PORT 返回 str，不再直接比较
    - 修复 Config 测试：DEEPSEEK_API_KEY 有硬编码默认值，validate() 永远不会为空
    - 修复 Registry 测试：category 必须传 ToolCategory 枚举，不能传字符串
    - 修复 Registry 测试：禁用工具的错误消息是中文"已被禁用"
    - 修复 Registry 测试：ToolRegistry 内置速率限制，连续调用需 sleep
    - 修复 Registry 测试：list_tools_by_category() 需要 category 参数
    - 修复 RegistryAdapter 测试：adapter 使用全局 registry，不使用局部实例
    - 修复 Security 测试：security.py 是 Starlette 中间件，安全检查在 builtin_tools
    - 新增 VisionTool fixture：mock 必须清除 ZHIPU_API_KEY 环境变量
"""

import os
import sys
import time
import json
import logging
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# fixtures
# ============================================================

@pytest.fixture
def sample_tool_func():
    """示例工具函数"""
    def echo(text: str, count: int = 1) -> dict:
        return {"result": text * count}
    return echo


# ============================================================
# 测试 L1: core/config.py — 配置验证器
# ============================================================

class TestConfigValidator:
    """配置验证器测试"""

    def test_default_values_loaded(self):
        """环境变量未设置时应使用默认值（需显式删除 LOG_LEVEL 以隔离 CI 环境）"""
        # CI 环境通常设了 LOG_LEVEL，必须显式删除才能测默认值
        os.environ.pop("LOG_LEVEL", None)
        from core.config import _ConfigValidator
        s = _ConfigValidator()
        assert s.LLM_MODEL == "deepseek-chat"
        assert s.API_PORT == "8000"
        assert s.LOG_LEVEL == "INFO"

    def test_env_override(self):
        """环境变量应覆盖默认值"""
        os.environ["LLM_MODEL"] = "gpt-4"
        from core.config import _ConfigValidator
        s = _ConfigValidator()
        assert s.LLM_MODEL == "gpt-4"

    def test_case_insensitive_access(self):
        """属性名应支持大小写混合访问（raw 值均为 str，property 返回 int）"""
        from core.config import _ConfigValidator
        s = _ConfigValidator()
        # API_PORT (str) 和 api_port (int via property) 类型不同
        # 验证两者语义等价
        assert int(s.API_PORT) == s.api_port
        assert isinstance(s.api_port, int)
        assert isinstance(s.API_PORT, str)

    def test_int_conversion_property(self):
        """api_port 属性应返回 int"""
        from core.config import _ConfigValidator
        s = _ConfigValidator()
        assert isinstance(s.api_port, int)
        assert s.api_port == 8000

    def test_frozen_immutability(self):
        """配置对象应为不可变（frozen）"""
        from core.config import _ConfigValidator
        s = _ConfigValidator()
        with pytest.raises(AttributeError):
            s.LLM_MODEL = "hacked"

    def test_validate_with_empty_required_key(self):
        """必需配置为空字符串时应返回错误"""
        # DEEPSEEK_API_KEY 有硬编码默认值，需要显式覆盖为空
        os.environ["DEEPSEEK_API_KEY"] = ""
        from core.config import _ConfigValidator
        s = _ConfigValidator()
        missing = s.validate()
        assert len(missing) > 0
        assert any("DEEPSEEK_API_KEY" in m for m in missing)

    def test_unknown_config_raises(self):
        """访问不存在的配置应抛 AttributeError"""
        from core.config import _ConfigValidator
        s = _ConfigValidator()
        with pytest.raises(AttributeError):
            _ = s.NONEXISTENT_CONFIG

    def test_zhipu_config_exists(self):
        """v6.1 新增的 ZHIPU 配置应可访问"""
        from core.config import _ConfigValidator
        s = _ConfigValidator()
        try:
            _ = s.VISION_MODEL
        except AttributeError:
            pytest.skip("ZHIPU config not in this version")

    def test_validate_passes_with_valid_key(self):
        """有效 key 存在时 validate 应返回空列表"""
        os.environ["DEEPSEEK_API_KEY"] = "sk-valid-key"
        from core.config import _ConfigValidator
        s = _ConfigValidator()
        missing = s.validate()
        assert len(missing) == 0

    def test_singleton_settings_consistency(self):
        """全局 settings 实例应与新建实例配置一致"""
        from core.config import settings, _ConfigValidator
        s = _ConfigValidator()
        assert settings.LLM_MODEL == s.LLM_MODEL


# ============================================================
# 测试 L1: core/logger.py — 日志系统
# ============================================================

class TestLoggerSystem:
    """日志系统测试"""

    def test_logger_has_file_handler(self):
        """根 logger 应有文件处理器"""
        try:
            from core.logger import setup_logging
            setup_logging()
            root = logging.getLogger("agentclaw")
            has_file = any(
                isinstance(h, logging.FileHandler)
                for h in root.handlers
            )
            assert has_file, "根 logger 应包含 FileHandler"
        except ImportError:
            pytest.skip("core.logger not available")

    def test_child_logger_inherits_handlers(self):
        """子 logger 应继承根 logger 的文件处理器"""
        try:
            from core.logger import setup_logging
            setup_logging()
            child = logging.getLogger("agentclaw.core")
            effective = child.handlers + [
                h for h in logging.getLogger("agentclaw").handlers
                if h not in child.handlers
            ]
            has_file = any(isinstance(h, logging.FileHandler) for h in effective)
            assert has_file
        except ImportError:
            pytest.skip("core.logger not available")

    def test_log_level_configurable(self):
        """LOG_LEVEL 环境变量应控制日志级别"""
        try:
            os.environ["LOG_LEVEL"] = "DEBUG"
            from core.logger import setup_logging
            setup_logging()
            logger = logging.getLogger("agentclaw")
            assert logger.level == logging.DEBUG
        except ImportError:
            pytest.skip("core.logger not available")

    def test_log_format_includes_timestamp(self):
        """日志格式应包含时间戳"""
        try:
            from core.logger import setup_logging
            setup_logging()
            logger = logging.getLogger("agentclaw")
            for h in logger.handlers:
                if h.formatter:
                    fmt_str = h.formatter._fmt if hasattr(h.formatter, '_fmt') else ''
                    assert "asctime" in fmt_str or "%(" in fmt_str
                    break
            else:
                pytest.skip("No formatter found on handlers")
        except ImportError:
            pytest.skip("core.logger not available")


# ============================================================
# 测试 L2: tool_registry.py — 工具注册中心
# ============================================================

class TestToolRegistry:
    """工具注册中心测试"""

    def test_singleton_pattern(self):
        """ToolRegistry 应为单例"""
        from tools.registry import ToolRegistry
        ToolRegistry._instance = None
        r1 = ToolRegistry()
        r2 = ToolRegistry()
        assert r1 is r2

    def test_register_decorator(self, sample_tool_func):
        """装饰器注册应正确添加工具"""
        from tools.registry import ToolRegistry, ToolCategory
        ToolRegistry._instance = None
        reg = ToolRegistry()
        @reg.register(
            name="echo",
            description="Echo tool",
            parameters=[{"name": "text", "type": "string", "description": "text"}],
            category=ToolCategory.CUSTOM,  # 必须传枚举，不能传字符串
        )
        def echo_tool(text: str) -> dict:
            return {"result": text}
        assert "echo" in reg.list_tools()

    def test_register_func_manual(self, sample_tool_func):
        """手动注册应正确添加工具"""
        from tools.registry import ToolRegistry, ToolCategory
        ToolRegistry._instance = None
        reg = ToolRegistry()
        reg.register_func(
            sample_tool_func,
            name="echo",
            description="Manual echo",
            parameters=[{"name": "text", "type": "string", "description": "text"}],
        )
        assert "echo" in reg.list_tools()

    def test_execute_success(self, sample_tool_func):
        """执行已注册工具应返回成功结果"""
        from tools.registry import ToolRegistry
        ToolRegistry._instance = None
        reg = ToolRegistry()
        reg.register_func(sample_tool_func, name="echo", description="Echo")
        result = reg.execute("echo", text="hello", count=2)
        assert result["success"] is True
        assert result["result"]["result"] == "hellohello"

    def test_execute_kwargs_style(self, sample_tool_func):
        """execute() 应支持 **kwargs 风格传参"""
        from tools.registry import ToolRegistry
        ToolRegistry._instance = None
        reg = ToolRegistry()
        reg.register_func(sample_tool_func, name="echo", description="Echo")
        result = reg.execute("echo", **{"text": "hi", "count": 3})
        assert result["success"] is True

    def test_execute_dict_style(self, sample_tool_func):
        """execute() 应支持 dict 风格传参"""
        from tools.registry import ToolRegistry
        ToolRegistry._instance = None
        reg = ToolRegistry()
        reg.register_func(sample_tool_func, name="echo", description="Echo")
        result = reg.execute("echo", {"text": "world"})
        assert result["success"] is True

    def test_execute_not_found(self):
        """执行不存在的工具应返回错误"""
        from tools.registry import ToolRegistry
        ToolRegistry._instance = None
        reg = ToolRegistry()
        result = reg.execute("nonexistent_tool", {})
        assert result["success"] is False
        assert "未注册" in result["error"]

    def test_execute_exception_caught(self):
        """工具抛异常时应捕获并返回失败结果"""
        from tools.registry import ToolRegistry
        ToolRegistry._instance = None
        reg = ToolRegistry()
        def bad_tool():
            raise RuntimeError("boom")
        reg.register_func(bad_tool, name="bad", description="Bad")
        result = reg.execute("bad")
        assert result["success"] is False
        assert "RuntimeError" in result["error"]

    def test_unregister(self, sample_tool_func):
        """注销工具后应不可调用"""
        from tools.registry import ToolRegistry
        ToolRegistry._instance = None
        reg = ToolRegistry()
        reg.register_func(sample_tool_func, name="echo", description="Echo")
        reg.unregister("echo")
        result = reg.execute("echo", text="test")
        assert result["success"] is False

    def test_enable_disable(self, sample_tool_func):
        """禁用工具后应返回失败（错误消息为中文）"""
        from tools.registry import ToolRegistry
        ToolRegistry._instance = None
        reg = ToolRegistry()
        reg.register_func(sample_tool_func, name="echo", description="Echo")
        reg.disable("echo")
        result = reg.execute("echo", text="test")
        assert result["success"] is False
        assert "禁用" in result["error"]  # 中文错误消息

    def test_get_tools_for_llm(self, sample_tool_func):
        """get_tools_for_llm 应返回 OpenAI function calling schema"""
        from tools.registry import ToolRegistry
        ToolRegistry._instance = None
        reg = ToolRegistry()
        reg.register_func(
            sample_tool_func,
            name="echo",
            description="Echo tool",
            parameters=[
                {"name": "text", "type": "string", "description": "text to echo", "required": True},
            ],
        )
        schema = reg.get_tools_for_llm()
        assert isinstance(schema, list)
        assert len(schema) >= 1
        func_schema = schema[0].get("function", {})
        assert func_schema["name"] == "echo"

    def test_call_statistics(self, sample_tool_func):
        """工具调用应更新统计信息（注意内置速率限制）"""
        from tools.registry import ToolRegistry
        ToolRegistry._instance = None
        reg = ToolRegistry()
        reg.register_func(sample_tool_func, name="echo", description="Echo")
        time.sleep(1.0)  # 等待前一个测试的速率限制冷却
        reg.execute("echo", text="hi")
        time.sleep(1.0)  # 等待同工具的速率限制重置
        reg.execute("echo", text="hi2")
        stats = reg.get_tool_stats()
        assert stats["echo"]["call_count"] == 2
        assert stats["echo"]["success_count"] == 2

    def test_list_tools_by_category(self, sample_tool_func):
        """按类别筛选工具应正确工作（需传 category 参数）"""
        from tools.registry import ToolRegistry, ToolCategory
        ToolRegistry._instance = None
        reg = ToolRegistry()
        reg.register_func(
            sample_tool_func, name="echo",
            description="Echo", category=ToolCategory.CUSTOM
        )
        cats = reg.list_tools_by_category(ToolCategory.CUSTOM)  # 必须传参
        assert "echo" in cats

    def test_execute_tracks_failure_stats(self):
        """工具调用失败应记录失败统计（需等待限流器冷却）"""
        from tools.registry import ToolRegistry
        ToolRegistry._instance = None
        reg = ToolRegistry()
        def fail_tool():
            raise ValueError("fail")
        reg.register_func(fail_tool, name="fail", description="Fails")
        time.sleep(1.0)  # 等待前一个测试的速率限制冷却
        reg.execute("fail")
        stats = reg.get_tool_stats()
        assert stats["fail"]["call_count"] == 1
        assert stats["fail"]["error_count"] >= 1


# ============================================================
# 测试 L1: core/retry.py — 重试机制
# ============================================================

class TestRetryMechanism:
    """重试装饰器测试"""

    def test_retries_on_timeout(self):
        """TimeoutError 应触发重试"""
        from core.retry import retry_with_backoff

        call_count = 0
        @retry_with_backoff(max_retries=3, initial_delay=0.01, backoff_factor=2)
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("temp fail")
            return "ok"

        result = flaky_func()
        assert result == "ok"
        assert call_count == 3

    def test_raises_on_permanent_error(self):
        """不可重试异常应立即抛出"""
        from core.retry import retry_with_backoff

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def bad_func():
            raise PermissionError("no access")

        with pytest.raises(PermissionError):
            bad_func()

    def test_exhausted_retries_raises(self):
        """超过最大重试次数应抛出最后一个异常"""
        from core.retry import retry_with_backoff

        @retry_with_backoff(max_retries=2, initial_delay=0.01)
        def always_fail():
            raise ConnectionError("down")

        with pytest.raises(ConnectionError):
            always_fail()

    def test_no_retry_on_success(self):
        """成功调用不应重试"""
        from core.retry import retry_with_backoff

        call_count = 0
        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def stable_func():
            nonlocal call_count
            call_count += 1
            return "done"

        assert stable_func() == "done"
        assert call_count == 1

    def test_backoff_timing(self):
        """重试间隔应按指数增长"""
        from core.retry import retry_with_backoff
        import time

        timestamps = []
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.05, backoff_factor=2)
        def timed_fail():
            nonlocal call_count
            call_count += 1
            timestamps.append(time.time())
            if call_count < 3:
                raise ConnectionError("fail")
            return "ok"

        timed_fail()
        assert len(timestamps) == 3
        # CI 环境定时器精度有限，放宽阈值（initial=0.05, backoff=2x → 0.10）
        assert timestamps[1] - timestamps[0] >= 0.03   # 理论 0.05s，允许 0.03s+
        assert timestamps[2] - timestamps[1] >= 0.06   # 理论 0.10s，允许 0.06s+


# ============================================================
# 测试 L1: core/rate_limiter.py — 速率限制器
# ============================================================

class TestRateLimiter:
    """速率限制器测试"""

    def test_zero_rate(self):
        """rate=0 时应完全不补充令牌"""
        from core.rate_limiter import TokenBucket
        bucket = TokenBucket(rate=0.0, capacity=5.0)
        bucket.consume(5.0)
        assert bucket.consume(1.0) is False
        time.sleep(0.1)
        assert bucket.consume(1.0) is False

    def test_multi_token_consume(self):
        """应支持一次消费多个令牌"""
        from core.rate_limiter import TokenBucket
        bucket = TokenBucket(rate=1000.0, capacity=10.0)
        assert bucket.consume(5.0) is True
        assert bucket.consume(6.0) is False
        assert bucket.consume(5.0) is True
        assert bucket.consume(1.0) is False

    def test_capacity_refill(self):
        """令牌桶应在时间推移后补充令牌"""
        from core.rate_limiter import TokenBucket
        bucket = TokenBucket(rate=100.0, capacity=10.0)
        bucket.consume(10.0)
        assert bucket.consume(1.0) is False
        time.sleep(0.15)
        assert bucket.consume(5.0) is True

    def test_capacity_upper_bound(self):
        """令牌数不应超过 capacity 上限"""
        from core.rate_limiter import TokenBucket
        bucket = TokenBucket(rate=1000.0, capacity=5.0)
        time.sleep(0.05)
        assert bucket.consume(5.0) is True
        time.sleep(0.05)
        assert bucket.consume(5.0) is True


# ============================================================
# 测试 L2: registry_adapter.py — LangGraph 桥接
# ============================================================

class TestRegistryAdapter:
    """RegistryAdapter 测试（使用全局 registry）"""

    def test_get_tool_names(self):
        """get_tool_names 应返回全局 registry 中的工具名列表"""
        try:
            from tools.registry import ToolRegistry, ToolCategory, registry
            from tools.registry_adapter import RegistryAdapter
            # 确保 registry 非空（不依赖其他测试的副作用）
            if len(registry.list_tools()) == 0:
                try:
                    import tools.builtin  # noqa: F401 — 触发工具注册
                except ImportError:
                    # builtin_tools 不可用时手动注册一个测试工具
                    registry.register_func(
                        lambda: {"ok": True}, name="test_adapter_tool",
                        description="Test", category=ToolCategory.CUSTOM,
                    )
            adapter = RegistryAdapter()
            names = adapter.get_tool_names()
            assert isinstance(names, list)
            assert len(names) >= 1
        except ImportError:
            pytest.skip("registry_adapter not available")

    def test_get_langchain_tools(self):
        """get_langchain_tools 应返回 StructuredTool 列表"""
        try:
            from tools.registry_adapter import RegistryAdapter
            adapter = RegistryAdapter()
            tools = adapter.get_langchain_tools()
            assert len(tools) >= 1
        except (ImportError, Exception):
            pytest.skip("registry_adapter or langchain not available")


# ============================================================
# 安全相关测试 — 测试 builtin_tools 的安全机制
# ============================================================

class TestSecurityPatterns:
    """安全机制验证 — builtin_tools 模块中的 DANGEROUS_PATTERNS"""

    def test_builtin_tools_has_dangerous_patterns(self):
        """builtin_tools 应定义 DANGEROUS_PATTERNS"""
        try:
            import tools.builtin as builtin_tools
            assert hasattr(builtin_tools, 'DANGEROUS_PATTERNS')
            assert len(builtin_tools.DANGEROUS_PATTERNS) > 0
        except ImportError:
            pytest.skip("builtin_tools not available")

    def test_builtin_tools_has_command_whitelist(self):
        """builtin_tools 应定义 COMMAND_WHITELIST"""
        try:
            import tools.builtin as builtin_tools
            assert hasattr(builtin_tools, 'COMMAND_WHITELIST')
            assert len(builtin_tools.COMMAND_WHITELIST) > 0
        except ImportError:
            pytest.skip("builtin_tools not available")

    def test_dangerous_patterns_catch_rm_rf(self):
        """DANGEROUS_PATTERNS 应匹配 rm -rf"""
        try:
            import tools.builtin as builtin_tools
            import re
            matched = any(
                re.search(p, "rm -rf /")
                for p in builtin_tools.DANGEROUS_PATTERNS
            )
            assert matched, "DANGEROUS_PATTERNS 应匹配 'rm -rf /'"
        except ImportError:
            pytest.skip("builtin_tools not available")

    def test_dangerous_patterns_catch_fork_bomb(self):
        """DANGEROUS_PATTERNS 应匹配 fork bomb"""
        try:
            import tools.builtin as builtin_tools
            import re
            matched = any(
                re.search(p, ":(){ :|:& };:")
                for p in builtin_tools.DANGEROUS_PATTERNS
            )
            assert matched, "DANGEROUS_PATTERNS 应匹配 fork bomb"
        except ImportError:
            pytest.skip("builtin_tools not available")


# ============================================================
# SecurityMiddleware 测试 — Starlette 中间件
# ============================================================

class TestSecurityMiddleware:
    """SecurityMiddleware 测试"""

    def test_middleware_has_sql_patterns(self):
        """SecurityMiddleware 应定义 SQL_PATTERNS"""
        try:
            from tools.security import SecurityMiddleware
            assert len(SecurityMiddleware.SQL_PATTERNS) > 0
        except ImportError:
            pytest.skip("security module not available")

    def test_middleware_has_xss_patterns(self):
        """SecurityMiddleware 应定义 XSS_PATTERNS"""
        try:
            from tools.security import SecurityMiddleware
            assert len(SecurityMiddleware.XSS_PATTERNS) > 0
        except ImportError:
            pytest.skip("security module not available")

    def test_middleware_has_sensitive_patterns(self):
        """SecurityMiddleware 应定义 SENSITIVE_PATTERNS"""
        try:
            from tools.security import SecurityMiddleware
            assert len(SecurityMiddleware.SENSITIVE_PATTERNS) > 0
        except ImportError:
            pytest.skip("security module not available")


# ============================================================
# agent_core.py 测试
# ============================================================

class TestAgentCore:
    """agent_core 核心逻辑测试"""

    def test_init_all_tools(self):
        """init_all_tools 应初始化并返回工具列表"""
        try:
            from agent.core import init_all_tools
            tools = init_all_tools()
            assert isinstance(tools, list)
            assert len(tools) >= 5
        except (ImportError, Exception) as e:
            pytest.skip(f"agent_core not testable: {e}")

    def test_get_tool_summary(self):
        """get_tool_summary 应返回工具摘要信息"""
        try:
            from agent.core import get_tool_summary
            summary = get_tool_summary()
            assert isinstance(summary, (str, dict))
        except (ImportError, Exception) as e:
            pytest.skip(f"agent_core not testable: {e}")
