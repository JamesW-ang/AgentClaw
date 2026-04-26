# ============================================================
# 配置验证器模块
# ============================================================
"""
集中式配置管理器，支持启动时验证和不可变单例模式

特性:
    - 环境变量自动加载
    - 必需配置验证
    - 不可变配置对象 (Frozen dataclass)
    - 灵活的属性访问接口

v6.1 变更:
    - 新增 ZHIPU_API_KEY / ZHIPU_BASE_URL（视觉/图片生成走智谱）
    - 新增 OPENAI_BASE_URL（OpenAI 兼容接口，默认指向 api.openai.com）
    - 新增 SERPAPI_KEY（web_search 三级降级需要）
"""

from dataclasses import dataclass
import os
import sys


@dataclass(frozen=True)
class _ConfigValidator:
    """
    中央化配置验证器
    
    配置项定义:
        (名称, 默认值, 是否必需, 描述)
    """
    
    _DEFINITIONS: tuple[tuple[str, str, bool, str], ...] = (
        # DeepSeek API 配置
        ("DEEPSEEK_API_KEY", "", True, "DeepSeek API Key"),
        ("DEEPSEEK_BASE_URL", "https://api.deepseek.com", False, "DeepSeek base URL"),
        
        # 智谱 Zhipu API 配置 (Vision GLM-4V / ImageGen CogView)
        ("ZHIPU_API_KEY", "", False, "Zhipu API Key"),
        ("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4", False, "Zhipu base URL"),
        
        # API 认证配置
        ("API_KEY", "", False, "API authentication key (optional)"),
        
        # LLM 模型配置
        ("LLM_MODEL", "deepseek-chat", False, "Default LLM model"),
        
        # OpenAI API 配置 (可选)
        ("OPENAI_API_KEY", "", False, "OpenAI API Key (optional)"),
        ("OPENAI_BASE_URL", "https://api.openai.com/v1", False, "OpenAI base URL"),
        
        # 视觉模型配置
        ("VISION_MODEL", "glm-4v-flash", False, "Vision model"),
        
        # SerpAPI 配置 (可选，用于 web_search)
        ("SERPAPI_KEY", "", False, "SerpAPI Key (optional, for web_search)"),
        
        # 数据库配置
        ("CHROMA_DB_PATH", "./data/chroma_db", False, "ChromaDB path"),
        
        # 服务端口配置
        ("API_PORT", "8000", False, "API port"),
        ("WEB_PORT", "7860", False, "Web UI port"),
        
        # 日志配置
        ("LOG_LEVEL", "INFO", False, "Log level"),
        ("LOG_DIR", "./data/logs", False, "Log directory"),
        
        # 工具执行配置
        ("TOOL_TIMEOUT", "30", False, "Tool timeout (sec)"),
        
        # Agent 配置
        ("REACT_MAX_ROUNDS", "5", False, "ReAct max rounds"),
    )

    def __post_init__(self):
        values = {
            name: os.getenv(name, default)
            for name, default, _, _ in self._DEFINITIONS
        }
        object.__setattr__(self, "_values", values)

    def __getattr__(self, name: str) -> str:
        values = object.__getattribute__(self, "_values")
        upper = name.upper()
        if upper in values:
            return values[upper]
        raise AttributeError(f"No config '{name}'")

    @property
    def api_port(self) -> int:
        return int(self.API_PORT)

    def validate(self) -> list[str]:
        return [
            f"X {name} -- {desc}"
            for name, default, required, desc in self._DEFINITIONS
            if required and not os.getenv(name, default)
        ]


# ============================================================
# 全局配置实例
# ============================================================

settings = _ConfigValidator()


def validate_on_startup() -> None:
    """启动时验证必需配置，缺失则退出"""
    missing = settings.validate()
    if missing:
        print("Startup FAILED: missing config", file=sys.stderr)
        for m in missing:
            print(m, file=sys.stderr)
        sys.exit(1)
