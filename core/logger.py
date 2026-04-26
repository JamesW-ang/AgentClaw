# ============================================================
# 日志管理模块
# ============================================================
"""
统一的日志系统，支持彩色控制台输出和旋转文件日志

功能:
    - 彩色控制台日志输出 (根据日志级别着色)
    - 文件日志存储 (午夜自动轮转，保留30天)
    - 自定义格式化器 (不同的控制台/文件格式)
    - 防止日志重复处理
    - 自动创建日志目录

使用方式:
    logger = get_logger(__name__)
    logger.info("message")
"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime


# ============================================================
# ANSI 颜色代码定义
# ============================================================

class _Color:
    """
    ANSI 颜色控制码
    
    用于在终端中显示彩色文本
    """
    RESET = "\033[0m"      # 重置样式
    DIM = "\033[2m"        # 暗淡 (用于时间戳和模块名)
    BLUE = "\033[34m"      # 蓝色 (DEBUG)
    GREEN = "\033[32m"     # 绿色 (INFO)
    YELLOW = "\033[33m"    # 黄色 (WARNING)
    RED = "\033[31m"       # 红色 (ERROR)


# 日志级别与颜色的映射
_LEVEL_COLORS = {
    logging.DEBUG: _Color.BLUE,
    logging.INFO: _Color.GREEN,
    logging.WARNING: _Color.YELLOW,
    logging.ERROR: _Color.RED,
}

# 文件日志中需要排除的内置属性
_RESERVED = {
    "name", "msg", "args", "created", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs",
    "message", "pathname", "process", "processName", "relativeCreated",
    "thread", "threadName", "exc_info", "exc_text", "stack_info",
    "getMessage", "getMessage"
}


# ============================================================
# 控制台日志格式化器 (带彩色)
# ============================================================

class ColorFormatter(logging.Formatter):
    """
    彩色控制台格式化器
    
    输出格式: [时间] [级别] [模块名] 消息
    
    特点:
        - 时间戳和模块名使用暗淡色
        - 日志级别根据类型着色
        - 占位符对齐，易于阅读
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录为彩色输出
        
        Args:
            record (logging.LogRecord): 日志记录对象
        
        Returns:
            str: 格式化后的彩色日志字符串
        
        示例输出:
            2026-04-24 10:30:45 INFO  [module_name] This is a message
        """
        # 提取时间戳 (格式: YYYY-MM-DD HH:MM:SS)
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        # 获取日志级别对应的颜色
        color = _LEVEL_COLORS.get(record.levelno, "")
        
        # 模块名称，左对齐到14个字符
        module = record.name[:14].ljust(14)
        
        # 日志级别，左对齐到5个字符
        level = record.levelname.ljust(5)
        
        # 组合各部分，使用颜色代码
        return (
            f"{_Color.DIM}{ts}{_Color.RESET} "
            f"{color}{level}{_Color.RESET} "
            f"{_Color.DIM}[{module}]{_Color.RESET} "
            f"{record.getMessage()}"
        )


# ============================================================
# 文件日志格式化器 (包含扩展属性)
# ============================================================

class FileFormatter(logging.Formatter):
    """
    文件日志格式化器
    
    输出格式: 时间 | 级别 | 模块 | 消息 | 扩展属性
    
    特点:
        - 包含毫秒级精度的时间戳
        - 记录所有自定义属性 (用于调试)
        - 易于解析的管道分隔符格式
        - 自动过滤内置属性
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录为文件输出
        
        Args:
            record (logging.LogRecord): 日志记录对象
        
        Returns:
            str: 格式化后的日志字符串
        
        示例输出:
            2026-04-24 10:30:45.123 INFO     | module_name | This is a message | user_id=12345 | request_id=abc
        """
        # 提取时间戳，包含毫秒 (格式: YYYY-MM-DD HH:MM:SS.mmm)
        from datetime import datetime
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 日志级别，左对齐到8个字符
        level = record.levelname.ljust(8)
        
        # 提取所有自定义属性 (非内置属性)
        extras = " | ".join(
            f"{k}={v}"
            for k, v in record.__dict__.items()
            if not k.startswith("_") and k not in _RESERVED
        )
        
        # 构建日志各部分
        parts = [
            ts,
            level,
            f"| {record.name}",
            f"| {record.getMessage()}"
        ]
        
        # 如果有扩展属性，添加到日志末尾
        if extras:
            parts.append(f"| {extras}")
        
        # 使用空格连接各部分
        return " ".join(parts)


# ============================================================
# 日志初始化标志和获取器
# ============================================================

# 全局标志：标记文件日志处理器是否已初始化
_initialized = False


def get_logger(name: str) -> logging.Logger:
    """
    获取或创建指定名称的日志记录器
    
    该函数实现了一个单例级别的日志系统：
        - 每个日志记录器只初始化一次
        - 全局只有一个文件处理器
        - 防止日志消息重复
    
    Args:
        name (str): 日志记录器名称 (通常为 __name__)
    
    Returns:
        logging.Logger: 配置好的日志记录器对象
    
    使用示例:
        logger = get_logger(__name__)
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
    """
    global _initialized
    
    # 获取或创建日志记录器
    logger = logging.getLogger(name)
    
    # 禁用事件传播，防止日志被传递给父记录器 (避免重复)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    
    # 如果日志记录器已有处理器，直接返回 (已初始化)
    if logger.handlers:
        return logger
    
    # ========== 初始化控制台处理器 ==========
    # 创建标准输出流处理器 (用于输出到终端)
    ch = logging.StreamHandler(sys.stdout)
    
    # 设置控制台日志格式化器 (带颜色)
    ch.setFormatter(ColorFormatter())
    ch.setLevel(logging.DEBUG)
    
    # 将控制台处理器添加到日志记录器
    logger.addHandler(ch)
    
    # ========== 初始化文件处理器 ==========
    # 全局只初始化一次文件处理器，避免多个旋转日志文件
    if not _initialized:
        _initialized = True
        
        # 日志目录路径
        log_dir = Path("./data/logs")
        
        # 创建日志目录 (不存在则自动创建，包括父目录)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建旋转文件处理器
        # - when="midnight": 每天午夜轮转
        # - backupCount=30: 保留30个备份文件
        # - encoding="utf-8": 使用 UTF-8 编码
        fh = TimedRotatingFileHandler(
            log_dir / "app.log",
            when="midnight",
            backupCount=30,
            encoding="utf-8"
        )
        
        # 设置文件日志格式化器
        fh.setFormatter(FileFormatter())
        fh.setLevel(logging.DEBUG)
        
        # 将文件处理器添加到根日志记录器
        # (这样所有子记录器都会输出到文件)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(fh)
    
    # 关键修复：将根logger的文件处理器也挂到子logger上
    # 因为 propagate=False，子logger的日志不会自动传到根logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, TimedRotatingFileHandler) and handler not in logger.handlers:
            logger.addHandler(handler)
    
    # 返回配置好的日志记录器
    return logger