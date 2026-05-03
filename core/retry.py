# ============================================================
# 重试机制模块
# ============================================================
"""
实现带指数退避和抖动的重试装饰器

功能:
    - 自动捕获和重试特定异常类型
    - 指数退避策略 (delay = initial_delay * backoff_factor^attempt)
    - 抖动机制 (避免雷鸣羊群问题)
    - 可配置的重试次数和初始延迟
    - 区分可重试和不可重试的异常

异常分类:
    - 可重试异常: TimeoutError, ConnectionError, OSError
    - 不可重试异常: PermissionError, ValueError, TypeError, KeyError, SyntaxError, RuntimeError

使用示例:
    @retry_with_backoff(max_retries=3, initial_delay=1.0)
    def fetch_data(url):
        return requests.get(url).json()
"""

import functools
import random
import time

from core.logger import get_logger

logger = get_logger("Retry")


# ============================================================
# 异常分类
# ============================================================

# 可重试的异常类型 (临时性故障)
DEFAULT_RETRY = (TimeoutError, ConnectionError, OSError)

# 不可重试的异常类型 (永久性故障)
NEVER_RETRY = (
    PermissionError,    # 权限错误，不应重试
    ValueError,         # 值错误，不应重试
    TypeError,          # 类型错误，不应重试
    KeyError,           # 键错误，不应重试
    SyntaxError,        # 语法错误，不应重试
    RuntimeError,       # 运行时错误，不应重试
)


# ============================================================
# 重试装饰器
# ============================================================

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    jitter: bool = True
):
    """
    重试装饰器，支持指数退避和抖动

    工作原理:
        1. 执行目标函数
        2. 若抛出可重试异常，则进行重试
        3. 若抛出不可重试异常，立即抛出
        4. 每次重试间增加延迟 (指数退避)
        5. 可选的抖动机制 (避免多个任务同时重试)

    参数说明:
        max_retries (int): 最大重试次数。默认 3 次
        initial_delay (float): 初始延迟时间 (秒)。默认 1.0 秒
        backoff_factor (float): 退避因子。每次重试乘以该值。默认 2.0
        jitter (bool): 是否添加抖动 (随机浮动 ±25%)。默认 True

    返回:
        function: 装饰后的函数

    异常处理:
        - 可重试异常 (DEFAULT_RETRY): 自动重试，直到达到最大重试次数
        - 不可重试异常 (NEVER_RETRY): 立即抛出，不进行重试

    延迟公式:
        delay = initial_delay * (backoff_factor ^ attempt)
        if jitter:
            delay = delay * random.uniform(0.75, 1.25)

    示例:
        @retry_with_backoff(max_retries=3, initial_delay=0.5, backoff_factor=2.0)
        def unstable_operation():
            # 可能抛出 TimeoutError 或 ConnectionError 的操作
            return fetch_from_api()
    """

    def decorator(func):
        """
        装饰器工厂函数

        Args:
            func: 被装饰的目标函数

        Returns:
            wrapper: 包装后的函数
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            装饰器包装函数

            处理流程:
                1. 循环执行，最多进行 max_retries+1 次尝试
                2. 第一次失败在索引 0，最后一次在索引 max_retries
                3. 捕获异常并根据类型决定是否重试
                4. 如果需要重试，计算延迟并等待

            Args:
                *args: 目标函数的位置参数
                **kwargs: 目标函数的关键字参数

            Returns:
                目标函数的返回值

            Raises:
                不可重试的异常或达到最大重试次数后的异常
            """

            # 循环执行，最多进行 max_retries+1 次尝试
            for attempt in range(max_retries + 1):
                try:
                    # 尝试执行目标函数
                    return func(*args, **kwargs)

                except NEVER_RETRY as e:
                    # 不可重试的异常，立即抛出
                    logger.error(f"[{func.__name__}] Non-retryable error: {type(e).__name__}: {e}")
                    raise

                except DEFAULT_RETRY as e:
                    # 可重试的异常

                    # 如果已达到最大重试次数，抛出异常
                    if attempt >= max_retries:
                        logger.error(
                            f"[{func.__name__}] Exhausted {max_retries} retries after {type(e).__name__}"
                        )
                        raise

                    # 计算延迟时间 (指数退避)
                    delay = initial_delay * (backoff_factor ** attempt)

                    # 如果启用抖动，添加随机浮动 (±25%)
                    if jitter:
                        delay *= random.uniform(0.75, 1.25)

                    # 记录重试信息
                    logger.warning(
                        f"[{func.__name__}] Retry {attempt + 1}/{max_retries} "
                        f"after {type(e).__name__}, waiting {delay:.2f}s"
                    )

                    # 等待指定的延迟时间
                    time.sleep(delay)

        return wrapper

    return decorator
