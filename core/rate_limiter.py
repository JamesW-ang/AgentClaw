# ============================================================
# 速率限制模块
# ============================================================
"""
实现基于令牌桶算法的速率限制系统

核心算法:
    令牌桶 (Token Bucket) 算法
    - 桶中存放令牌，初始时桶满
    - 定时补充令牌 (速率: rate 个/秒)
    - 每次调用消耗 N 个令牌
    - 令牌不足时，请求被限制

特点:
    - 支持突发流量 (容量参数控制)
    - 线程安全 (使用互斥锁)
    - 支持等待或直接返回
    - 可自定义速率和容量
    - 装饰器支持函数级限制

使用示例:
    # 创建限制器
    limiter = TokenBucket(rate=60.0, capacity=10.0)

    # 方式 1: 直接消费
    if limiter.consume(tokens=1):
        print("请求被接受")

    # 方式 2: 阻塞等待
    limiter.wait(tokens=1)

    # 方式 3: 装饰器
    @rate_limit(rpm=60)
    def my_function():
        pass
"""

import threading
import time
from collections.abc import Callable
from functools import wraps

# ============================================================
# 令牌桶算法实现
# ============================================================

class TokenBucket:
    """
    令牌桶速率限制器

    工作原理:
        1. 桶初始化时填满令牌 (capacity 个)
        2. 令牌以恒定速率补充 (rate 个/秒)
        3. 每次请求消耗 N 个令牌
        4. 若令牌不足，则请求被限制

    参数:
        rate (float): 令牌生成速率 (令牌/秒)。默认 60.0
        capacity (float): 桶的最大容量 (令牌数)。默认 10.0

    特点:
        - 线程安全: 使用互斥锁保护共享状态
        - 允许突发: 初始容量允许立即处理多个请求
        - 高精度: 使用 monotonic 时间，不受系统时钟调整影响

    性能考虑:
        - 突发容量通常设置为速率的 1/6 到 1/10
        - 例如: 60 RPM 的限制器，容量应为 10-1
    """

    def __init__(self, rate: float = 60.0, capacity: float = 10.0):
        """
        初始化令牌桶

        Args:
            rate (float): 每秒生成的令牌数
            capacity (float): 桶的最大容量 (防止令牌过度累积)
        """
        self.rate = rate              # 令牌生成速率 (令牌/秒)
        self.capacity = capacity      # 桶的最大容量 (令牌)
        self.tokens = capacity        # 当前令牌数 (初始为满)
        self.last_refill = time.monotonic()  # 最后补充令牌的时间
        self.lock = threading.Lock()  # 线程同步锁

    def consume(self, tokens: float = 1.0) -> bool:
        """
        尝试消费令牌 (非阻塞)

        算法步骤:
            1. 计算距上次补充以来的时间
            2. 根据时间和速率补充新令牌
            3. 确保令牌数不超过容量
            4. 检查是否有足够令牌
            5. 若足够，消费并返回 True
            6. 若不足，返回 False

        Args:
            tokens (float): 要消费的令牌数。默认 1.0

        Returns:
            bool:
                True - 成功消费令牌
                False - 令牌不足，消费失败

        示例:
            bucket = TokenBucket(rate=10, capacity=5)
            if bucket.consume(1):
                print("请求被接受")
            else:
                print("速率限制")
        """
        with self.lock:
            # 获取当前时间
            now = time.monotonic()

            # 计算时间间隔 (秒)
            elapsed = now - self.last_refill

            # 补充令牌: 已有令牌 + 时间内生成的令牌
            # min() 确保不超过容量
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )

            # 更新最后补充时间
            self.last_refill = now

            # 检查是否有足够令牌
            if self.tokens >= tokens:
                # 消费令牌
                self.tokens -= tokens
                return True

            # 令牌不足
            return False

    def wait(self, tokens: float = 1.0) -> None:
        """
        等待直到有足够令牌 (阻塞)

        算法:
            - 循环检查是否有足够令牌
            - 若不足，睡眠 0.1 秒后重试
            - 直到成功消费或无限等待

        Args:
            tokens (float): 要消费的令牌数。默认 1.0

        使用示例:
            limiter = TokenBucket(rate=10, capacity=5)
            limiter.wait(1)  # 阻塞直到令牌可用
            print("请求已被接受")

        注意:
            - 该方法会阻塞当前线程
            - 适合对延迟不敏感的操作
            - 生产环境建议使用 consume() 后异步重试
        """
        # 循环直到消费成功
        while not self.consume(tokens):
            # 睡眠 0.1 秒，避免忙轮询
            time.sleep(0.1)


# ============================================================
# 全局限制器实例
# ============================================================

# LLM API 请求限制器
# 配置: 60 RPM (每分钟 60 个请求)
# - rate = 1.0 (每秒 1 个令牌 = 每分钟 60 个)
# - capacity = 5 (允许瞬间突发 5 个请求)
_llm_limiter = TokenBucket(rate=1.0, capacity=5)

# 网络搜索 API 请求限制器
# 配置: 20 RPM (每分钟 20 个请求)
# - rate = 0.33 (每秒 0.33 个令牌 ≈ 每分钟 20 个)
# - capacity = 3 (允许瞬间突发 3 个请求)
_search_limiter = TokenBucket(rate=0.33, capacity=3)


# ============================================================
# 速率限制装饰器
# ============================================================

def rate_limit(
    limiter: TokenBucket | None = None,
    rpm: int = 60
):
    """
    函数级速率限制装饰器

    使用令牌桶算法限制函数的调用频率。
    装饰器会在函数执行前等待足够的令牌。

    参数:
        limiter (Optional[TokenBucket]): 预先创建的限制器实例
            - 若提供，使用该实例
            - 若为 None，根据 rpm 参数创建新实例

        rpm (int): 每分钟请求数 (requests per minute)
            - 仅在 limiter=None 时使用
            - 默认 60 RPM
            - 例如: rpm=120 表示每分钟允许 120 个请求

    返回:
        function: 装饰后的函数

    工作流程:
        1. 装饰器被应用到函数
        2. 每次调用函数时，先尝试消费令牌
        3. 若令牌不足，阻塞等待
        4. 令牌充足后，执行原函数
        5. 返回函数结果

    使用示例:
        # 方式 1: 使用 rpm 参数
        @rate_limit(rpm=60)
        def call_llm_api():
            return requests.post(...)

        # 方式 2: 使用预创建的限制器
        my_limiter = TokenBucket(rate=1.0, capacity=5)

        @rate_limit(limiter=my_limiter)
        def my_function():
            pass

    线程安全:
        - TokenBucket 内部使用互斥锁
        - 多线程环境下安全使用
        - 多个函数可共享同一个限制器
    """
    # 若未提供限制器，根据 rpm 参数创建
    if limiter is None:
        # 将 RPM 转换为 RPS (requests per second)
        rate = rpm / 60.0

        # 容量设置为速率的 1/6 (允许适度突发)
        capacity = max(rpm // 6, 1)

        # 创建新限制器实例
        limiter = TokenBucket(rate=rate, capacity=capacity)

    def decorator(func: Callable) -> Callable:
        """
        装饰器工厂函数

        Args:
            func: 被装饰的函数

        Returns:
            wrapper: 包装后的函数
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            包装函数，实现速率限制

            Args:
                *args: 原函数的位置参数
                **kwargs: 原函数的关键字参数

            Returns:
                原函数的返回值
            """
            # 阻塞等待令牌
            limiter.wait(tokens=1)

            # 令牌充足，执行原函数
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================
# 使用示例
# ============================================================

# 示例 1: 直接使用装饰器 (自动创建限制器)
# @rate_limit(rpm=60)
# def fetch_from_llm():
#     pass

# 示例 2: 使用全局限制器
# @rate_limit(limiter=_llm_limiter)
# def call_llm():
#     pass
