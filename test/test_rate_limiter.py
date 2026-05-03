"""令牌桶速率限制器测试"""
import threading
import time

from core.rate_limiter import TokenBucket


class TestTokenBucket:

    def test_initial_capacity(self):
        """桶初始应为满的"""
        bucket = TokenBucket(rate=100.0, capacity=10.0)
        assert bucket.tokens == 10.0

    def test_consume_reduces_tokens(self):
        """每次消费应减少1个令牌"""
        bucket = TokenBucket(rate=100.0, capacity=10.0)
        assert bucket.consume(1.0) is True
        assert bucket.tokens == 9.0

    def test_rejects_when_empty(self):
        """令牌不足时应返回 False"""
        bucket = TokenBucket(rate=0.0, capacity=1.0)
        bucket.consume(1.0)
        assert bucket.consume(1.0) is False

    def test_refills_over_time(self):
        """令牌应随时间自动补充"""
        bucket = TokenBucket(rate=1000.0, capacity=5.0)
        bucket.consume(5.0)
        assert bucket.consume(1.0) is False
        time.sleep(0.02)
        assert bucket.consume(1.0) is True

    def test_capacity_capped(self):
        """令牌数不应超过容量上限"""
        bucket = TokenBucket(rate=10000.0, capacity=5.0)
        time.sleep(0.1)
        assert bucket.consume(1.0) is True
        assert bucket.tokens <= 5.0

    def test_wait_blocks_until_available(self):
        """wait() 应阻塞直到令牌可用"""
        bucket = TokenBucket(rate=500.0, capacity=1.0)
        bucket.consume(1.0)
        start = time.monotonic()
        bucket.wait(1.0)
        elapsed = time.monotonic() - start
        assert elapsed < 0.2
        assert bucket.tokens <= 1.0

    def test_thread_safety(self):
        """并发消费不应损坏桶状态"""
        bucket = TokenBucket(rate=10000.0, capacity=100.0)
        errors = []
        def worker():
            try:
                for _ in range(50):
                    bucket.consume(1.0)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert bucket.tokens >= 0
