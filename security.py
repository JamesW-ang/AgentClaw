# security.py - 安全中间件
import re
import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from core.logger import get_logger

logger = get_logger("Security")


class SecurityMiddleware(BaseHTTPMiddleware):
    """API安全防护中间件"""

    rate_limit = 30
    rate_window = 60
    request_counts = defaultdict(list)

    SENSITIVE_PATTERNS = [
        "password", "secret", "api_key", "token",
        "private_key", "credential", "database_url",
    ]

    SQL_PATTERNS = [
        r"(?i)SELECT\s.+FROM",
        r"(?i)DROP\s+TABLE",
        r"(?i)INSERT\s+INTO",
        r"(?i)DELETE\s+FROM",
        r"(?i)UPDATE\s+.+SET",
        r"(?i)UNION\s+SELECT",
        r"(?i);--",
        r"(?i)'\s*OR\s+'",
    ]

    XSS_PATTERNS = [
        r"<script", r"javascript:", r"on\w+\s*=",
        r"<iframe", r"<img\s+.*onerror",
    ]

    MAX_CONTENT_LENGTH = 5000

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host

        if not self._check_rate_limit(client_ip):
            logger.warning(f"频率限制触发: ip={client_ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后再试"}
            )

        if request.method == "POST":
            body = await request.body()
            body_text = body.decode("utf-8", errors="ignore")

            if len(body_text) > self.MAX_CONTENT_LENGTH:
                logger.warning(f"请求体过长: {len(body_text)}字符")
                return JSONResponse(
                    status_code=413,
                    content={"detail": "请求内容过长"}
                )

            for pattern in self.SQL_PATTERNS:
                if re.search(pattern, body_text):
                    logger.warning(f"SQL注入检测: ip={client_ip}")
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "请求包含非法字符"}
                    )

            for pattern in self.XSS_PATTERNS:
                if re.search(pattern, body_text):
                    logger.warning(f"XSS攻击检测: ip={client_ip}")
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "请求包含非法内容"}
                    )

            for pattern in self.SENSITIVE_PATTERNS:
                if pattern.lower() in body_text.lower():
                    logger.warning(f"敏感信息检测: ip={client_ip}")
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "请求包含敏感信息"}
                    )

        response = await call_next(request)
        return response

    def _check_rate_limit(self, ip: str) -> bool:
        now = time.time()
        self.request_counts[ip] = [
            t for t in self.request_counts[ip]
            if now - t < self.rate_window
        ]
        if len(self.request_counts[ip]) >= self.rate_limit:
            return False
        self.request_counts[ip].append(now)
        return True
