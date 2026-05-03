"""
AgentClaw v6 — 统一启动入口（代理脚本）
实际实现在 scripts/main.py
"""
import os
import sys

# 确保项目根目录在 Python 路径中
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from scripts.main import main  # noqa: E402

if __name__ == "__main__":
    main()
