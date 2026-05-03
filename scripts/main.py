"""
AgentClaw v6 — 统一启动入口
Docker 容器启动时执行此文件，同时拉起 API 服务和 Web UI。

v6 变更:
    - 启动时初始化 agent_core（注册全部工具）
    - 可选启动自主学习系统
"""

import signal
import subprocess
import sys
import threading
import time

# Fail-Fast: 启动前验证配置
from core.config import validate_on_startup

validate_on_startup()

from core.logger import get_logger  # noqa: E402

logger = get_logger("main")

# 子进程引用，用于优雅退出
_processes = []


def init_agent_core():
    """初始化 AgentClaw 核心（注册全部工具 + 三链联动）"""
    logger.info("初始化 AgentClaw 核心...")
    try:
        from agent.core import get_tool_summary, init_all_tools, init_chains
        init_all_tools()
        summary = get_tool_summary()
        logger.info(f"工具初始化完成: {summary['total']} 个工具")
        for name in summary['tools']:
            logger.info(f"  - {name}")

        # Phase 1: 初始化三链联动
        error_chain, trace_chain = init_chains()
        if error_chain is not None:
            logger.info("三链联动系统已启动")
        else:
            logger.warning("三链联动未启动 (不影响主功能)")
    except Exception as e:
        logger.error(f"AgentCore 初始化失败: {e}")


def run_api_server():
    """启动 FastAPI 服务 (端口 8000)"""
    logger.info("正在启动 API Server on :8000 ...")
    proc = subprocess.Popen(
        [sys.executable, "api_server.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    _processes.append(proc)
    proc.wait()
    logger.warning("API Server 已退出 (code=%d)", proc.returncode)


def run_web_ui():
    """启动 Gradio Web UI (端口 7860)"""
    logger.info("正在启动 Web UI on :7860 ...")
    proc = subprocess.Popen(
        [sys.executable, "demo_ui.py"],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    _processes.append(proc)
    proc.wait()
    logger.warning("Web UI 已退出 (code=%d)", proc.returncode)


def shutdown(signum, frame):
    """优雅关闭所有子进程"""
    logger.info("收到退出信号，正在关闭所有服务 ...")
    for proc in _processes:
        if proc.poll() is None:
            proc.terminate()
    sys.exit(0)


def main():
    # 注册信号处理
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 初始化核心（注册全部工具）
    init_agent_core()

    logger.info("=" * 50)
    logger.info("  AgentClaw v6 启动中")
    logger.info("  API:   http://localhost:8000/docs")
    logger.info("  Web:   http://localhost:7860")
    logger.info("=" * 50)

    # 线程启动两个服务
    api_thread = threading.Thread(target=run_api_server, name="api-server", daemon=True)
    ui_thread = threading.Thread(target=run_web_ui, name="web-ui", daemon=True)

    api_thread.start()
    time.sleep(2)
    ui_thread.start()

    # 主线程等待
    try:
        while True:
            alive = [p for p in _processes if p.poll() is None]
            if not alive:
                logger.error("所有服务已退出，主进程结束")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
