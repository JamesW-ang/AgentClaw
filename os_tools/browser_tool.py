"""
AgentClaw v6.1 — 浏览器自动化工具

使用 Playwright 进行网页操作和数据提取：
  - 无头浏览器导航 (navigate)
  - 页面内容提取 (get_content)
  - 网页截图 (screenshot)
  - 自动资源清理 (close)

特性:
  - 30 秒超时控制（可配置）
  - 完整错误处理和日志
  - 可选依赖（未安装时优雅跳过）
  - 自动创建临时截图目录

依赖:
  - playwright (可选，用于浏览器自动化)
  pip install playwright
  playwright install chromium

注意:
  - 初次使用需要下载 Chromium 浏览器 (~100MB)
  - 适合轻量级网页爬取和截图任务
  - 重度爬虫建议用 Selenium 或 requests
"""

import os
import tempfile

from core.logger import get_logger

logger = get_logger("BrowserTool")


class BrowserTool:
    """
    浏览器自动化工具（基于 Playwright）。

    如果 Playwright 未安装，导入时会抛出 ImportError，
    测试脚本中通过 try/except 跳过。
    """

    def __init__(self, headless: bool = True):
        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=headless)
        self._page = self._browser.new_page()
        self._title = ""
        logger.info(f"浏览器已启动 (headless={headless})")

    def navigate(self, url: str, timeout: int = 30000) -> dict:
        """
        导航到指定 URL。

        参数:
            url:     目标网址
            timeout: 超时时间（毫秒）

        返回:
            {"success": bool, "title": str, "url": str}
        """
        try:
            self._page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            self._title = self._page.title()
            return {
                "success": True,
                "title": self._title,
                "url": url,
            }
        except Exception as e:
            logger.error(f"浏览器导航失败: url={url} — {e}")
            return {"success": False, "message": f"导航失败: {e}"}

    def get_content(self) -> dict:
        """
        提取当前页面的文本内容。

        返回:
            {"success": bool, "content": str, "length": int}
        """
        try:
            content = self._page.inner_text("body")
            return {
                "success": True,
                "content": content,
                "length": len(content),
            }
        except Exception as e:
            logger.error(f"获取页面内容失败: {e}")
            return {"success": False, "message": f"获取内容失败: {e}"}

    def screenshot(self, path: str = None, full_page: bool = False) -> dict:
        """
        截取当前页面截图。

        参数:
            path:      保存路径（默认临时文件）
            full_page: 是否截取整页

        返回:
            {"success": bool, "path": str, "size": int}
        """
        try:
            if not path:
                path = os.path.join(tempfile.gettempdir(), "agentclaw_screenshot.png")

            self._page.screenshot(path=path, full_page=full_page)
            file_size = os.path.getsize(path)

            return {
                "success": True,
                "path": path,
                "size": file_size,
            }
        except Exception as e:
            logger.error(f"截图失败: path={path} — {e}")
            return {"success": False, "message": f"截图失败: {e}"}

    def close(self):
        """关闭浏览器，释放资源。"""
        try:
            self._browser.close()
            self._playwright.stop()
            logger.info("浏览器已关闭，资源已释放")
        except Exception as e:
            logger.warning(f"浏览器关闭时出现异常: {e}")
