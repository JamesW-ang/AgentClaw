# ============================================================
# Level 2 多模态: 智能输入路由器
# ============================================================
# 功能: 检测用户输入类型 (纯文本 / 文本+图片 / 纯图片)
# 特性: 自动路由到对应的处理函数
#      支持 URL / base64 / 本地路径三种图片来源识别
# ============================================================

import os
import re
import base64
from typing import Callable, Optional
from urllib.parse import urlparse

# ============================================================
# 常量定义
# ============================================================

# 允许的图片文件扩展名
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"}

# 常见图床域名模式 - 用于识别图片 URL
IMAGE_HOSTING_PATTERNS = [
    "imgur.com",
    "imgbb.com",
    "postimg.cc",
    "img1.baidu.com",
    "img2.baidu.com",  # 百度图片
    "picsum.photos",
    "unsplash.com",
    "pexels.com",
    "placekitten.com",
    "placehold.co",
    "i.imgur.com",
    "i.postimg.cc",
    "cdn.",
    "static.",
    "upload.",
    "image.",
]

# Base64 图片 data URI 正则表达式
# 匹配 data:image/{type};base64,{base64_data} 格式的字符串
BASE64_IMAGE_RE = re.compile(
    r'data:image/(png|jpe?g|webp|gif|bmp|svg\+xml);base64,[A-Za-z0-9+/=]+',
    re.IGNORECASE,
)

# ============================================================
# 多模态输入路由器类
# ============================================================


class MultimodalRouter:
    """
    多模态输入路由器 — 智能检测用户输入类型并分发到对应处理器。
    
    核心功能:
    1. 检测用户输入是否包含文本和/或图片
    2. 自动识别图片来源类型 (URL / base64 / 本地路径)
    3. 根据输入类型路由到相应的处理函数
    
    支持检测的图片来源:
    - HTTP/HTTPS URL (包含图床域名或图片扩展名)
    - Base64 编码的 data URI (格式: data:image/{type};base64,{data})
    - 本地文件路径 (以图片扩展名结尾，支持 ~ 符号)
    
    使用示例:
        router = MultimodalRouter()
        input_type = router.detect_input_type(user_input)
        result = router.route(user_input, text_handler, vision_handler)
    """

    def detect_input_type(self, user_input: str) -> str:
        """
        检测用户输入的类型。
        
        检测逻辑:
        1. 提取文本中的所有图片引用 (URL / base64 / 本地路径)
        2. 移除图片相关内容后检查是否还有有意义的文本
        3. 根据文本和图片的存在情况判断输入类型
        
        参数:
            user_input (str): 用户输入的文本
            
        返回:
            str: 检测到的输入类型
                - "text_only": 仅包含文本
                - "text_with_image": 包含文本和图片
                - "image_only": 仅包含图片
        """
        # 处理空输入
        if not user_input or not user_input.strip():
            return "text_only"
        
        # 提取文本中的所有图片
        images = self._extract_images(user_input)
        
        # 移除所有图片相关内容后，检查是否还有有意义的文本
        cleaned = self._strip_image_content(user_input)
        has_text = len(cleaned.strip()) > 0
        has_image = len(images) > 0
        
        # 根据文本和图片的存在情况返回对应的类型
        if has_image and has_text:
            return "text_with_image"
        elif has_image and not has_text:
            return "image_only"
        else:
            return "text_only"

    def route(
        self,
        user_input: str,
        text_handler: Callable,
        vision_handler: Optional[Callable] = None
    ) -> dict:
        """
        根据输入类型路由到对应的处理器。
        
        路由规则:
        - 纯文本: 调用 text_handler
        - 包含图片: 调用 vision_handler (如果未指定则使用 text_handler)
        
        参数:
            user_input (str): 用户输入的文本
            text_handler (Callable): 纯文本处理函数，接收字符串参数
            vision_handler (Callable, 可选): 视觉处理函数
                                           接收字典参数 {"text": str, "images": list}
                                           如果不提供则默认使用 text_handler
            
        返回:
            dict: 包含以下键的字典
                - input_type (str): 检测到的输入类型
                - handler_type (str): 实际调用的处理器类型 ("text" 或 "vision")
                - extracted_images (list): 提取到的图片列表
                - result: 处理器的返回值
        """
        # 检测输入类型
        input_type = self.detect_input_type(user_input)
        
        # 提取文本中的图片
        images = self._extract_images(user_input)
        
        # 如果未指定视觉处理器，则使用文本处理器
        handler = vision_handler or text_handler
        
        # 根据输入类型调用对应的处理器
        if input_type == "text_only":
            # 纯文本输入 - 调用文本处理器
            result = text_handler(user_input)
            handler_type = "text"
        elif input_type in ("text_with_image", "image_only"):
            # 包含图片的输入 - 调用视觉处理器
            # 将图片信息和文本分离后一起传给视觉处理器
            vision_input = {
                "text": self._strip_image_content(user_input),
                "images": images,
            }
            result = handler(vision_input)
            handler_type = "vision"
        else:
            # 默认情况 - 调用文本处理器
            result = text_handler(user_input)
            handler_type = "text"
        
        # 返回路由结果
        return {
            "input_type": input_type,
            "handler_type": handler_type,
            "extracted_images": images,
            "result": result,
        }

    def _extract_images(self, text: str) -> list:
        """
        从文本中提取所有图片引用 (URL / base64 / 本地路径)。
        
        提取顺序:
        1. Base64 data URI - 通过正则表达式匹配
        2. HTTP/HTTPS URL - 通过正则表达式和域名检查
        3. 本地文件路径 - 通过正则表达式和扩展名检查
        
        参数:
            text (str): 要扫描的文本
            
        返回:
            list: 提取到的图片列表，每项为字典:
                {
                    "type": "base64" | "url" | "local_path",
                    "source": 图片源
                }
        """
        images = []
        
        # 1. 检测 Base64 data URI 格式的图片
        for match in BASE64_IMAGE_RE.finditer(text):
            images.append({
                "type": "base64",
                "source": match.group()
            })
        
        # 2. 检测 HTTP/HTTPS URL
        url_pattern = re.compile(r'https?://[^\s<>"\'\)]+')
        for match in url_pattern.finditer(text):
            url = match.group()
            # 验证该 URL 确实指向图片
            if self._is_image_url(url):
                images.append({
                    "type": "url",
                    "source": url
                })
        
        # 3. 检测本地文件路径
        # 匹配 ~/, ./, ../ 或绝对路径开头的文件路径
        path_pattern = re.compile(r'(?:^|[\s,;])((?:~/|[./])[^\s<>"\'\)]+\.\w+)')
        for match in path_pattern.finditer(text):
            path = match.group(1)
            _, ext = os.path.splitext(path)
            # 验证文件扩展名是否为图片格式
            if ext.lower() in IMAGE_EXTENSIONS:
                # 转换为绝对路径 (展开 ~ 符号)
                abs_path = os.path.abspath(os.path.expanduser(path))
                images.append({
                    "type": "local_path",
                    "source": abs_path
                })
        
        return images

    def _is_image_url(self, url: str) -> bool:
        """
        判断给定的 URL 是否指向图片资源。
        
        检查方式:
        1. 检查 URL 路径的文件扩展名
        2. 检查 URL 的域名是否为已知图床域名
        
        参数:
            url (str): 要检查的 URL
            
        返回:
            bool: 如果 URL 指向图片则返回 True，否则返回 False
        """
        try:
            # 解析 URL
            parsed = urlparse(url)
            path_lower = parsed.path.lower()
            
            # 检查路径的文件扩展名
            _, ext = os.path.splitext(path_lower)
            if ext in IMAGE_EXTENSIONS:
                return True
            
            # 检查是否为已知图床域名
            domain = parsed.hostname or ""
            for pattern in IMAGE_HOSTING_PATTERNS:
                if pattern in domain:
                    return True
        except Exception:
            # 解析异常时返回 False
            pass
        
        return False

    def _strip_image_content(self, text: str) -> str:
        """
        移除文本中的所有图片引用，保留纯文本部分。
        
        移除内容:
        1. Base64 data URI
        2. 独立一行的图片 URL
        3. 本地路径（如果以图片扩展名结尾）
        
        参数:
            text (str): 原始文本
            
        返回:
            str: 移除图片引用后的纯文本
        """
        # 移除 Base64 data URI
        cleaned = BASE64_IMAGE_RE.sub("", text)
        
        # 移除独立 URL 行 (如果是指向图片的)
        lines = cleaned.split("\n")
        filtered = []
        for line in lines:
            stripped = line.strip()
            # 检查是否为 URL 行
            if stripped.startswith(("http://", "https://")):
                # 只有当该 URL 指向图片时才移除
                if not self._is_image_url(stripped):
                    filtered.append(line)
            else:
                filtered.append(line)
        
        return "\n".join(filtered).strip()


# ============================================================
# 演示 / 测试
# ============================================================

if __name__ == "__main__":
    # 打印标题
    print("=" * 60)
    print("多模态输入路由器 — 演示")
    print("=" * 60)
    
    # 初始化路由器
    router = MultimodalRouter()
    
    # 定义模拟处理器
    def text_handler(input_data):
        """纯文本处理器"""
        text = input_data if isinstance(input_data, str) else input_data.get("text", "")
        return f"[文本处理] 收到 {len(text)} 字符的文本输入"
    
    def vision_handler(input_data):
        """视觉处理器"""
        img_count = len(input_data.get("images", []))
        text = input_data.get("text", "")
        return f"[视觉处理] 检测到 {img_count} 张图片, 文本 {len(text)} 字符"
    
    # ==================== 场景 1: 纯文本输入 ====================
    print("\n[场景 1] 纯文本输入")
    text_only = "请帮我解释什么是 Transformer 架构?"
    decision = router.detect_input_type(text_only)
    print(f" 检测类型: {decision}")
    result = router.route(text_only, text_handler, vision_handler)
    print(f" 路由结果: {result['result']}")
    assert decision == "text_only"
    assert result["handler_type"] == "text"
    
    # ==================== 场景 2: 文本 + 图片 URL ====================
    print("\n[场景 2] 文本 + 图片 URL")
    text_with_image = (
        "请分析这张图片的内容:\n"
        "https://img1.baidu.com/it/u=1369931113&fm=253\n"
        "重点关注颜色和构图"
    )
    decision = router.detect_input_type(text_with_image)
    print(f" 检测类型: {decision}")
    result = router.route(text_with_image, text_handler, vision_handler)
    print(f" 路由结果: {result['result']}")
    print(f" 提取图片: {result['extracted_images']}")
    assert decision == "text_with_image"
    assert result["handler_type"] == "vision"
    
    # ==================== 场景 3: 纯图片 (base64) ====================
    print("\n[场景 3] 纯图片 (base64)")
    fake_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    decision = router.detect_input_type(fake_b64)
    print(f" 检测类型: {decision}")
    result = router.route(fake_b64, text_handler, vision_handler)
    print(f" 路由结果: {result['result']}")
    assert decision == "image_only"
    assert result["handler_type"] == "vision"
    
    # ==================== 场景 4: 本地图片路径检测 ====================
    print("\n[场景 4] 本地图片路径检测")
    local_input = "帮我看看这张图: ~/Desktop/photo.png"
    decision = router.detect_input_type(local_input)
    print(f" 检测类型: {decision}")
    result = router.route(local_input, text_handler, vision_handler)
    print(f" 路由结果: {result['result']}")
    assert decision == "text_with_image"
    
    # 演示完成
    print("\n" + "=" * 60)
    print("全部 4 个场景演示完成!")
    print("=" * 60)