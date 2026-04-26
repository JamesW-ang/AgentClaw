"""
AgentClaw 多模态视觉工具 - 生产级完整版
接入真实 VLM API，支持 OpenAI 兼容接口（GPT-4o / GLM-4V / DeepSeek / Qwen-VL 等）

配置方式（二选一）:
    方式1: 在项目根目录创建 .env 文件:
        ZHIPU_API_KEY=xxx（推荐，用于 GLM-4V/CogView）
        OPENAI_API_KEY=sk-xxx（备选，用于 GPT-4o 等）

    方式2: export 环境变量
        export ZHIPU_API_KEY="xxx"
        export OPENAI_API_KEY="sk-xxx"

支持的图片输入方式:
    1. 本地文件路径: "/path/to/image.jpg"
    2. 网络URL: "https://example.com/image.png"
    3. Base64: "data:image/jpeg;base64,..."
"""

import os
import base64
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 自动加载 .env 文件（项目根目录优先，当前目录次之）
try:
    from dotenv import load_dotenv
    for env_path in [
        Path(__file__).parent / ".env",   # 脚本同级目录
        Path.cwd() / ".env",               # 当前工作目录
        Path.home() / ".env",               # 用户目录
    ]:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            break
except ImportError:
    pass  # 没有 python-dotenv 也继续运行

from dataclasses import dataclass, field
from enum import Enum

from core.logger import get_logger
logger = get_logger("VisionTool")

# ============================================================
# 配置
# ============================================================

# 从环境变量读取配置（优先智谱，回退 OpenAI）
OPENAI_API_KEY = os.environ.get("ZHIPU_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("ZHIPU_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
VISION_MODEL = os.environ.get("VISION_MODEL", "glm-4v-flash")

# 支持的图片格式
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
# 最大图片大小 20MB
MAX_IMAGE_SIZE = 20 * 1024 * 1024
# 图片编码最大分辨率（超过则缩放）
MAX_PIXELS = 4096 * 4096  # 约 1600 万像素


# ============================================================
# 数据模型
# ============================================================

@dataclass
class ImageRegion:
    """图片区域定义"""
    name: str          # 区域名称
    x: int             # 左上角 x
    y: int             # 左上角 y
    width: int         # 宽
    height: int        # 高


@dataclass
class DetectionObject:
    """检测到的对象"""
    label: str         # 对象名称
    confidence: float  # 置信度 0-1
    bbox: Optional[Dict[str, int]] = None  # 边界框 {"x":, "y":, "w":, "h":}


@dataclass
class VisionResult:
    """视觉分析结果"""
    description: str                   # 图片描述
    objects: List[DetectionObject]     # 检测到的对象
    confidence: float                  # 整体置信度
    metadata: Dict[str, Any]           # 额外元数据
    raw_response: Optional[dict]       # API 原始响应
    model: str = ""                    # 使用的模型
    latency: float = 0.0               # 耗时


# ============================================================
# 多模态视觉工具
# ============================================================

class MultiModalVisionTool:
    """
    AgentClaw 多模态视觉工具
    
    功能:
        - 单图分析（描述、对象检测、OCR）
        - 多图对比分析
        - 区域缩放分析
        - 视频帧分析
    
    用法:
        tool = MultiModalVisionTool()
        result = tool.analyze("/path/to/image.jpg")
        print(result.description)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        timeout: int = 60,
    ):
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = base_url or OPENAI_BASE_URL
        self.model = model or VISION_MODEL

        # 自动检测智谱 Key 并设置 GLM 模型默认值
        if self.api_key and not model:
            zhipu_key = os.environ.get("ZHIPU_API_KEY", "")
            if zhipu_key and self.api_key.startswith(zhipu_key[:8]):
                if not os.environ.get("VISION_MODEL"):
                    self.model = "glm-4v-flash"
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None
        self._call_count = 0
        self._total_latency = 0.0

    @property
    def is_configured(self) -> bool:
        """检查是否已配置 API Key"""
        return bool(self.api_key)

    @property
    def avg_latency(self) -> float:
        return self._total_latency / max(self._call_count, 1)

    def _get_client(self):
        """懒加载 OpenAI 客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
                logger.info(f"OpenAI 客户端初始化成功 (model={self.model})")
            except ImportError:
                raise ImportError(
                    "需要安装 openai 库: pip install openai\n"
                    "如果不需要真实 API 调用，可使用 mock 模式测试。"
                )
        return self._client

    # ----------------------------------------------------------
    # 图片预处理
    # ----------------------------------------------------------

    def _load_image(self, source: str) -> Tuple[str, str]:
        """
        加载图片，返回 (media_type, base64_data)
        
        Args:
            source: 本地路径 / URL / base64
        
        Returns:
            (media_type, base64_data)
        """
        # 情况1: 已经是 base64 data URI
        if source.startswith("data:image/"):
            parts = source.split(";base64,", 1)
            media_type = parts[0].replace("data:", "")
            return media_type, parts[1]
        
        # 情况2: URL
        if source.startswith("http://") or source.startswith("https://"):
            return self._download_image(source)
        
        # 情况3: 本地文件
        if os.path.isfile(source):
            return self._encode_local_image(source)
        
        # 情况4: 纯 base64 字符串
        try:
            decoded = base64.b64decode(source)
            return "image/jpeg", base64.b64encode(decoded).decode("utf-8")
        except Exception:
            pass
        
        raise ValueError(f"无法识别的图片来源: {source[:100]}")

    def _encode_local_image(self, file_path: str) -> Tuple[str, str]:
        """编码本地图片为 base64"""
        path = Path(file_path)
        
        # 检查格式
        ext = path.suffix.lower()
        if ext not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(f"不支持的图片格式: {ext}，支持: {SUPPORTED_IMAGE_FORMATS}")
        
        # 检查大小
        file_size = path.stat().st_size
        if file_size > MAX_IMAGE_SIZE:
            raise ValueError(f"图片过大: {file_size / 1024 / 1024:.1f}MB（最大 {MAX_IMAGE_SIZE / 1024 / 1024:.0f}MB）")
        
        # MIME 类型
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp", ".bmp": "image/bmp",
        }
        media_type = mime_map.get(ext, "image/jpeg")
        
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        
        logger.info(f"图片编码成功: {path.name} ({file_size / 1024:.1f}KB)")
        return media_type, data

    def _download_image(self, url: str) -> Tuple[str, str]:
        """下载网络图片"""
        import requests
        resp = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 AgentClaw/1.0"
        })
        resp.raise_for_status()
        
        content_type = resp.headers.get("Content-Type", "image/jpeg")
        media_type = content_type.split(";")[0].strip()
        
        data = base64.b64encode(resp.content).decode("utf-8")
        logger.info(f"图片下载成功: {url[:60]}... ({len(resp.content) / 1024:.1f}KB)")
        return media_type, data

    @property
    def _is_glm_model(self) -> bool:
        """判断是否是智谱 GLM 模型（API 格式与 OpenAI 有差异）"""
        return "glm" in self.model.lower()

    def _build_image_content(self, source: str, detail: str = "auto") -> dict:
        """构建 vision API 的 image content（自动适配 GLM/OpenAI 格式）"""
        media_type, b64_data = self._load_image(source)
        # 智谱 GLM-4V API 同样要求 data URI 前缀
        image_url_entry = {"url": f"data:{media_type};base64,{b64_data}"}
        # detail 参数是 OpenAI 专属，智谱 GLM 不支持，会导致 1210 错误
        if detail != "auto" and not self._is_glm_model:
            image_url_entry["detail"] = detail
        return {
            "type": "image_url",
            "image_url": image_url_entry
        }

    def _build_image_content_b64(self, media_type: str, b64_data: str) -> dict:
        """从已有的 base64 数据构建 image content（适配 GLM）"""
        image_url_entry = {"url": f"data:{media_type};base64,{b64_data}"}
        # detail 参数是 OpenAI 专属，GLM 不支持
        if not self._is_glm_model:
            image_url_entry["detail"] = "high"
        return {"type": "image_url", "image_url": image_url_entry}

    def _crop_image_b64(self, source: str, region: ImageRegion) -> Tuple[str, str]:
        """裁剪图片区域，返回 (media_type, base64_data)"""
        try:
            from PIL import Image
            import io
            
            media_type, b64_data = self._load_image(source)
            img_data = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_data))
            
            # 裁剪
            cropped = img.crop((region.x, region.y, region.x + region.width, region.y + region.height))
            
            # 保存到内存
            buf = io.BytesIO()
            output_format = "JPEG" if media_type in ("image/jpeg",) else "PNG"
            cropped.save(buf, format=output_format)
            result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            out_media = "image/jpeg" if output_format == "JPEG" else "image/png"
            
            logger.info(f"裁剪区域 [{region.name}] {region.width}x{region.height}px")
            return out_media, result_b64
            
        except ImportError:
            # 没有 PIL，返回原图并提示
            logger.warning("PIL 未安装，无法裁剪图片。安装: pip install Pillow")
            return self._load_image(source)

    # ----------------------------------------------------------
    # 核心 API 调用
    # ----------------------------------------------------------

    def _call_vision_api(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """
        调用 OpenAI 兼容的视觉 API
        
        Returns:
            {"content": str, "usage": dict, "model": str}
        """
        client = self._get_client()
        
        start_time = time.time()
        # 构建请求参数（GLM 与 OpenAI 差异处理）
        kwargs = {
            "model": self.model,
            "messages": messages,
        }
        # GLM 模型参数适配：
        #   - max_tokens 上限 1024（glm-4v-flash 限制）
        #   - 不传 temperature（GLM 不支持或行为不同）
        if self._is_glm_model:
            kwargs["max_tokens"] = min(max_tokens or self.max_tokens, 1024)
        else:
            kwargs["max_tokens"] = max_tokens or self.max_tokens
            kwargs["temperature"] = temperature
        logger.info(f"API 请求: model={self.model}, max_tokens={kwargs.get('max_tokens')}, is_glm={self._is_glm_model}")
        response = client.chat.completions.create(**kwargs)
        latency = time.time() - start_time
        
        self._call_count += 1
        self._total_latency += latency
        
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
        
        logger.info(f"API 调用完成 ({latency:.2f}s, {usage['total_tokens']} tokens)")
        
        return {
            "content": content,
            "usage": usage,
            "model": response.model,
            "latency": round(latency, 3),
        }

    def _parse_json_response(self, text: str) -> dict:
        """尝试从 LLM 响应中提取 JSON"""
        # 尝试提取 ```json ... ``` 代码块
        import re
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取花括号内容
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass
        
        return {"raw": text}

    # ----------------------------------------------------------
    # 公共分析方法
    # ----------------------------------------------------------

    def analyze(
        self,
        image_source: str,
        prompt: Optional[str] = None,
        detail_level: str = "high",
    ) -> VisionResult:
        """
        分析单张图片
        
        Args:
            image_source: 图片路径/URL/base64
            prompt: 自定义分析提示词（默认自动生成）
            detail_level: "high"（详细）或 "low"（快速）
        
        Returns:
            VisionResult
        """
        if not self.is_configured:
            return self._mock_analyze(image_source)
        
        if prompt is None:
            prompt = (
                "请详细分析这张图片，用中文回答。请包括以下内容：\n"
                "1. 整体描述（场景、主题、风格）\n"
                "2. 关键对象列表（名称和位置）\n"
                "3. 文字内容（如有）\n"
                "4. 颜色和构图\n"
                "5. 特殊细节或异常\n\n"
                "请用以下 JSON 格式返回：\n"
                '{\n'
                '  "description": "整体描述",\n'
                '  "objects": [{"label": "对象名", "confidence": 0.9, "bbox": {"x":0,"y":0,"w":100,"h":100}}],\n'
                '  "text_content": "图片中的文字",\n'
                '  "colors": ["主色调"],\n'
                '  "style": "图片风格",\n'
                '  "confidence": 0.9\n'
                '}'
            )
        
        start_time = time.time()
        
        try:
            image_content = self._build_image_content(image_source, detail=detail_level)
            
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的图像分析助手。请准确、详细地分析图片内容，返回 JSON 格式的结构化结果。"
                },
                {
                    "role": "user",
                    "content": [image_content, {"type": "text", "text": prompt}]
                }
            ]
            
            api_result = self._call_vision_api(messages)
            parsed = self._parse_json_response(api_result["content"])
            
            # 构建 DetectionObject 列表
            objects = []
            for obj in parsed.get("objects", []):
                objects.append(DetectionObject(
                    label=obj.get("label", "unknown"),
                    confidence=float(obj.get("confidence", 0.0)),
                    bbox=obj.get("bbox"),
                ))
            
            return VisionResult(
                description=parsed.get("description", api_result["content"]),
                objects=objects,
                confidence=float(parsed.get("confidence", 0.8)),
                metadata={
                    "text_content": parsed.get("text_content", ""),
                    "colors": parsed.get("colors", []),
                    "style": parsed.get("style", ""),
                },
                raw_response=api_result,
                model=api_result["model"],
                latency=round(time.time() - start_time, 3),
            )
            
        except Exception as e:
            logger.error(f"图片分析失败: {e}")
            return VisionResult(
                description=f"分析失败: {e}",
                objects=[],
                confidence=0.0,
                metadata={"error": str(e)},
                raw_response=None,
                model=self.model,
                latency=round(time.time() - start_time, 3),
            )

    def compare(
        self,
        image_sources: List[str],
        prompt: Optional[str] = None,
    ) -> VisionResult:
        """
        对比分析多张图片
        
        Args:
            image_sources: 图片路径/URL 列表
            prompt: 自定义对比提示词
        
        Returns:
            VisionResult
        """
        if not self.is_configured:
            return self._mock_compare(image_sources)
        
        if prompt is None:
            prompt = (
                f"请对比这 {len(image_sources)} 张图片，用中文回答：\n"
                "1. 每张图片的简要描述\n"
                "2. 它们之间的主要区别\n"
                "3. 它们之间的共同点\n"
                "4. 总体评价\n\n"
                "请用以下 JSON 格式返回：\n"
                '{\n'
                '  "descriptions": ["图1描述", "图2描述", ...],\n'
                '  "differences": ["区别1", "区别2", ...],\n'
                '  "similarities": ["共同点1", "共同点2", ...],\n'
                '  "summary": "总体评价"\n'
                '}'
            )
        
        start_time = time.time()
        
        try:
            content_parts = []
            for i, src in enumerate(image_sources, 1):
                content_parts.append(self._build_image_content(src))
                content_parts.append({
                    "type": "text",
                    "text": f"[图片 {i}]"
                })
            content_parts.append({"type": "text", "text": prompt})
            
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的图像对比分析助手。请仔细对比图片之间的异同。"
                },
                {
                    "role": "user",
                    "content": content_parts
                }
            ]
            
            api_result = self._call_vision_api(messages)
            parsed = self._parse_json_response(api_result["content"])
            
            description = parsed.get("summary", api_result["content"])
            if parsed.get("differences"):
                description = "区别: " + "; ".join(parsed["differences"][:5]) + "\n" + description
            
            return VisionResult(
                description=description,
                objects=[],
                confidence=0.85,
                metadata={
                    "descriptions": parsed.get("descriptions", []),
                    "differences": parsed.get("differences", []),
                    "similarities": parsed.get("similarities", []),
                },
                raw_response=api_result,
                model=api_result["model"],
                latency=round(time.time() - start_time, 3),
            )
            
        except Exception as e:
            logger.error(f"图片对比失败: {e}")
            return VisionResult(
                description=f"对比失败: {e}",
                objects=[],
                confidence=0.0,
                metadata={"error": str(e)},
                raw_response=None,
                model=self.model,
                latency=round(time.time() - start_time, 3),
            )

    def analyze_regions(
        self,
        image_source: str,
        regions: List[ImageRegion],
        prompt: Optional[str] = None,
    ) -> Dict[str, VisionResult]:
        """
        分区域分析图片（缩放查看细节）
        
        Args:
            image_source: 图片路径/URL
            regions: 区域定义列表
            prompt: 自定义分析提示词
        
        Returns:
            {区域名称: VisionResult}
        """
        if prompt is None:
            prompt = "请详细描述这个区域的细节内容，包括文字、图标、颜色、布局等。"
        
        start_time = time.time()
        results = {}
        
        for region in regions:
            try:
                # 裁剪区域
                media_type, b64_data = self._crop_image_b64(image_source, region)
                
                if self.is_configured:
                    # 真实 API 分析
                    image_content = self._build_image_content_b64(media_type, b64_data)
                    messages = [
                        {"role": "system", "content": "你是一个专业的图像细节分析助手。"},
                        {"role": "user", "content": [
                            image_content,
                            {"type": "text", "text": f"[区域: {region.name} ({region.width}x{region.height}px)]\n{prompt}"}
                        ]}
                    ]
                    api_result = self._call_vision_api(messages)
                    
                    results[region.name] = VisionResult(
                        description=api_result["content"],
                        objects=[],
                        confidence=0.85,
                        metadata={"region": {"x": region.x, "y": region.y, "w": region.width, "h": region.height}},
                        raw_response=api_result,
                        model=api_result["model"],
                    )
                else:
                    results[region.name] = VisionResult(
                        description=f"[Mock] 区域 {region.name} 的分析结果",
                        objects=[],
                        confidence=0.5,
                        metadata={"region": {"x": region.x, "y": region.y, "w": region.width, "h": region.height}},
                        raw_response=None,
                    )
                    
            except Exception as e:
                results[region.name] = VisionResult(
                    description=f"分析失败: {e}",
                    objects=[],
                    confidence=0.0,
                    metadata={"error": str(e)},
                    raw_response=None,
                )
        
        logger.info(f"区域分析完成: {len(results)}/{len(regions)} 个区域 ({time.time() - start_time:.2f}s)")
        return results

    def ocr(
        self,
        image_source: str,
        languages: Optional[List[str]] = None,
    ) -> VisionResult:
        """
        OCR 文字识别
        
        Args:
            image_source: 图片路径/URL
            languages: 语言提示（如 ["中文", "英文"]）
        
        Returns:
            VisionResult（description 中包含识别的文字）
        """
        lang_hint = ""
        if languages:
            lang_hint = f"图片中可能包含以下语言: {', '.join(languages)}。\n"
        
        prompt = (
            f"{lang_hint}"
            "请仔细识别这张图片中的所有文字内容。\n"
            "请保持原文的格式、大小写和标点符号。\n"
            "按从上到下、从左到右的顺序排列。\n\n"
            "请用以下 JSON 格式返回：\n"
            '{\n'
            '  "full_text": "完整文字内容",\n'
            '  "blocks": [{"text": "文本块", "region": "位置描述"}],\n'
            '  "languages": ["检测到的语言"],\n'
            '  "confidence": 0.95\n'
            '}'
        )
        
        result = self.analyze(image_source, prompt=prompt)
        
        # 如果 metadata 里有 parsed 数据，提取文字
        if result.raw_response:
            parsed = self._parse_json_response(result.raw_response["content"])
            if "full_text" in parsed:
                result.description = parsed["full_text"]
                result.metadata["blocks"] = parsed.get("blocks", [])
                result.metadata["languages"] = parsed.get("languages", [])
        
        return result

    def analyze_video_frame(
        self,
        frame_source: str,
        frame_index: int = 0,
        context: str = "",
    ) -> VisionResult:
        """
        分析视频帧
        
        Args:
            frame_source: 帧图片路径/URL
            frame_index: 帧序号
            context: 上下文描述（如视频主题）
        
        Returns:
            VisionResult
        """
        prompt = (
            f"这是视频的第 {frame_index} 帧。"
            f"{f'视频主题: {context}' if context else ''}\n"
            "请分析这一帧的内容，包括场景、人物动作、UI 元素等。"
        )
        return self.analyze(frame_source, prompt=prompt)

    # ----------------------------------------------------------
    # Mock 模式（未配置 API Key 时使用）
    # ----------------------------------------------------------

    def _mock_analyze(self, image_source: str) -> VisionResult:
        """Mock 单图分析"""
        return VisionResult(
            description=f"[Mock] 这是对 {image_source} 的分析（未配置 API Key，使用模拟数据）",
            objects=[
                DetectionObject(label="object1", confidence=0.9),
                DetectionObject(label="object2", confidence=0.8),
            ],
            confidence=0.9,
            metadata={"mode": "mock", "image": image_source},
            raw_response=None,
            model="mock",
        )

    def _mock_compare(self, sources: List[str]) -> VisionResult:
        """Mock 多图对比"""
        return VisionResult(
            description=f"[Mock] 对比 {len(sources)} 张图片（未配置 API Key）",
            objects=[],
            confidence=0.7,
            metadata={"mode": "mock", "images": sources},
            raw_response=None,
            model="mock",
        )

    # ----------------------------------------------------------
    # 状态信息
    # ----------------------------------------------------------

    def get_stats(self) -> dict:
        """获取工具统计信息"""
        return {
            "model": self.model,
            "configured": self.is_configured,
            "call_count": self._call_count,
            "avg_latency": f"{self.avg_latency:.3f}s",
            "base_url": self.base_url,
        }


# ============================================================
# 注册为 AgentClaw 内置工具（通过 tool_registry）
# ============================================================

def create_vision_tool_instance() -> MultiModalVisionTool:
    """创建全局视觉工具实例"""
    return MultiModalVisionTool()

# 全局实例
_vision_tool: Optional[MultiModalVisionTool] = None

def get_vision_tool() -> MultiModalVisionTool:
    """获取全局视觉工具实例"""
    global _vision_tool
    if _vision_tool is None:
        _vision_tool = MultiModalVisionTool()
    return _vision_tool


# ============================================================
# AgentClaw 工具注册（可选，需要 tool_registry.py）
# ============================================================

try:
    from tool_registry import registry, ToolCategory
    
    def _vision_analyze(image_path: str, prompt: str = "") -> dict:
        """分析图片"""
        tool = get_vision_tool()
        result = tool.analyze(image_path, prompt=prompt if prompt else None)
        return {
            "description": result.description,
            "objects": [{"label": o.label, "confidence": o.confidence} for o in result.objects],
            "confidence": result.confidence,
            "latency": result.latency,
        }
    
    def _vision_ocr(image_path: str, languages: str = "中文,英文") -> dict:
        """OCR 识别"""
        tool = get_vision_tool()
        result = tool.ocr(image_path, languages=languages.split(","))
        return {
            "text": result.description,
            "confidence": result.confidence,
            "blocks": result.metadata.get("blocks", []),
        }
    
    def _vision_compare(image_paths: str, prompt: str = "") -> dict:
        """对比多张图片"""
        tool = get_vision_tool()
        paths = [p.strip() for p in image_paths.split(",")]
        result = tool.compare(paths, prompt=prompt if prompt else None)
        return {
            "description": result.description,
            "differences": result.metadata.get("differences", []),
            "similarities": result.metadata.get("similarities", []),
        }
    
    registry.register_func(
        _vision_analyze,
        name="vision_analyze",
        description="分析图片内容，返回描述、检测对象和置信度。支持本地文件和URL。",
        category=ToolCategory.CUSTOM,
    )
    
    registry.register_func(
        _vision_ocr,
        name="vision_ocr",
        description="OCR 文字识别，从图片中提取文字内容。",
        category=ToolCategory.CUSTOM,
    )
    
    registry.register_func(
        _vision_compare,
        name="vision_compare",
        description="对比多张图片的差异（图片路径用逗号分隔）。",
        category=ToolCategory.CUSTOM,
    )
    
    logger.info("视觉工具已注册到 AgentClaw ToolRegistry")
    
except ImportError:
    logger.info("tool_registry 未找到，视觉工具以独立模式运行（不注册到 AgentClaw）")


# ============================================================
# 主测试程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AgentClaw 多模态视觉工具 - 生产级测试")
    print("=" * 60)
    
    tool = MultiModalVisionTool()
    
    # 显示配置状态
    stats = tool.get_stats()
    print(f"\n配置状态:")
    print(f"  API Key: {'已配置 ✅' if stats['configured'] else '未配置 ⚠️'}")
    print(f"  模型: {stats['model']}")
    print(f"  Base URL: {stats['base_url']}")
    print(f"  模式: {'真实 API' if stats['configured'] else 'Mock 模拟'}")
    
    if not stats['configured']:
        print("\n⚠️  未检测到 OPENAI_API_KEY 环境变量")
        print("   以下测试将使用 Mock 模拟数据")
        print("   配置方法:")
        print("   export OPENAI_API_KEY='sk-your-key'")
        print("   export OPENAI_BASE_URL='https://api.openai.com/v1'  # 可选")
        print("   export VISION_MODEL='gpt-4o'                        # 可选")
    
    # ----------------------------------------------------------
    # 测试 1: 单图分析
    # ----------------------------------------------------------
    print("\n" + "-" * 40)
    print("测试 1: 单图分析")
    
    # 用 AI 生成一张测试图片
    test_image = "/tmp/agentclaw_test_scene.png"
    if not os.path.exists(test_image):
        print("  正在生成测试图片...")
        try:
            import subprocess
            result = subprocess.run([
                "z-ai-generate", "-p",
                "A clean technology workspace with a laptop showing code, "
                "a cup of coffee, and a small plant on a wooden desk, "
                "soft lighting, photorealistic",
                "-o", test_image, "-s", "1024x1024"
            ], capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"  图片生成失败，使用 Mock 模式")
                test_image = None
        except FileNotFoundError:
            print("  z-ai-generate 未找到，跳过真实图片测试")
            test_image = None
        except subprocess.TimeoutExpired:
            print("  图片生成超时，跳过真实图片测试")
            test_image = None
    
    if test_image and os.path.exists(test_image):
        print(f"  分析图片: {test_image}")
        result = tool.analyze(test_image)
        print(f"  描述: {result.description[:200]}")
        print(f"  置信度: {result.confidence}")
        print(f"  对象: {[o.label for o in result.objects]}")
        print(f"  耗时: {result.latency}s")
        if result.model != "mock":
            print(f"  模型: {result.model}")
    else:
        result = tool.analyze("test_image.jpg")
        print(f"  [Mock] {result.description}")
    
    # ----------------------------------------------------------
    # 测试 2: OCR
    # ----------------------------------------------------------
    print("\n" + "-" * 40)
    print("测试 2: OCR 文字识别")
    
    test_ocr_image = "/tmp/agentclaw_test_text.png"
    if not os.path.exists(test_ocr_image):
        try:
            result = subprocess.run([
                "z-ai-generate", "-p",
                "A white card with black text: 'Hello AgentClaw!' "
                "and below it '多模态视觉工具 v2.0', clean design",
                "-o", test_ocr_image, "-s", "1024x1024"
            ], capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                test_ocr_image = None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            test_ocr_image = None
    
    if test_ocr_image and os.path.exists(test_ocr_image):
        result = tool.ocr(test_ocr_image)
        print(f"  识别文字: {result.description[:200]}")
        print(f"  置信度: {result.confidence}")
    else:
        result = tool.ocr("test_text.jpg")
        print(f"  [Mock] {result.description}")
    
    # ----------------------------------------------------------
    # 测试 3: 多图对比
    # ----------------------------------------------------------
    print("\n" + "-" * 40)
    print("测试 3: 多图对比")
    
    result = tool.compare(["image_a.jpg", "image_b.jpg"])
    print(f"  结果: {result.description[:200]}")
    
    # ----------------------------------------------------------
    # 测试 4: 区域分析
    # ----------------------------------------------------------
    print("\n" + "-" * 40)
    print("测试 4: 区域缩放分析")
    
    regions = [
        ImageRegion(name="左上角", x=0, y=0, width=400, height=400),
        ImageRegion(name="右下角", x=624, y=624, width=400, height=400),
    ]
    
    if test_image and os.path.exists(test_image):
        results = tool.analyze_regions(test_image, regions)
        for name, r in results.items():
            print(f"  [{name}]: {r.description[:100]}")
    else:
        results = tool.analyze_regions("demo.jpg", regions)
        for name, r in results.items():
            status = "失败" if r.metadata.get("error") else "Mock"
            print(f"  [{name}]: {status}")
    
    # ----------------------------------------------------------
    # 测试 5: 工具统计
    # ----------------------------------------------------------
    print("\n" + "-" * 40)
    print("工具统计:")
    stats = tool.get_stats()
    print(f"  总调用: {stats['call_count']} 次")
    print(f"  平均延迟: {stats['avg_latency']}")
    
    # ----------------------------------------------------------
    # 测试 6: ToolRegistry 集成
    # ----------------------------------------------------------
    try:
        from tool_registry import registry as reg
        print("\n" + "-" * 40)
        print("ToolRegistry 集成:")
        vision_tools = [t for t in reg.list_tools() if "vision" in t]
        print(f"  已注册视觉工具: {vision_tools}")
        
        if vision_tools:
            schema = reg.get_tools_for_llm()
            for s in schema:
                if "vision" in s["function"]["name"]:
                    print(f"    - {s['function']['name']}: {s['function']['description'][:50]}")
    except ImportError:
        pass
    
    # ----------------------------------------------------------
    # 清理
    # ----------------------------------------------------------
    for f in ["/tmp/agentclaw_test_scene.png", "/tmp/agentclaw_test_text.png"]:
        if os.path.exists(f):
            os.remove(f)
    
    print("\n" + "=" * 60)
    print("测试完成!")
    if stats['configured']:
        print("✅ 真实 API 模式 — 视觉工具已就绪")
    else:
        print("⚠️  Mock 模式 — 配置 API Key 后启用真实视觉分析")
        print("   export OPENAI_API_KEY='sk-your-key'")
    print("=" * 60)
