"""
AgentClaw v6.1 — 多模态智能路由器

根据用户意图自动选择最佳视觉分析模式：
  - OCR: 文字识别和提取
  - DESCRIBE: 图片内容描述
  - ANALYZE: 深度分析和理解
  - COMPARE: 多图对比分析
  - CUSTOM: 自定义视觉任务

特性:
  - 基于规则的意图识别（关键词匹配 + 置信度）
  - 可扩展的模式定义
  - 中英文双语支持
  - 自适应路由提示词生成

依赖:
  - dataclasses (内置)
  - re (内置)
"""
import re
import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum

from core.logger import get_logger

logger = get_logger("MultimodalRouter")


# ============================================================
# 枚举 & 数据结构
# ============================================================

class VisionMode(Enum):
    """视觉分析模式枚举"""
    OCR = "ocr"
    DESCRIBE = "describe"
    ANALYZE = "analyze"
    COMPARE = "compare"
    CUSTOM = "text"


@dataclass
class RouteResult:
    """路由结果数据类
    
    Attributes:
        mode: 推荐的视觉分析模式
        prompt: 生成的提示词
        confidence: 匹配置信度 (0-1)
        reason: 路由理由说明
    """
    mode: VisionMode
    prompt: str
    confidence: float
    reason: str


class MultimodalRouter:
    """
    基于规则的视觉意图路由器

    根据用户输入自动判断应该使用哪种视觉分析模式:
    - OCR: 提取图片中的文字
    - DESCRIBE: 描述图片内容
    - ANALYZE: 深度分析图片
    - COMPARE: 对比多张图片
    - CUSTOM: 自定义问题
    """

    # 模式关键词映射
    MODE_RULES = [
        (VisionMode.OCR, ["ocr", "OCR", "文字", "识别文字", "提取文字", "文字识别",
                          "text recognition", "读取文字", "文字内容", "提取文本",
                          "识别文本", "截图文字", "图片文字", "照片文字"]),
        (VisionMode.COMPARE, ["对比", "比较", "差异", "不同", "区别", "compare",
                             "difference", "变化", "前后"]),
        (VisionMode.ANALYZE, ["分析", "详细分析", "深度分析", "analyze", "analysis",
                             "技术分析", "质量", "缺陷", "问题", "诊断",
                             "专业分析", "详细描述"]),
        (VisionMode.DESCRIBE, ["描述", "describe", "内容", "什么", "看到了什么",
                              "图中有什么", "画面", "图片内容", "这是什么"]),
    ]

    def route(self, query: str) -> RouteResult:
        """
        路由用户查询到最佳视觉分析模式

        Args:
            query: 用户输入的查询文本

        Returns:
            RouteResult 包含 mode, prompt, confidence, reason
        """
        if not query or not query.strip():
            return RouteResult(
                mode=VisionMode.DESCRIBE,
                prompt="请详细描述这张图片的内容、构图和主要元素。",
                confidence=1.0,
                reason="空查询，使用默认描述模式"
            )

        query_lower = query.lower()

        # 遍历规则，找到最匹配的模式
        best_mode = VisionMode.CUSTOM
        best_score = 0
        best_reason = ""

        for mode, keywords in self.MODE_RULES:
            score = sum(1 for kw in keywords if kw.lower() in query_lower)
            if score > best_score:
                best_score = score
                best_mode = mode
                best_reason = f"匹配关键词: {[kw for kw in keywords if kw.lower() in query_lower]}"

        # 构建模式对应的 prompt
        mode_prompts = {
            VisionMode.OCR: "请对这张图片进行 OCR 文字识别，提取所有可见文字内容。",
            VisionMode.COMPARE: "请仔细对比这些图片，找出差异之处并详细说明。",
            VisionMode.ANALYZE: "请对这张图片进行深度分析，包括内容、风格、潜在含义和技术细节。",
            VisionMode.DESCRIBE: "请详细描述这张图片的内容、构图和主要元素。",
            VisionMode.CUSTOM: "",
        }

        prompt = mode_prompts.get(best_mode, "")

        # 如果是 OCR/DESCRIBE/ANALYZE，自动追加用户的问题
        if best_mode in (VisionMode.OCR, VisionMode.DESCRIBE, VisionMode.ANALYZE, VisionMode.COMPARE):
            if query.strip() and best_mode != VisionMode.CUSTOM:
                prompt += f"\n用户补充问题: {query.strip()}"

        confidence = min(0.5 + best_score * 0.15, 0.95) if best_score > 0 else 0.5

        if best_mode == VisionMode.CUSTOM and not prompt:
            prompt = query.strip()  # 直接使用用户的原始问题

        logger.info(f"[MultimodalRouter] 路由结果: mode={best_mode.value}, confidence={confidence:.2f}, reason={best_reason}")

        return RouteResult(
            mode=best_mode,
            prompt=prompt,
            confidence=confidence,
            reason=best_reason or "无关键词匹配，使用自定义模式"
        )


# 全局单例
_router = None

def get_multimodal_router() -> MultimodalRouter:
    global _router
    if _router is None:
        _router = MultimodalRouter()
    return _router
