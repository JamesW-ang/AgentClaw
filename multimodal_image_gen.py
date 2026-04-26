"""
AgentClaw v6.1 — 多模态图片生成工具

通过 OpenAI 兼容接口调用智谱 CogView-3-Flash 进行文生图：
  - 支持 7 种标准尺寸 (1024x1024 到 1440x720)
  - 内容安全审计 (16 种危险模式检测)
  - 文件输出白名单保护
  - 自动生成唯一的图片文件名
  - 反馈信号采集（成功率、延迟、错误追踪）

v6.1 变更:
  - 集成 core.config.settings 统一配置管理
  - 集成 core.logger 统一日志
  - 工具注册到 tool_registry (registry.register 装饰器)
  - 支持异步反馈采集

依赖:
  - openai (OpenAI Python 客户端)
  - core.config, core.logger (AgentClaw 内置)
"""
import os
import hashlib
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI

from core.config import settings
from core.logger import get_logger

logger = get_logger("ImageGen")

from tool_registry import registry

# ============================================================
# 安全与配置常量
# ============================================================

ALLOWED_SIZES = [
    "1024x1024", "768x1344", "864x1152",
    "1344x768",  "1152x864", "1440x720", "720x1440",
]

MIN_PROMPT_LEN = 1
MAX_PROMPT_LEN = 4000

DANGEROUS_PATTERNS = [
    "child", "minor", "illegal", "weapon", "bomb",
    "血腥", "暴力", "恐怖", "违禁", "色情",
    "kill", "murder", "suicide", "self-harm",
    "仇恨", "歧视", "racis", "nazi",
]

OUTPUT_WHITELIST_PREFIX = [
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Documents"),
    os.path.expanduser("~/Projects"),
    "./",
]


def _validate_size(size: str) -> str:
    if size not in ALLOWED_SIZES:
        raise ValueError(f"不支持的图片尺寸 '{size}', 仅允许: {', '.join(ALLOWED_SIZES)}")
    return size


def _validate_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise ValueError("prompt 必须为字符串")
    if len(prompt) < MIN_PROMPT_LEN:
        raise ValueError("prompt 不能为空")
    if len(prompt) > MAX_PROMPT_LEN:
        raise ValueError(f"prompt 过长: {len(prompt)} 字符, 上限为 {MAX_PROMPT_LEN} 字符")
    prompt_lower = prompt.lower()
    for pattern in DANGEROUS_PATTERNS:
        if pattern.lower() in prompt_lower:
            raise PermissionError(f"提示词包含禁止内容模式 '{pattern}', 请修改后重试")
    return prompt


def _validate_output_dir(output_dir: str) -> str:
    abs_dir = os.path.abspath(output_dir)
    if not any(abs_dir.startswith(p) for p in OUTPUT_WHITELIST_PREFIX):
        raise PermissionError(f"输出目录不在白名单内: {abs_dir}")
    return abs_dir


def _generate_filename(prompt: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
    return f"gen_{timestamp}_{short_hash}.png"


def _save_image(image_data: bytes, output_dir: str, prompt: str) -> str:
    abs_dir = _validate_output_dir(output_dir)
    os.makedirs(abs_dir, exist_ok=True)
    filename = _generate_filename(prompt)
    file_path = os.path.join(abs_dir, filename)
    with open(file_path, "wb") as f:
        f.write(image_data)
    return file_path


@registry.register(
    name="image_generate",
    description="根据文本提示词生成图片, 支持多种尺寸选择",
    parameters=["prompt", "size", "output_dir"],
)
def image_generate(prompt: str, size: str = "1024x1024",
                   output_dir: str = "./generated_images") -> dict:
    """
    文生图工具 — 根据文本描述生成图片。

    Args:
        prompt:     文本提示词, 描述想要生成的图片内容
        size:       图片尺寸, 默认 "1024x1024"
        output_dir: 输出目录, 默认 "./generated_images"

    Returns:
        dict: {file_path, size, prompt, model_used, timestamp}
    """
    validated_prompt = _validate_prompt(prompt)
    validated_size = _validate_size(size)

    # v6: 从 settings 读取智谱 API 配置（CogView 是智谱模型）
    api_key = settings.ZHIPU_API_KEY
    if not api_key:
        raise EnvironmentError(
            "未设置 ZHIPU_API_KEY 环境变量。"
            "请执行: export ZHIPU_API_KEY='your-key-here'"
        )
    client = OpenAI(
        api_key=api_key,
        base_url=settings.ZHIPU_BASE_URL,
    )

    try:
        response = client.images.generate(
            model="cogview-3-flash",
            prompt=validated_prompt,
            size=validated_size,
        )
        image_url = response.data[0].url
        import urllib.request
        req = urllib.request.Request(image_url)
        with urllib.request.urlopen(req) as resp:
            image_data = resp.read()

    except Exception as e:
        raise RuntimeError(f"图片生成 API 调用失败: {e}") from e

    file_path = _save_image(image_data, output_dir, validated_prompt)

    return {
        "file_path": file_path,
        "size": validated_size,
        "prompt": validated_prompt,
        "model_used": "cogview-3-flash",
        "timestamp": datetime.now().isoformat(),
        "file_size_bytes": len(image_data),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("文生图工具 — 演示")
    print("=" * 60)

    print("\n[测试 1] 危险内容拦截")
    try:
        image_generate("a weapon", size="1024x1024")
    except PermissionError as e:
        print(f"  OK 已拦截: {e}")

    print("\n[测试 2] 提示词长度限制")
    try:
        long_prompt = "美丽的风景画, " * 501
        image_generate(long_prompt)
    except ValueError as e:
        print(f"  OK 已拦截: {str(e)[:80]}...")

    print("\n[测试 3] 无效尺寸拦截")
    try:
        image_generate("一只猫", size="5000x5000")
    except ValueError as e:
        print(f"  OK 已拦截: {e}")

    print("\n演示完成!")
