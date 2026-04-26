# c03_multimodal_vlm.py - Level 2 多模态: VLM 视觉理解工具
# 通过 OpenAI 兼容接口调用智谱 GLM-4V 系列视觉模型,
# 支持本地图片 (base64) 和 URL 两种输入方式, 提供 OCR/描述/分析等输出类型。
import os
import base64
from pathlib import Path
from openai import OpenAI

# 自动加载 .env 文件（项目根目录优先，当前目录次之）
try:
    from dotenv import load_dotenv
    for env_path in [
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
        Path.home() / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            break
except ImportError:
    pass  # 没有 python-dotenv 也继续运行

# 从 tool_registry 导入注册装饰器
from tool_registry import registry

# ============================================================
# 安全配置 — 与 file_read 保持一致的白名单/黑名单机制
# ============================================================
IMAGE_WHITELIST_PREFIX = [
    os.path.expanduser("~/Desktop"),
    os.path.expanduser("~/Documents"),
    os.path.expanduser("~/Projects"),
    "./",   # 当前工作目录
]

IMAGE_BLACKLIST = [
    "/etc/", "/var/log/", "/proc/", "/sys/",
    ".ssh/", ".git/", ".env",
    "id_rsa", "id_ed25519", "authorized_keys",
]

# 允许的图片扩展名
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# 文件大小上限 20MB
MAX_IMAGE_SIZE = 20 * 1024 * 1024

# 支持的输出类型
VALID_OUTPUT_TYPES = {"text", "ocr", "describe", "analyze"}


def _validate_image_path(image_path: str) -> str:
    """验证图片路径安全性, 返回绝对路径"""
    abs_path = os.path.abspath(image_path)

    # 白名单检查
    if not any(abs_path.startswith(p) for p in IMAGE_WHITELIST_PREFIX):
        raise PermissionError(f"图片路径不在白名单内: {abs_path}")

    # 黑名单检查
    for blocked in IMAGE_BLACKLIST:
        if blocked in abs_path:
            raise PermissionError(f"禁止访问敏感路径: {blocked} 存在于 {abs_path}")

    # 扩展名检查
    _, ext = os.path.splitext(abs_path)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"不支持的图片格式 '{ext}', 仅允许: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # 文件存在性与大小检查
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"图片文件不存在: {abs_path}")

    file_size = os.path.getsize(abs_path)
    if file_size > MAX_IMAGE_SIZE:
        raise ValueError(f"图片文件过大: {file_size / 1024 / 1024:.1f}MB > 20MB")

    return abs_path


def _encode_image_base64(image_path: str) -> str:
    """将本地图片编码为 base64, 返回 data URI 字符串"""
    _, ext = os.path.splitext(image_path)
    # 映射扩展名到 MIME 类型
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    }
    mime = mime_map.get(ext.lower(), "image/png")

    with open(image_path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime};base64,{b64_data}"


def _build_message_content(image_source: str, question: str,
                           output_type: str) -> list:
    """根据图片来源构建 OpenAI 多模态消息体"""
    content = []

    # 判断图片来源: URL 还是 data URI (base64)
    if image_source.startswith(("http://", "https://")):
        content.append({"type": "image_url", "image_url": {"url": image_source}})
    elif image_source.startswith("data:"):
        content.append({"type": "image_url", "image_url": {"url": image_source}})
    else:
        raise ValueError("图片源格式无效, 仅支持 URL 或 base64 data URI")

    # 根据输出类型构造提示语
    type_prompts = {
        "ocr":      "请对这张图片进行 OCR 文字识别, 提取所有可见文字内容。",
        "describe": "请详细描述这张图片的内容、构图和主要元素。",
        "analyze":  "请对这张图片进行深度分析, 包括内容、风格、潜在含义等。",
        "text":     "",  # 通用模式, 使用用户自定义问题
    }
    prompt = type_prompts.get(output_type, "")
    if question:
        prompt += question
    content.append({"type": "text", "text": prompt.strip() or "请分析这张图片。"})

    return content


@registry.register(
    name="vision_analyze",
    description="使用 VLM 视觉模型分析图片内容, 支持 OCR/描述/分析等多种模式",
    parameters=["image_path", "question", "model", "output_type"],
)
def vision_analyze(image_path: str, question: str,
                   model: str = None, output_type: str = "text") -> dict:
    """
    VLM 视觉理解工具 — 分析图片并返回结构化结果。

    Args:
        image_path: 图片路径 (本地文件) 或图片 URL
        question:   针对图片提出的问题
        model:      模型名称, 默认读取 VISION_MODEL 环境变量, 回退到 glm-4v-flash
        output_type: 输出类型 — "text"(默认)/"ocr"/"describe"/"analyze"

    Returns:
        dict: {analysis_type, content, model_used}
    """
    # 参数校验
    if question is not None and not isinstance(question, str):
        raise ValueError("question 参数必须为字符串")

    # text 模式需要 question, 其他模式可用内置 prompt
    if output_type == "text" and not question:
        raise ValueError("text 模式下 question 参数不能为空")

    if output_type not in VALID_OUTPUT_TYPES:
        raise ValueError(
            f"无效的 output_type '{output_type}', "
            f"仅允许: {', '.join(VALID_OUTPUT_TYPES)}"
        )

    # 确定模型
    model_name = model or os.getenv("VISION_MODEL", "glm-4v-flash")

    # 判断图片来源: URL 直接使用, 本地路径需验证并编码
    if image_path.startswith(("http://", "https://")):
        image_source = image_path
    else:
        abs_path = _validate_image_path(image_path)
        image_source = _encode_image_base64(abs_path)

    # 构造消息
    messages = [{
        "role": "user",
        "content": _build_message_content(image_source, question, output_type),
    }]

    # 调用 OpenAI 兼容接口 (智谱)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "未设置 OPENAI_API_KEY 环境变量。"
            "请执行: export OPENAI_API_KEY='your-key-here'"
        )
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=4096,
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()

    return {
        "analysis_type": output_type,
        "content": content,
        "model_used": model_name,
    }


# ============================================================
# 演示 / 测试
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("VLM 视觉理解工具 — 演示")
    print("=" * 60)

    # 场景 1: 安全拦截测试
    print("\n[测试 1] 安全拦截 — 尝试读取 /etc/passwd")
    try:
        vision_analyze("/etc/passwd", "这是什么?")
    except PermissionError as e:
        print(f"  ✓ 已拦截: {e}")

    # 场景 2: URL 图片分析
    print("\n[测试 2] URL 图片描述模式")
    try:
        result = vision_analyze(
            image_path="https://img1.baidu.com/it/u=1369931113,3388870256&fm=253",
            question="请描述这张图片",
            output_type="describe",
        )
        print(f"  模型: {result['model_used']}")
        print(f"  结果: {result['content'][:200]}...")
    except Exception as e:
        print(f"  (网络或 API 异常, 跳过): {e}")

    # 场景 3: OCR 模式 (本地文件 — 需要实际图片)
    print("\n[测试 3] OCR 模式 (本地图片)")
    try:
        result = vision_analyze(
            image_path="./test_image.png",
            question=None,
            output_type="ocr",
        )
        print(f"  OCR 结果: {result['content'][:200]}")
    except FileNotFoundError as e:
        print(f"  (本地文件不存在, 预期行为): {e}")
    except Exception as e:
        print(f"  异常: {e}")

    print("\n演示完成!")
