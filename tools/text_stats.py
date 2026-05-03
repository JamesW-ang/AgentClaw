"""文本统计工具 — 统计字符数、单词数、行数"""
from core.logger import get_logger
from tools.registry import ToolCategory, registry

logger = get_logger("text_stats_tool")


@registry.register(
    name="text_stats",
    description="统计文本的字符数（含/不含空格）、单词数、行数等信息",
    parameters=[
        {"name": "text", "type": "string", "description": "要统计的文本内容", "required": True},
        {"name": "count_spaces", "type": "boolean", "description": "是否统计空格字符", "required": False, "default": True},
    ],
    category=ToolCategory.CUSTOM,
)
def text_stats(text: str, count_spaces: bool = True) -> dict:
    """统计文本信息"""
    lines = text.splitlines()
    line_count = len(lines)
    word_count = len(text.split())

    if count_spaces:
        char_count = len(text)
    else:
        char_count = len(text.replace(" ", "").replace("\t", ""))

    return {
        "char_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
        "count_spaces": count_spaces,
    }
