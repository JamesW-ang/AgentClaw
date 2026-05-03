"""
AgentClaw XML 配置管理工具模块

提供 VisionParams.xml / Point.xml 配置文件的读取、写入、差异对比功能，
通过 tool_registry 注册为 Agent 可调用的工具。

工具列表:
    1. xml_config_read  — 读取 XML 配置文件，自动识别 vision/motion 类型
    2. xml_config_write — 写入/修改 XML 配置参数（含备份、校验、原子写入）
    3. xml_config_diff  — 与默认基准值对比，列出差异项

使用方式:
    from tools.xml_config import register_xml_config_tools
    register_xml_config_tools()

依赖:
    标准库: xml.etree.ElementTree, json, shutil, uuid, os
"""

import json
import os
import shutil
import sys
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

# 确保工作目录正确
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# 尝试导入项目日志模块，失败则回退到标准 logging（保证模块一定能加载）
try:
    from core.logger import get_logger
    _logger = get_logger("xml_config_tool")
except Exception:
    import logging
    _logger = logging.getLogger("xml_config_tool")
    if not _logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-5s [xml_config_tool] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        _logger.addHandler(_handler)
        _logger.setLevel(logging.DEBUG)

# 统一别名，后续代码无需改动
logger = _logger


# ============================================================
# 默认值 & 校验范围
# ============================================================

# VisionParams.xml 默认参数值（用于 diff 对比）
VISION_DEFAULTS: dict[str, Any] = {
    "CannyLow": 50,
    "CannyHigh": 150,
    "Clahe": 2.0,
    "MinArea": 100,
    "Confidence": 0.8,
    "NmsIou": 0.45,
}

# 参数校验范围: {参数名: (最小值, 最大值, 类型)}
PARAM_RANGES: dict[str, tuple[float, float, type]] = {
    "CannyLow": (10, 200, int),
    "CannyHigh": (50, 300, int),
    "Clahe": (0.5, 5.0, float),
    "MinArea": (10, 1000, int),
    "Confidence": (0.1, 0.99, float),
    "NmsIou": (0.1, 0.9, float),
}


# ============================================================
# 内部辅助函数
# ============================================================

def _detect_config_type(root: ET.Element) -> str:
    """
    自动检测 XML 配置文件类型。

    Args:
        root: XML 根节点

    Returns:
        "vision" 或 "motion"
    """
    # VisionParams.xml 通常以 <VisionParams> 为根节点，内含 <Param> 子节点
    if root.tag == "VisionParams":
        return "vision"
    # Point.xml 通常以 <PointGroup> 为根节点，包含 Motion / IO / PLC 子组
    if root.tag == "PointGroup" or root.find("PointGroup") is not None:
        return "motion"
    # 备用: 检查是否包含 VisionParams 常见参数名
    for child in root:
        if child.tag == "Param" or child.tag in ("CannyLow", "CannyHigh", "Clahe"):
            return "vision"
        if child.tag in ("Motion", "IO", "PLC"):
            return "motion"
    logger.warning(f"无法识别配置类型，根节点: {root.tag}，默认按 vision 处理")
    return "vision"


def _parse_vision_xml(root: ET.Element) -> dict[str, Any]:
    """
    解析 VisionParams.xml，提取所有参数名-值对。

    Args:
        root: XML 根节点（通常是 <VisionParams>）

    Returns:
        参数字典 {参数名: 值(已转型)}
    """
    params = {}
    for child in root:
        name = child.tag
        text = child.text.strip() if child.text and child.text.strip() else None
        if text is None:
            continue
        # 自动转型: 整数 / 浮点 / 字符串
        params[name] = _coerce_value(text)
    return params


def _parse_point_xml(root: ET.Element) -> dict[str, Any]:
    """
    解析 Point.xml，提取 Motion/IO/PLC 各组坐标及引脚信息。

    Args:
        root: XML 根节点（通常是 <PointGroup>）

    Returns:
        分组参数字典，结构如:
        {
            "Motion": {"X": 100.0, "Y": 200.0, "Z": 50.0},
            "IO": {"Pin1": "ON", "Pin2": "OFF"},
            "PLC": {...}
        }
    """
    result = {}
    actual_root = root.find("PointGroup") if root.find("PointGroup") is not None else root

    for group in actual_root:
        group_name = group.tag  # Motion / IO / PLC
        group_params = {}
        for child in group:
            group_params[child.tag] = _coerce_value(child.text.strip()) if child.text else None
        if group_params:
            result[group_name] = group_params
    return result


def _coerce_value(text: str) -> Any:
    """
    将文本值自动转型为合适的 Python 类型。

    优先级: int > float > bool > str

    Args:
        text: 原始文本值

    Returns:
        转型后的值
    """
    # 尝试整数
    try:
        return int(text)
    except (ValueError, TypeError):
        pass
    # 尝试浮点数
    try:
        return float(text)
    except (ValueError, TypeError):
        pass
    # 尝试布尔值
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    # 字符串
    return text


def _validate_params(params: dict[str, Any]) -> list[str]:
    """
    校验参数是否在允许范围内。

    对于超出范围的参数，记录警告但不阻止写入。

    Args:
        params: 待校验参数字典

    Returns:
        警告消息列表（空列表表示全部合规）
    """
    warnings = []
    for name, value in params.items():
        if name not in PARAM_RANGES:
            continue
        low, high, expected_type = PARAM_RANGES[name]
        # 类型检查
        try:
            typed_val = expected_type(value)
        except (ValueError, TypeError):
            warnings.append(
                f"参数 '{name}' 类型错误: 期望 {expected_type.__name__}, "
                f"实际值 '{value}'"
            )
            continue
        # 范围检查
        if typed_val < low or typed_val > high:
            warnings.append(
                f"参数 '{name}' 超出范围: 当前值 {typed_val}, "
                f"允许范围 [{low}, {high}]"
            )
    return warnings


def _backup_file(file_path: str) -> str:
    """
    创建配置文件备份（保留时间戳后缀）。

    Args:
        file_path: 原始文件路径

    Returns:
        备份文件路径
    """
    p = Path(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = p.parent / f"{p.stem}.bak.{timestamp}{p.suffix}"
    shutil.copy2(str(p), str(backup_path))
    logger.info(f"配置备份已创建: {backup_path}")
    return str(backup_path)


def _atomic_write(file_path: str, xml_content: str) -> None:
    """
    原子写入: 先写入临时文件，再替换目标文件。

    避免写入过程中断导致文件损坏。
    使用 uuid 生成唯一临时文件名，避免 tempfile.NamedTemporaryFile
    在某些 Python 3.12 环境下的兼容性问题。

    Args:
        file_path: 目标文件路径
        xml_content: XML 字符串内容
    """
    dir_path = str(Path(file_path).parent)
    tmp_path = os.path.join(dir_path, f".tmp_{uuid.uuid4().hex[:12]}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(xml_content)
        os.replace(tmp_path, str(file_path))
        logger.info(f"原子写入完成: {file_path}")
    except Exception:
        # 写入失败时清理临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ============================================================
# 工具1: xml_config_read
# ============================================================

def xml_config_read(config_path: str) -> dict:
    """
    读取 XML 配置文件，自动识别类型并提取所有参数。

    支持 VisionParams.xml（视觉参数）和 Point.xml（运动/IO/PLC 坐标）。

    Args:
        config_path: XML 配置文件的完整路径

    Returns:
        {
            "success": True/False,
            "result": {参数名: 值, ...},
            "file_path": 文件路径,
            "config_type": "vision" / "motion",
            "error": None / 错误描述
        }
    """
    logger.info(f"读取 XML 配置: {config_path}")

    # 路径校验
    p = Path(config_path)
    if not p.exists():
        return {
            "success": False,
            "result": None,
            "file_path": config_path,
            "config_type": None,
            "error": f"文件不存在: {config_path}",
        }

    try:
        tree = ET.parse(str(p))
        root = tree.getroot()
    except ET.ParseError as e:
        return {
            "success": False,
            "result": None,
            "file_path": config_path,
            "config_type": None,
            "error": f"XML 解析失败: {e}",
        }

    # 自动检测类型
    config_type = _detect_config_type(root)
    logger.info(f"配置类型: {config_type}, 文件: {p.name}")

    # 按类型解析
    if config_type == "vision":
        result = _parse_vision_xml(root)
    else:
        result = _parse_point_xml(root)

    return {
        "success": True,
        "result": result,
        "file_path": str(p.resolve()),
        "config_type": config_type,
        "error": None,
    }


# ============================================================
# 工具2: xml_config_write
# ============================================================

def xml_config_write(config_path: str, params: str) -> dict:
    """
    写入/修改 XML 配置文件中的参数。

    特性:
        - 写入前自动备份原文件（.bak.{时间戳}）
        - 参数范围校验（超范围会警告但仍写入）
        - 原子写入（先写临时文件再替换，防止中断损坏）

    Args:
        config_path: XML 配置文件的完整路径
        params: JSON 字符串格式的参数字典，例如 '{"CannyLow": 30, "CannyHigh": 120}'

    Returns:
        {
            "success": True/False,
            "result": {已更新的参数名: 值, ...},
            "backup_path": 备份文件路径,
            "validation_warnings": [警告列表],
            "error": None / 错误描述
        }
    """
    logger.info(f"写入 XML 配置: {config_path}")

    # 路径校验
    p = Path(config_path)
    if not p.exists():
        return {
            "success": False,
            "result": None,
            "backup_path": None,
            "validation_warnings": [],
            "error": f"文件不存在: {config_path}",
        }

    # 解析 JSON 参数
    try:
        update_params = json.loads(params)
        if not isinstance(update_params, dict):
            raise ValueError("params 必须是 JSON 对象")
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "success": False,
            "result": None,
            "backup_path": None,
            "validation_warnings": [],
            "error": f"参数解析失败: {e}",
        }

    # 参数校验（仅警告，不阻止写入）
    validation_warnings = _validate_params(update_params)
    for w in validation_warnings:
        logger.warning(f"参数校验警告: {w}")

    try:
        tree = ET.parse(str(p))
        root = tree.getroot()
    except ET.ParseError as e:
        return {
            "success": False,
            "result": None,
            "backup_path": None,
            "validation_warnings": validation_warnings,
            "error": f"XML 解析失败: {e}",
        }

    config_type = _detect_config_type(root)

    # 备份原文件
    try:
        backup_path = _backup_file(str(p))
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "backup_path": None,
            "validation_warnings": validation_warnings,
            "error": f"备份失败: {e}",
        }

    # 根据类型更新参数
    try:
        if config_type == "vision":
            _update_vision_xml(root, update_params)
        else:
            _update_point_xml(root, update_params)
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "backup_path": backup_path,
            "validation_warnings": validation_warnings,
            "error": f"参数更新失败: {e}",
        }

    # 原子写入
    try:
        xml_content = _prettify_xml(root)
        _atomic_write(str(p), xml_content)
    except Exception as e:
        return {
            "success": False,
            "result": None,
            "backup_path": backup_path,
            "validation_warnings": validation_warnings,
            "error": f"写入失败: {e}",
        }

    logger.info(f"配置写入成功: {len(update_params)} 个参数已更新")

    return {
        "success": True,
        "result": update_params,
        "backup_path": backup_path,
        "validation_warnings": validation_warnings,
        "error": None,
    }


def _update_vision_xml(root: ET.Element, params: dict[str, Any]) -> None:
    """
    更新 VisionParams XML 中的参数节点。

    已存在的节点更新值，不存在的节点新增。

    Args:
        root: XML 根节点
        params: 待更新的参数字典
    """
    for name, value in params.items():
        # 查找现有节点
        found = None
        for child in root:
            if child.tag == name:
                found = child
                break
        if found is not None:
            found.text = str(value)
        else:
            # 新增节点
            new_elem = ET.SubElement(root, name)
            new_elem.text = str(value)
            logger.debug(f"新增参数节点: {name}={value}")


def _update_point_xml(root: ET.Element, params: dict[str, Any]) -> None:
    """
    更新 Point XML 中的坐标/引脚节点。

    params 中的键格式: "组名.参数名"，例如 "Motion.X", "IO.Pin1"

    Args:
        root: XML 根节点
        params: 待更新的参数字典
    """
    actual_root = root.find("PointGroup") if root.find("PointGroup") is not None else root

    for key, value in params.items():
        parts = key.split(".", 1)
        if len(parts) == 2:
            group_name, param_name = parts
        else:
            # 如果没有组前缀，遍历所有组查找该参数
            group_name, param_name = None, key

        updated = False
        for group in actual_root:
            if group_name and group.tag != group_name:
                continue
            for child in group:
                if child.tag == param_name:
                    child.text = str(value)
                    updated = True
                    break
            if updated:
                break

        if not updated:
            logger.warning(f"未找到参数 '{key}'，跳过更新")


def _prettify_xml(root: ET.Element, indent: str = "  ") -> str:
    """
    将 XML 节点格式化为带缩进的字符串（写入前美化）。

    Args:
        root: XML 根节点
        indent: 缩进字符串

    Returns:
        格式化后的 XML 字符串
    """
    def _indent(elem, level=0):
        i = "\n" + level * indent
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + indent
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    _indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


# ============================================================
# 工具3: xml_config_diff
# ============================================================

def xml_config_diff(config_path: str) -> dict:
    """
    对比当前配置与默认基准值，列出差异项。

    目前仅支持 VisionParams.xml 类型的配置文件与内置默认值对比。
    对于 motion 类型，返回所有当前值（无默认基准）。

    Args:
        config_path: XML 配置文件的完整路径

    Returns:
        {
            "success": True/False,
            "result": {
                "config_type": "vision" / "motion",
                "total_params": 参数总数,
                "diff_count": 差异数量,
                "diffs": [
                    {"param": "CannyLow", "current": 30, "default": 50, "status": "modified"},
                    ...
                ],
                "identical": ["Clahe", ...],
                "summary": "总结描述"
            },
            "error": None / 错误描述
        }
    """
    logger.info(f"对比 XML 配置: {config_path}")

    # 先读取当前配置
    read_result = xml_config_read(config_path)
    if not read_result["success"]:
        return {
            "success": False,
            "result": None,
            "error": read_result["error"],
        }

    config_type = read_result["config_type"]
    current_params = read_result["result"]

    if config_type != "vision":
        # motion 类型暂无默认基准，返回全部当前值
        return {
            "success": True,
            "result": {
                "config_type": config_type,
                "total_params": len(current_params),
                "diff_count": 0,
                "diffs": [],
                "identical": list(current_params.keys()),
                "summary": f"运动配置文件，共 {len(current_params)} 个参数，暂无默认基准对比",
                "current_values": current_params,
            },
            "error": None,
        }

    # 与 VisionParams 默认值对比
    diffs = []
    identical = []
    all_keys = set(list(current_params.keys()) + list(VISION_DEFAULTS.keys()))

    for key in sorted(all_keys):
        current_val = current_params.get(key)
        default_val = VISION_DEFAULTS.get(key)

        if current_val is None:
            # 配置中缺失该参数
            diffs.append({
                "param": key,
                "current": None,
                "default": default_val,
                "status": "missing",
            })
        elif default_val is None:
            # 默认值中没有（新增参数）
            diffs.append({
                "param": key,
                "current": current_val,
                "default": None,
                "status": "new",
            })
        elif current_val != default_val:
            # 值不同
            diffs.append({
                "param": key,
                "current": current_val,
                "default": default_val,
                "status": "modified",
            })
        else:
            identical.append(key)

    summary = (
        f"视觉配置对比完成: 共 {len(all_keys)} 个参数，"
        f"{len(diffs)} 个差异（{len([d for d in diffs if d['status'] == 'modified'])} 个修改，"
        f"{len([d for d in diffs if d['status'] == 'missing'])} 个缺失，"
        f"{len([d for d in diffs if d['status'] == 'new'])} 个新增），"
        f"{len(identical)} 个一致"
    )
    logger.info(summary)

    return {
        "success": True,
        "result": {
            "config_type": config_type,
            "total_params": len(all_keys),
            "diff_count": len(diffs),
            "diffs": diffs,
            "identical": identical,
            "summary": summary,
        },
        "error": None,
    }


# ============================================================
# 工具注册
# ============================================================

def register_xml_config_tools():
    """
    注册 XML 配置管理工具到 tool_registry。

    将 xml_config_read / xml_config_write / xml_config_diff
    三个工具注册为 Agent 可调用的工具。

    调用方式（在 agent_core.py 中）:
        import tools.xml_config
        xml_config_tool.register_xml_config_tools()
    """
    from tools.registry import ToolCategory, registry

    # 工具1: 读取 XML 配置
    registry.register_func(
        xml_config_read,
        name="xml_config_read",
        description=(
            "读取 AOI 设备的 XML 配置文件（VisionParams.xml 或 Point.xml）。"
            "自动识别配置类型（vision 视觉参数 / motion 运动坐标），"
            "返回所有参数的名称和当前值。"
        ),
        parameters=[
            {
                "name": "config_path",
                "type": "string",
                "description": "XML 配置文件的完整路径（如 /config/VisionParams.xml）",
                "required": True,
            },
        ],
        category=ToolCategory.CUSTOM,
        timeout=30,
    )

    # 工具2: 写入 XML 配置
    registry.register_func(
        xml_config_write,
        name="xml_config_write",
        description=(
            "修改 AOI 设备的 XML 配置参数。支持 VisionParams.xml 视觉参数和 Point.xml 运动坐标。"
            "写入前自动备份原文件，支持参数范围校验（超范围会警告但仍写入），"
            "使用原子写入防止中断损坏。params 为 JSON 字符串格式，"
            "例如 '{\"CannyLow\": 30, \"CannyHigh\": 120}'。"
        ),
        parameters=[
            {
                "name": "config_path",
                "type": "string",
                "description": "XML 配置文件的完整路径",
                "required": True,
            },
            {
                "name": "params",
                "type": "string",
                "description": (
                    "JSON 字符串格式的参数字典，例如 '{\"CannyLow\": 30, \"CannyHigh\": 120}'。"
                    "对于 Point.xml 使用 '组名.参数名' 格式，如 '{\"Motion.X\": 100.0}'"
                ),
                "required": True,
            },
        ],
        category=ToolCategory.CUSTOM,
        timeout=60,
    )

    # 工具3: 配置差异对比
    registry.register_func(
        xml_config_diff,
        name="xml_config_diff",
        description=(
            "将当前 XML 配置与内置默认基准值进行对比，列出所有差异项。"
            "VisionParams.xml 对比内置视觉默认值，Point.xml 返回全部当前值。"
            "适用于检查设备参数是否被调偏、排查参数漂移问题。"
        ),
        parameters=[
            {
                "name": "config_path",
                "type": "string",
                "description": "XML 配置文件的完整路径",
                "required": True,
            },
        ],
        category=ToolCategory.CUSTOM,
        timeout=30,
    )

    logger.info("XML 配置管理工具已注册 (xml_config_read / xml_config_write / xml_config_diff)")


# ============================================================
# 自测
# ============================================================

def main():
    """xml_config 工具模块自测"""
    import tempfile

    print("=" * 60)
    print("xml_config_tool 自测")
    print("=" * 60)

    # 创建临时测试目录
    test_dir = tempfile.mkdtemp(prefix="xml_config_test_")
    print(f"\n测试目录: {test_dir}")

    # ---- 创建测试 VisionParams.xml ----
    vision_xml = """<?xml version="1.0" encoding="UTF-8"?>
<VisionParams>
    <CannyLow>30</CannyLow>
    <CannyHigh>120</CannyHigh>
    <Clahe>2.5</Clahe>
    <MinArea>200</MinArea>
    <Confidence>0.85</Confidence>
    <NmsIou>0.45</NmsIou>
</VisionParams>"""
    vision_path = os.path.join(test_dir, "VisionParams.xml")
    with open(vision_path, "w", encoding="utf-8") as f:
        f.write(vision_xml)

    # ---- 创建测试 Point.xml ----
    point_xml = """<?xml version="1.0" encoding="UTF-8"?>
<PointGroup>
    <Motion>
        <X>150.5</X>
        <Y>200.0</Y>
        <Z>50.0</Z>
    </Motion>
    <IO>
        <Pin1>ON</Pin1>
        <Pin2>OFF</Pin2>
    </IO>
    <PLC>
        <Register1>100</Register1>
        <Register2>200</Register2>
    </PLC>
</PointGroup>"""
    point_path = os.path.join(test_dir, "Point.xml")
    with open(point_path, "w", encoding="utf-8") as f:
        f.write(point_xml)

    # ===== 测试1: xml_config_read =====
    print("\n--- 测试 xml_config_read ---")

    result = xml_config_read(vision_path)
    print(f"[VisionParams] success={result['success']}, type={result['config_type']}")
    print(f"  参数: {result['result']}")
    assert result["success"] is True
    assert result["config_type"] == "vision"
    assert result["result"]["CannyLow"] == 30

    result = xml_config_read(point_path)
    print(f"[Point.xml] success={result['success']}, type={result['config_type']}")
    print(f"  参数: {result['result']}")
    assert result["success"] is True
    assert result["config_type"] == "motion"
    assert result["result"]["Motion"]["X"] == 150.5

    # 文件不存在
    result = xml_config_read("/nonexistent/path.xml")
    print(f"[不存在文件] success={result['success']}, error={result['error']}")
    assert result["success"] is False

    # ===== 测试2: xml_config_write =====
    print("\n--- 测试 xml_config_write ---")

    # 正常写入（参数在范围内）
    result = xml_config_write(
        vision_path,
        '{"CannyLow": 60, "CannyHigh": 180, "Clahe": 3.0}'
    )
    print(f"[写入] success={result['success']}")
    print(f"  已更新: {result['result']}")
    print(f"  备份: {result['backup_path']}")
    print(f"  警告: {result['validation_warnings']}")
    if not result["success"]:
        print(f"  错误: {result['error']}")
    assert result["success"] is True, f"写入失败: {result['error']}"
    assert result["result"]["CannyLow"] == 60
    assert len(result["validation_warnings"]) == 0

    # 验证文件确实被更新
    result2 = xml_config_read(vision_path)
    assert result2["result"]["CannyLow"] == 60
    assert result2["result"]["Clahe"] == 3.0

    # 写入超范围参数（应该成功但有警告）
    result = xml_config_write(
        vision_path,
        '{"CannyLow": 5, "Confidence": 1.5}'
    )
    print(f"[超范围写入] success={result['success']}")
    print(f"  警告: {result['validation_warnings']}")
    assert result["success"] is True
    assert len(result["validation_warnings"]) == 2

    # Point.xml 写入
    result = xml_config_write(
        point_path,
        '{"Motion.X": 300.0, "IO.Pin1": "OFF"}'
    )
    print(f"[Point写入] success={result['success']}")
    assert result["success"] is True

    # 验证 Point.xml 更新
    result2 = xml_config_read(point_path)
    assert result2["result"]["Motion"]["X"] == 300.0
    assert result2["result"]["IO"]["Pin1"] == "OFF"

    # ===== 测试3: xml_config_diff =====
    print("\n--- 测试 xml_config_diff ---")

    result = xml_config_diff(vision_path)
    print(f"[对比] success={result['success']}")
    assert result["success"] is True
    diff_info = result["result"]
    print(f"  总参数: {diff_info['total_params']}")
    print(f"  差异数: {diff_info['diff_count']}")
    print("  差异项:")
    for d in diff_info["diffs"]:
        print(f"    {d['param']}: 当前={d['current']}, 默认={d['default']}, 状态={d['status']}")
    print(f"  一致项: {diff_info['identical']}")
    print(f"  总结: {diff_info['summary']}")

    # Point.xml 对比（motion 类型无默认基准）
    result = xml_config_diff(point_path)
    print(f"\n[Point对比] success={result['success']}")
    print(f"  总结: {result['result']['summary']}")
    assert result["success"] is True

    # ===== 测试4: 注册到 tool_registry =====
    print("\n--- 测试工具注册 ---")
    from tools.registry import registry
    ToolRegistry = registry.__class__
    ToolRegistry.reset()
    from tools.registry import registry as fresh_registry
    register_xml_config_tools()

    tools = fresh_registry.list_tools()
    print(f"已注册工具: {tools}")
    assert "xml_config_read" in tools
    assert "xml_config_write" in tools
    assert "xml_config_diff" in tools

    for t in tools:
        info = fresh_registry.get_tool(t)
        print(f"  [{info.category.value}] {t}: {info.description[:40]}...")

    # 清理
    shutil.rmtree(test_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("全部测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
