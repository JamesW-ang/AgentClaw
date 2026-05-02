"""
AgentClaw v6.1 — AOI 智能闭环工作流

基于 LangGraph StateGraph 的多 Agent 协作流水线，实现 AOI 缺陷检测闭环调优：
  Stage 1: defect_analyst   — 缺陷分析师：执行检测 → LLM 分析判定
  Stage 2: param_optimizer  — 参数优化师：RAG 检索相似案例 → LLM 推理推荐参数
  Stage 3: config_executor  — 配置执行器：写入 XML → 重新检测
  Stage 4: verifier         — 复检验证器：对比调参前后 → 输出最终评估

流程路由:
  START → defect_analyst → should_tune?
    → YES:  param_optimizer → config_executor → verifier → END
    → NO:   END (产品合格或无需调参)

依赖:
    pip install langgraph langchain-core langchain-openai
"""

import re
import json
import time
import sys
from pathlib import Path
from typing import TypedDict, Optional, List, Dict, Any

# 确保工作目录正确（以脚本所在目录为基准）
SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# 尝试导入项目日志模块，失败则回退到标准 logging（保证模块一定能加载）
try:
    from core.logger import get_logger
    _logger = get_logger("aoi_workflow")
except Exception:
    import logging
    _logger = logging.getLogger("aoi_workflow")
    if not _logger.handlers:
        _handler = logging.StreamHandler()
        _handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-5s [aoi_workflow] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        _logger.addHandler(_handler)
        _logger.setLevel(logging.DEBUG)

# 统一别名，后续代码无需改动
logger = _logger


# ============================================================
# 工作流状态定义
# ============================================================

class AOIWorkflowState(TypedDict, total=False):
    """AOI 闭环工作流状态，各节点通过此状态传递数据"""
    messages: list                          # LangGraph 消息列表
    image_path: str                         # 待检测图片路径
    detection_mode: str                     # 检测模式 (traditional/deeplearning/hybrid)
    config_path: Optional[str]              # XML 配置文件路径
    initial_result: Optional[dict]          # 首次检测结果 (aoi_detect_for_agent 返回值)
    rag_cases: Optional[List[dict]]         # RAG 检索到的相似历史调参案例
    recommended_params: Optional[dict]      # LLM 推荐的参数调整方案
    xml_write_result: Optional[dict]        # xml_config_write 写入结果
    reverify_result: Optional[dict]         # 调参后复检结果
    final_verdict: Optional[dict]           # 最终评估结论
    stage: str                              # 当前阶段追踪


# ============================================================
# 延迟初始化 LLM（避免 import 时创建连接）
# ============================================================

_llm = None


def _get_llm():
    """延迟创建 LLM 实例，首次调用时初始化"""
    global _llm
    if _llm is None:
        try:
            from langchain_openai import ChatOpenAI
            from core.config import settings
            _llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                temperature=0,
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL,
            )
            logger.info("AOI 工作流 LLM 已初始化")
        except Exception as e:
            logger.error(f"LLM 初始化失败: {e}")
            raise
    return _llm


# ============================================================
# RAG 知识库：从 aoi_cases.json 加载历史调参案例
# ============================================================

_aoi_cases_cache: Optional[List[dict]] = None


def _load_aoi_cases() -> List[dict]:
    """加载 data/aoi_cases.json 历史调参案例（带缓存）"""
    global _aoi_cases_cache
    if _aoi_cases_cache is not None:
        return _aoi_cases_cache

    cases_path = SCRIPT_DIR / "data" / "aoi_cases.json"
    try:
        with open(cases_path, "r", encoding="utf-8") as f:
            _aoi_cases_cache = json.load(f)
        logger.info(f"已加载 {len(_aoi_cases_cache)} 条历史调参案例")
    except Exception as e:
        logger.warning(f"加载 aoi_cases.json 失败: {e}，使用空列表")
        _aoi_cases_cache = []
    return _aoi_cases_cache


def _search_similar_cases(query: str, top_k: int = 5) -> List[dict]:
    """
    基于关键词重叠评分的简单 RAG 检索。

    对 query 分词后与每条案例的 defect_type / phenomenon / root_cause 字段
    计算关键词重叠比例，返回 top_k 条最相似案例。

    Args:
        query: 检索查询文本（通常为缺陷描述）
        top_k: 返回数量

    Returns:
        按相似度排序的案例列表，每条附加 score 字段
    """
    cases = _load_aoi_cases()
    if not cases:
        return []

    # 中文分词：简单按标点/空白分割，去停用词
    stop_words = {"的", "了", "在", "是", "和", "与", "中", "有", "被", "将", "对", "等",
                  "为", "到", "也", "或", "上", "下", "不", "这", "那", "就", "都",
                  "一", "个", "从", "以", "其", "已", "而", "由", "能", "所", "可",
                  "会", "但", "更", "较", "于", "则"}

    def tokenize(text: str) -> set:
        tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", text.lower()))
        return tokens - stop_words

    query_tokens = tokenize(query)
    if not query_tokens:
        return cases[:top_k]

    scored = []
    for case in cases:
        # 合并案例的多个文本字段进行匹配
        case_text = " ".join([
            case.get("defect_type", ""),
            case.get("phenomenon", ""),
            case.get("root_cause", ""),
            case.get("tuning_rationale", ""),
        ])
        case_tokens = tokenize(case_text)

        # Jaccard 相似度
        intersection = query_tokens & case_tokens
        union = query_tokens | case_tokens
        score = len(intersection) / len(union) if union else 0.0

        # 缺陷类型精确匹配加权
        defect_type = case.get("defect_type", "")
        if defect_type and defect_type in query:
            score += 0.15

        scored.append({**case, "score": round(score, 4)})

    # 按分数降序排列，取 top_k
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ============================================================
# 参数提取：从 LLM 回复中解析 JSON 参数块
# ============================================================

def _extract_params_from_llm(text: str) -> Optional[Dict[str, Any]]:
    """
    从 LLM 回复文本中提取参数 JSON。

    支持以下格式:
      1. ```json ... ``` 代码块
      2. { ... } 内联 JSON
      3. 参数名=值 的逐行格式（如 CannyLow: 30）

    Args:
        text: LLM 生成的文本

    Returns:
        提取到的参数字典，失败返回 None
    """
    # 尝试提取 ```json ... ``` 代码块
    json_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_block:
        try:
            params = json.loads(json_block.group(1).strip())
            if isinstance(params, dict):
                return params
        except json.JSONDecodeError:
            pass

    # 尝试提取 { ... } JSON 对象
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            params = json.loads(json_match.group(0))
            if isinstance(params, dict):
                return params
        except json.JSONDecodeError:
            pass

    # 尝试逐行解析 "参数名: 值" 或 "参数名：值" 格式
    known_params = {
        "CannyLow": int, "CannyHigh": int, "Clahe": float,
        "MinArea": int, "Confidence": float, "NmsIou": float,
        "cannylow": int, "cannyhigh": int, "clahe": float,
        "minarea": int, "confidence": float, "nmsiou": float,
    }
    params = {}
    for line in text.split("\n"):
        line = line.strip().strip("-").strip("*").strip()
        match = re.match(
            r"['\"]?(\w+)['\"]?\s*[:：=]\s*['\"]?([\d.]+)['\"]?",
            line
        )
        if match:
            key, val_str = match.group(1), match.group(2)
            # 标准化键名（首字母大写）
            key_norm = key.capitalize()
            if key_norm in known_params:
                try:
                    params[key_norm] = known_params[key_norm](val_str)
                except (ValueError, TypeError):
                    pass

    return params if params else None


# ============================================================
# 检测报告格式化工具
# ============================================================

def _format_detection_report(result: dict) -> str:
    """将 aoi_detect_for_agent 返回值格式化为可读报告文本"""
    if not result or "error" in result:
        return f"检测失败: {result.get('error', '未知错误')}" if result else "无检测结果"

    lines = [
        f"任务编号: {result.get('task_id', 'N/A')}",
        f"检测模式: {result.get('mode', 'N/A')}",
        f"图像尺寸: {result.get('image_size', 'N/A')}",
        f"处理耗时: {result.get('detection_time_ms', 0):.1f}ms",
        f"判定结果: {'合格 PASS' if result.get('pass') else '不合格 FAIL'}",
        f"缺陷总数: {result.get('total_defects', 0)}（严重: {result.get('critical_defects', 0)}）",
    ]

    defects = result.get("defects", [])
    if defects:
        lines.append("")
        lines.append("缺陷明细:")
        for d in defects:
            lines.append(
                f"  [{d.get('id', '?')}] {d.get('type', '?')} "
                f"| 等级: {d.get('severity', '?')} "
                f"| 置信度: {d.get('confidence', 0):.3f} "
                f"| 位置: {d.get('location', '?')} "
                f"| {d.get('description', '')}"
            )
    else:
        lines.append("未检测到缺陷")

    return "\n".join(lines)


def _format_cases_for_prompt(cases: List[dict]) -> str:
    """将 RAG 检索到的案例格式化为 LLM prompt 文本"""
    if not cases:
        return "无相似历史案例"

    parts = []
    for i, case in enumerate(cases, 1):
        params_before = case.get("params_before", {})
        params_after = case.get("params_after", {})
        # 计算参数变化
        changes = {}
        for k, v in params_after.items():
            if k in params_before and params_before[k] != v:
                changes[k] = f"{params_before[k]} → {v}"

        parts.append(
            f"--- 案例 {case.get('case_id', f'#{i}')} "
            f"(相似度: {case.get('score', 'N/A')}) ---\n"
            f"缺陷类型: {case.get('defect_type', '?')}\n"
            f"现象: {case.get('phenomenon', '')[:200]}\n"
            f"根因: {case.get('root_cause', '')[:200]}\n"
            f"参数变化: {json.dumps(changes, ensure_ascii=False) if changes else '无变化'}\n"
            f"调参理由: {case.get('tuning_rationale', '')[:200]}\n"
            f"效果: {case.get('result', '')[:200]}"
        )
    return "\n\n".join(parts)


# ============================================================
# Stage 1: defect_analyst — 缺陷分析师
# ============================================================

DEFECT_ANALYST_PROMPT = """你是AOI缺陷分析专家。分析以下PCB检测报告，判断：
1. 产品是否合格（有无严重/致命缺陷）
2. 如果不合格，哪些缺陷需要通过调参改善
3. 给出具体的缺陷类型和严重程度评估

输出格式：
- 判定：合格/不合格
- 需要调参：是/否
- 缺陷摘要：...
- 调参建议方向：...（如需要调参）"""


def defect_analyst(state: AOIWorkflowState) -> dict:
    """
    Stage 1: 缺陷分析师节点

    执行 AOI 检测，使用 LLM 分析检测结果，判定是否需要调参。
    """
    logger.info("=" * 50)
    logger.info("Stage 1: 缺陷分析师 开始")
    t0 = time.perf_counter()

    image_path = state.get("image_path", "")
    detection_mode = state.get("detection_mode", "traditional")

    # 1.1 调用 AOI 检测引擎（通过 tool_registry 统一调度）
    try:
        from tools.registry import registry
        # 确保工具已注册
        if "aoi_detect" not in registry.list_tools():
            from aoi.engine import register_aoi_tools
            register_aoi_tools()
        exec_result = registry.execute("aoi_detect", {
            "image_path": image_path,
            "mode": detection_mode,
        })
        if exec_result.get("success"):
            result = exec_result["result"]
        else:
            raise RuntimeError(exec_result.get("error", "AOI 检测执行失败"))
    except Exception as e:
        logger.error(f"AOI 检测调用失败: {e}")
        return {
            "initial_result": {"error": str(e)},
            "stage": "defect_analyst_error",
            "messages": [f"[缺陷分析师] 检测失败: {e}"],
        }

    # 1.2 格式化报告
    report_text = _format_detection_report(result)
    logger.info(f"检测结果: {report_text[:200]}...")

    # 1.3 LLM 分析
    try:
        llm = _get_llm()
        from langchain_core.messages import SystemMessage, HumanMessage

        analysis = llm.invoke([
            SystemMessage(content=DEFECT_ANALYST_PROMPT),
            HumanMessage(content=f"检测报告如下：\n\n{report_text}"),
        ])
        analysis_text = analysis.content
        logger.info(f"LLM 分析完成: {analysis_text[:200]}...")
    except Exception as e:
        logger.warning(f"LLM 分析失败，使用规则判定: {e}")
        analysis_text = f"检测分析（规则判定）: {report_text}"

    # 1.4 更新状态
    elapsed = time.perf_counter() - t0
    logger.info(f"Stage 1 完成，耗时 {elapsed:.2f}s")

    return {
        "initial_result": result,
        "stage": "defect_analyst",
        "messages": [f"[缺陷分析师]\n{analysis_text}"],
    }


# ============================================================
# 路由决策: should_tune — 是否需要调参
# ============================================================

def should_tune(state: AOIWorkflowState) -> str:
    """
    条件路由节点：根据缺陷分析结果决定是否需要调参。

    路由规则:
      - 检测结果为合格（pass=True）且无严重缺陷 → "end"（无需调参）
      - 检测出错 → "end"
      - 存在严重缺陷 → "tune"（需要调参）
    """
    result = state.get("initial_result")

    # 检测出错，直接结束
    if not result or "error" in result:
        logger.info("路由决策: 检测出错，结束流程")
        return "end"

    # 产品合格且无严重缺陷，无需调参
    if result.get("pass", True) and result.get("critical_defects", 0) == 0:
        logger.info("路由决策: 产品合格，无需调参")
        return "end"

    # 存在严重缺陷，需要调参
    critical = result.get("critical_defects", 0)
    total = result.get("total_defects", 0)
    logger.info(f"路由决策: 严重缺陷={critical}，总缺陷={total}，进入调参流程")
    return "tune"


# ============================================================
# Stage 2: param_optimizer — 参数优化师
# ============================================================

PARAM_OPTIMIZER_PROMPT = """你是AOI参数优化专家。根据检测报告和历史调参案例，推荐最优参数调整方案。

当前参数：{current_params}
检测报告：{detection_report}
相似历史案例：
{rag_results}

请分析以上信息，推荐具体的参数调整。输出JSON格式：
{{"CannyLow": 值, "CannyHigh": 值, "Clahe": 值, "MinArea": 值, "Confidence": 值, "NmsIou": 值}}
每个参数必须给出具体数值，不需要调整的参数可以省略。
调整理由：..."""


def param_optimizer(state: AOIWorkflowState) -> dict:
    """
    Stage 2: 参数优化师节点

    1. 检索 RAG 相似历史案例
    2. 读取当前 XML 配置（如有）
    3. LLM 结合案例 + 检测报告推理推荐参数
    4. 提取结构化参数
    """
    logger.info("=" * 50)
    logger.info("Stage 2: 参数优化师 开始")
    t0 = time.perf_counter()

    initial_result = state.get("initial_result", {})
    config_path = state.get("config_path")

    # 2.1 从检测结果提取缺陷描述用于 RAG 检索
    defect_desc_parts = []
    for d in initial_result.get("defects", []):
        defect_desc_parts.append(
            f"{d.get('type', '')} {d.get('severity', '')} {d.get('description', '')}"
        )
    defect_query = " ".join(defect_desc_parts[:5])  # 取前 5 个缺陷作为查询
    if not defect_query:
        defect_query = initial_result.get("summary", "未知缺陷")

    # 2.2 RAG 检索相似案例（通过 tool_registry 统一调度）
    try:
        from tools.registry import registry
        if "knowledge_search" not in registry.list_tools():
            from tools.searcher import register_rag_tool
            register_rag_tool()
        rag_result = registry.execute("knowledge_search", {
            "query": defect_query,
            "top_k": 5,
        })
        if rag_result.get("success") and rag_result.get("result", {}).get("details"):
            similar_cases = rag_result["result"]["details"]
        else:
            similar_cases = _search_similar_cases(defect_query, top_k=5)
    except Exception:
        similar_cases = _search_similar_cases(defect_query, top_k=5)
    rag_text = _format_cases_for_prompt(similar_cases)
    logger.info(f"RAG 检索到 {len(similar_cases)} 条相似案例")

    # 2.3 读取当前 XML 配置（通过 tool_registry 统一调度）
    current_params = {
        "CannyLow": 50, "CannyHigh": 150, "Clahe": 2.0,
        "MinArea": 100, "Confidence": 0.8, "NmsIou": 0.45,
    }
    if config_path:
        try:
            from tools.registry import registry
            if "xml_config_read" not in registry.list_tools():
                from tools.xml_config import register_xml_config_tools
                register_xml_config_tools()
            read_result = registry.execute("xml_config_read", {"config_path": config_path})
            if read_result.get("success") and read_result.get("result"):
                current_params.update(read_result["result"])
                logger.info(f"已读取当前配置: {current_params}")
            else:
                logger.warning(f"XML 配置读取失败: {read_result.get('error')}")
        except Exception as e:
            logger.warning(f"读取 XML 配置异常: {e}")

    # 2.4 LLM 推理推荐参数
    report_text = _format_detection_report(initial_result)
    try:
        llm = _get_llm()
        from langchain_core.messages import SystemMessage, HumanMessage

        prompt = PARAM_OPTIMIZER_PROMPT.format(
            current_params=json.dumps(current_params, ensure_ascii=False),
            detection_report=report_text,
            rag_results=rag_text,
        )
        response = llm.invoke([
            SystemMessage(content="你是AOI参数优化专家。请严格按照要求输出JSON格式的参数调整方案。"),
            HumanMessage(content=prompt),
        ])
        llm_output = response.content
        logger.info(f"LLM 参数推荐: {llm_output[:300]}...")
    except Exception as e:
        logger.error(f"LLM 参数推理失败: {e}")
        llm_output = ""

    # 2.5 提取结构化参数
    recommended_params = _extract_params_from_llm(llm_output)
    if not recommended_params:
        logger.warning("无法从 LLM 回复中提取参数，使用默认微调策略")
        # 降级：根据缺陷类型使用经验规则
        defect_types = set(d.get("type", "") for d in initial_result.get("defects", []))
        if "短路" in defect_types:
            recommended_params = {"CannyLow": 30}
        elif "断路" in defect_types:
            recommended_params = {"MinArea": 50, "CannyLow": 40}
        elif "焊桥" in defect_types:
            recommended_params = {"CannyHigh": 200, "Confidence": 0.85}
        else:
            recommended_params = {"CannyLow": 40, "MinArea": 50}

    # 2.6 将推荐参数与当前参数合并（推荐值覆盖当前值）
    final_params = {**current_params, **recommended_params}
    logger.info(f"最终推荐参数: {final_params}")

    elapsed = time.perf_counter() - t0
    logger.info(f"Stage 2 完成，耗时 {elapsed:.2f}s")

    return {
        "rag_cases": similar_cases,
        "recommended_params": final_params,
        "stage": "param_optimizer",
        "messages": [
            f"[参数优化师]\n推荐参数: {json.dumps(final_params, ensure_ascii=False)}\n"
            f"LLM 分析:\n{llm_output[:500]}"
        ],
    }


# ============================================================
# Stage 3: config_executor — 配置执行器
# ============================================================

def config_executor(state: AOIWorkflowState) -> dict:
    """
    Stage 3: 配置执行器节点

    1. 将推荐参数写入 XML 配置文件（如有配置路径）
    2. 使用新参数重新执行 AOI 检测
    3. 记录写入结果和复检结果
    """
    logger.info("=" * 50)
    logger.info("Stage 3: 配置执行器 开始")
    t0 = time.perf_counter()

    image_path = state.get("image_path", "")
    detection_mode = state.get("detection_mode", "traditional")
    config_path = state.get("config_path")
    recommended_params = state.get("recommended_params", {})

    # 3.1 写入 XML 配置（通过 tool_registry 统一调度）
    xml_write_result = None
    if config_path:
        try:
            from tools.registry import registry
            if "xml_config_write" not in registry.list_tools():
                from tools.xml_config import register_xml_config_tools
                register_xml_config_tools()
            params_json = json.dumps(recommended_params, ensure_ascii=False)
            xml_write_result = registry.execute("xml_config_write", {
                "config_path": config_path,
                "params": params_json,
            })
            if xml_write_result.get("success"):
                logger.info(f"XML 配置写入成功: {xml_write_result.get('result')}")
            else:
                logger.warning(f"XML 配置写入失败: {xml_write_result.get('error')}")
        except Exception as e:
            logger.error(f"XML 配置写入异常: {e}")
            xml_write_result = {"success": False, "error": str(e)}
    else:
        logger.info("未提供配置路径，跳过 XML 写入，仅使用推荐参数重新检测")

    # 3.2 使用新参数重新检测（通过 tool_registry 统一调度）
    try:
        from tools.registry import registry
        if "aoi_detect" not in registry.list_tools():
            from aoi.engine import register_aoi_tools
            register_aoi_tools()
        exec_result = registry.execute("aoi_detect", {
            "image_path": image_path,
            "mode": detection_mode,
            "canny_low": int(recommended_params.get("CannyLow", 50)),
            "canny_high": int(recommended_params.get("CannyHigh", 150)),
            "clahe_clip": float(recommended_params.get("Clahe", 2.0)),
            "min_area": int(recommended_params.get("MinArea", 100)),
            "conf_thresh": float(recommended_params.get("Confidence", 0.8)),
            "iou_thresh": float(recommended_params.get("NmsIou", 0.45)),
        })
        if exec_result.get("success"):
            reverify_result = exec_result["result"]
        else:
            raise RuntimeError(exec_result.get("error", "复检测执行失败"))
        logger.info(
            f"复检完成: pass={reverify_result.get('pass')}, "
            f"defects={reverify_result.get('total_defects', 0)}, "
            f"critical={reverify_result.get('critical_defects', 0)}"
        )
    except Exception as e:
        logger.error(f"复检测调用失败: {e}")
        reverify_result = {"error": str(e)}

    elapsed = time.perf_counter() - t0
    logger.info(f"Stage 3 完成，耗时 {elapsed:.2f}s")

    return {
        "xml_write_result": xml_write_result,
        "reverify_result": reverify_result,
        "stage": "config_executor",
        "messages": [
            f"[配置执行器]\n"
            f"参数已应用: {json.dumps(recommended_params, ensure_ascii=False)}\n"
            f"复检结果: {_format_detection_report(reverify_result)[:300]}"
        ],
    }


# ============================================================
# Stage 4: verifier — 复检验证器
# ============================================================

VERIFIER_PROMPT = """你是AOI复检验证专家。对比调参前后的检测结果，评估改善效果。

调参前结果：{before}
推荐参数：{params}
调参后结果：{after}

评估维度：
1. 缺陷检出数量变化
2. 严重缺陷是否减少
3. 是否引入新的误报
4. 总体是否改善

输出：
- 改善判定：改善/未改善/需人工确认
- 改善幅度：...
- 风险提示：...
- 最终建议：..."""


def verifier(state: AOIWorkflowState) -> dict:
    """
    Stage 4: 复检验证器节点

    对比调参前后检测结果，使用 LLM 评估改善效果并输出最终结论。
    """
    logger.info("=" * 50)
    logger.info("Stage 4: 复检验证器 开始")
    t0 = time.perf_counter()

    initial_result = state.get("initial_result", {})
    recommended_params = state.get("recommended_params", {})
    reverify_result = state.get("reverify_result", {})

    before_text = _format_detection_report(initial_result)
    after_text = _format_detection_report(reverify_result)
    params_text = json.dumps(recommended_params, ensure_ascii=False)

    # 规则预判（无需 LLM 也能给出基本结论）
    before_defects = initial_result.get("total_defects", 0)
    before_critical = initial_result.get("critical_defects", 0)
    after_defects = reverify_result.get("total_defects", 0)
    after_critical = reverify_result.get("critical_defects", 0)

    rule_verdict = {
        "before_defects": before_defects,
        "before_critical": before_critical,
        "after_defects": after_defects,
        "after_critical": after_critical,
        "defect_reduction": max(0, before_defects - after_defects),
        "critical_reduction": max(0, before_critical - after_critical),
        "improved": after_critical < before_critical or reverify_result.get("pass", False),
    }

    # LLM 深度评估
    llm_verdict = ""
    try:
        llm = _get_llm()
        from langchain_core.messages import SystemMessage, HumanMessage

        prompt = VERIFIER_PROMPT.format(
            before=before_text,
            params=params_text,
            after=after_text,
        )
        response = llm.invoke([
            SystemMessage(content="你是AOI复检验证专家，请客观评估改善效果。"),
            HumanMessage(content=prompt),
        ])
        llm_verdict = response.content
        logger.info(f"LLM 验证完成: {llm_verdict[:200]}...")
    except Exception as e:
        logger.warning(f"LLM 验证失败，使用规则判定: {e}")
        if rule_verdict["improved"]:
            llm_verdict = f"规则判定: 缺陷从{before_defects}降至{after_defects}，有所改善。建议人工确认。"
        else:
            llm_verdict = f"规则判定: 缺陷未明显减少（{before_defects}→{after_defects}），建议调整策略后重试。"

    # 综合最终结论
    final_verdict = {
        "rule_analysis": rule_verdict,
        "llm_evaluation": llm_verdict,
        "recommended_params": recommended_params,
        "conclusion": "改善" if rule_verdict["improved"] else "需进一步调优",
    }

    elapsed = time.perf_counter() - t0
    logger.info(f"Stage 4 完成，耗时 {elapsed:.2f}s")
    logger.info(f"最终结论: {final_verdict['conclusion']}")

    return {
        "final_verdict": final_verdict,
        "stage": "verifier",
        "messages": [
            f"[复检验证器]\n"
            f"改善统计: 缺陷 {before_defects}→{after_defects}, "
            f"严重 {before_critical}→{after_critical}\n"
            f"LLM 评估:\n{llm_verdict[:500]}"
        ],
    }


# ============================================================
# 工作流构建
# ============================================================

def create_aoi_workflow():
    """
    创建并编译 AOI 闭环工作流。

    流程:
      START → defect_analyst → should_tune?
        → "end":  END
        → "tune": param_optimizer → config_executor → verifier → END

    Returns:
        编译后的 LangGraph CompiledGraph 实例
    """
    from langgraph.graph import StateGraph, START, END

    # 创建状态图
    builder = StateGraph(AOIWorkflowState)

    # 注册节点
    builder.add_node("defect_analyst", defect_analyst)
    builder.add_node("param_optimizer", param_optimizer)
    builder.add_node("config_executor", config_executor)
    builder.add_node("verifier", verifier)

    # 添加边
    builder.add_edge(START, "defect_analyst")

    # 条件分支: defect_analyst 之后根据 should_tune 路由
    builder.add_conditional_edges(
        "defect_analyst",
        should_tune,
        {
            "tune": "param_optimizer",
            "end": END,
        },
    )

    # 调参流水线（线性）
    builder.add_edge("param_optimizer", "config_executor")
    builder.add_edge("config_executor", "verifier")
    builder.add_edge("verifier", END)

    # 编译
    workflow = builder.compile()
    logger.info("AOI 闭环工作流编译完成")
    return workflow


# ============================================================
# 便捷入口函数
# ============================================================

def run_aoi_closed_loop(
    image_path: str,
    detection_mode: str = "traditional",
    config_path: str = None,
) -> dict:
    """
    一键运行 AOI 闭环检测流程。

    创建工作流实例、传入初始状态、执行完整流水线，返回最终状态。

    Args:
        image_path: 待检测的 PCB 图片路径
        detection_mode: 检测模式 (traditional/deeplearning/hybrid)
        config_path: XML 配置文件路径（可选，为 None 时跳过 XML 写入）

    Returns:
        最终工作流状态字典，包含:
          - initial_result: 首次检测结果
          - recommended_params: 推荐参数（如经历调参流程）
          - reverify_result: 复检结果（如经历调参流程）
          - final_verdict: 最终评估（如经历调参流程）
          - messages: 各阶段消息日志
    """
    logger.info(f"启动 AOI 闭环检测: image={image_path}, mode={detection_mode}")
    t_start = time.perf_counter()

    # 创建工作流
    workflow = create_aoi_workflow()

    # 初始状态
    initial_state: AOIWorkflowState = {
        "messages": [],
        "image_path": image_path,
        "detection_mode": detection_mode,
        "config_path": config_path,
        "initial_result": None,
        "rag_cases": None,
        "recommended_params": None,
        "xml_write_result": None,
        "reverify_result": None,
        "final_verdict": None,
        "stage": "init",
    }

    # 执行工作流
    try:
        final_state = workflow.invoke(initial_state)
    except Exception as e:
        logger.error(f"工作流执行异常: {e}")
        final_state = {
            **initial_state,
            "stage": "error",
            "messages": [f"[错误] 工作流执行失败: {e}"],
        }

    elapsed = time.perf_counter() - t_start
    logger.info(f"AOI 闭环检测完成，总耗时 {elapsed:.2f}s，最终阶段: {final_state.get('stage', '?')}")

    return final_state


# ============================================================
# 自测入口
# ============================================================

if __name__ == "__main__":
    # 结构性测试：验证工作流能正确编译，不依赖 LLM/API
    print("=" * 60)
    print("AOI 智能闭环工作流 — 结构性自测")
    print("=" * 60)

    # 测试 1: RAG 案例加载
    print("\n[测试1] RAG 案例加载")
    cases = _load_aoi_cases()
    print(f"  已加载 {len(cases)} 条历史案例")
    assert len(cases) == 20, f"期望 20 条案例，实际 {len(cases)} 条"

    # 测试 2: RAG 检索
    print("\n[测试2] RAG 相似案例检索")
    results = _search_similar_cases("BGA焊盘短路误判", top_k=3)
    print(f"  检索到 {len(results)} 条相似案例")
    for r in results:
        print(f"    {r.get('case_id', '?')} | {r.get('defect_type', '?')} | score={r.get('score', 0)}")
    assert len(results) > 0, "RAG 检索结果不应为空"

    # 测试 3: 参数提取
    print("\n[测试3] LLM 回复参数提取")
    test_texts = [
        '{"CannyLow": 30, "CannyHigh": 120, "Clahe": 3.0}',
        '```json\n{"MinArea": 50, "Confidence": 0.7}\n```',
        'CannyLow: 25\nCannyHigh: 130\nClahe: 4.0',
    ]
    for text in test_texts:
        params = _extract_params_from_llm(text)
        print(f"  输入: {text[:50]}...")
        print(f"  提取: {params}")
        assert params is not None, f"参数提取失败: {text}"

    # 测试 4: 工作流编译
    print("\n[测试4] 工作流编译")
    workflow = create_aoi_workflow()
    print(f"  节点: {list(workflow.nodes.keys())}")
    expected_nodes = {"defect_analyst", "param_optimizer", "config_executor", "verifier"}
    actual_nodes = set(workflow.nodes.keys())
    assert expected_nodes.issubset(actual_nodes), f"缺少节点: {expected_nodes - actual_nodes}"

    # 测试 5: 检测报告格式化
    print("\n[测试5] 检测报告格式化")
    mock_result = {
        "task_id": "TEST-001",
        "mode": "传统算法",
        "pass": False,
        "total_defects": 3,
        "critical_defects": 1,
        "detection_time_ms": 45.2,
        "defects": [
            {"id": "T-SC-0001", "type": "短路", "severity": "严重",
             "confidence": 0.85, "location": "(100,200,30,20)", "description": "疑似短路"},
        ],
        "summary": "不合格，缺陷总数=3，严重缺陷=1",
    }
    report = _format_detection_report(mock_result)
    print(f"  报告预览: {report[:150]}...")
    assert "短路" in report
    assert "严重" in report

    # 测试 6: should_tune 路由
    print("\n[测试6] 路由决策逻辑")
    # 合格产品 → end
    route = should_tune({"initial_result": {"pass": True, "critical_defects": 0}})
    print(f"  合格产品 → {route}")
    assert route == "end"

    # 严重缺陷 → tune
    route = should_tune({"initial_result": {"pass": False, "critical_defects": 2, "total_defects": 5}})
    print(f"  不合格产品 → {route}")
    assert route == "tune"

    # 错误结果 → end
    route = should_tune({"initial_result": {"error": "图片不存在"}})
    print(f"  检测出错 → {route}")
    assert route == "end"

    print("\n" + "=" * 60)
    print("全部自测通过!")
    print("=" * 60)
    print("\n工作流拓扑:")
    print("  START → defect_analyst → should_tune?")
    print("    → end:   END")
    print("    → tune:  param_optimizer → config_executor → verifier → END")
