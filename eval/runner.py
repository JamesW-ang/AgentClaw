"""
Agent 系统评估框架 — 自动化测试 + 量化指标

用法:
    python eval_runner.py

输出:
    - 终端实时打印测试结果
    - 生成 eval_report.json (原始数据)
    - 生成 eval_report.md (人类可读报告)

评估维度:
    1. 任务完成率: Agent 是否成功完成目标任务
    2. 工具选择准确率: Agent 是否选择了正确的工具
    3. 工具参数正确率: 工具调用时参数是否正确
    4. 平均响应延迟: 从输入到输出的时间
    5. 错误恢复率: 遇到错误后能否自动恢复
    6. ReAct 循环效率: 平均需要几轮工具调用完成任务
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class TaskType(Enum):
    SINGLE_TOOL = "single_tool"           # 单工具调用
    MULTI_TOOL = "multi_tool"             # 多工具链式调用
    COND_BRANCH = "cond_branch"           # 条件分支判断
    ERROR_RECOVERY = "error_recovery"     # 错误恢复场景
    MULTI_AGENT = "multi_agent"           # 多 Agent 协作
    RAG_RETRIEVAL = "rag_retrieval"       # RAG 检索增强


class EvalResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class ToolCall:
    """记录一次工具调用"""
    tool_name: str
    args: dict[str, Any]
    actual_result: Any
    expected_tool: str
    expected_args: dict[str, Any]
    is_correct_tool: bool
    is_correct_args: bool
    duration_ms: float = 0.0


@dataclass
class TestCase:
    """单个测试用例"""
    id: str
    name: str
    task_type: TaskType
    description: str
    input_message: str
    expected_tools: list[str]              # 期望调用的工具列表 (按顺序)
    expected_args_list: list[dict]         # 期望的工具参数列表
    expected_output_contains: list[str]    # 期望输出包含的关键词
    tags: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    """单个用例的测试结果"""
    case_id: str
    case_name: str
    task_type: str
    result: str                            # EvalResult value
    total_duration_ms: float
    tool_calls: list[dict]                 # 工具调用记录
    actual_tools_used: list[str]
    correct_tools_count: int
    total_expected_tools: int
    correct_args_count: int
    total_args_count: int
    react_rounds: int                      # ReAct 循环轮数
    error_occurred: bool
    error_recovered: bool
    raw_output: str
    notes: str = ""


@dataclass
class EvalReport:
    """完整评估报告"""
    run_id: str
    timestamp: str
    total_cases: int
    passed: int
    failed: int
    partial: int
    errors: int
    task_completion_rate: float            # 任务完成率
    tool_selection_accuracy: float         # 工具选择准确率
    tool_args_accuracy: float              # 工具参数正确率
    avg_latency_ms: float                  # 平均延迟
    avg_react_rounds: float                # 平均 ReAct 轮数
    error_recovery_rate: float             # 错误恢复率
    results_by_type: dict[str, dict]       # 按任务类型分组统计
    case_results: list[dict]               # 每个用例的详细结果
    summary: str = ""                      # 总结文本


class AgentEvaluator:
    """
    Agent 评估器

    支持两种模式:
    1. 实际 Agent 调用模式: 调用真实的 agent 执行测试用例
    2. Mock 模式: 使用预设的模拟结果 (用于开发调试)

    集成方式:
        # 在你的项目中:
        from eval.runner import AgentEvaluator, TestCase

        evaluator = AgentEvaluator(agent_invoke_fn=your_agent_invoke)

        # 添加测试用例
        evaluator.add_case(TestCase(
            id="tc_001",
            name="单工具调用-查询当前参数",
            task_type=TaskType.SINGLE_TOOL,
            description="测试 Agent 能否正确调用 get_current_params 工具",
            input_message="帮我查一下当前的 AOI 检测参数",
            expected_tools=["get_current_params"],
            expected_args_list=[{}],
            expected_output_contains=["曝光", "增益"],
        ))

        # 运行评估
        report = evaluator.run()
        evaluator.save_report("/path/to/output")
    """

    def __init__(self, agent_invoke_fn=None, cases: list[TestCase] | None = None,
                 use_mock: bool = False):
        """
        Args:
            agent_invoke_fn: Agent 调用函数，签名为 (message: str) -> (output: str, tool_calls: list)
            cases: 预设测试用例列表
            use_mock: 是否使用模拟模式
        """
        self.agent_invoke_fn = agent_invoke_fn
        self.cases: list[TestCase] = cases or []
        self.use_mock = use_mock
        self.results: list[CaseResult] = []

    def add_case(self, case: TestCase):
        self.cases.append(case)

    def add_cases(self, cases: list[TestCase]):
        self.cases.extend(cases)

    def run(self) -> EvalReport:
        """运行所有测试用例，返回评估报告"""
        print(f"\n{'='*60}")
        print(f"  Agent 系统评估 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  测试用例数: {len(self.cases)}")
        print(f"  模式: {'Mock' if self.use_mock else '实际调用'}")
        print(f"{'='*60}\n")

        self.results = []
        for i, case in enumerate(self.cases, 1):
            print(f"[{i}/{len(self.cases)}] {case.name} ...", end=" ", flush=True)
            result = self._run_single_case(case)
            self.results.append(result)
            status_icon = {"pass": "✓", "fail": "✗", "partial": "◐", "error": "!"}
            print(f"{status_icon.get(result.result, '?')} [{result.result.upper()}] "
                  f"{result.total_duration_ms:.0f}ms")

        report = self._build_report()
        self._print_summary(report)
        return report

    def _run_single_case(self, case: TestCase) -> CaseResult:
        """执行单个测试用例"""
        start_time = time.time()
        raw_output = ""
        actual_tools = []
        tool_call_details = []
        error_occurred = False
        error_recovered = False

        try:
            if self.use_mock:
                raw_output, tool_call_details = self._mock_invoke(case)
            elif self.agent_invoke_fn:
                raw_output, tool_call_details = self.agent_invoke_fn(case.input_message)
            else:
                raw_output, tool_call_details = " evaluator 未配置", []

            actual_tools = [tc.get("tool_name", tc.get("name", "unknown"))
                            for tc in tool_call_details]

        except Exception as e:
            error_occurred = True
            raw_output = f"ERROR: {str(e)}"
            # 检查是否恢复 (如果后续还有输出则认为恢复)
            if raw_output and "ERROR" not in raw_output[6:]:
                error_recovered = True

        duration_ms = (time.time() - start_time) * 1000

        # 计算指标
        correct_tools = self._check_tool_selection(case, actual_tools)
        correct_args = self._check_tool_args(case, tool_call_details)
        contains_keywords = all(kw in raw_output for kw in case.expected_output_contains)

        # 判定结果
        if error_occurred and not error_recovered:
            result = EvalResult.ERROR
        elif correct_tools == len(case.expected_tools) and contains_keywords:
            result = EvalResult.PASS
        elif correct_tools > 0 or contains_keywords:
            result = EvalResult.PARTIAL
        else:
            result = EvalResult.FAIL

        return CaseResult(
            case_id=case.id,
            case_name=case.name,
            task_type=case.task_type.value,
            result=result.value,
            total_duration_ms=round(duration_ms, 1),
            tool_calls=tool_call_details,
            actual_tools_used=actual_tools,
            correct_tools_count=correct_tools,
            total_expected_tools=len(case.expected_tools),
            correct_args_count=correct_args,
            total_args_count=len(case.expected_args_list) if case.expected_args_list else 0,
            react_rounds=len(tool_call_details),
            error_occurred=error_occurred,
            error_recovered=error_recovered,
            raw_output=raw_output[:500],
        )

    def _check_tool_selection(self, case: TestCase, actual_tools: list[str]) -> int:
        """检查工具选择是否正确"""
        correct = 0
        for expected in case.expected_tools:
            # 支持模糊匹配: 实际工具名包含期望工具名
            for actual in actual_tools:
                if expected.lower() in actual.lower():
                    correct += 1
                    break
        return correct

    def _check_tool_args(self, case: TestCase, tool_calls: list[dict]) -> int:
        """检查工具参数是否正确"""
        if not case.expected_args_list:
            return 0
        correct = 0
        for i, tc in enumerate(tool_calls):
            if i >= len(case.expected_args_list):
                break
            expected_args = case.expected_args_list[i]
            actual_args = tc.get("args", tc.get("arguments", {}))
            if isinstance(actual_args, str):
                try:
                    actual_args = json.loads(actual_args)
                except (json.JSONDecodeError, TypeError):
                    actual_args = {}
            match = True
            for key, val in expected_args.items():
                if key not in actual_args or str(val) not in str(actual_args[key]):
                    match = False
                    break
            if match:
                correct += 1
        return correct

    def _mock_invoke(self, case: TestCase):
        """模拟 Agent 调用 (用于开发和调试)"""
        import random
        time.sleep(random.uniform(0.3, 1.5))

        mock_output = f"针对 '{case.input_message}' 的处理结果: "
        mock_tools = []

        for i, tool in enumerate(case.expected_tools):
            args = case.expected_args_list[i] if i < len(case.expected_args_list) else {}
            mock_tools.append({
                "tool_name": tool,
                "args": args,
                "result": "mock_result",
            })
            mock_output += f"已调用 {tool}, "

        mock_output += "任务完成。"
        for kw in case.expected_output_contains:
            if kw not in mock_output:
                mock_output += f" 相关指标: {kw}。"

        return mock_output, mock_tools

    def _build_report(self) -> EvalReport:
        """构建评估报告"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.result == "pass")
        failed = sum(1 for r in self.results if r.result == "fail")
        partial = sum(1 for r in self.results if r.result == "partial")
        errors = sum(1 for r in self.results if r.result == "error")

        total_tools = sum(r.total_expected_tools for r in self.results)
        correct_tools = sum(r.correct_tools_count for r in self.results)
        total_args = sum(r.total_args_count for r in self.results)
        correct_args = sum(r.correct_args_count for r in self.results)

        latencies = [r.total_duration_ms for r in self.results if r.total_duration_ms > 0]
        rounds = [r.react_rounds for r in self.results if r.react_rounds > 0]

        error_cases = [r for r in self.results if r.error_occurred]
        recovered_cases = [r for r in error_cases if r.error_recovered]

        # 按任务类型分组统计
        results_by_type = {}
        for r in self.results:
            if r.task_type not in results_by_type:
                results_by_type[r.task_type] = {"total": 0, "pass": 0, "avg_ms": 0, "cases": []}
            results_by_type[r.task_type]["total"] += 1
            results_by_type[r.task_type]["pass"] += (1 if r.result == "pass" else 0)
            results_by_type[r.task_type]["avg_ms"] += r.total_duration_ms
            results_by_type[r.task_type]["cases"].append(r.case_id)

        for v in results_by_type.values():
            v["rate"] = round(v["pass"] / v["total"] * 100, 1) if v["total"] > 0 else 0
            v["avg_ms"] = round(v["avg_ms"] / v["total"], 1) if v["total"] > 0 else 0

        summary = self._generate_summary(
            passed, failed, partial, errors, total,
            correct_tools, total_tools, correct_args, total_args,
            latencies, rounds, error_cases, recovered_cases,
        )

        return EvalReport(
            run_id=str(int(time.time())),
            timestamp=datetime.now().isoformat(),
            total_cases=total,
            passed=passed,
            failed=failed,
            partial=partial,
            errors=errors,
            task_completion_rate=round((passed + partial * 0.5) / total * 100, 1) if total > 0 else 0,
            tool_selection_accuracy=round(correct_tools / total_tools * 100, 1) if total_tools > 0 else 0,
            tool_args_accuracy=round(correct_args / total_args * 100, 1) if total_args > 0 else 0,
            avg_latency_ms=round(sum(latencies) / len(latencies), 1) if latencies else 0,
            avg_react_rounds=round(sum(rounds) / len(rounds), 1) if rounds else 0,
            error_recovery_rate=round(len(recovered_cases) / len(error_cases) * 100, 1) if error_cases else 100,
            results_by_type=results_by_type,
            case_results=[asdict(r) for r in self.results],
            summary=summary,
        )

    def _generate_summary(self, passed, failed, partial, errors, total,
                          correct_tools, total_tools, correct_args, total_args,
                          latencies, rounds, error_cases, recovered_cases) -> str:
        completion = round((passed + partial * 0.5) / total * 100, 1) if total > 0 else 0
        lines = [
            f"本次评估共 {total} 个测试用例，通过 {passed} 个，部分通过 {partial} 个，"
            f"失败 {failed} 个，异常 {errors} 个。",
            f"综合任务完成率: {completion}%。",
        ]
        if total_tools > 0:
            lines.append(f"工具选择准确率: {round(correct_tools/total_tools*100,1)}%，"
                         f"参数正确率: {round(correct_args/total_args*100,1)}%。")
        if latencies:
            lines.append(f"平均响应延迟: {round(sum(latencies)/len(latencies),0)}ms，"
                         f"最大延迟: {round(max(latencies),0)}ms。")
        if rounds:
            lines.append(f"平均 ReAct 循环轮数: {round(sum(rounds)/len(rounds),1)}。")
        if error_cases:
            lines.append(f"错误恢复率: {round(len(recovered_cases)/len(error_cases)*100,1)}%。")
        return " ".join(lines)

    def _print_summary(self, report: EvalReport):
        """打印评估摘要"""
        print(f"\n{'='*60}")
        print("  评估结果摘要")
        print(f"{'='*60}")
        print(f"  任务完成率:       {report.task_completion_rate}%")
        print(f"  工具选择准确率:   {report.tool_selection_accuracy}%")
        print(f"  工具参数正确率:   {report.tool_args_accuracy}%")
        print(f"  平均响应延迟:     {report.avg_latency_ms}ms")
        print(f"  平均 ReAct 轮数:  {report.avg_react_rounds}")
        print(f"  错误恢复率:       {report.error_recovery_rate}%")
        print(f"{'─'*60}")

        if report.results_by_type:
            print("  按任务类型:")
            for t, stats in report.results_by_type.items():
                print(f"    {t:20s}  通过率 {stats['rate']:5.1f}%  "
                      f"({stats['pass']}/{stats['total']})  "
                      f"平均 {stats['avg_ms']:.0f}ms")
        print(f"{'='*60}\n")

    def save_report(self, output_dir: str):
        """保存报告到文件"""
        if not self.results:
            print("[WARN] 尚未运行评估，请先调用 run()")
            return

        report = self._build_report()
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        # JSON 原始数据
        json_path = path / "eval_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        print(f"[JSON] {json_path}")

        # Markdown 人类可读报告
        md_path = path / "eval_report.md"
        md_content = self._render_markdown(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[MD]   {md_path}")

    def _render_markdown(self, report: EvalReport) -> str:
        """渲染 Markdown 格式报告"""
        lines = [
            "# Agent 系统评估报告",
            "",
            f"**评估时间**: {report.timestamp}",
            f"**评估 ID**: {report.run_id}",
            f"**测试用例数**: {report.total_cases}",
            "",
            "## 综合指标",
            "",
            "| 指标 | 数值 |",
            "|------|------|",
            f"| 任务完成率 | {report.task_completion_rate}% |",
            f"| 工具选择准确率 | {report.tool_selection_accuracy}% |",
            f"| 工具参数正确率 | {report.tool_args_accuracy}% |",
            f"| 平均响应延迟 | {report.avg_latency_ms}ms |",
            f"| 平均 ReAct 轮数 | {report.avg_react_rounds} |",
            f"| 错误恢复率 | {report.error_recovery_rate}% |",
            "",
            "## 结果分布",
            "",
            "| 状态 | 数量 | 占比 |",
            "|------|------|------|",
            f"| 通过 | {report.passed} | {round(report.passed/report.total_cases*100,1) if report.total_cases else 0}% |",
            f"| 部分通过 | {report.partial} | {round(report.partial/report.total_cases*100,1) if report.total_cases else 0}% |",
            f"| 失败 | {report.failed} | {round(report.failed/report.total_cases*100,1) if report.total_cases else 0}% |",
            f"| 异常 | {report.errors} | {round(report.errors/report.total_cases*100,1) if report.total_cases else 0}% |",
            "",
        ]

        if report.results_by_type:
            lines += [
                "## 分类型统计",
                "",
                "| 任务类型 | 通过率 | 平均延迟 | 用例数 |",
                "|---------|--------|---------|--------|",
            ]
            for t, stats in report.results_by_type.items():
                lines.append(f"| {t} | {stats['rate']}% | {stats['avg_ms']}ms | {stats['total']} |")
            lines.append("")

        # 每个用例的详细结果
        lines += [
            "## 用例详情",
            "",
        ]
        for cr in report.case_results:
            status_icon = {"pass": "✅", "fail": "❌", "partial": "◐️", "error": "⚠️"}
            icon = status_icon.get(cr["result"], "❓")
            lines.append(f"### {icon} {cr['case_name']}")
            lines.append(f"- **类型**: {cr['task_type']}")
            lines.append(f"- **结果**: {cr['result'].upper()}")
            lines.append(f"- **延迟**: {cr['total_duration_ms']}ms")
            lines.append(f"- **ReAct 轮数**: {cr['react_rounds']}")
            lines.append(f"- **工具选择**: {cr['correct_tools_count']}/{cr['total_expected_tools']} 正确")
            if cr["total_args_count"] > 0:
                lines.append(f"- **参数正确**: {cr['correct_args_count']}/{cr['total_args_count']}")
            lines.append(f"- **实际工具**: {', '.join(cr['actual_tools_used']) or '无'}")
            if cr["error_occurred"]:
                lines.append(f"- **错误恢复**: {'是' if cr['error_recovered'] else '否'}")
            lines.append(f"- **输出摘要**: {cr['raw_output'][:200]}")
            lines.append("")

        lines += [
            "## 总结",
            "",
            f"{report.summary}",
            "",
        ]

        return "\n".join(lines)


# ========================================
# 集成到 agent_core.py 的示例
# ========================================
"""
在你的项目中集成评估框架:

1. 在 agent_core.py 中暴露一个 invoke 函数:

    def eval_invoke(message: str):
        result = agent.invoke({"messages": [HumanMessage(content=message)]})
        output = result["messages"][-1].content
        # 从中间步骤提取工具调用记录
        tool_calls = []
        for msg in result["messages"]:
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "tool_name": tc["name"],
                        "args": tc["args"],
                    })
        return output, tool_calls

2. 运行评估:

    from eval.runner import AgentEvaluator
    from eval.cases import get_aoi_eval_cases

    evaluator = AgentEvaluator(agent_invoke_fn=eval_invoke)
    evaluator.add_cases(get_aoi_eval_cases())
    report = evaluator.run()
    evaluator.save_report("./eval_output")
"""


if __name__ == "__main__":
    # 快速演示: 使用 Mock 模式运行
    from eval.cases import get_agent_eval_cases

    evaluator = AgentEvaluator(use_mock=True)
    evaluator.add_cases(get_agent_eval_cases())
    report = evaluator.run()
    evaluator.save_report("./eval_output")

