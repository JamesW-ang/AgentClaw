# ============================================================
# AgentClaw 评估模块单元测试
# ============================================================
"""
覆盖:
    - TextSimilarity: tokenize / jaccard / overlap / bm25 / keyword_overlap
    - AgentEvaluator: RAG 评估 / 工具选择 / 延迟统计
    - 数据结构: RAGSample / ToolCallSample / E2ELatencySample
    - EvaluationReport: 序列化

标记: pytest -m eval（不需要 LLM API, CI 友好）
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# TextSimilarity 测试
# ============================================================

class TestTextSimilarity:
    """文本相似度工具测试"""

    def test_tokenize_english(self):
        """英文分词应返回小写单词列表"""
        from eval.agent_evaluator import TextSimilarity
        tokens = TextSimilarity.tokenize("Hello World Python")
        assert "hello" in tokens
        assert "world" in tokens
        assert "python" in tokens

    def test_tokenize_chinese_bigram(self):
        """中文应生成 bigram"""
        from eval.agent_evaluator import TextSimilarity
        tokens = TextSimilarity.tokenize("数字孪生")
        assert "数字" in tokens
        assert "字孪" in tokens
        assert "孪生" in tokens

    def test_tokenize_mixed(self):
        """中英混合分词"""
        from eval.agent_evaluator import TextSimilarity
        tokens = TextSimilarity.tokenize("AgentClaw是Python框架")
        # \w+ 会把中英混在一起匹配, 所以至少有英文 token
        assert any("agent" in t or "python" in t for t in tokens)
        assert len(tokens) >= 3

    def test_tokenize_single_chinese_char(self):
        """中文 bigram 生成"""
        from eval.agent_evaluator import TextSimilarity
        tokens = TextSimilarity.tokenize("你好A")
        # 两个汉字至少生成一个 bigram
        assert "你好" in tokens
        # \\w+ 把 \"你好A\" 整体匹配为单个 token, 英文部分单独匹配
        assert len(tokens) >= 1

    def test_jaccard_identical(self):
        """相同文本 Jaccard = 1.0"""
        from eval.agent_evaluator import TextSimilarity
        t = TextSimilarity.tokenize("hello world")
        assert TextSimilarity.jaccard(t, t) == 1.0

    def test_jaccard_disjoint(self):
        """完全不同文本 Jaccard = 0.0"""
        from eval.agent_evaluator import TextSimilarity
        a = TextSimilarity.tokenize("hello world")
        b = TextSimilarity.tokenize("foo bar baz")
        assert TextSimilarity.jaccard(a, b) == 0.0

    def test_jaccard_partial(self):
        """部分重叠"""
        from eval.agent_evaluator import TextSimilarity
        a = TextSimilarity.tokenize("hello world python")
        b = TextSimilarity.tokenize("hello world java")
        score = TextSimilarity.jaccard(a, b)
        assert 0.0 < score < 1.0

    def test_jaccard_empty(self):
        """空文本 Jaccard = 0.0"""
        from eval.agent_evaluator import TextSimilarity
        assert TextSimilarity.jaccard([], ["hello"]) == 0.0
        assert TextSimilarity.jaccard([], []) == 0.0

    def test_overlap_identical(self):
        """相同文本重叠系数 = 1.0"""
        from eval.agent_evaluator import TextSimilarity
        t = TextSimilarity.tokenize("hello world test")
        assert TextSimilarity.overlap_coefficient(t, t) == 1.0

    def test_overlap_subset(self):
        """子集重叠系数 = 1.0"""
        from eval.agent_evaluator import TextSimilarity
        a = TextSimilarity.tokenize("hello world")
        b = TextSimilarity.tokenize("hello world test extra")
        assert TextSimilarity.overlap_coefficient(a, b) == 1.0

    def test_bm25_positive(self):
        """BM25 评分应为正数"""
        from eval.agent_evaluator import TextSimilarity
        q = TextSimilarity.tokenize("python agent framework")
        d = TextSimilarity.tokenize("python agent framework is great")
        assert TextSimilarity.bm25_score(q, d) > 0

    def test_bm25_empty(self):
        """空文本 BM25 = 0.0"""
        from eval.agent_evaluator import TextSimilarity
        assert TextSimilarity.bm25_score([], ["hello"]) == 0.0
        assert TextSimilarity.bm25_score(["hello"], []) == 0.0

    def test_keyword_overlap_full(self):
        """关键词完全覆盖 = 1.0"""
        from eval.agent_evaluator import TextSimilarity
        assert TextSimilarity.keyword_overlap("hello world test", "hello world test") == 1.0

    def test_keyword_overlap_empty_ref(self):
        """空参考文本 = 0.0"""
        from eval.agent_evaluator import TextSimilarity
        assert TextSimilarity.keyword_overlap("hello", "") == 0.0

    def test_keyword_overlap_partial(self):
        """部分关键词覆盖"""
        from eval.agent_evaluator import TextSimilarity
        score = TextSimilarity.keyword_overlap("hello world", "hello world test extra")
        assert 0.0 < score < 1.0


# ============================================================
# AgentEvaluator - RAG 评估
# ============================================================

class TestRAGEvaluation:
    """RAG 评估测试"""

    def _make_evaluator(self):
        from eval.agent_evaluator import AgentEvaluator
        return AgentEvaluator(mode="fast", persist_dir=tempfile.mkdtemp())

    def test_rag_empty(self):
        """无样本时 RAG 指标全部为 0"""
        ev = self._make_evaluator()
        report = ev.evaluate()
        assert report.rag is not None
        assert report.rag.context_precision == 0.0
        assert report.rag.context_recall == 0.0
        assert report.rag.answer_relevancy == 0.0
        assert report.rag.faithfulness == 0.0
        assert report.rag.answer_correctness == -1.0

    def test_rag_perfect_match(self):
        """完美匹配应得高分 (recall/faithfulness/correctness)"""
        ev = self._make_evaluator()
        ev.add_rag_sample(
            question="AgentClaw 支持哪些语言?",
            contexts=["AgentClaw 使用 Python 开发, 基于 LangGraph 框架。"],
            answer="AgentClaw 使用 Python 开发, 基于 LangGraph 框架。",
            ground_truth="AgentClaw 使用 Python 开发, 基于 LangGraph 框架。",
        )
        report = ev.evaluate()
        assert report.rag.context_recall > 0.5
        assert report.rag.faithfulness > 0.5
        assert report.rag.answer_correctness > 0.5

    def test_rag_hallucination(self):
        """幻觉回答 Faithfulness 应低分"""
        ev = self._make_evaluator()
        ev.add_rag_sample(
            question="AgentClaw 是什么?",
            contexts=["AgentClaw 是一个 Python AI Agent 框架。"],
            answer="AgentClaw 是一个由 Google 开发的商业级 AI Agent 框架, 支持实时语音交互。",
        )
        report = ev.evaluate()
        # 回答包含"Google"、"商业级"、"实时语音交互"等上下文中没有的内容
        assert report.rag.faithfulness < 0.8

    def test_rag_no_ground_truth(self):
        """无 ground_truth 时 answer_correctness = -1"""
        ev = self._make_evaluator()
        ev.add_rag_sample(
            question="test",
            contexts=["context"],
            answer="answer",
        )
        report = ev.evaluate()
        assert report.rag.answer_correctness == -1.0

    def test_rag_multiple_samples(self):
        """多条样本应正确聚合"""
        ev = self._make_evaluator()
        for i in range(5):
            ev.add_rag_sample(
                question=f"问题 {i}?",
                contexts=[f"上下文 {i} 的内容"],
                answer=f"答案是 {i} 的内容",
                ground_truth=f"答案是 {i} 的内容",
            )
        report = ev.evaluate()
        assert report.rag.context_precision > 0.0
        assert report.rag.context_recall > 0.0


# ============================================================
# AgentEvaluator - 工具选择评估
# ============================================================

class TestToolEvaluation:
    """工具选择评估测试"""

    def _make_evaluator(self):
        from eval.agent_evaluator import AgentEvaluator
        return AgentEvaluator(mode="fast", persist_dir=tempfile.mkdtemp())

    def test_tool_empty(self):
        """无样本时工具指标为 0"""
        ev = self._make_evaluator()
        report = ev.evaluate()
        assert report.tool.total_calls == 0
        assert report.tool.selection_accuracy == 0.0

    def test_tool_all_correct(self):
        """全部正确: accuracy = 1.0"""
        ev = self._make_evaluator()
        ev.add_tool_sample("q1", "web_search", "web_search", True, 1.0)
        ev.add_tool_sample("q2", "calculator", "calculator", True, 0.1)
        report = ev.evaluate()
        assert report.tool.selection_accuracy == 1.0
        assert report.tool.correct_calls == 2

    def test_tool_all_wrong(self):
        """全部错误: accuracy = 0.0"""
        ev = self._make_evaluator()
        ev.add_tool_sample("q1", "web_search", "file_read", False, 0.5)
        ev.add_tool_sample("q2", "calculator", "web_search", False, 1.0)
        report = ev.evaluate()
        assert report.tool.selection_accuracy == 0.0

    def test_tool_partial_accuracy(self):
        """部分正确"""
        ev = self._make_evaluator()
        for _ in range(7):
            ev.add_tool_sample("q", "web_search", "web_search", True)
        for _ in range(3):
            ev.add_tool_sample("q", "web_search", "file_read", False)
        report = ev.evaluate()
        assert report.tool.selection_accuracy == 0.7

    def test_tool_per_tool_accuracy(self):
        """每个工具的准确率应正确计算"""
        ev = self._make_evaluator()
        ev.add_tool_sample("q1", "web_search", "web_search", True)
        ev.add_tool_sample("q2", "web_search", "web_search", True)
        ev.add_tool_sample("q3", "web_search", "file_read", False)
        ev.add_tool_sample("q4", "calculator", "calculator", True)
        report = ev.evaluate()
        assert "web_search" in report.tool.per_tool_accuracy
        assert report.tool.per_tool_accuracy["calculator"] == 1.0
        # web_search 有 2 正确 + 1 错误(file_read),  注意错选的算在 file_read 上
        assert report.tool.per_tool_accuracy["web_search"] > 0


# ============================================================
# AgentEvaluator - 延迟评估
# ============================================================

class TestLatencyEvaluation:
    """延迟评估测试"""

    def _make_evaluator(self):
        from eval.agent_evaluator import AgentEvaluator
        return AgentEvaluator(mode="fast", persist_dir=tempfile.mkdtemp())

    def test_percentile_basic(self):
        """百分位数计算正确"""
        from eval.agent_evaluator import AgentEvaluator
        assert AgentEvaluator._percentile([1, 2, 3, 4, 5], 0.5) == 3.0
        assert AgentEvaluator._percentile([10, 20, 30], 0.50) == 20.0

    def test_percentile_empty(self):
        """空列表百分位数 = 0.0"""
        from eval.agent_evaluator import AgentEvaluator
        assert AgentEvaluator._percentile([], 0.95) == 0.0

    def test_percentile_single(self):
        """单元素百分位数 = 该值"""
        from eval.agent_evaluator import AgentEvaluator
        assert AgentEvaluator._percentile([42.0], 0.99) == 42.0

    def test_latency_metrics(self):
        """延迟指标应正确计算"""
        ev = self._make_evaluator()
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        for lat in latencies:
            ev.add_latency_sample("q", lat)
        report = ev.evaluate()
        assert report.latency is not None
        assert report.latency.mean == 3.0
        assert report.latency.min == 1.0
        assert report.latency.max == 5.0
        assert report.latency.sample_count == 5
        assert report.latency.p50 > 0

    def test_latency_no_samples(self):
        """无样本时延迟为 None"""
        ev = self._make_evaluator()
        report = ev.evaluate()
        assert report.latency is None

    def test_tool_latency(self):
        """各工具延迟应正确分组"""
        ev = self._make_evaluator()
        ev.add_tool_sample("q1", "web_search", "web_search", True, 1.0)
        ev.add_tool_sample("q2", "web_search", "web_search", True, 2.0)
        ev.add_tool_sample("q3", "calculator", "calculator", True, 0.05)
        report = ev.evaluate()
        assert "web_search" in report.tool_latency
        assert "calculator" in report.tool_latency
        assert report.tool_latency["web_search"].mean == 1.5
        assert report.tool_latency["calculator"].mean == 0.05


# ============================================================
# AgentEvaluator - 综合测试
# ============================================================

class TestEvaluatorIntegration:
    """评估器集成测试"""

    def test_mixed_samples(self):
        """混合样本应全部正确评估"""
        from eval.agent_evaluator import AgentEvaluator
        ev = AgentEvaluator(mode="fast", persist_dir=tempfile.mkdtemp())

        ev.add_rag_sample("q?", ["ctx"], "ans", "gt")
        ev.add_tool_sample("q", "t1", "t1", True, 0.5)
        ev.add_latency_sample("q", 1.0, 2, True)

        report = ev.evaluate()
        assert report.sample_count == 3
        assert report.rag is not None
        assert report.tool.total_calls == 1
        assert report.latency is not None

    def test_report_serialization(self):
        """报告应能正确序列化为 JSON"""
        from eval.agent_evaluator import AgentEvaluator, asdict
        ev = AgentEvaluator(mode="fast", persist_dir=tempfile.mkdtemp())
        ev.add_rag_sample("q?", ["ctx"], "ans", "gt")
        report = ev.evaluate()

        data = json.loads(json.dumps(asdict(report), default=str))
        assert "rag" in data
        assert data["mode"] == "fast"
        assert data["sample_count"] == 1

    def test_print_report_no_crash(self):
        """print_report 不应崩溃"""
        from eval.agent_evaluator import AgentEvaluator
        ev = AgentEvaluator(mode="fast", persist_dir=tempfile.mkdtemp())
        ev.add_rag_sample("q?", ["ctx"], "ans", "gt")
        report = ev.evaluate()
        ev.print_report(report)  # should not raise

    def test_load_from_feedback(self):
        """从 FeedbackCollector 加载"""
        from dataclasses import dataclass

        from eval.agent_evaluator import AgentEvaluator

        ev = AgentEvaluator(mode="fast", persist_dir=tempfile.mkdtemp())

        # 模拟 FeedbackSignal
        @dataclass
        class FakeSignal:
            task_id: str = "task-1"
            tool_name: str = "web_search"
            success: bool = True
            latency: float = 1.5
            error_type: str = None
            context: str = "test context"

        signals = [FakeSignal() for _ in range(5)]
        signals[4].success = False
        ev.load_from_feedback(signals)
        report = ev.evaluate()

        assert report.latency is not None
        assert report.latency.sample_count == 5
        assert report.agent_success_rate == 0.8

    def test_save_report_creates_file(self):
        """评估报告应保存为文件"""
        from eval.agent_evaluator import AgentEvaluator
        tmpdir = tempfile.mkdtemp()
        ev = AgentEvaluator(mode="fast", persist_dir=tmpdir)
        ev.add_rag_sample("q?", ["ctx"], "ans")
        ev.evaluate()

        files = os.listdir(tmpdir)
        assert len(files) == 1
        assert files[0].startswith("eval_report_")

    def test_agent_success_rate(self):
        """任务成功率应正确计算"""
        from eval.agent_evaluator import AgentEvaluator
        ev = AgentEvaluator(mode="fast", persist_dir=tempfile.mkdtemp())
        for _ in range(8):
            ev.add_latency_sample("q", 1.0, 1, True)
        for _ in range(2):
            ev.add_latency_sample("q", 1.0, 1, False)
        report = ev.evaluate()
        assert report.agent_success_rate == 0.8

    def test_avg_reasoning_steps(self):
        """平均推理步数应正确计算"""
        from eval.agent_evaluator import AgentEvaluator
        ev = AgentEvaluator(mode="fast", persist_dir=tempfile.mkdtemp())
        ev.add_latency_sample("q1", 1.0, 3, True)
        ev.add_latency_sample("q2", 2.0, 1, True)
        ev.add_latency_sample("q3", 1.5, 2, True)
        report = ev.evaluate()
        assert report.avg_reasoning_steps == 2.0


if __name__ == "__main__":
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
