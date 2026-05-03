# ============================================================
# AgentClaw — 学习系统单元测试
# ============================================================

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFeedbackSignal:
    """反馈信号数据结构"""

    def test_defaults(self):
        from learning.feedback import FeedbackSignal
        signal = FeedbackSignal(
            task_id="t1", tool_name="web_search",
            success=True, latency=0.5,
        )
        assert signal.task_id == "t1"
        assert signal.tool_name == "web_search"
        assert signal.success is True
        assert signal.latency == 0.5
        assert signal.timestamp > 0

    def test_error_type_default(self):
        from learning.feedback import FeedbackSignal
        s = FeedbackSignal(task_id="t1", tool_name="calc", success=True, latency=0.1)
        assert s.error_type is None

    def test_context_default(self):
        from learning.feedback import FeedbackSignal
        s = FeedbackSignal(task_id="t1", tool_name="calc", success=True, latency=0.1)
        assert s.context == ""


class TestFeedbackCollector:
    """反馈采集器"""

    @pytest.fixture
    def collector(self):
        from learning.feedback import FeedbackCollector
        with tempfile.TemporaryDirectory() as tmp:
            c = FeedbackCollector(persist_dir=tmp)
            yield c

    def test_collect(self, collector):
        from learning.feedback import FeedbackSignal
        collector.collect(FeedbackSignal("t1", "web_search", True, 0.5))
        recent = collector.get_recent()
        assert len(recent) == 1
        assert recent[0].tool_name == "web_search"

    def test_get_recent_limit(self, collector):
        from learning.feedback import FeedbackSignal
        for i in range(10):
            collector.collect(FeedbackSignal(f"t{i}", "calc", True, 0.1))
        recent = collector.get_recent(n=3)
        assert len(recent) == 3

    def test_get_recent_empty(self, collector):
        recent = collector.get_recent()
        assert recent == []

    def test_get_by_tool(self, collector):
        from learning.feedback import FeedbackSignal
        collector.collect(FeedbackSignal("t1", "search", True, 0.5))
        collector.collect(FeedbackSignal("t2", "calc", True, 0.1))
        collector.collect(FeedbackSignal("t3", "search", False, 1.0))
        search_signals = collector.get_by_tool("search")
        assert len(search_signals) == 2
        calc_signals = collector.get_by_tool("calc")
        assert len(calc_signals) == 1

    def test_get_by_tool_empty(self, collector):
        assert collector.get_by_tool("nonexistent") == []

    def test_persist_and_reload(self, collector):
        from learning.feedback import FeedbackCollector, FeedbackSignal
        collector.collect(FeedbackSignal("t1", "search", True, 0.5))
        collector.collect(FeedbackSignal("t2", "calc", False, 1.0))

        # Trigger internal save
        collector._save()

        # New collector from the same persist_dir (attribute is persist_dir, not _persist_dir)
        collector2 = FeedbackCollector(persist_dir=collector.persist_dir)
        recent = collector2.get_recent()
        assert len(recent) == 2


class TestStrategy:
    """执行策略"""

    def test_creation(self):
        from learning.learner import Strategy
        s = Strategy(
            name="test_strat",
            trigger_pattern="web_search",
            tool_sequence=["web_search", "file_read"],
            success_rate=0.8,
        )
        assert s.name == "test_strat"
        assert s.success_rate == 0.8
        assert s.anti_pattern is False

    def test_effectiveness_below_3(self):
        from learning.learner import Strategy
        s = Strategy("s", "t", ["t"], 0.8, confidence=0.6)
        assert s.usage_count == 0
        eff = s.effectiveness
        assert eff == 0.8 * 0.6

    def test_effectiveness_above_3(self):
        from learning.learner import Strategy
        s = Strategy("s", "t", ["t"], 0.8, confidence=0.6)
        s.usage_count = 4
        s.success_after_recommend = 3
        eff = s.effectiveness
        assert eff == (3 / 4) * 0.6

    def test_record_outcome_success(self):
        from learning.learner import Strategy
        s = Strategy("s", "t", ["t"], 0.8)
        s.record_outcome(success=True)
        assert s.usage_count == 1
        assert s.success_after_recommend == 1

    def test_record_outcome_failure(self):
        from learning.learner import Strategy
        s = Strategy("s", "t", ["t"], 0.8)
        s.record_outcome(success=False)
        assert s.usage_count == 1
        assert s.success_after_recommend == 0

    def test_anti_pattern_flag(self):
        from learning.learner import Strategy
        s = Strategy("bad", "t", ["t"], 0.1, anti_pattern=True)
        assert s.anti_pattern is True
        assert "ANTI" in repr(s)

    def test_repr(self):
        from learning.learner import Strategy
        s = Strategy("test", "t", ["t"], 0.9, confidence=0.8)
        r = repr(s)
        assert "STRAT" in r
        assert "test" in r

    def test_update_timestamp(self):
        from learning.learner import Strategy
        s = Strategy("s", "t", ["t"], 0.8)
        old = s.last_updated
        time.sleep(0.01)
        s.record_outcome(success=True)
        assert s.last_updated > old


class TestExperienceLearner:
    """经验学习器"""

    @pytest.fixture
    def learner(self):
        from learning.feedback import FeedbackCollector
        from learning.learner import ExperienceLearner
        collector = FeedbackCollector()
        return ExperienceLearner(collector)

    def test_learn_from_empty(self, learner):
        learner.learn_from_history([])
        # should not crash; strategies dict exists
        assert learner.strategies is not None

    def test_learn_from_signals(self, learner):
        from learning.feedback import FeedbackSignal
        signals = [
            FeedbackSignal("t1", "web_search", True, 0.5, context="search python"),
            FeedbackSignal("t2", "calculator", True, 0.3, context="calculate math"),
            FeedbackSignal("t3", "web_search", True, 0.4, context="search documentation"),
        ]
        learner.learn_from_history(signals)
        # Even without sequences of 2+, should not crash
        assert learner.strategies is not None

    def test_anti_pattern_detection(self, learner):
        from learning.learner import Strategy
        ap = Strategy(
            name="cycle_search",
            trigger_pattern="web_search",
            tool_sequence=["web_search", "web_search"],
            success_rate=0.2,
            anti_pattern=True,
        )
        learner.strategies[ap.name] = ap
        result = learner.check_anti_pattern(["web_search", "web_search", "calculator"])
        assert result is not None
        assert result.name == "cycle_search"

    def test_anti_pattern_no_match(self, learner):
        from learning.learner import Strategy
        ap = Strategy("bad", "x", ["x", "y"], 0.1, anti_pattern=True)
        learner.strategies[ap.name] = ap
        result = learner.check_anti_pattern(["a", "b", "c"])
        assert result is None

    def test_anti_pattern_skips_non_anti(self, learner):
        from learning.learner import Strategy
        good = Strategy("good", "a", ["a", "b"], 0.9, anti_pattern=False)
        learner.strategies[good.name] = good
        result = learner.check_anti_pattern(["a", "b"])
        assert result is None

    def test_get_strategy_empty(self, learner):
        result = learner.get_strategy("search")
        assert result is None

    def test_get_strategy_exact_match(self, learner):
        from learning.learner import Strategy
        learner.strategies["search_tool"] = Strategy(
            "search_tool", "search_query", ["web_search"], 0.9,
        )
        result = learner.get_strategy("search_query")
        assert result is not None
        assert result.name == "search_tool"


class TestEvolutionManager:
    """进化管理器"""

    @pytest.fixture
    def evolution(self):
        from learning.evolution import EvolutionManager
        from learning.feedback import FeedbackCollector
        from learning.learner import ExperienceLearner
        collector = FeedbackCollector()
        learner = ExperienceLearner(collector)
        optimizer = MagicMock()
        optimizer.route_weights = {"web_search": 1.0, "calculator": 0.8}
        optimizer.analyze_failure_patterns.return_value = {}
        return EvolutionManager(collector, learner, optimizer)

    def test_initial_state(self, evolution):
        assert evolution._running is False
        assert evolution._cycle_count == 0

    def test_check_current_pattern_no_strategies(self, evolution):
        result = evolution.check_current_pattern(["web_search"])
        assert result is None

    def test_check_current_pattern_with_strategy(self, evolution):
        from learning.learner import Strategy
        s = Strategy("search_pattern", "search", ["web_search", "file_read"], 0.9, anti_pattern=True)
        evolution.learner.strategies[s.name] = s
        result = evolution.check_current_pattern(["web_search", "file_read"])
        assert result is not None

    def test_get_evolution_report_empty(self, evolution):
        report = evolution.get_evolution_report()
        assert "total_cycles" in report
        assert "overall_success_rate" in report

    def test_get_metrics(self, evolution):
        metrics = evolution.get_metrics()
        assert "cycles_completed" in metrics
        assert "strategy_count" in metrics
        assert "anti_pattern_count" in metrics

    def test_run_evolution_cycle(self, evolution):
        evolution._run_evolution_cycle()
        assert evolution._cycle_count == 1
        report = evolution.get_evolution_report()
        assert report["total_cycles"] == 1

    def test_stop(self, evolution):
        evolution._running = True
        evolution.stop()
        assert evolution._running is False

    def test_start_stop_loop(self, evolution):
        evolution.start_evolution_loop(interval=0.01)
        assert evolution._running is True
        time.sleep(0.08)
        evolution.stop()
        assert evolution._running is False
        assert evolution._cycle_count >= 1

    def test_get_metrics_after_cycle(self, evolution):
        evolution._run_evolution_cycle()
        metrics = evolution.get_metrics()
        assert metrics["cycles_completed"] == 1
