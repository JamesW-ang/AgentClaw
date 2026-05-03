"""
进化管理器 v2 — 协调反馈收集、经验学习和自适应优化的完整生命周期

v2 升级 (不破坏现有架构):
    1. 策略效果评估 — 每轮进化检查推荐策略是否真的提升了成功率
    2. 反模式预警 — 进化循环中检查当前工具调用是否命中反模式
    3. 失败模式分析 — 调用 AdaptiveOptimizer 的分类报告
    4. 进化指标追踪 — 记录每轮进化的关键指标, 追踪长期趋势
    5. 工具偏好学习 — 在进化循环中更新工具偏好矩阵
    6. 进化报告 — 生成人类可读的进化报告

向后兼容:
    EvolutionManager(collector, learner, optimizer) — 构造函数不变
    start_evolution_loop(interval) — 签名不变
    stop() — 签名不变
"""

import time
from collections import defaultdict

from core.logger import get_logger
from learning.feedback import FeedbackCollector
from learning.learner import ExperienceLearner, Strategy
from learning.optimizer import AdaptiveOptimizer

logger = get_logger("evolution_manager")


class EvolutionManager:
    """
    进化管理器 v2

    v1 → v2 变更:
        - _run_evolution_cycle: 4步 → 7步 (新增效果评估/反模式检查/报告生成)
        - 新增 evaluate_strategies — 策略效果评估
        - 新增 get_evolution_report — 进化报告
        - 新增 get_metrics — 进化指标

    v2 进化循环 (每轮):
        1. 收集反馈数据 (最近 200 条)
        2. 学习成功策略 + 反模式 (ExperienceLearner v2)
        3. 更新路由权重 (AdaptiveOptimizer v2, 自适应 alpha)
        4. 更新工具偏好矩阵 (AdaptiveOptimizer v2)
        5. 优化 Prompt 模板 (AdaptiveOptimizer v2, 失败分类)
        6. 评估策略效果 (检查推荐后成功率)
        7. 生成进化报告
    """

    def __init__(
        self,
        collector: FeedbackCollector,
        learner: ExperienceLearner,
        optimizer: AdaptiveOptimizer,
    ):
        self.collector = collector
        self.learner = learner
        self.optimizer = optimizer
        self._running = False
        self._interval = 3600

        # v2: 进化指标追踪
        self._cycle_count = 0
        self._metrics_history: list[dict] = []
        self._last_cycle_time = 0.0

        logger.info("EvolutionManager v2 初始化完成")

    def start_evolution_loop(self, interval: int = 3600):
        """
        启动后台进化循环 (向后兼容, 签名不变)
        """
        self._interval = interval
        self._running = True

        def _loop():
            while self._running:
                time.sleep(interval)
                try:
                    self._run_evolution_cycle()
                except Exception as e:
                    logger.error(f"进化循环异常: {e}", exc_info=True)

        thread = __import__("threading").Thread(target=_loop, daemon=True)
        thread.start()
        logger.info(f"进化循环已启动, 间隔 {interval} 秒")

    def stop(self):
        """停止进化循环"""
        self._running = False
        logger.info("进化循环已停止")

    # ============================================================
    # v2 进化循环 (7步)
    # ============================================================

    def _run_evolution_cycle(self):
        """执行一轮完整进化循环 (v2: 7步)"""
        cycle_start = time.time()
        self._cycle_count += 1
        cycle_id = self._cycle_count

        logger.info(f"{'='*50}")
        logger.info(f"进化循环 #{cycle_id} 开始")
        logger.info(f"{'='*50}")

        # ==================== 步骤 1: 收集反馈数据 ====================
        feedback = list(self.collector.get_recent(200))
        if not feedback:
            logger.info("无反馈数据, 跳过本轮")
            return

        success_count = sum(1 for f in feedback if f.success)
        failure_count = len(feedback) - success_count
        success_rate = success_count / len(feedback)
        logger.info(f"反馈统计: {len(feedback)} 条 (成功 {success_count}, "
                     f"失败 {failure_count}, 成功率 {success_rate:.1%})")

        # ==================== 步骤 2: 学习策略 + 反模式 ====================
        self.learner.learn_from_history(feedback)

        strat_count = sum(1 for s in self.learner.strategies.values() if not s.anti_pattern)
        anti_count = sum(1 for s in self.learner.strategies.values() if s.anti_pattern)
        logger.info(f"策略库: {strat_count} 正向策略, {anti_count} 反模式")

        # ==================== 步骤 3: 更新路由权重 ====================
        for signal in feedback[-50:]:
            self.optimizer.update_route_weights(signal.tool_name, signal.success)

        # 路由权重 Top5
        weights = self.optimizer.route_weights
        if weights:
            top_routes = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info("路由权重 Top5: " + ", ".join(
                f"{k}={v:.3f}" for k, v in top_routes))

        # ==================== 步骤 4: 更新工具偏好矩阵 ====================
        for signal in feedback[-100:]:
            if signal.context:
                # 从上下文中提取关键词作为 context
                context_keywords = self._extract_context_keywords(signal.context)
                for ctx in context_keywords:
                    self.optimizer.update_tool_preference(
                        signal.tool_name, ctx, signal.success
                    )

        # ==================== 步骤 5: 优化 Prompt 模板 ====================
        self.optimizer.optimize_prompt("default", feedback)

        # ==================== 步骤 6: 评估策略效果 ====================
        self._evaluate_strategies(feedback)

        # ==================== 步骤 7: 生成进化报告 ====================
        cycle_duration = time.time() - cycle_start
        report = self._build_cycle_report(
            cycle_id, feedback, cycle_duration
        )
        self._metrics_history.append(report)

        # 打印摘要
        logger.info(f"进化循环 #{cycle_id} 完成 (耗时 {cycle_duration:.2f}s)")
        logger.info(f"摘要: 策略 {strat_count} 条, "
                     f"整体成功率 {success_rate:.1%}, "
                     f"最佳路由 {max(weights.items(), key=lambda x: x[1])[0] if weights else 'N/A'}")

        self._last_cycle_time = cycle_start

    # ============================================================
    # v2: 策略效果评估
    # ============================================================

    def _evaluate_strategies(self, feedback: list):
        """
        评估已推荐策略的实际效果

        对每条反馈, 检查是否有策略被推荐给对应 task_id,
        如果有, 记录实际执行结果到 strategy.record_outcome()
        """
        # 按工具名分组最近的反馈
        tool_feedback = defaultdict(list)
        for f in feedback[-100:]:
            tool_feedback[f.tool_name].append(f)

        evaluated = 0
        for name, strat in self.learner.strategies.items():
            if strat.anti_pattern or strat.usage_count == 0:
                continue

            # 检查策略中的工具是否有最近的反馈
            for tool in strat.tool_sequence:
                if tool in tool_feedback:
                    recent = tool_feedback[tool][-5:]
                    successes = sum(1 for f in recent if f.success)
                    if successes > 0:
                        strat.record_outcome(success=True)
                        evaluated += 1
                    elif len(recent) > 0:
                        strat.record_outcome(success=False)
                        evaluated += 1

        if evaluated > 0:
            # 显示效果最好的策略
            active = [(n, s) for n, s in self.learner.strategies.items()
                      if not s.anti_pattern and s.usage_count >= 3]
            if active:
                top3 = sorted(active, key=lambda x: x[1].effectiveness, reverse=True)[:3]
                logger.info("策略效果 Top3: " + ", ".join(
                    f"{s.name}(eff={s.effectiveness:.2f})" for _, s in top3))

    # ============================================================
    # v2: 反模式预警 (供外部调用)
    # ============================================================

    def check_current_pattern(self, tool_sequence: list) -> Strategy | None:
        """
        检查当前工具调用序列是否命中反模式

        可在 ReAct 循环中调用, 实时预警

        Args:
            tool_sequence: 当前已执行的工具名称列表

        Returns:
            命中的反模式 Strategy, 或 None
        """
        anti = self.learner.check_anti_pattern(tool_sequence)
        if anti:
            logger.warning(f"反模式预警: 当前调用序列命中 '{anti.name}', "
                           f"建议避免继续此路径")
        return anti

    # ============================================================
    # v2: 进化报告
    # ============================================================

    def get_evolution_report(self) -> dict:
        """
        生成完整的进化报告 (人类可读)

        Returns:
            {
                "total_cycles": int,
                "total_feedback": int,
                "strategies": int,
                "anti_patterns": int,
                "overall_success_rate": float,
                "top_strategies": [...],
                "route_weights": {...},
                "failure_analysis": {...},
                "recent_trend": "improving" | "stable" | "declining",
                "cycle_history": [...]
            }
        """
        feedback = self.collector.get_recent(500)
        success_count = sum(1 for f in feedback if f.success)
        total = max(len(feedback), 1)

        # 策略统计
        active_strategies = [
            (n, s) for n, s in self.learner.strategies.items()
            if not s.anti_pattern
        ]
        anti_patterns = [
            (n, s) for n, s in self.learner.strategies.items()
            if s.anti_pattern
        ]
        top_strats = sorted(active_strategies, key=lambda x: x[1].effectiveness, reverse=True)[:5]

        # 失败分析
        failure_report = self.optimizer.analyze_failure_patterns()

        # 趋势判断
        trend = self._compute_trend()

        return {
            "total_cycles": self._cycle_count,
            "total_feedback_processed": total,
            "strategies": len(active_strategies),
            "anti_patterns": len(anti_patterns),
            "overall_success_rate": round(success_count / total, 3),
            "top_strategies": [
                {
                    "name": n,
                    "effectiveness": round(s.effectiveness, 3),
                    "success_rate": round(s.success_rate, 3),
                    "confidence": round(s.confidence, 3),
                    "usage_count": s.usage_count,
                }
                for n, s in top_strats
            ],
            "route_weights": dict(sorted(
                self.optimizer.route_weights.items(),
                key=lambda x: x[1], reverse=True
            )[:10]),
            "failure_analysis": failure_report,
            "recent_trend": trend,
            "recent_cycles": self._metrics_history[-5:] if self._metrics_history else [],
        }

    def get_metrics(self) -> dict:
        """获取关键进化指标 (轻量版, 适合监控面板)"""
        feedback = self.collector.get_recent(200)
        success_count = sum(1 for f in feedback if f.success)

        return {
            "cycles_completed": self._cycle_count,
            "feedback_count": len(feedback),
            "success_rate": round(success_count / max(len(feedback), 1), 3),
            "strategy_count": sum(1 for s in self.learner.strategies.values() if not s.anti_pattern),
            "anti_pattern_count": sum(1 for s in self.learner.strategies.values() if s.anti_pattern),
            "route_count": len(self.optimizer.route_weights),
            "last_cycle_time": self._last_cycle_time,
            "trend": self._compute_trend(),
        }

    # ============================================================
    # 内部方法
    # ============================================================

    @staticmethod
    def _extract_context_keywords(context: str) -> list[str]:
        """从执行上下文中提取关键词 (用于工具偏好矩阵)"""
        if not context:
            return []
        # 简单分词 + 过滤短词
        keywords = [w.lower() for w in context.split() if len(w) >= 3]
        return keywords[:5]

    def _compute_trend(self) -> str:
        """
        判断成功率趋势

        对比最近 3 轮进化的成功率:
            - 持续上升 → "improving"
            - 持续下降 → "declining"
            - 其他 → "stable"
        """
        history = self._metrics_history[-3:]
        if len(history) < 2:
            return "stable"

        rates = [h.get("success_rate", 0.5) for h in history]
        improving = all(rates[i] < rates[i + 1] for i in range(len(rates) - 1))
        declining = all(rates[i] > rates[i + 1] for i in range(len(rates) - 1))

        if improving:
            return "improving"
        elif declining:
            return "declining"
        return "stable"

    def _build_cycle_report(self, cycle_id: int, feedback: list, duration: float) -> dict:
        """构建单轮进化报告"""
        success_count = sum(1 for f in feedback if f.success)

        return {
            "cycle_id": cycle_id,
            "timestamp": time.time(),
            "feedback_count": len(feedback),
            "success_rate": round(success_count / max(len(feedback), 1), 3),
            "strategy_count": sum(1 for s in self.learner.strategies.values() if not s.anti_pattern),
            "anti_pattern_count": sum(1 for s in self.learner.strategies.values() if s.anti_pattern),
            "duration": round(duration, 2),
        }
