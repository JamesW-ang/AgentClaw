"""
经验学习系统 v2 — 从历史反馈中学习成功路径和策略

v2 升级 (不破坏现有架构):
    1. 时间衰减加权 — 近期模式权重更高，老模式自然淡出 (half_life=7天)
    2. TF-IDF 语义匹配 — 替代子字符串匹配，支持模糊任务类型推荐
    3. 反模式检测 — 从失败路径中识别"不该做什么"的工具组合
    4. 统计指标 — support / confidence(Wilson下限) / lift 三维度评估
    5. 策略效果追踪 — 推荐后追踪实际成功率 (A/B思维)

向后兼容:
    Strategy(name, trigger_pattern, tool_sequence, success_rate) 仍可用
    新字段均有默认值
"""

import time
import math
from collections import defaultdict, Counter
from typing import Optional, List, Dict, Tuple

from learning.feedback import FeedbackCollector, FeedbackSignal

from core.logger import get_logger
logger = get_logger("experience_learner")


# ============================================================
# 策略数据类 v2
# ============================================================


class Strategy:
    """
    执行策略 v2 — 表示一个工具调用模式及其统计特征

    向后兼容: 原有4参数构造函数仍可用，新字段均有默认值

    新增统计指标:
        confidence: Wilson 置信区间下限 (95%), 比简单成功率更鲁棒
        support: 时间衰减加权支持度 (近期出现频率)
        lift: 提升度 vs 随机期望 (>1 表示正相关)
        anti_pattern: 是否为反模式 (从失败路径学到)
        effectiveness: 实际推荐效果 (推荐后成功率 * 置信度)
    """

    def __init__(
        self,
        name: str,
        trigger_pattern: str,
        tool_sequence: list,
        success_rate: float,
        confidence: float = 0.5,
        support: float = 0.0,
        lift: float = 1.0,
        anti_pattern: bool = False,
    ):
        self.name = name
        self.trigger_pattern = trigger_pattern
        self.tool_sequence = tool_sequence
        self.success_rate = success_rate
        # v2 新增字段
        self.confidence = confidence
        self.support = support
        self.lift = lift
        self.anti_pattern = anti_pattern
        self.first_seen = time.time()
        self.last_updated = time.time()
        self.usage_count = 0                  # 被推荐次数
        self.success_after_recommend = 0      # 推荐后成功次数

    @property
    def effectiveness(self) -> float:
        """策略实际效果评分 (0-1)

        推荐不足 3 次时, 使用 score = success_rate * confidence (先验估计)
        推荐超过 3 次后, 使用 score = (实际成功率) * confidence (后验修正)
        """
        if self.usage_count < 3:
            return self.success_rate * self.confidence
        return (self.success_after_recommend / self.usage_count) * self.confidence

    def record_outcome(self, success: bool):
        """记录一次推荐后的实际执行结果 (供 A/B 效果追踪)"""
        self.usage_count += 1
        if success:
            self.success_after_recommend += 1
        self.last_updated = time.time()

    def __repr__(self):
        kind = "ANTI" if self.anti_pattern else "STRAT"
        return (f"Strategy({kind} {self.name}, "
                f"rate={self.success_rate:.2f}, conf={self.confidence:.2f}, "
                f"lift={self.lift:.2f}, eff={self.effectiveness:.2f})")


# ============================================================
# 经验学习器 v2
# ============================================================


class ExperienceLearner:
    """
    经验学习器 v2 — 从历史反馈中自动学习成功策略和失败反模式

    v1 → v2 变更:
        - _find_patterns          → _find_patterns_weighted (时间衰减)
        - get_strategy (子字符串)  → get_strategy (TF-IDF + 精确匹配 双层)
        - 新增 _find_anti_patterns (失败路径挖掘)
        - 新增 check_anti_pattern  (实时反模式检测)
        - 新增 _prune_stale_strategies (自动淘汰过期策略)
        - 新增 Wilson 置信度 / lift 指标
        - 新增策略效果追踪 (record_outcome)

    时间衰减公式:
        weight = exp(-lambda * age)
        lambda = ln(2) / half_life
        默认 half_life = 7 天 → 7天前的数据权重衰减为 0.5

    Wilson 置信区间下限:
        用于替代简单成功率, 在样本量少时更保守
        公式: (p + z²/2n - z*sqrt((p(1-p) + z²/4n) / n)) / (1 + z²/n)
    """

    # 时间衰减参数
    HALF_LIFE = 7 * 86400                        # 半衰期 7 天
    DECAY_LAMBDA = math.log(2) / HALF_LIFE       # 衰减系数

    # 匹配参数
    SIMILARITY_THRESHOLD = 0.3                    # TF-IDF 匹配最低阈值

    def __init__(self, collector: FeedbackCollector):
        self.collector = collector
        self.strategies: Dict[str, Strategy] = {}
        self._pattern_cache: list = []
        self._strategy_vectors: Dict[str, dict] = {}   # TF-IDF 向量缓存
        logger.info("ExperienceLearner v2 初始化完成")

    # ============================================================
    # 主学习接口 (向后兼容)
    # ============================================================

    def learn_from_history(self, feedback_history: Optional[list] = None):
        """
        从反馈历史中学习 (v2: 含反模式 + 时间衰减 + 统计评估)

        兼容 v1: 无参调用仍可用, 内部自动获取最近 500 条
        """
        if feedback_history is None:
            feedback_history = self.collector.get_recent(500)

        if not feedback_history:
            logger.debug("无反馈历史，跳过学习")
            return

        now = time.time()
        successes = [s for s in feedback_history if s.success]
        failures = [s for s in feedback_history if not s.success]
        overall_rate = len(successes) / len(feedback_history)

        logger.info(f"开始学习: {len(successes)} 成功 / {len(failures)} 失败 "
                     f"(整体成功率 {overall_rate:.1%})")

        # 1. 从成功路径中学习策略 (时间衰减加权)
        success_paths = self._extract_paths(successes)
        patterns = self._find_patterns_weighted(success_paths, now)

        # 2. 从失败路径中学习反模式
        failure_paths = self._extract_paths(failures)
        anti_patterns = self._find_anti_patterns(failure_paths, now)

        # 3. 生成/更新正向策略
        new_count = 0
        for pattern in patterns:
            strategy = self._generate_strategy_v2(
                pattern, successes, overall_rate, now
            )
            if strategy:
                existing = self.strategies.get(pattern["name"])
                if existing:
                    self._merge_strategy(existing, strategy)
                else:
                    self.strategies[pattern["name"]] = strategy
                    new_count += 1
                self._strategy_vectors.pop(pattern["name"], None)

        # 4. 生成/更新反模式
        for ap in anti_patterns:
            ap_name = f"ANTI:{ap['name']}"
            ap_strategy = Strategy(
                name=ap_name,
                trigger_pattern=ap["sequence"][0],
                tool_sequence=ap["sequence"],
                success_rate=0.0,
                confidence=ap.get("confidence", 0.3),
                support=ap.get("weighted_support", 0),
                anti_pattern=True,
            )
            existing = self.strategies.get(ap_name)
            if existing:
                self._merge_strategy(existing, ap_strategy)
            else:
                self.strategies[ap_name] = ap_strategy
                new_count += 1

        # 5. 淘汰过期且效果差的策略
        pruned = self._prune_stale_strategies(now)

        # 日志
        strat_count = sum(1 for s in self.strategies.values() if not s.anti_pattern)
        anti_count = sum(1 for s in self.strategies.values() if s.anti_pattern)
        logger.info(f"学习完成: {len(patterns)} 正向模式, {len(anti_patterns)} 反模式, "
                     f"新增 {new_count} 条, 淘汰 {pruned} 条")
        logger.info(f"策略库: {strat_count} 正向 + {anti_count} 反模式 = {len(self.strategies)} 总计")

    # ============================================================
    # 策略推荐 (v2: TF-IDF 语义匹配)
    # ============================================================

    def get_strategy(self, task_type: str) -> Optional[Strategy]:
        """
        根据任务类型获取推荐策略 (v2: 双层匹配)

        匹配优先级:
            1. 精确匹配: task_type 包含在策略名称/触发模式中 (高优先级)
            2. 语义匹配: TF-IDF 余弦相似度 > 阈值 (模糊匹配)
            3. 均返回 effectiveness 最高的策略

        反模式过滤: 推荐结果自动排除反模式
        """
        if not task_type or not self.strategies:
            return None

        candidates = {k: v for k, v in self.strategies.items() if not v.anti_pattern}
        if not candidates:
            return None

        # 层1: 精确匹配
        exact_matches = []
        for name, strat in candidates.items():
            if task_type.lower() in name.lower() or task_type.lower() in strat.trigger_pattern.lower():
                exact_matches.append(strat)

        if exact_matches:
            return max(exact_matches, key=lambda s: s.effectiveness)

        # 层2: TF-IDF 语义匹配
        query_vec = self._text_to_tfidf(task_type)
        if not query_vec:
            return None

        best_match = None
        best_score = self.SIMILARITY_THRESHOLD

        for name, strat in candidates.items():
            strat_vec = self._get_strategy_vector(name, strat)
            if not strat_vec:
                continue
            sim = self._cosine_sim(query_vec, strat_vec)

            # 综合考虑相似度和策略效果
            eff_bonus = strat.effectiveness * 0.2
            combined = sim + eff_bonus

            if combined > best_score + best_score * 0.2:
                best_score = combined
                best_match = strat

        return best_match

    def check_anti_pattern(self, tool_sequence: list) -> Optional[Strategy]:
        """
        检查工具调用序列是否命中已知反模式

        用法:
            anti = learner.check_anti_pattern(["web_search", "calculator", "web_search"])
            if anti:
                logger.warning(f"命中反模式: {anti.name} (建议避免)")

        Returns:
            命中的反模式 Strategy, 或 None
        """
        if not tool_sequence:
            return None

        seq_str = "->".join(tool_sequence)
        for name, strat in self.strategies.items():
            if not strat.anti_pattern:
                continue
            # 检查是否以反模式开头 (前缀匹配)
            pattern_prefix = "->".join(strat.tool_sequence[:2])
            if seq_str.startswith(pattern_prefix):
                return strat

        return None

    # ============================================================
    # 路径提取 (兼容 v1)
    # ============================================================

    def _extract_paths(self, signals: list) -> list:
        """按 task_id 分组，提取工具调用序列 (≥2步)"""
        groups = defaultdict(list)
        for s in signals:
            groups[s.task_id].append(s)
        return [tools for tools in groups.values() if len(tools) >= 2]

    # ============================================================
    # v2: 时间衰减加权模式挖掘
    # ============================================================

    def _time_decay(self, timestamp: float, now: float) -> float:
        """指数时间衰减: weight = exp(-lambda * age)"""
        age = now - timestamp
        if age < 0:
            return 1.0
        return math.exp(-self.DECAY_LAMBDA * age)

    def _find_patterns_weighted(
        self,
        paths: list,
        now: float,
        min_len: int = 2,
        min_support: float = 2.0,
    ) -> list:
        """
        发现高频工具调用模式 (v2: 时间衰减加权)

        与 v1 的区别:
            v1: 纯计数, min_count=3 → 老数据和新数据同等权重
            v2: 加权 support, half_life=7天 → 近期模式更容易被发现

        算法:
            对每条路径, 用路径最新时间戳计算衰减权重
            累加每个 n-gram 序列的加权支持度
            过滤 support >= min_support 的模式
        """
        seq_weighted_support = defaultdict(float)

        for path in paths:
            tools = [s.tool_name for s in path]
            path_time = max(s.timestamp for s in path)
            weight = self._time_decay(path_time, now)

            for length in range(min_len, min(len(tools) + 1, 4)):
                for i in range(len(tools) - length + 1):
                    seq = tuple(tools[i:i + length])
                    seq_weighted_support[seq] += weight

        patterns = [
            {
                "name": "->".join(seq),
                "sequence": list(seq),
                "count": sum(1 for p in paths if any(s.tool_name in seq for s in p)),
                "weighted_support": round(w_support, 3),
            }
            for seq, w_support in seq_weighted_support.items()
            if w_support >= min_support
        ]

        logger.debug(f"发现 {len(patterns)} 个时间加权高频模式")
        return patterns

    # ============================================================
    # v2: 反模式检测
    # ============================================================

    def _find_anti_patterns(
        self,
        paths: list,
        now: float,
        min_len: int = 2,
        min_support: float = 1.5,
    ) -> list:
        """
        从失败路径中发现反模式

        算法: 与正向模式相同的 N-gram 挖掘, 但输入为失败路径
        反模式的 confidence 基于加权支持度缩放到 0-1 区间
        """
        seq_weighted_support = defaultdict(float)

        for path in paths:
            tools = [s.tool_name for s in path]
            path_time = max(s.timestamp for s in path)
            weight = self._time_decay(path_time, now)

            for length in range(min_len, min(len(tools) + 1, 4)):
                for i in range(len(tools) - length + 1):
                    seq = tuple(tools[i:i + length])
                    seq_weighted_support[seq] += weight

        anti_patterns = [
            {
                "name": "->".join(seq),
                "sequence": list(seq),
                "weighted_support": round(w_support, 3),
                "failure_rate": 1.0,
                "confidence": min(w_support / 5.0, 0.9),
            }
            for seq, w_support in seq_weighted_support.items()
            if w_support >= min_support
        ]

        logger.debug(f"发现 {len(anti_patterns)} 个反模式")
        return anti_patterns

    # ============================================================
    # v2: 策略生成 (Wilson 置信度 + Lift)
    # ============================================================

    def _generate_strategy_v2(
        self,
        pattern: dict,
        successes: list,
        overall_rate: float,
        now: float,
    ) -> Optional[Strategy]:
        """
        生成策略 (v2: 含 confidence / support / lift)

        统计指标:
            confidence — Wilson 置信区间下限 (95%)
                比简单成功率更鲁棒: 样本少时置信度自动降低
                公式: (p + z²/2n - z*sqrt((p(1-p) + z²/4n) / n)) / (1 + z²/n)

            lift — 提升度 (strategy_rate / overall_rate)
                > 1: 策略正相关于成功
                < 1: 策略与成功无正相关
                = 1: 策略成功率等于随机水平
        """
        seq = pattern["sequence"]
        weighted_support = pattern.get("weighted_support", 0)

        # 匹配的成功记录
        relevant = [s for s in successes if any(s.tool_name == t for t in seq)]
        n = len(relevant)
        rate = n / max(len(successes), 1)

        # Wilson 置信区间下限 (95%, z=1.96)
        if n > 0:
            p = rate
            z = 1.96
            denom = 1 + z ** 2 / n
            center = (p + z ** 2 / (2 * n)) / denom
            spread = z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
            confidence = max(0.0, center - spread)
        else:
            confidence = 0.1

        # Lift
        lift = rate / max(overall_rate, 0.01)

        return Strategy(
            name=pattern["name"],
            trigger_pattern=seq[0],
            tool_sequence=seq,
            success_rate=round(rate, 3),
            confidence=round(confidence, 3),
            support=weighted_support,
            lift=round(lift, 2),
        )

    # ============================================================
    # v2: 策略合并与淘汰
    # ============================================================

    def _merge_strategy(self, existing: Strategy, new_data: Strategy):
        """合并新旧策略 (EMA 平滑, alpha=0.3)"""
        alpha = 0.3
        existing.success_rate = round(
            alpha * new_data.success_rate + (1 - alpha) * existing.success_rate, 3
        )
        existing.confidence = round(
            alpha * new_data.confidence + (1 - alpha) * existing.confidence, 3
        )
        existing.support = round(
            alpha * new_data.support + (1 - alpha) * existing.support, 3
        )
        existing.lift = round(
            alpha * new_data.lift + (1 - alpha) * existing.lift, 2
        )
        existing.last_updated = time.time()

    def _prune_stale_strategies(self, now: float, max_age: float = 30 * 86400) -> int:
        """淘汰过期且效果差的策略 (>30天未更新 且 effectiveness < 0.3)"""
        stale = []
        for name, strat in self.strategies.items():
            age = now - strat.last_updated
            if age > max_age and strat.effectiveness < 0.3:
                stale.append(name)

        for name in stale:
            del self.strategies[name]
            self._strategy_vectors.pop(name, None)

        if stale:
            logger.debug(f"淘汰 {len(stale)} 条过期策略: {stale[:3]}{'...' if len(stale) > 3 else ''}")

        return len(stale)

    # ============================================================
    # v2: TF-IDF 语义匹配引擎
    # ============================================================

    def _text_to_tfidf(self, text: str) -> Optional[dict]:
        """将文本转为 TF-IDF 稀疏向量 (纯 Python, 零依赖)"""
        tokens = text.lower().split()
        if not tokens:
            return None
        total = len(tokens)
        tf = Counter(tokens)
        return {word: count / total for word, count in tf.items()}

    def _get_strategy_vector(self, name: str, strat: Strategy) -> Optional[dict]:
        """获取策略的 TF-IDF 向量 (带内存缓存)"""
        if name in self._strategy_vectors:
            return self._strategy_vectors[name]
        text = f"{strat.name} {strat.trigger_pattern} {' '.join(strat.tool_sequence)}"
        vec = self._text_to_tfidf(text)
        self._strategy_vectors[name] = vec
        return vec

    @staticmethod
    def _cosine_sim(vec_a: dict, vec_b: dict) -> float:
        """计算两个稀疏 TF-IDF 向量的余弦相似度"""
        common = set(vec_a.keys()) & set(vec_b.keys())
        if not common:
            return 0.0
        dot = sum(vec_a[k] * vec_b[k] for k in common)
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return dot / (norm_a * norm_b)
