"""
AgentClaw Agent 评估模块 v1

标准化的 Agent 能力评估框架，覆盖面试高频考点：

    1. RAG 评估 (类 RAGAS)
       - Context Precision: 检索到的上下文中有多少是真正有用的
       - Context Recall: 答案所需信息在上下文中的覆盖率
       - Answer Relevancy: 回答与问题的相关程度
       - Faithfulness: 回答是否忠于上下文（不幻觉）

    2. 工具选择准确率
       - Tool Selection Accuracy: Agent 是否选对了工具
       - Tool Call Latency: 每个工具的调用延迟分布

    3. 端到端延迟
       - E2E Latency: P50 / P95 / P99 百分位统计

    4. Agent 决策质量
       - Step Efficiency: 完成任务的平均推理步数
       - Success Rate: 任务成功率

两种评估模式:
    - fast: 纯文本统计 (无需 LLM, 秒级出结果)
    - llm: 调用 LLM 做语义级评估 (更精准, 需要 API Key)

使用方式:
    # 快速评估 (无需 LLM)
    evaluator = AgentEvaluator(mode="fast")
    evaluator.add_rag_sample(
        question="AgentClaw 有哪些等级?",
        contexts=["Level 1 基础问答", "Level 2 工具增强", "Level 3 多Agent协作"],
        answer="AgentClaw 分为4个等级: Level 1-4",
        ground_truth="AgentClaw 分为4个等级: 基础问答、工具增强、多Agent协作、自主进化"
    )
    report = evaluator.evaluate()

    # 从 FeedbackCollector 导入历史数据
    evaluator = AgentEvaluator(mode="fast")
    evaluator.load_from_feedback(collector)
    report = evaluator.evaluate()
"""

import math
import time
import json
import os
import re
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict


# ============================================================
# 数据结构
# ============================================================

@dataclass
class RAGSample:
    """单条 RAG 评估样本"""
    question: str
    contexts: List[str]
    answer: str
    ground_truth: str = ""
    tool_name: str = "knowledge_search"


@dataclass
class ToolCallSample:
    """单条工具调用样本"""
    question: str
    expected_tool: str
    actual_tool: str
    success: bool = True
    latency: float = 0.0


@dataclass
class E2ELatencySample:
    """单条端到端延迟样本"""
    question: str
    total_latency: float
    tool_call_count: int = 1
    success: bool = True


@dataclass
class RAGMetrics:
    """RAG 评估指标"""
    context_precision: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    answer_correctness: float = -1.0


@dataclass
class ToolMetrics:
    """工具选择指标"""
    selection_accuracy: float = 0.0
    total_calls: int = 0
    correct_calls: int = 0
    per_tool_accuracy: Dict[str, float] = field(default_factory=dict)


@dataclass
class LatencyMetrics:
    """延迟指标"""
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0
    sample_count: int = 0


@dataclass
class EvaluationReport:
    """完整评估报告"""
    timestamp: float = field(default_factory=time.time)
    mode: str = "fast"
    sample_count: int = 0
    rag: Optional[RAGMetrics] = None
    tool: Optional[ToolMetrics] = None
    latency: Optional[LatencyMetrics] = None
    tool_latency: Dict[str, LatencyMetrics] = field(default_factory=dict)
    agent_success_rate: float = -1.0
    avg_reasoning_steps: float = -1.0


# ============================================================
# 文本相似度工具 (不依赖外部库)
# ============================================================

class TextSimilarity:
    """纯文本相似度计算, 无外部依赖"""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """分词: 英文单词 + 中文字符 bigram"""
        words = re.findall(r'\w+', text.lower())
        for segment in re.findall(r'[\u4e00-\u9fff]+', text):
            for i in range(len(segment) - 1):
                words.append(segment[i:i+2])
            if len(segment) == 1:
                words.append(segment)
        return words

    @staticmethod
    def jaccard(tokens_a: List[str], tokens_b: List[str]) -> float:
        """Jaccard 相似度"""
        set_a, set_b = set(tokens_a), set(tokens_b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    @staticmethod
    def overlap_coefficient(tokens_a: List[str], tokens_b: List[str]) -> float:
        """重叠系数 (交集 / 较小集合)"""
        set_a, set_b = set(tokens_a), set(tokens_b)
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / min(len(set_a), len(set_b))

    @staticmethod
    def bm25_score(query_tokens: List[str], doc_tokens: List[str],
                   k1: float = 1.5, b: float = 0.75) -> float:
        """简化 BM25 评分"""
        if not query_tokens or not doc_tokens:
            return 0.0
        tf = Counter(query_tokens)
        dl = len(doc_tokens)
        avg_dl = max(len(query_tokens), 1)
        score = 0.0
        for term in set(query_tokens):
            term_freq = sum(1 for t in doc_tokens if t == term)
            idf = math.log(1 + (len(query_tokens) - tf[term] + 0.5) / (tf[term] + 0.5) + 1)
            numerator = term_freq * (k1 + 1)
            denominator = term_freq + k1 * (1 - b + b * dl / avg_dl)
            score += idf * numerator / denominator
        return score

    @staticmethod
    def keyword_overlap(text: str, reference: str) -> float:
        """关键词覆盖率: reference 中的关键词在 text 中出现的比例"""
        if not reference:
            return 0.0
        ref_tokens = TextSimilarity.tokenize(reference)
        if not ref_tokens:
            return 0.0
        text_tokens = set(TextSimilarity.tokenize(text))
        hits = sum(1 for t in ref_tokens if t in text_tokens)
        return hits / len(ref_tokens)


# ============================================================
# LLM 评估器 (可选, llm 模式使用)
# ============================================================

class LLMJudge:
    """
    LLM-as-Judge: 用 LLM 对回答质量进行语义级打分。

    评估维度:
    - Faithfulness: 回答是否忠于上下文 (0-1)
    - Relevancy:   回答与问题的相关性 (0-1)
    - Correctness: 回答的正确性 (0-1)
    """

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self._client = None
        self._model = model
        try:
            from core.config import settings
            self._api_key = api_key or settings.DEEPSEEK_API_KEY
            self._base_url = base_url or settings.DEEPSEEK_BASE_URL
            self._model = model or settings.LLM_MODEL
        except Exception:
            self._api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
            self._base_url = base_url or os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            )

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        except ImportError:
            raise RuntimeError("需要 openai 包: pip install openai")

    def _score(self, prompt: str) -> float:
        """通用评分: 发送 prompt, 解析 0-1 分数"""
        self._ensure_client()
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            score = float(re.search(r'[\d.]+', resp.choices[0].message.content).group())
            return max(0.0, min(1.0, score))
        except Exception:
            return -1.0

    def score_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """评估回答是否忠于上下文 (0-1)"""
        ctx_text = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        prompt = (
            "你是一个严格的评估员。请判断以下回答是否完全基于给定的上下文信息。\n"
            "如果回答中包含上下文中未提及的信息（幻觉），请给出低分。\n\n"
            f"上下文:\n{ctx_text}\n\n"
            f"回答: {answer}\n\n"
            "请只输出一个 0 到 1 之间的数字（保留两位小数），不要输出其他内容。"
        )
        return self._score(prompt)

    def score_relevancy(self, question: str, answer: str) -> float:
        """评估回答与问题的相关性 (0-1)"""
        prompt = (
            "你是一个评估员。请判断以下回答与问题的相关程度。\n"
            "相关度高: 直接回答了问题，信息准确\n"
            "相关度低: 回答偏离主题或无关\n\n"
            f"问题: {question}\n"
            f"回答: {answer}\n\n"
            "请只输出一个 0 到 1 之间的数字（保留两位小数），不要输出其他内容。"
        )
        return self._score(prompt)

    def score_correctness(self, answer: str, ground_truth: str) -> float:
        """评估回答的正确性 (0-1)"""
        prompt = (
            "你是一个严格的评估员。请判断回答与参考答案的一致程度。\n"
            "核心信息一致即为高分，格式差异不影响。\n\n"
            f"参考答案: {ground_truth}\n"
            f"回答: {answer}\n\n"
            "请只输出一个 0 到 1 之间的数字（保留两位小数），不要输出其他内容。"
        )
        return self._score(prompt)


# ============================================================
# 核心评估器
# ============================================================

class AgentEvaluator:
    """
    AgentClaw 评估引擎

    支持两种模式:
        - fast: 纯文本统计 (无需 LLM API, 秒级完成)
        - llm:  文本统计 + LLM 语义评估 (需要 API Key)

    评估维度:
        1. RAG 质量: Context Precision / Recall / Answer Relevancy / Faithfulness
        2. 工具选择: Selection Accuracy / Per-tool Accuracy
        3. 延迟分布: P50 / P95 / P99 / Mean / Std
        4. Agent 决策: Success Rate / Avg Reasoning Steps
    """

    def __init__(self, mode: str = "fast", persist_dir: str = "./eval_data"):
        self.mode = mode
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self._rag_samples: List[RAGSample] = []
        self._tool_samples: List[ToolCallSample] = []
        self._latency_samples: List[E2ELatencySample] = []

        self._llm_judge: Optional[LLMJudge] = None
        if mode == "llm":
            try:
                self._llm_judge = LLMJudge()
            except Exception as e:
                print(f"[WARNING] LLM 评估器初始化失败, 回退到 fast 模式: {e}")
                self.mode = "fast"

    # ============================================================
    # 样本录入
    # ============================================================

    def add_rag_sample(self, question: str, contexts: List[str],
                       answer: str, ground_truth: str = "",
                       tool_name: str = "knowledge_search"):
        """添加一条 RAG 评估样本"""
        self._rag_samples.append(RAGSample(
            question=question, contexts=contexts,
            answer=answer, ground_truth=ground_truth, tool_name=tool_name,
        ))

    def add_tool_sample(self, question: str, expected_tool: str,
                        actual_tool: str, success: bool = True,
                        latency: float = 0.0):
        """添加一条工具调用样本"""
        self._tool_samples.append(ToolCallSample(
            question=question, expected_tool=expected_tool,
            actual_tool=actual_tool, success=success, latency=latency,
        ))

    def add_latency_sample(self, question: str, total_latency: float,
                           tool_call_count: int = 1, success: bool = True):
        """添加一条端到端延迟样本"""
        self._latency_samples.append(E2ELatencySample(
            question=question, total_latency=total_latency,
            tool_call_count=tool_call_count, success=success,
        ))

    def load_from_feedback(self, feedback_list):
        """从 FeedbackCollector 导入历史反馈数据"""
        for signal in feedback_list:
            self._latency_samples.append(E2ELatencySample(
                question=signal.context or signal.task_id,
                total_latency=signal.latency,
                success=signal.success,
            ))

    def load_from_file(self, file_path: str):
        """从 JSON 文件加载评估样本"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("rag_samples", []):
            self.add_rag_sample(**item)
        for item in data.get("tool_samples", []):
            self.add_tool_sample(**item)
        for item in data.get("latency_samples", []):
            self.add_latency_sample(**item)

    # ============================================================
    # RAG 评估
    # ============================================================

    def _eval_rag_fast(self) -> RAGMetrics:
        """快速模式: 纯文本统计评估 RAG 质量"""
        if not self._rag_samples:
            return RAGMetrics()

        precision_scores = []
        recall_scores = []
        relevancy_scores = []
        faithfulness_scores = []
        correctness_scores = []

        for sample in self._rag_samples:
            q_tokens = TextSimilarity.tokenize(sample.question)
            a_tokens = TextSimilarity.tokenize(sample.answer)
            combined_context = " ".join(sample.contexts)
            ctx_tokens = TextSimilarity.tokenize(combined_context)

            # Context Precision: 上下文与问题的相关度
            if sample.contexts:
                ctx_relevance = sum(
                    TextSimilarity.bm25_score(q_tokens, TextSimilarity.tokenize(c))
                    for c in sample.contexts
                ) / len(sample.contexts)
                precision_scores.append(min(ctx_relevance / 5.0, 1.0))

            # Context Recall: answer 信息在 context 中的覆盖率
            if ctx_tokens:
                recall_scores.append(
                    TextSimilarity.keyword_overlap(combined_context, sample.answer)
                )

            # Answer Relevancy: Jaccard + BM25 加权
            relevancy = TextSimilarity.jaccard(q_tokens, a_tokens)
            bm25_rel = TextSimilarity.bm25_score(q_tokens, a_tokens) / 5.0
            relevancy_scores.append(
                min(0.6 * relevancy + 0.4 * min(bm25_rel, 1.0), 1.0)
            )

            # Faithfulness: 回答内容在上下文中的覆盖率
            if ctx_tokens and a_tokens:
                faithfulness_scores.append(
                    TextSimilarity.keyword_overlap(sample.answer, combined_context)
                )

            # Answer Correctness (需要 ground_truth)
            if sample.ground_truth:
                gt_tokens = TextSimilarity.tokenize(sample.ground_truth)
                correctness_scores.append(
                    TextSimilarity.overlap_coefficient(a_tokens, gt_tokens)
                )

        return RAGMetrics(
            context_precision=statistics.mean(precision_scores) if precision_scores else 0.0,
            context_recall=statistics.mean(recall_scores) if recall_scores else 0.0,
            answer_relevancy=statistics.mean(relevancy_scores) if relevancy_scores else 0.0,
            faithfulness=statistics.mean(faithfulness_scores) if faithfulness_scores else 0.0,
            answer_correctness=statistics.mean(correctness_scores) if correctness_scores else -1.0,
        )

    def _eval_rag_llm(self) -> RAGMetrics:
        """LLM 模式: 用 LLM-as-Judge 评估 RAG 质量"""
        fast_metrics = self._eval_rag_fast()
        if not self._llm_judge or not self._rag_samples:
            return fast_metrics

        faith_scores, relevancy_scores, correctness_scores = [], [], []

        for sample in self._rag_samples:
            faith = self._llm_judge.score_faithfulness(sample.answer, sample.contexts)
            if faith >= 0:
                faith_scores.append(faith)

            rel = self._llm_judge.score_relevancy(sample.question, sample.answer)
            if rel >= 0:
                relevancy_scores.append(rel)

            if sample.ground_truth:
                corr = self._llm_judge.score_correctness(sample.answer, sample.ground_truth)
                if corr >= 0:
                    correctness_scores.append(corr)

        return RAGMetrics(
            context_precision=fast_metrics.context_precision,
            context_recall=fast_metrics.context_recall,
            answer_relevancy=(
                statistics.mean(relevancy_scores) if relevancy_scores
                else fast_metrics.answer_relevancy
            ),
            faithfulness=(
                statistics.mean(faith_scores) if faith_scores
                else fast_metrics.faithfulness
            ),
            answer_correctness=(
                statistics.mean(correctness_scores) if correctness_scores
                else fast_metrics.answer_correctness
            ),
        )

    # ============================================================
    # 工具选择评估
    # ============================================================

    def _eval_tool_selection(self) -> ToolMetrics:
        """评估工具选择准确率"""
        if not self._tool_samples:
            return ToolMetrics()

        total = len(self._tool_samples)
        correct = sum(1 for s in self._tool_samples if s.actual_tool == s.expected_tool)

        tool_correct: Dict[str, int] = defaultdict(int)
        tool_total: Dict[str, int] = defaultdict(int)
        for sample in self._tool_samples:
            tool_total[sample.actual_tool] += 1
            if sample.actual_tool == sample.expected_tool:
                tool_correct[sample.actual_tool] += 1

        per_tool = {
            tool: cnt / tool_total[tool]
            for tool, cnt in tool_correct.items() if tool_total[tool] > 0
        }

        return ToolMetrics(
            selection_accuracy=correct / total if total > 0 else 0.0,
            total_calls=total,
            correct_calls=correct,
            per_tool_accuracy=per_tool,
        )

    # ============================================================
    # 延迟评估
    # ============================================================

    @staticmethod
    def _percentile(data: List[float], p: float) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)

    def _eval_latency(self, latencies: List[float]) -> LatencyMetrics:
        """计算延迟分布指标"""
        if not latencies:
            return LatencyMetrics()
        return LatencyMetrics(
            p50=self._percentile(latencies, 0.50),
            p95=self._percentile(latencies, 0.95),
            p99=self._percentile(latencies, 0.99),
            mean=statistics.mean(latencies),
            min=min(latencies),
            max=max(latencies),
            std=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            sample_count=len(latencies),
        )

    def _eval_all_latency(self) -> Tuple[Optional[LatencyMetrics], Dict[str, LatencyMetrics]]:
        """评估所有延迟指标 (端到端 + 每个工具)"""
        e2e_latencies = [s.total_latency for s in self._latency_samples]
        e2e_metrics = self._eval_latency(e2e_latencies) if e2e_latencies else None

        tool_latencies: Dict[str, List[float]] = defaultdict(list)
        for sample in self._tool_samples:
            tool_latencies[sample.actual_tool].append(sample.latency)

        tool_metrics = {
            tool: self._eval_latency(lats)
            for tool, lats in tool_latencies.items()
        }
        return e2e_metrics, tool_metrics

    # ============================================================
    # 综合评估
    # ============================================================

    def evaluate(self) -> EvaluationReport:
        """执行完整评估并生成报告"""
        print(f"[Evaluator] 开始评估 (模式: {self.mode})")
        start = time.time()

        rag_metrics = self._eval_rag_llm() if self.mode == "llm" else self._eval_rag_fast()
        tool_metrics = self._eval_tool_selection()
        e2e_metrics, tool_latency = self._eval_all_latency()

        success_count = sum(1 for s in self._latency_samples if s.success)
        agent_sr = success_count / len(self._latency_samples) if self._latency_samples else -1.0
        avg_steps = statistics.mean(
            [s.tool_call_count for s in self._latency_samples]
        ) if self._latency_samples else -1.0

        duration = time.time() - start
        print(f"[Evaluator] 评估完成 (耗时 {duration:.2f}s)")

        report = EvaluationReport(
            mode=self.mode,
            sample_count=len(self._rag_samples) + len(self._tool_samples) + len(self._latency_samples),
            rag=rag_metrics,
            tool=tool_metrics,
            latency=e2e_metrics,
            tool_latency=tool_latency,
            agent_success_rate=agent_sr,
            avg_reasoning_steps=avg_steps,
        )
        self._save_report(report)
        return report

    # ============================================================
    # 报告输出
    # ============================================================

    def _save_report(self, report: EvaluationReport):
        os.makedirs(self.persist_dir, exist_ok=True)
        path = os.path.join(self.persist_dir, f"eval_report_{int(report.timestamp)}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2, default=str)
        print(f"[Evaluator] 报告已保存: {path}")

    def print_report(self, report: EvaluationReport):
        """打印人类可读的评估报告"""
        print("\n" + "=" * 60)
        print(f"  AgentClaw 评估报告 (模式: {report.mode})")
        print(f"  样本总量: {report.sample_count}")
        print("=" * 60)

        if report.rag:
            print(f"\n[RAG] 评估 ({len(self._rag_samples)} 样本):")
            print(f"  Context Precision:    {report.rag.context_precision:.3f}")
            print(f"  Context Recall:       {report.rag.context_recall:.3f}")
            print(f"  Answer Relevancy:     {report.rag.answer_relevancy:.3f}")
            print(f"  Faithfulness:         {report.rag.faithfulness:.3f}")
            if report.rag.answer_correctness >= 0:
                print(f"  Answer Correctness:   {report.rag.answer_correctness:.3f}")
            else:
                print(f"  Answer Correctness:   N/A (无 ground_truth)")

        if report.tool:
            print(f"\n[TOOL] 工具选择评估 ({report.tool.total_calls} 样本):")
            print(f"  Selection Accuracy:   {report.tool.selection_accuracy:.1%}")
            print(f"  正确/总数:            {report.tool.correct_calls}/{report.tool.total_calls}")
            if report.tool.per_tool_accuracy:
                print(f"  各工具准确率:")
                for tool, acc in sorted(report.tool.per_tool_accuracy.items(),
                                        key=lambda x: x[1], reverse=True):
                    print(f"    {tool}: {acc:.1%}")

        if report.latency:
            print(f"\n[LATENCY] 端到端延迟 ({report.latency.sample_count} 样本):")
            print(f"  P50:                  {report.latency.p50:.3f}s")
            print(f"  P95:                  {report.latency.p95:.3f}s")
            print(f"  P99:                  {report.latency.p99:.3f}s")
            print(f"  Mean +/- Std:         {report.latency.mean:.3f}s +/- {report.latency.std:.3f}s")
            print(f"  Min / Max:            {report.latency.min:.3f}s / {report.latency.max:.3f}s")

        if report.tool_latency:
            print(f"\n[LATENCY] 各工具延迟:")
            for tool_name, m in sorted(report.tool_latency.items()):
                print(f"  {tool_name}: P50={m.p50:.3f}s  P95={m.p95:.3f}s  Mean={m.mean:.3f}s")

        if report.agent_success_rate >= 0:
            print(f"\n[AGENT] 决策质量:")
            print(f"  任务成功率:           {report.agent_success_rate:.1%}")
        if report.avg_reasoning_steps >= 0:
            print(f"  平均推理步数:         {report.avg_reasoning_steps:.1f}")

        print("\n" + "=" * 60)


# ============================================================
# 快速演示
# ============================================================

if __name__ == "__main__":
    print("AgentClaw Agent 评估模块演示\n")
    evaluator = AgentEvaluator(mode="fast")

    # 1. RAG 评估样本
    evaluator.add_rag_sample(
        question="AgentClaw 有哪些等级?",
        contexts=[
            "Level 1 基础问答: 使用 DeepSeek API 实现对话功能。",
            "Level 2 工具增强: 集成计算器、文件读写、命令执行等工具。",
            "Level 3 多Agent协作: 使用 AgentOrchestrator 实现多Agent调度。",
            "Level 4 自主进化: 通过反思循环自动优化 Agent 行为。",
        ],
        answer="AgentClaw 分为4个等级: 基础问答、工具增强、多Agent协作和自主进化。",
        ground_truth="AgentClaw 分为4个等级: Level 1 基础问答、Level 2 工具增强、"
                     "Level 3 多Agent协作、Level 4 自主进化。",
    )
    evaluator.add_rag_sample(
        question="安全机制包括哪些?",
        contexts=[
            "安全机制包括路径白名单、文件黑名单、命令白名单和危险模式检测。",
            "所有安全检查必须 raise PermissionError, 不能 return error dict。",
        ],
        answer="安全机制包括路径白名单、文件黑名单、命令白名单和危险模式检测。"
              "所有安全检查通过 PermissionError 实现。",
        ground_truth="安全机制包括: 路径白名单、文件黑名单、命令白名单、危险模式检测。"
                     "检查失败时抛出 PermissionError。",
    )
    evaluator.add_rag_sample(
        question="RAG 模块支持什么向量数据库?",
        contexts=[
            "向量存储 (内存版, 可扩展为 FAISS/Chroma)",
            "RAG 模块支持 TXT/Markdown/JSON 文档加载和 TF-IDF 向量检索。",
        ],
        answer="RAG 模块默认使用内存版向量存储，可扩展为 FAISS 或 Chroma。",
        ground_truth="默认使用 InMemoryVectorStore (内存版), 可扩展为 FAISS 或 Chroma。",
    )

    # 2. 工具选择样本
    evaluator.add_tool_sample("帮我搜索Python最新版本", "web_search", "web_search", True, 1.2)
    evaluator.add_tool_sample("计算 123 * 456", "calculator", "calculator", True, 0.05)
    evaluator.add_tool_sample("读取 config.yaml", "file_read", "file_read", True, 0.3)
    evaluator.add_tool_sample("查看系统内存", "sys_monitor", "sys_monitor", True, 0.8)
    evaluator.add_tool_sample("生成一张猫的图片", "image_generate", "image_generate", True, 5.2)
    evaluator.add_tool_sample("帮我搜索Python最新版本", "web_search", "file_read", False, 0.3)
    evaluator.add_tool_sample("这个代码有什么问题?", "code_execute", "web_search", False, 1.5)

    # 3. 延迟样本
    import random
    random.seed(42)
    for _ in range(50):
        latency = max(0.1, random.gauss(2.5, 1.2))
        evaluator.add_latency_sample("test", latency,
                                     tool_call_count=random.randint(1, 5),
                                     success=random.random() > 0.1)

    # 4. 执行评估
    report = evaluator.evaluate()
    evaluator.print_report(report)
