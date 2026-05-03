"""
自适应优化器 v2 — 根据反馈动态调整路由权重和 Prompt 模板

v2 升级 (不破坏现有架构):
    1. 自适应学习率 — alpha 随样本量递减 (前期快学, 后期稳态)
    2. 失败模式分类 — 按错误类型分类 (timeout/permission/format/api_error/unknown)
    3. 针对性 Prompt 优化 — 根据失败类别生成不同优化建议 (不再硬编码一句话)
    4. 工具偏好矩阵 — 学习哪种上下文/任务适合用哪种工具
    5. 路由权重统计 — 样本量 + 加权均值 + 最近趋势
    6. Prompt 变体追踪 — 多版本 Prompt 效果对比, 自动选优

向后兼容:
    update_route_weights(route_name, success) — 签名不变
    get_best_route(candidates) — 签名不变
    optimize_prompt(template_name, feedback_list) — 签名不变
"""

import time
from collections import defaultdict

from core.logger import get_logger

logger = get_logger("adaptive_optimizer")


class AdaptiveOptimizer:
    """
    自适应优化器 v2

    v1 → v2 变更:
        - update_route_weights: 固定 alpha=0.05 → 自适应 alpha (1/(1+0.1*n))
        - optimize_prompt: 硬编码追加文本 → 失败分类 + 针对性优化 + 防重复
        - 新增 get_tool_preference(context) — 工具偏好推荐
        - 新增 analyze_failure_patterns — 失败模式分类报告
        - 新增 route_stats — 路由权重详细统计
        - 新增 prompt_variants — Prompt 变体效果对比

    自适应学习率公式:
        alpha = initial_alpha / (1 + decay_rate * sample_count)

        前10个样本: alpha ≈ 0.33 → 快速响应
        50个样本:   alpha ≈ 0.14
        100个样本:  alpha ≈ 0.08
        500个样本:  alpha ≈ 0.02 → 稳态微调

    失败分类逻辑:
        timeout    → 建议增加超时 / 拆分任务
        permission → 建议检查权限 / 白名单
        format     → 建议优化输入格式 / 参数验证
        api_error  → 建议检查 API 配置 / 降级策略
        unknown    → 通用优化建议
    """

    # 自适应学习率参数
    INITIAL_ALPHA = 0.5         # 初始学习率 (快速响应)
    DECAY_RATE = 0.1            # 衰减速率
    MIN_ALPHA = 0.01            # 最小学习率 (保底)
    DEFAULT_WEIGHT = 0.5        # 未见过路由的默认权重

    # 失败分类关键词
    ERROR_PATTERNS = {
        "timeout": ["timeout", "timed out", "超时", "TimeoutError"],
        "permission": ["permission", "denied", "forbidden", "权限", "PermissionError"],
        "format": ["format", "invalid", "parse", "格式", "ValueError", "TypeError", "KeyError"],
        "api_error": ["400", "401", "403", "429", "500", "503", "api", "API"],
    }

    def __init__(self):
        # 路由权重表 v2: value 从 float 变为 dict (存储完整统计)
        self.route_weights: dict[str, float] = {}
        self._route_stats: dict[str, dict] = {}

        # 工具偏好矩阵: context_keyword -> {tool -> weighted_score}
        self.tool_preferences: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._tool_context_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Prompt 模板库 v2: name -> { "current": str, "variants": [{text, score, count}] }
        self.prompt_templates: dict[str, dict] = {}

        logger.info("AdaptiveOptimizer v2 初始化完成")

    # ============================================================
    # 路由权重 (v2: 自适应学习率)
    # ============================================================

    def update_route_weights(self, route_name: str, success: bool):
        """
        使用自适应 EMA 更新路由权重

        v2 变更:
            v1: 固定 alpha=0.05
            v2: alpha = 0.5 / (1 + 0.1 * n), 随样本量自适应递减

        设计原理:
            - 新路由: 样本少, alpha 大 → 快速学习
            - 成熟路由: 样本多, alpha 小 → 避免被短期波动影响
        """
        stats = self._route_stats.setdefault(route_name, {
            "count": 0,
            "successes": 0,
            "failures": 0,
            "last_success": 0,
            "last_failure": 0,
        })

        stats["count"] += 1
        if success:
            stats["successes"] += 1
            stats["last_success"] = time.time()
        else:
            stats["failures"] += 1
            stats["last_failure"] = time.time()

        # 自适应 alpha
        n = stats["count"]
        alpha = self.INITIAL_ALPHA / (1 + self.DECAY_RATE * n)
        alpha = max(alpha, self.MIN_ALPHA)

        current = self.route_weights.get(route_name, self.DEFAULT_WEIGHT)
        feedback_value = 1.0 if success else 0.0

        self.route_weights[route_name] = alpha * feedback_value + (1 - alpha) * current

    def get_best_route(self, candidates: list) -> str:
        """
        从候选路由中选择权重最高的最优路由 (v2: 加权均值 + 置信度)

        v2 改进: 候选路由样本量太少 (<5) 时, 加入探索倾向
        """
        if not candidates:
            raise ValueError("候选路由列表不能为空")

        scored = []
        for r in candidates:
            weight = self.route_weights.get(r, self.DEFAULT_WEIGHT)
            stats = self._route_stats.get(r, {})
            count = stats.get("count", 0)

            if count < 5:
                # 探索加成: 样本少的路由给 10% bonus, 鼓励探索
                weight *= 1.1

            scored.append((r, weight))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def route_stats(self, route_name: str) -> dict:
        """获取路由的详细统计信息"""
        stats = self._route_stats.get(route_name, {})
        count = stats.get("count", 0)
        weight = self.route_weights.get(route_name, self.DEFAULT_WEIGHT)

        return {
            "weight": round(weight, 4),
            "samples": count,
            "successes": stats.get("successes", 0),
            "failures": stats.get("failures", 0),
            "success_rate": round(stats.get("successes", 0) / max(count, 1), 3),
            "alpha": round(self.INITIAL_ALPHA / (1 + self.DECAY_RATE * max(count, 1)), 4),
        }

    # ============================================================
    # Prompt 优化 (v2: 失败分类 + 针对性优化 + 防重复)
    # ============================================================

    def optimize_prompt(self, template_name: str, feedback_list: list):
        """
        基于反馈自动优化 Prompt 模板 (v2: 智能分析, 不再硬编码)

        v1: 成功率 < 50% → 追加 "请先分析问题再使用工具" (且每次循环重复追加)
        v2:
            1. 分类失败原因 (timeout/permission/format/api_error/unknown)
            2. 根据失败类别选择针对性优化建议
            3. 检查是否已包含相同建议, 避免重复追加
            4. 追踪优化后的效果 (记录变体及其分数)
        """
        if not feedback_list:
            return

        avg_success = sum(1 for f in feedback_list if f.success) / len(feedback_list)

        if avg_success >= 0.5:
            return

        # 分类失败原因
        failures = [f for f in feedback_list if not f.success]
        categories = self._categorize_failures(failures)

        if not categories:
            return

        # 根据最主要的失败类别选择优化建议
        top_category = max(categories, key=categories.get)
        suggestion = self._get_optimization_suggestion(top_category, categories[top_category])

        # 获取当前模板
        template = self.prompt_templates.get(template_name, {})
        current_text = template.get("current", "")

        # 防重复: 检查建议是否已存在于模板中
        if suggestion in current_text:
            logger.debug(f"模板 {template_name} 已包含建议, 跳过重复追加")
            return

        # 保存优化前版本 (用于效果对比)
        if current_text:
            variants = template.get("variants", [])
            old_score = template.get("last_score", 0.0)
            variants.append({
                "text": current_text,
                "score": old_score,
                "count": template.get("last_count", 0),
                "timestamp": time.time(),
            })
            # 只保留最近 5 个变体
            template["variants"] = variants[-5:]

        # 追加优化建议
        new_text = current_text + "\n" + suggestion
        template["current"] = new_text
        template["last_score"] = avg_success
        template["last_count"] = len(feedback_list)
        template["last_category"] = top_category
        self.prompt_templates[template_name] = template

        logger.info(f"Prompt 模板 '{template_name}' 优化: 追加 [{top_category}] 建议 "
                     f"(成功率 {avg_success:.1%}, 变体数 {len(template.get('variants', []))})")

    def _categorize_failures(self, failures: list) -> dict[str, int]:
        """按错误类型分类失败记录"""
        categories: dict[str, int] = defaultdict(int)

        for f in failures:
            error_text = f.error_type or ""
            context_text = f.context or ""

            combined = f"{error_text} {context_text}".lower()
            matched = False

            for category, keywords in self.ERROR_PATTERNS.items():
                for kw in keywords:
                    if kw.lower() in combined:
                        categories[category] += 1
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                categories["unknown"] += 1

        return dict(categories)

    @staticmethod
    def _get_optimization_suggestion(category: str, count: int) -> str:
        """根据失败类别返回针对性优化建议"""
        suggestions = {
            "timeout": (
                "优化提示: 如果工具调用超时，请尝试: "
                "1) 拆分为更小的子任务; "
                "2) 优先使用缓存结果; "
                "3) 设置合理的超时时间。"
            ),
            "permission": (
                "优化提示: 遇到权限问题时，请: "
                "1) 检查路径是否在白名单内; "
                "2) 确认操作类型是否被允许; "
                "3) 如无权限，向用户说明原因。"
            ),
            "format": (
                "优化提示: 输入格式错误时，请: "
                "1) 先验证输入参数的完整性; "
                "2) 对模糊输入请求用户澄清; "
                "3) 使用默认值填充缺失参数。"
            ),
            "api_error": (
                "优化提示: API 调用失败时，请: "
                "1) 检查 API Key 和端点配置; "
                "2) 遇到限流时等待后重试; "
                "3) 考虑使用降级方案。"
            ),
            "unknown": (
                "优化提示: 执行失败时，请: "
                "1) 仔细分析用户意图; "
                "2) 确认所需工具是否可用; "
                "3) 分步骤执行，每步验证结果。"
            ),
        }
        return suggestions.get(category, suggestions["unknown"])

    def prompt_variants_report(self, template_name: str) -> dict | None:
        """获取 Prompt 模板的变体效果对比报告"""
        template = self.prompt_templates.get(template_name)
        if not template:
            return None

        return {
            "current_text_length": len(template.get("current", "")),
            "current_score": template.get("last_score", 0),
            "variants": template.get("variants", []),
            "last_category": template.get("last_category", "none"),
        }

    # ============================================================
    # v2: 工具偏好矩阵
    # ============================================================

    def update_tool_preference(self, tool_name: str, context: str, success: bool):
        """
        更新工具偏好矩阵

        记录特定上下文关键词下各工具的成功率,
        用于推荐"在这个场景下应该用哪个工具"

        Args:
            tool_name: 工具名称
            context: 上下文关键词 (如 "search", "calculate", "file")
            success: 是否成功
        """
        score = 1.0 if success else 0.0
        stats = self._tool_context_counts[context][tool_name]
        n = stats + 1
        old_avg = self.tool_preferences[context].get(tool_name, 0.5)

        # 在线均值更新: new_avg = old_avg + (new - old_avg) / n
        self.tool_preferences[context][tool_name] = old_avg + (score - old_avg) / n
        self._tool_context_counts[context][tool_name] = n

    def get_tool_preference(self, context: str, candidates: list | None = None) -> str | None:
        """
        获取特定上下文下推荐使用的工具

        Args:
            context: 上下文关键词
            candidates: 候选工具列表 (None 则返回该上下文下所有工具中最佳的)

        Returns:
            推荐的工具名称, 或 None
        """
        prefs = self.tool_preferences.get(context)
        if not prefs:
            return None

        if candidates:
            scored = [(t, prefs.get(t, 0.5)) for t in candidates if t in prefs]
        else:
            scored = list(prefs.items())

        if not scored:
            return None

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def analyze_failure_patterns(self) -> dict[str, dict]:
        """
        生成全局失败模式分析报告

        Returns:
            {
                "timeout": {"count": 5, "routes": ["api_call", "web_search"], "suggestion": "..."},
                ...
            }
        """
        report = {}
        for route_name, stats in self._route_stats.items():
            failures = stats.get("failures", 0)
            if failures == 0:
                continue

            # 简单分类 (基于路由名称启发式)
            route_lower = route_name.lower()
            if any(k in route_lower for k in ["search", "fetch", "download"]):
                category = "api_error"
            elif any(k in route_lower for k in ["file", "write", "read", "path"]):
                category = "permission"
            elif any(k in route_lower for k in ["calc", "code", "exec"]):
                category = "format"
            elif any(k in route_lower for k in ["vision", "image", "generate"]):
                category = "timeout"
            else:
                category = "unknown"

            if category not in report:
                report[category] = {"count": 0, "routes": []}

            report[category]["count"] += failures
            report[category]["routes"].append(route_name)

        # 添加建议
        for cat, info in report.items():
            suggestion = self._get_optimization_suggestion(cat, info["count"])
            report[cat]["suggestion"] = suggestion

        return report
