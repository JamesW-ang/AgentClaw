"""
AgentClaw v6 — 评估系统入口（向后兼容包装器）
实际实现在 eval/runner.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from eval.cases import get_agent_eval_cases
from eval.runner import AgentEvaluator

if __name__ == "__main__":
    evaluator = AgentEvaluator(use_mock=True)
    evaluator.add_cases(get_agent_eval_cases())
    report = evaluator.run()
    evaluator.save_report("./eval_output")
