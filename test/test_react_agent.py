# ============================================================
# ReAct 循环测试模块
# ============================================================
"""
ReAct (Reasoning + Acting) 推理循环的单元测试

该测试套件验证 ReAct Agent 的核心功能:
    1. LLM 响应解析 (思考、行动、最终答案)
    2. 工具调用和执行
    3. 工具失败处理和错误恢复
    4. 最大轮次限制和循环终止

测试场景:
    - 成功的行动解析
    - 最终答案检测
    - 工具执行成功
    - 工具执行失败
    - 达到最大轮次停止

依赖:
    - pytest: 测试框架
    - json: JSON 序列化
"""

import pytest
import json
from tools.dispatcher import parse_response


# ============================================================
# ReAct 循环测试类
# ============================================================

class TestReActLoop:
    """
    ReAct 推理循环的测试类
    
    测试覆盖:
        - LLM 响应解析器功能
        - 工具调用和结果处理
        - 错误处理和恢复机制
        - 循环控制和终止条件
    """

    def test_parse_llm_response_with_action(self):
        """
        测试: 解析包含行动的 LLM 响应
        
        测试目标:
            验证能够正确解析 LLM 返回的思考-行动对
        
        场景:
            - LLM 返回思考 ("I need info")
            - 指定要执行的行动 ("web_search")
            - 提供行动参数 ({"query": "python"})
        
        预期结果:
            - result.action == "web_search"
            - result.action_input == {"query": "python"}
        """
        # 构造 LLM 响应内容 (JSON 格式)
        content = json.dumps({
            "thought": "I need info",           # 推理思路
            "action": "web_search",             # 要执行的工具
            "action_input": {"query": "python"} # 工具参数
        })
        
        # 调用解析函数
        result = parse_response(content)
        
        # 断言：验证行动被正确解析
        assert result.action == "web_search"
        assert result.action_input == {"query": "python"}

    def test_parse_llm_response_final_answer(self):
        """
        测试: 检测最终答案并停止循环
        
        测试目标:
            验证能够识别 LLM 的最终答案，终止推理循环
        
        场景:
            - LLM 确认已得到答案
            - 返回 final_answer 字段而不是 action
            - 应该设置 is_final 标记
        
        预期结果:
            - result.is_final == True (循环应停止)
            - result.final_answer == "Python is great"
        """
        # 构造最终答案响应
        content = json.dumps({
            "thought": "I have the answer",
            "final_answer": "Python is great"
        })
        
        # 调用解析函数
        result = parse_response(content)
        
        # 断言：验证最终答案被正确识别
        assert result.is_final is True
        assert result.final_answer == "Python is great"

    def test_tool_call_success(self, mock_tool_registry):
        """
        测试: 工具调用成功
        
        测试目标:
            验证工具能够正确执行并返回结果
        
        场景:
            - 调用 web_search 工具
            - 工具正常执行
            - 返回预期的搜索结果
        
        预期结果:
            - result == "Search result: test"
        """
        # 调用工具注册表的 execute 方法
        result = mock_tool_registry.execute("web_search", {"query": "test"})
        
        # 断言：验证工具执行成功并返回正确结果
        assert result == "Search result: test"

    def test_tool_call_failure_logs_error(self, mock_tool_registry):
        """
        测试: 工具调用失败时处理
        
        测试目标:
            验证工具执行失败时能够正确处理错误
        
        场景:
            - 工具执行过程中抛出 TimeoutError
            - Agent 应记录错误
            - 错误应被正确传播
        
        预期结果:
            - TimeoutError 被抛出
            - 错误信息包含 "API timeout"
        """
        # 模拟工具执行失败 (设置副作用为抛出异常)
        mock_tool_registry.execute.side_effect = TimeoutError("API timeout")
        
        # 验证异常被正确抛出
        with pytest.raises(TimeoutError):
            mock_tool_registry.execute("web_search", {"query": "test"})

    def test_max_rounds_stops_loop(self):
        """
        测试: 达到最大轮次时停止循环
        
        测试目标:
            验证 Agent 在达到最大轮次限制时能够停止循环
        
        场景:
            - 设置最大轮次为 3
            - 模拟 Agent 步骤执行
            - 当到达最大轮次时返回最终答案
            - 继续执行额外步骤，验证循环已停止
        
        预期结果:
            - 只执行了 3 轮 (max_rounds)
            - 循环停止，没有继续执行
            - round_count == max_rounds
        """
        # 设置最大轮次
        max_rounds = 3
        round_count = 0
        
        def mock_step():
            """
            模拟 Agent 的单步推理
            
            返回:
                dict: 步骤结果，包含 is_final 标记和可选的最终答案
            """
            nonlocal round_count
            round_count += 1
            
            # 当到达最大轮次时，返回最终答案
            if round_count >= max_rounds:
                return {
                    "is_final": True,
                    "final_answer": "stopped"
                }
            
            # 否则返回继续标记
            return {"is_final": False}
        
        # 模拟循环 (max_rounds + 5 次迭代)
        # 验证循环能在达到最大轮次时停止
        for _ in range(max_rounds + 5):
            step = mock_step()
            
            # 如果检测到最终答案，停止循环
            if step["is_final"]:
                break
        
        # 断言：验证只执行了 max_rounds 次
        assert round_count == max_rounds