# ==========================================
# ReAct 模式工具调度器
# ==========================================
# 该模块实现了一个基于 ReAct (Reasoning + Acting) 模式的工具调度系统
# 用于与 DeepSeek LLM 交互，支持函数调用和工具执行

import json
from openai import OpenAI
from tools.registry import registry, ToolInfo, ToolParameter
from core.config import settings


# ==================== ReAct 响应解析 ====================
# 从 LLM 返回的 JSON 中提取 action 或 final_answer
# 用于 ReAct 循环的"思考-行动-观察"控制流

import json
from typing import NamedTuple


class ParsedResponse(NamedTuple):
    """ReAct 响应解析结果（不可变，线程安全）"""
    action: str = ""           # 要执行的工具名称
    action_input: dict = {}    # 工具参数
    is_final: bool = False     # 是否为最终答案
    final_answer: str = ""     # 最终答案内容


def parse_response(content: str) -> ParsedResponse:
    """
    解析 LLM 返回的 ReAct 格式 JSON 响应

    输入格式:
        {"thought": "...", "action": "web_search", "action_input": {"query": "..."}}
    或:
        {"thought": "...", "final_answer": "..."}

    Args:
        content: LLM 返回的 JSON 字符串

    Returns:
        ParsedResponse: 包含 action 或 final_answer 的解析结果

    Raises:
        json.JSONDecodeError: 当 LLM 返回的不是合法 JSON 时抛出
    """
    data = json.loads(content)

    # 判断是否为最终答案（终止循环）
    if "final_answer" in data:
        return ParsedResponse(
            is_final=True,
            final_answer=data["final_answer"]
        )

    # 否则提取 action（继续循环）
    return ParsedResponse(
        action=data.get("action", ""),
        action_input=data.get("action_input", {})
    )



class ReActDispatcher:
    """
    ReAct 模式调度器类
    
    该类负责：
    1. 维护工具注册表
    2. 与 DeepSeek LLM 交互
    3. 管理多轮对话和工具调用
    """

    def __init__(self):
        self.registry = registry
        self.client = OpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL
        )
        self.model = settings.LLM_MODEL

    def _get_tool_schema(self):
        """
        从工具注册表生成 OpenAI 函数调用 schema
        修复: 正确处理 ToolInfo dataclass 和 ToolParameter dataclass
        """
        schemas = []
        for name, info in self.registry._tools.items():
            # info 是 ToolInfo dataclass，不是 dict
            if not isinstance(info, ToolInfo):
                continue

            properties = {}
            required = []

            for p in info.parameters:
                # p 是 ToolParameter dataclass，不是 str/dict
                pname = p.name
                properties[pname] = {
                    "type": p.type if p.type in ("string", "number", "boolean", "integer", "array", "object") else "string",
                    "description": p.description
                }
                if p.required:
                    required.append(pname)

            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })

        return schemas

    def run(self, user_query: str, max_rounds: int = 5):
        """
        执行 ReAct 循环
        """
        messages = [{"role": "user", "content": user_query}]
        tools = self._get_tool_schema()

        for round_i in range(max_rounds):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                temperature=0.1
            )

            msg = resp.choices[0].message
            messages.append(msg)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    # execute 接受 (tool_name, args_dict) 或 (tool_name, **kwargs)
                    result = self.registry.execute(tc.function.name, args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False)
                    })
            else:
                return {
                    "answer": msg.content,
                    "rounds": round_i + 1
                }

        return {
            "answer": messages[-1].content,
            "rounds": max_rounds
        }
