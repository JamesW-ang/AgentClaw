"""
GuardedChatModel - LangChain ChatModel wrapper for LLMGuard
让 LLMGuard 的容错能力无缝接入 LangGraph create_react_agent

v6.1.5 修复:
    1-3. (v6.1.4)
    4. json.loads 容错: LLM 返回破损 JSON 时自动修复 (unterminated string/missing brace/trailing comma)
"""
import json
import logging
from collections.abc import Callable, Generator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

logger = logging.getLogger(__name__)


def _repair_tool_args(raw: str, tool_name: str) -> dict:
    """Best-effort repair of malformed JSON from LLM tool-call arguments."""
    if not raw or not raw.strip():
        return {}
    # Try common fixes for truncated JSON
    import re
    raw = raw.strip()
    # Fix 1: unclosed string — add closing quote + brace
    if raw.count('"') % 2 != 0:
        raw += '"'
    # Fix 2: missing closing brace
    open_b = raw.count("{") - raw.count("}")
    open_s = raw.count("[") - raw.count("]")
    raw += "]" * open_s + "}" * open_b
    # Fix 3: trailing comma before closing brace/bracket
    raw = re.sub(r',(\s*[}\]])', r'\1', raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning(f"[GuardedChatModel] Could not repair args for {tool_name}, using empty dict")
        return {}


def _convert_lc_tool_calls_to_openai(tool_calls: list[dict]) -> list[dict]:
    """Convert LangChain tool_calls format to OpenAI API format.

    LangChain: [{"name": "...", "args": {...}, "id": "...", "type": "tool_call"}]
    OpenAI:    [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "{...}"}}]
    """
    converted = []
    for tc in tool_calls:
        fn_name = tc.get("name", "")
        fn_args = tc.get("args", {})
        if not isinstance(fn_args, str):
            fn_args = json.dumps(fn_args, ensure_ascii=False)
        converted.append({
            "id": tc.get("id", ""),
            "type": "function",
            "function": {"name": fn_name, "arguments": fn_args},
        })
    return converted


class GuardedChatModel(BaseChatModel):
    """
    LangChain ChatModel that delegates to LLMGuard for fault-tolerant LLM calls.
    """

    guard: Any = Field(description="LLMGuard instance")
    temperature: float = 0.0
    max_tokens: int = 2000

    @property
    def _llm_type(self) -> str:
        return "guarded-chat-model"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RunnableBinding:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return RunnableBinding(bound=self, kwargs=kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using LLMGuard"""
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        response_format = kwargs.pop("response_format", None)

        # 修复1：LangChain Pydantic 工具 -> OpenAI dict 格式
        if tools is not None:
            tools = [convert_to_openai_tool(t) for t in tools]

        # Convert LangChain messages to OpenAI format
        openai_messages = self._messages_to_openai(messages)

        result = self.guard.chat(
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

        if result.is_error:
            content = result.recovery_hint or result.error_message or "处理失败，请重试"
            logger.warning(f"[GuardedChatModel] LLMGuard returned error: {result.error_type}")
            message = AIMessage(content=content)
        else:
            content = result.content
            tool_calls = None
            raw = getattr(result, 'raw_response', None)
            if raw and hasattr(raw, 'choices') and raw.choices:
                ai_msg = raw.choices[0].message
                if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
                    tool_calls = []
                    for tc in ai_msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            raw_args = tc.function.arguments or ""
                            logger.warning(
                                f"[GuardedChatModel] LLM returned malformed tool-call "
                                f"arguments for {tc.function.name}: {raw_args[:120]}"
                            )
                            args = _repair_tool_args(raw_args, tc.function.name)
                        tool_calls.append({
                            "name": tc.function.name,
                            "args": args,
                            "id": tc.id,
                        })
            if tool_calls:
                message = AIMessage(content=content, tool_calls=tool_calls)
            else:
                message = AIMessage(content=content)

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation], llm_output={
            "model": result.model,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "is_fallback": result.is_fallback,
            "fallback_level": result.fallback_level,
            "attempts": result.attempts,
        })

    @staticmethod
    def _messages_to_openai(messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Convert LangChain messages to OpenAI API format, preserving tool_call_id."""
        openai_messages = []
        for msg in messages:
            if msg.type == "system":
                openai_messages.append({"role": "system", "content": msg.content})
            elif msg.type == "tool":
                tc_id = getattr(msg, 'tool_call_id', None) or ""
                openai_messages.append({"role": "tool", "content": msg.content, "tool_call_id": tc_id})
            elif msg.type == "ai":
                entry = {"role": "assistant", "content": msg.content}
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    entry["tool_calls"] = _convert_lc_tool_calls_to_openai(msg.tool_calls)
                openai_messages.append(entry)
            else:
                openai_messages.append({"role": "user", "content": msg.content})
        return openai_messages

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Generator[ChatGenerationChunk, None, None]:
        """Stream a response using LLMGuard with streaming enabled."""
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        response_format = kwargs.pop("response_format", None)

        if tools is not None:
            tools = [convert_to_openai_tool(t) for t in tools]

        openai_messages = self._messages_to_openai(messages)

        result = self.guard.chat(
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            stop=stop,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

        if result.is_error:
            content = result.recovery_hint or result.error_message or "处理失败，请重试"
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=content)
            )
            return

        stream_iter = result.stream
        if stream_iter is None:
            return

        try:
            for chunk in stream_iter:
                if len(chunk.choices) == 0:
                    continue

                delta = chunk.choices[0].delta

                content = delta.content or ""

                tool_call_chunks = []
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        tool_call_chunks.append(
                            {
                                "name": tc.function.name if tc.function else None,
                                "args": tc.function.arguments if tc.function else None,
                                "id": tc.id,
                                "index": tc.index,
                            }
                        )

                if not content and not tool_call_chunks:
                    continue

                message_chunk = AIMessageChunk(
                    content=content,
                    tool_call_chunks=tool_call_chunks if tool_call_chunks else None,
                )
                yield ChatGenerationChunk(message=message_chunk)
        except Exception as exc:
            logger.error(f"[GuardedChatModel] Stream error: {exc}")
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="\n\n[Stream interrupted, please retry]")
            )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate - delegates to sync _generate"""
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
