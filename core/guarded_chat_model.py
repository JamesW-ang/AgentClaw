"""
GuardedChatModel - LangChain ChatModel wrapper for LLMGuard
让 LLMGuard 的容错能力无缝接入 LangGraph create_react_agent
"""
import logging
from typing import Any, Callable, Dict, List, Optional, Union, Sequence, Generator
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import BaseTool
from pydantic import Field

logger = logging.getLogger(__name__)


class GuardedChatModel(BaseChatModel):
    """
    LangChain ChatModel that delegates to LLMGuard for fault-tolerant LLM calls.
    
    Usage:
        from core.guarded_chat_model import GuardedChatModel
        from core.llm_guard import LLMGuard, get_llm_guard
        
        guard = get_llm_guard(default_model="deepseek-chat", backup_models=["deepseek-reasoner"])
        model = GuardedChatModel(guard=guard, temperature=0)
        
        # Use with LangGraph
        agent = create_react_agent(model=model, tools=tools, ...)
    """
    
    guard: Any = Field(description="LLMGuard instance")
    temperature: float = 0.0
    max_tokens: int = 2000
    
    @property
    def _llm_type(self) -> str:
        return "guarded-chat-model"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> RunnableBinding:
        """
        Bind tools to the model. Returns a RunnableBinding that stores tools
        and passes them to _generate / _stream via kwargs.
        """
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return RunnableBinding(bound=self, kwargs=kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using LLMGuard"""
        # Extract LangGraph/function-calling parameters from kwargs
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        response_format = kwargs.pop("response_format", None)

        # Convert LangChain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            role = "system" if msg.type == "system" else (
                "assistant" if msg.type == "ai" else "user"
            )
            openai_messages.append({"role": role, "content": msg.content})

        # Call LLMGuard with tool definition forwarding
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
            # Return the fallback message from LLMGuard
            content = result.recovery_hint or result.error_message or "处理失败，请重试"
            logger.warning(f"[GuardedChatModel] LLMGuard returned error: {result.error_type}")
        else:
            content = result.content

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
    
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Generator[ChatGenerationChunk, None, None]:
        """Stream a response using LLMGuard with streaming enabled."""
        # Extract LangGraph/function-calling parameters from kwargs
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        response_format = kwargs.pop("response_format", None)

        # Convert LangChain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            role = "system" if msg.type == "system" else (
                "assistant" if msg.type == "ai" else "user"
            )
            openai_messages.append({"role": role, "content": msg.content})

        # Call LLMGuard with stream=True
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
            # Fallback/error case: yield a single chunk with the error message
            content = result.recovery_hint or result.error_message or "处理失败，请重试"
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=content)
            )
            return

        # Iterate the raw OpenAI stream
        stream_iter = result.stream
        if stream_iter is None:
            return

        try:
            for chunk in stream_iter:
                if len(chunk.choices) == 0:
                    continue

                delta = chunk.choices[0].delta

                # Content delta (text tokens)
                content = delta.content or ""

                # Tool call chunks (incremental function calling)
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

                # Only yield if there's actual content or tool calls
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
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate - delegates to sync _generate (LLMGuard handles async internally)"""
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
