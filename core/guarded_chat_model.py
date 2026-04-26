"""
GuardedChatModel - LangChain ChatModel wrapper for LLMGuard
让 LLMGuard 的容错能力无缝接入 LangGraph create_react_agent
"""
import logging
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.language_models.chat_models import BaseChatModel
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
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using LLMGuard"""
        # Convert LangChain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            role = "system" if msg.type == "system" else (
                "assistant" if msg.type == "ai" else "user"
            )
            openai_messages.append({"role": role, "content": msg.content})
        
        # Call LLMGuard
        result = self.guard.chat(
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
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
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate - delegates to sync _generate (LLMGuard handles async internally)"""
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
