"""NVIDIA NIM LLM client wrapper using LangChain."""

from __future__ import annotations

from collections.abc import Iterator

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from rag_agent.config import settings
from rag_agent.utils.logger import get_logger

logger = get_logger()


class LLMClient:
    """Wraps NVIDIA chat model for generation with retry and streaming.

    Uses LangChain's ChatNVIDIA which handles API auth, retries,
    and the OpenAI-compatible format natively.
    """

    def __init__(self) -> None:
        self._model = ChatNVIDIA(
            model=settings.chat_model,
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        self.model_name = settings.chat_model
        logger.info(f"LLM client initialized: model={self.model_name}")

    def generate(self, messages: list[dict]) -> str:
        """Generate a response for a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            The model's response text.
        """
        logger.debug(f"Generating response for {len(messages)} messages")
        response = self._model.invoke(
            [(m["role"], m["content"]) for m in messages]
        )
        return response.content

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Stream a response token by token.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Yields:
            Text tokens as they are generated.
        """
        logger.debug(f"Streaming response for {len(messages)} messages")
        for chunk in self._model.stream(
            [(m["role"], m["content"]) for m in messages]
        ):
            if chunk.content:
                yield chunk.content

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict] | None = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Convenience method for system + user + history chat.

        Args:
            system_prompt: The system instruction.
            user_message: The user's question.
            history: Optional conversation history.
            stream: If True, return an iterator of tokens.

        Returns:
            Response text or token iterator.
        """
        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_message})

        if stream:
            return self.stream(messages)
        else:
            return self.generate(messages)
