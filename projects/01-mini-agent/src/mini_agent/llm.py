"""Thin wrapper around the OpenAI SDK.

Works with any OpenAI-compatible API: OpenAI, NVIDIA NIM, local vLLM, etc.
No LangChain — just the raw SDK, so you can see every API call.
"""

from __future__ import annotations

from openai import OpenAI

from mini_agent.config import settings


def _build_client() -> OpenAI:
    """Create an OpenAI client pointed at the configured base URL."""
    return OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )


# Module-level client — reused across calls
_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Return the shared OpenAI client (lazy init)."""
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send messages to the LLM and return the response text.

    Args:
        messages: List of {"role": "...", "content": "..."} dicts.
        model: Override the default model.
        temperature: Override the default temperature.
        max_tokens: Override the default max_tokens.

    Returns:
        The model's text response.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature or settings.temperature,
        max_tokens=max_tokens or settings.max_tokens,
    )
    return response.choices[0].message.content or ""


def chat_stream(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
):
    """Stream tokens from the LLM.

    Yields one token string at a time.
    """
    client = get_client()
    stream = client.chat.completions.create(
        model=model or settings.llm_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature or settings.temperature,
        max_tokens=max_tokens or settings.max_tokens,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
