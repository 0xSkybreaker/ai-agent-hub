"""Streaming utilities for LLM output."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator


def collect_stream(stream: Iterator[str]) -> str:
    """Collect all tokens from a stream into a single string.

    Args:
        stream: Token iterator from LLM streaming.

    Returns:
        Concatenated response text.
    """
    buffer: list[str] = []
    for token in stream:
        buffer.append(token)
    return "".join(buffer)


async def async_collect_stream(stream: AsyncIterator[str]) -> str:
    """Async version of collect_stream.

    Args:
        stream: Async token iterator.

    Returns:
        Concatenated response text.
    """
    buffer: list[str] = []
    async for token in stream:
        buffer.append(token)
    return "".join(buffer)
