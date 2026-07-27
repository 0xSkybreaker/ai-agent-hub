"""Shared test fixtures for agentic-rag."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure rag-agent is importable from tests (needed for some tool tests)
_RAG_AGENT_SRC = Path(__file__).resolve().parents[2] / "02-rag-agent" / "src"
if str(_RAG_AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(_RAG_AGENT_SRC))


@pytest.fixture
def mock_llm_client():
    """Mock the OpenAI SDK client for direct injection into the agent."""
    mock = MagicMock()
    return mock


@pytest.fixture
def mock_retriever():
    """Mock the Retriever from rag-agent."""
    with patch("agentic_rag.tools._retriever") as mock:
        yield mock


@pytest.fixture
def mock_vector_store():
    """Mock the ChromaVectorStore from rag-agent."""
    with patch("agentic_rag.tools._vector_store") as mock:
        yield mock
