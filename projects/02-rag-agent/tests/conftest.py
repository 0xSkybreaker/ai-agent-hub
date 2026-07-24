"""Shared pytest fixtures for RAG Agent tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mock_settings():
    """Mock application settings for testing."""
    with patch("rag_agent.config.settings") as mock:
        mock.nvidia_api_key = "test-api-key"
        mock.nvidia_base_url = "https://integrate.api.nvidia.com/v1"
        mock.chat_model = "test-model"
        mock.embedding_model = "test-embed-model"
        mock.chroma_persist_dir = "./data/test_chroma"
        mock.chroma_persist_path = Path("./data/test_chroma")
        mock.collection_name = "test_collection"
        mock.chunk_size = 500
        mock.chunk_overlap = 50
        mock.top_k = 3
        mock.similarity_threshold = 0.0
        mock.max_tokens = 512
        mock.temperature = 0.1
        mock.host = "127.0.0.1"
        mock.port = 8000
        mock.log_level = "DEBUG"
        mock.max_conversation_turns = 5
        mock.tracker_db_path = "./data/test_tracker.db"
        mock.tracker_db_path_resolved = Path("./data/test_tracker.db")
        yield mock


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing loaders."""
    content = "This is a test document.\n\nIt has multiple paragraphs.\n\nThis is the third paragraph."
    file_path = temp_dir / "test.txt"
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def sample_markdown_file(temp_dir):
    """Create a sample markdown file for testing loaders."""
    content = """# Test Document

## Section 1

This is the first section content.

## Section 2

This is the second section with **bold** and *italic* text.
"""
    file_path = temp_dir / "test.md"
    file_path.write_text(content)
    return str(file_path)


@pytest.fixture
def sample_documents():
    """Create sample Document objects for testing chunking."""
    from rag_agent.ingestion.base import Document

    return [
        Document(
            content="This is document one. " * 50,
            metadata={
                "source_path": "/test/doc1.txt",
                "file_name": "doc1.txt",
                "file_type": "txt",
            },
        ),
        Document(
            content="This is document two. " * 50,
            metadata={
                "source_path": "/test/doc2.txt",
                "file_name": "doc2.txt",
                "file_type": "txt",
            },
        ),
    ]


@pytest.fixture
def mock_openai_client():
    """Mock the OpenAI client for NVIDIA API calls."""
    with patch("openai.OpenAI") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance

        # Mock chat completion
        chat_response = MagicMock()
        chat_choice = MagicMock()
        chat_message = MagicMock()
        chat_message.content = "This is a mock response."
        chat_choice.message = chat_message
        chat_response.choices = [chat_choice]
        client_instance.chat.completions.create.return_value = chat_response

        # Mock embeddings
        embed_response = MagicMock()
        embed_data = MagicMock()
        embed_data.embedding = [0.1, 0.2, 0.3]
        embed_response.data = [embed_data]
        client_instance.embeddings.create.return_value = embed_response

        yield client_instance
