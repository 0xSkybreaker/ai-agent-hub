"""Tests for the retriever module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestRetriever:
    def test_retrieve_returns_documents(self):
        from rag_agent.retrieval.retriever import Retriever, RetrievedDocument

        mock_store = MagicMock()
        mock_store.search.return_value = [
            ("doc text 1", {"file_name": "test.txt", "source_path": "/test.txt"}, 0.95),
            ("doc text 2", {"file_name": "test2.txt", "source_path": "/test2.txt"}, 0.85),
        ]

        mock_embed = MagicMock()

        retriever = Retriever(mock_store, mock_embed)
        results = retriever.retrieve("test query", top_k=2)

        assert len(results) == 2
        assert isinstance(results[0], RetrievedDocument)
        assert results[0].score == 0.95
        assert results[0].metadata["file_name"] == "test.txt"

    def test_retrieve_respects_threshold(self):
        from rag_agent.retrieval.retriever import Retriever

        mock_store = MagicMock()
        mock_store.search.return_value = [
            ("doc text 1", {"file_name": "test.txt"}, 0.95),
            ("doc text 2", {"file_name": "test2.txt"}, 0.35),
        ]

        mock_embed = MagicMock()
        retriever = Retriever(mock_store, mock_embed)
        results = retriever.retrieve("test query", top_k=5, threshold=0.5)

        assert len(results) == 1
        assert results[0].score == 0.95

    def test_format_context(self):
        from rag_agent.retrieval.retriever import Retriever, RetrievedDocument

        retriever = Retriever(MagicMock(), MagicMock())
        docs = [
            RetrievedDocument(
                text="First document content.",
                metadata={"file_name": "doc1.txt", "page_number": 1},
                score=0.95,
            ),
            RetrievedDocument(
                text="Second document content.",
                metadata={"file_name": "doc2.txt"},
                score=0.85,
            ),
        ]

        context = retriever.format_context(docs)

        assert "doc1.txt" in context
        assert "Page 1" in context
        assert "doc2.txt" in context
        assert "0.95" in context

    def test_format_context_empty(self):
        from rag_agent.retrieval.retriever import Retriever

        retriever = Retriever(MagicMock(), MagicMock())
        context = retriever.format_context([])
        assert "No relevant documents" in context
