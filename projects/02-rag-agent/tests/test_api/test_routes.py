"""Tests for FastAPI routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_app():
    """Create a test FastAPI app with mocked dependencies."""
    # Mock all singletons before importing the app
    with (
        patch("rag_agent.api.dependencies._embedding_client", None),
        patch("rag_agent.api.dependencies._llm_client", None),
        patch("rag_agent.api.dependencies._vector_store", None),
        patch("rag_agent.api.dependencies._retriever", None),
        patch("rag_agent.api.dependencies._generator", None),
        patch("rag_agent.api.dependencies._indexer", None),
        patch("rag_agent.api.dependencies._memory", None),
    ):
        from rag_agent.api.server import create_app
        app = create_app()
        yield app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestQueryEndpoint:
    def test_query_success(self, client):
        from rag_agent.generation.generator import GenerationResult
        from rag_agent.generation.citations import SourceCitation

        mock_result = GenerationResult(
            answer="Test answer.",
            sources=[
                SourceCitation(
                    file_name="test.txt",
                    source_path="/test.txt",
                    page_number=None,
                    chunk_index=0,
                    excerpt="test excerpt...",
                    relevance_score=0.95,
                ),
            ],
            retrieved_count=1,
            model="test-model",
        )

        with patch("rag_agent.api.routes.get_generator") as mock_gen_fn:
            mock_gen = MagicMock()
            mock_gen.generate.return_value = mock_result
            mock_gen_fn.return_value = mock_gen

            with patch("rag_agent.api.routes.get_memory") as mock_mem_fn:
                mock_mem = MagicMock()
                mock_mem.create_session.return_value = "test-session"
                mock_mem.get_history.return_value = []
                mock_mem_fn.return_value = mock_mem

                response = client.post(
                    "/api/v1/query",
                    json={"question": "test question", "top_k": 3},
                )

                assert response.status_code == 200
                data = response.json()
                assert data["answer"] == "Test answer."
                assert len(data["sources"]) == 1

    def test_query_empty_question(self, client):
        response = client.post("/api/v1/query", json={"question": ""})
        assert response.status_code == 422  # Validation error


class TestIngestEndpoint:
    def test_ingest_path(self, client):
        from rag_agent.vector_store.indexer import IndexResult

        mock_result = IndexResult(
            source="/test/file.txt",
            chunks_created=5,
            status="indexed",
            message="Successfully indexed 5 chunks",
        )

        with patch("rag_agent.api.routes.get_indexer") as mock_fn:
            mock_idx = MagicMock()
            mock_idx.index_file.return_value = mock_result
            mock_fn.return_value = mock_idx

            response = client.post(
                "/api/v1/ingest/path",
                params={"source_path": "/test/file.txt"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "indexed"
            assert data["chunks_created"] == 5


class TestMemoryEndpoint:
    def test_clear_memory(self, client):
        with patch("rag_agent.api.routes.get_memory") as mock_fn:
            mock_mem = MagicMock()
            mock_fn.return_value = mock_mem

            response = client.delete("/api/v1/memory/test-session")
            assert response.status_code == 200
            assert response.json()["status"] == "cleared"


class TestDeleteDocumentEndpoint:
    def test_delete_document(self, client):
        with patch("rag_agent.api.routes.get_indexer") as mock_fn:
            mock_idx = MagicMock()
            mock_idx.remove_document.return_value = 10
            mock_fn.return_value = mock_idx

            response = client.delete("/api/v1/documents/test/file.txt")
            assert response.status_code == 200
            assert response.json()["chunks_removed"] == 10
