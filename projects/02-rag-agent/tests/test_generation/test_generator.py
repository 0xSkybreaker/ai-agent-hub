"""Tests for the RAG generator."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestGenerator:
    def test_generate_with_documents(self):
        from rag_agent.generation.generator import Generator, GenerationResult
        from rag_agent.retrieval.retriever import RetrievedDocument

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RetrievedDocument(
                text="The revenue forecast is $10M.",
                metadata={"file_name": "report.pdf", "source_path": "/report.pdf"},
                score=0.95,
            ),
        ]
        mock_retriever.format_context.return_value = "[Document 1 — Source: report.pdf]\nThe revenue forecast is $10M."

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Based on the report, the revenue forecast is $10M [Document 1]."
        mock_llm.model_name = "test-model"

        generator = Generator(mock_retriever, mock_llm)
        result = generator.generate("What is the revenue forecast?")

        assert isinstance(result, GenerationResult)
        assert "$10M" in result.answer
        assert len(result.sources) == 1
        assert result.sources[0].file_name == "report.pdf"

    def test_generate_no_documents(self):
        from rag_agent.generation.generator import Generator

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []

        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"

        generator = Generator(mock_retriever, mock_llm)
        result = generator.generate("What is the revenue forecast?")

        assert "don't have enough information" in result.answer.lower()
        assert len(result.sources) == 0
        assert result.retrieved_count == 0

    def test_generate_with_history(self):
        from rag_agent.generation.generator import Generator, GenerationResult
        from rag_agent.retrieval.retriever import RetrievedDocument

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RetrievedDocument(
                text="Important data here.",
                metadata={"file_name": "data.txt", "source_path": "/data.txt"},
                score=0.90,
            ),
        ]
        mock_retriever.format_context.return_value = "[Document 1 — Source: data.txt]\nImportant data here."

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Answer referencing data."
        mock_llm.model_name = "test-model"

        generator = Generator(mock_retriever, mock_llm)
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
        ]
        result = generator.generate("What is the data?", history=history)

        assert isinstance(result, GenerationResult)
        assert "Answer" in result.answer


class TestCitations:
    def test_extract_citations(self):
        from rag_agent.generation.citations import extract_citations, SourceCitation
        from rag_agent.retrieval.retriever import RetrievedDocument

        docs = [
            RetrievedDocument(
                text="Content from doc A." * 20,
                metadata={
                    "file_name": "doc_a.pdf",
                    "source_path": "/path/doc_a.pdf",
                    "page_number": 5,
                    "chunk_index": 2,
                },
                score=0.95,
            ),
        ]

        citations = extract_citations(docs)

        assert len(citations) == 1
        assert isinstance(citations[0], SourceCitation)
        assert citations[0].file_name == "doc_a.pdf"
        assert citations[0].page_number == 5
        assert citations[0].relevance_score == 0.95

    def test_deduplicate_same_source(self):
        from rag_agent.generation.citations import extract_citations
        from rag_agent.retrieval.retriever import RetrievedDocument

        docs = [
            RetrievedDocument(
                text="First chunk.",
                metadata={"file_name": "doc.pdf", "source_path": "/doc.pdf"},
                score=0.95,
            ),
            RetrievedDocument(
                text="Second chunk.",
                metadata={"file_name": "doc.pdf", "source_path": "/doc.pdf"},
                score=0.85,
            ),
        ]

        citations = extract_citations(docs)
        assert len(citations) == 1  # Deduplicated

    def test_format_citations_for_display(self):
        from rag_agent.generation.citations import SourceCitation, format_citations_for_display

        citations = [
            SourceCitation(
                file_name="report.pdf",
                source_path="/report.pdf",
                page_number=3,
                chunk_index=1,
                excerpt="The revenue forecast...",
                relevance_score=0.95,
            ),
        ]

        formatted = format_citations_for_display(citations)
        assert "report.pdf" in formatted
        assert "Page 3" in formatted

    def test_format_citations_empty(self):
        from rag_agent.generation.citations import format_citations_for_display
        assert format_citations_for_display([]) == ""
