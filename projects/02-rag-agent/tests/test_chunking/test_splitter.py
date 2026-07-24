"""Tests for text chunking and metadata."""

from __future__ import annotations


class TestDocumentSplitter:
    def test_split_documents(self, sample_documents):
        from rag_agent.chunking.splitter import DocumentSplitter

        splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split(sample_documents)

        assert len(chunks) > 0
        # Each chunk should have metadata
        for chunk in chunks:
            assert chunk.text
            assert "source_path" in chunk.metadata
            assert "chunk_index" in chunk.metadata
            assert "chunk_count" in chunk.metadata

    def test_split_single(self, sample_documents):
        from rag_agent.chunking.splitter import DocumentSplitter

        splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_single(sample_documents[0])

        assert len(chunks) > 0
        assert all(c.metadata["file_type"] == "txt" for c in chunks)

    def test_chunk_size_respected(self, sample_documents):
        from rag_agent.chunking.splitter import DocumentSplitter

        splitter = DocumentSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split(sample_documents)

        for chunk in chunks:
            assert len(chunk.text) <= 250  # Some tolerance


class TestMetadataEnrichment:
    def test_enrich_chunk_metadata(self):
        from rag_agent.chunking.metadata import enrich_chunk_metadata

        meta = enrich_chunk_metadata(
            metadata={"source_path": "/test/doc.txt", "file_name": "doc.txt"},
            chunk_text="Hello world",
            chunk_index=3,
        )

        assert meta["chunk_index"] == 3
        assert "content_hash" in meta
        assert "ingested_at" in meta
        assert meta["char_count"] == len("Hello world")
