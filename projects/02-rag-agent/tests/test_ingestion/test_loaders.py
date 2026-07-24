"""Tests for document loaders."""

from __future__ import annotations


class TestTextLoader:
    def test_load_txt_file(self, sample_text_file):
        from rag_agent.ingestion.text_loader import TextLoader

        loader = TextLoader()
        docs = loader.load(sample_text_file)

        assert len(docs) == 1
        assert "test document" in docs[0].content.lower()
        assert docs[0].metadata["file_type"] == "txt"

    def test_supported_extensions(self):
        from rag_agent.ingestion.text_loader import TextLoader

        loader = TextLoader()
        assert ".txt" in loader.supported_extensions


class TestMarkdownLoader:
    def test_load_md_file(self, sample_markdown_file):
        from rag_agent.ingestion.text_loader import MarkdownLoader

        loader = MarkdownLoader()
        docs = loader.load(sample_markdown_file)

        assert len(docs) == 1
        assert "Section 1" in docs[0].content
        assert docs[0].metadata["file_type"] == "md"
        assert docs[0].metadata["title"] == "Test Document"

    def test_supported_extensions(self):
        from rag_agent.ingestion.text_loader import MarkdownLoader

        loader = MarkdownLoader()
        assert ".md" in loader.supported_extensions


class TestLoaderRegistry:
    def test_register_and_get(self):
        from rag_agent.ingestion.base import LoaderRegistry
        from rag_agent.ingestion.text_loader import TextLoader

        registry = LoaderRegistry()
        registry.register(TextLoader)

        loader = registry.get_loader("test.txt")
        assert isinstance(loader, TextLoader)

    def test_unsupported_extension(self):
        from rag_agent.ingestion.base import LoaderRegistry

        registry = LoaderRegistry()
        try:
            registry.get_loader("test.xyz")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_list_extensions(self):
        from rag_agent.ingestion.base import LoaderRegistry
        from rag_agent.ingestion.text_loader import TextLoader

        registry = LoaderRegistry()
        registry.register(TextLoader)
        assert ".txt" in registry.list_extensions()
