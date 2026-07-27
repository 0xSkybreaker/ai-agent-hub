"""Tests for RAG-aware tool functions."""

from __future__ import annotations

from unittest.mock import MagicMock

from agentic_rag.tools import (
    Tool,
    ToolRegistry,
    create_registry,
    tool_list_documents,
    tool_search_documents,
)


class TestTool:
    """Tests for the Tool dataclass."""

    def test_to_openai_schema(self):
        """Tool.to_openai_schema() should produce valid OpenAI format."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={
                "query": {
                    "type": "string",
                    "description": "A query parameter",
                }
            },
            func=lambda query: f"Result: {query}",
        )

        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert schema["function"]["description"] == "A test tool"
        assert "query" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["required"] == ["query"]


class TestToolRegistry:
    """Tests for the ToolRegistry."""

    def test_register_and_get_schemas(self):
        """Registering tools and getting schemas should work."""
        registry = ToolRegistry()
        tool = Tool(
            name="my_tool",
            description="Does something",
            parameters={
                "input": {"type": "string", "description": "Input text"}
            },
            func=lambda input: input.upper(),
        )
        registry.register(tool)

        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "my_tool"

    def test_execute_tool(self):
        """Executing a registered tool should call its function."""
        registry = ToolRegistry()
        tool = Tool(
            name="greet",
            description="Greets someone",
            parameters={
                "name": {"type": "string", "description": "Name to greet"}
            },
            func=lambda name: f"Hello, {name}!",
        )
        registry.register(tool)

        result = registry.execute("greet", {"name": "World"})
        assert result == "Hello, World!"

    def test_execute_unknown_tool(self):
        """Executing an unknown tool should return an error message."""
        registry = ToolRegistry()
        result = registry.execute("nonexistent", {})
        assert "Error" in result
        assert "nonexistent" in result

    def test_execute_tool_error(self):
        """Tool errors should be caught and returned as strings."""
        registry = ToolRegistry()
        tool = Tool(
            name="broken",
            description="Always fails",
            parameters={},
            func=lambda: 1 / 0,
        )
        registry.register(tool)
        result = registry.execute("broken", {})
        assert "Tool error" in result
        assert "ZeroDivisionError" in result

    def test_list_tools(self):
        """list_tools() should return a human-readable summary."""
        registry = ToolRegistry()
        registry.register(Tool(
            name="search",
            description="Search the knowledge base",
            parameters={"query": {"type": "string", "description": "Query"}},
            func=lambda query: query,
        ))
        summary = registry.list_tools()
        assert "search" in summary
        assert "Search the knowledge base" in summary

    def test_has_tool(self):
        """has() should check tool existence."""
        registry = ToolRegistry()
        registry.register(Tool(
            name="exists",
            description="",
            parameters={},
            func=lambda: "",
        ))
        assert registry.has("exists") is True
        assert registry.has("missing") is False


class TestCreateRegistry:
    """Tests for the factory function that creates all tools."""

    def test_registry_has_all_tools(self):
        """create_registry should include all three RAG tools."""
        registry = create_registry()
        assert registry.has("search_documents")
        assert registry.has("get_document")
        assert registry.has("list_documents")
        assert len(registry._tools) == 3


class TestToolSearchDocuments:
    """Tests for the search_documents tool."""

    def test_search_returns_formatted_results(self, mock_retriever):
        """search_documents should return formatted context string."""
        # Create mock RetrievedDocuments
        mock_doc1 = MagicMock()
        mock_doc1.text = "RAG combines retrieval with generation."
        mock_doc1.metadata = {"source_path": "/about_rag.md", "file_name": "about_rag.md"}
        mock_doc1.score = 0.92

        mock_doc2 = MagicMock()
        mock_doc2.text = "Benefits include reduced hallucination."
        mock_doc2.metadata = {"source_path": "/about_rag.md", "file_name": "about_rag.md"}
        mock_doc2.score = 0.85

        mock_retriever.retrieve.return_value = [mock_doc1, mock_doc2]
        mock_retriever.format_context.return_value = (
            "[Document 1 — Source: about_rag.md (relevance: 0.92)]\n"
            "RAG combines retrieval with generation.\n\n"
            "---\n\n"
            "[Document 2 — Source: about_rag.md (relevance: 0.85)]\n"
            "Benefits include reduced hallucination."
        )

        result = tool_search_documents(query="What is RAG?")

        assert "Search results for" in result
        assert "Found 2 document" in result
        mock_retriever.retrieve.assert_called_once()

    def test_search_no_results(self, mock_retriever):
        """When no documents match, return helpful message."""
        mock_retriever.retrieve.return_value = []

        result = tool_search_documents(query="nonexistent topic")

        assert "No documents matched" in result
        assert "Suggestions" in result


class TestToolListDocuments:
    """Tests for the list_documents tool."""

    def test_list_empty(self, mock_vector_store):
        """When no documents indexed, return helpful message."""
        mock_vector_store.get_collection_stats.return_value = {
            "chunk_count": 0,
            "unique_sources": 0,
            "collection_name": "documents",
        }

        result = tool_list_documents()

        assert "No documents indexed yet" in result

    def test_list_with_documents(self, mock_vector_store):
        """When documents exist, list them."""
        mock_vector_store.get_collection_stats.return_value = {
            "chunk_count": 10,
            "unique_sources": 2,
            "collection_name": "documents",
        }
        # Mock the ChromaDB internal store access
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"source_path": "/a.md", "file_name": "a.md"},
                {"source_path": "/a.md", "file_name": "a.md"},
                {"source_path": "/b.md", "file_name": "b.md"},
            ]
        }
        mock_vector_store._store._collection = mock_collection

        result = tool_list_documents()

        assert "Knowledge Base Summary" in result
        assert "Total chunks: 10" in result
