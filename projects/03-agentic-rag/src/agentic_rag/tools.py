"""RAG-aware tools for the Agentic RAG agent.

Each tool wraps a capability from rag-agent (retrieve, inspect documents, list
available knowledge) and exposes it as a function-callable tool the agent can use.

The Tool dataclass and ToolRegistry are adapted from 01-mini-agent's pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agentic_rag.config import settings


# ── Tool definition ──────────────────────────────────────────────


@dataclass
class Tool:
    """A tool that the Agent can call.

    Follows the exact same interface as 01-mini-agent's Tool.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema properties
    func: Callable[..., Any]

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI function-calling schema dict."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys()),
                },
            },
        }


# ── Registry ─────────────────────────────────────────────────────


class ToolRegistry:
    """Holds all available tools. Register → get schemas → execute by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas for all registered tools."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def has(self, name: str) -> bool:
        return name in self._tools

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name with the given arguments.

        Returns the result as a string (suitable for feeding back to the LLM).
        """
        if name not in self._tools:
            return f"Error: tool '{name}' not found. Available: {list(self._tools.keys())}"

        try:
            result = self._tools[name].func(**arguments)
            return str(result)
        except Exception as e:
            return f"Tool error: {type(e).__name__}: {e}"

    def list_tools(self) -> str:
        """Return a human-readable summary of all tools."""
        lines = []
        for tool in self._tools.values():
            params = ", ".join(tool.parameters.keys()) if tool.parameters else "(none)"
            lines.append(f"  • {tool.name}({params}) — {tool.description}")
        return "\n".join(lines)


# ── Tool implementations ─────────────────────────────────────────
#
# Each tool function takes keyword arguments matching its schema
# and returns a string. The string is fed directly back to the LLM
# as the tool result, so it must be human-readable and informative.


def _get_retriever():
    """Lazy-import and create the Retriever from rag-agent."""
    from rag_agent.embeddings.nvidia_embeddings import EmbeddingClient
    from rag_agent.retrieval.retriever import Retriever
    from rag_agent.vector_store.chroma_store import ChromaVectorStore

    embedding_client = EmbeddingClient()
    vector_store = ChromaVectorStore(embedding_client)
    return Retriever(vector_store, embedding_client)


# Module-level singleton — created on first tool call
_retriever = None


def _ensure_retriever():
    global _retriever
    if _retriever is None:
        _retriever = _get_retriever()
    return _retriever


def _get_vector_store():
    """Lazy-import and create the ChromaVectorStore from rag-agent."""
    from rag_agent.embeddings.nvidia_embeddings import EmbeddingClient
    from rag_agent.vector_store.chroma_store import ChromaVectorStore

    return ChromaVectorStore(EmbeddingClient())


_vector_store = None


def _ensure_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = _get_vector_store()
    return _vector_store


# ── Tool 1: search_documents ─────────────────────────────────────


def tool_search_documents(query: str, top_k: int = 5) -> str:
    """Search the document knowledge base for relevant information.

    Use this as your PRIMARY tool for finding information. It performs
    semantic search — it finds documents whose MEANING matches your query,
    not just keyword matching.

    Tips for good searches:
    - Be specific: "benefits of RAG over fine-tuning" not just "RAG"
    - Try different phrasings if first results aren't relevant
    - Use short, focused queries rather than long questions
    - For multi-part questions, search each part separately

    Returns formatted results with [Document N] markers and relevance scores.
    """
    retriever = _ensure_retriever()
    k = top_k or settings.top_k

    try:
        documents = retriever.retrieve(query=query, top_k=k)

        if not documents:
            return (
                f"No documents matched the query '{query}'.\n\n"
                "Suggestions:\n"
                "- Try different keywords or a broader query\n"
                "- Use list_documents to see what's available\n"
                "- Check if documents have been indexed yet"
            )

        context = retriever.format_context(documents)

        header = (
            f"Search results for: '{query}'\n"
            f"Found {len(documents)} document(s):\n\n"
        )
        return header + context

    except Exception as e:
        return f"Search failed: {type(e).__name__}: {e}"


# ── Tool 2: get_document ─────────────────────────────────────────


def tool_get_document(source_path: str) -> str:
    """Get the full content of a specific document by its source path.

    Use this to deep-dive into a document that looks promising from
    search_documents results. Returns all chunks from the document.

    The source_path can be the full path shown in search results,
    or just the filename (e.g., "about_rag.md").
    """
    store = _ensure_vector_store()

    try:
        # Try exact match first, then try partial match
        results = store.search(query=source_path, top_k=100, filter_metadata={})

        # Filter results that match the source_path
        chunks = [
            (text, meta, score)
            for text, meta, score in results
            if source_path in meta.get("source_path", "")
            or source_path == meta.get("file_name", "")
        ]

        if not chunks:
            # Try searching by filename only
            filename = source_path.split("/")[-1].split("\\")[-1]
            chunks = [
                (text, meta, score)
                for text, meta, score in results
                if filename == meta.get("file_name", "")
            ]

        if not chunks:
            return (
                f"Document '{source_path}' not found in the index.\n\n"
                f"Use list_documents to see all indexed documents."
            )

        # Build a readable view of the document
        lines = [f"Document: {source_path}"]
        lines.append(f"Chunks: {len(chunks)}")
        lines.append(f"(relevance scores range: {min(c[2] for c in chunks):.2f} — {max(c[2] for c in chunks):.2f})")
        lines.append("=" * 60)
        lines.append("")

        for i, (text, meta, score) in enumerate(chunks, 1):
            page = meta.get("page_number", "")
            page_info = f" | Page {page}" if page else ""
            lines.append(f"--- Chunk {i}{page_info} (score: {score:.3f}) ---")
            lines.append(text)
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"Error getting document: {type(e).__name__}: {e}"


# ── Tool 3: list_documents ───────────────────────────────────────


def tool_list_documents() -> str:
    """List all documents currently indexed in the knowledge base.

    Use this BEFORE searching to understand what information is available.
    Shows each document's name, source path, and chunk count.

    This helps you:
    - Know what topics are covered
    - Formulate better search queries
    - Decide if you need more documents indexed
    """
    store = _ensure_vector_store()

    try:
        stats = store.get_collection_stats()
        hashes = store.get_source_hashes()

        if stats["chunk_count"] == 0:
            return (
                "No documents indexed yet.\n\n"
                "Use rag-agent to index documents first:\n"
                "  python -m rag_agent index <path-to-documents>"
            )

        lines = [
            f"Knowledge Base Summary",
            f"{'=' * 60}",
            f"Collection: {stats['collection_name']}",
            f"Total chunks: {stats['chunk_count']}",
            f"Unique sources: {stats['unique_sources']}",
            f"",
            f"Indexed Documents:",
        ]

        # Get source details
        collection = store._store._collection
        all_results = collection.get()

        # Group by source_path
        sources: dict[str, dict] = {}
        if all_results and all_results["metadatas"]:
            for meta in all_results["metadatas"]:
                sp = meta.get("source_path", "unknown")
                fn = meta.get("file_name", "unknown")
                if sp not in sources:
                    sources[sp] = {"file_name": fn, "chunks": 0}
                sources[sp]["chunks"] += 1

        for i, (source_path, info) in enumerate(sorted(sources.items()), 1):
            lines.append(f"  [{i}] {info['file_name']}")
            lines.append(f"      Path: {source_path}")
            lines.append(f"      Chunks: {info['chunks']}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing documents: {type(e).__name__}: {e}"


# ── Build the registry ───────────────────────────────────────────


def create_registry() -> ToolRegistry:
    """Create and return a ToolRegistry with all RAG tools."""
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="search_documents",
            description=(
                "Search the document knowledge base for information relevant to a query. "
                "This is your primary tool for finding answers. "
                "Returns formatted document chunks with [Document N] markers and relevance scores. "
                "If results are poor or irrelevant, try reformulating your query with different keywords. "
                "For complex questions, search each sub-question separately."
            ),
            parameters={
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Be specific and use keywords that would appear "
                        "in relevant documents. Example: 'RAG benefits over fine-tuning' "
                        "rather than just 'RAG'."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 20)",
                },
            },
            func=tool_search_documents,
        )
    )

    registry.register(
        Tool(
            name="get_document",
            description=(
                "Get the full content of a specific document by its source path or filename. "
                "Use this to deep-dive into a document that looks promising from search results. "
                "Returns all chunks from that document concatenated."
            ),
            parameters={
                "source_path": {
                    "type": "string",
                    "description": (
                        "The source path or filename of the document. "
                        "Use the path shown in search_documents results or list_documents output. "
                        "Example: 'about_rag.md' or 'data/uploads/report.pdf'"
                    ),
                },
            },
            func=tool_get_document,
        )
    )

    registry.register(
        Tool(
            name="list_documents",
            description=(
                "List all documents currently available in the knowledge base. "
                "Use this BEFORE searching to understand what topics are covered. "
                "Shows document names, paths, and chunk counts."
            ),
            parameters={},
            func=tool_list_documents,
        )
    )

    return registry
