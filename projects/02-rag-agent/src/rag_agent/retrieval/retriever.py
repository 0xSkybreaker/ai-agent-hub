"""Semantic retrieval over the vector store."""

from __future__ import annotations

from dataclasses import dataclass

from rag_agent.config import settings
from rag_agent.embeddings.nvidia_embeddings import EmbeddingClient
from rag_agent.utils.logger import get_logger
from rag_agent.vector_store.chroma_store import ChromaVectorStore

logger = get_logger()


@dataclass
class RetrievedDocument:
    """A document retrieved from the vector store."""

    text: str
    metadata: dict
    score: float


class Retriever:
    """Retrieves relevant document chunks for a user query."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_client: EmbeddingClient,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_client = embedding_client

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[RetrievedDocument]:
        """Retrieve top-k relevant document chunks for a query.

        Args:
            query: The user's question.
            top_k: Number of results to return (default: settings.top_k).
            threshold: Minimum relevance score (default: settings.similarity_threshold).

        Returns:
            List of RetrievedDocument sorted by relevance.
        """
        k = top_k or settings.top_k
        thresh = threshold if threshold is not None else settings.similarity_threshold

        results = self._vector_store.search(query=query, top_k=k)

        docs = []
        for text, metadata, score in results:
            if score >= thresh:
                docs.append(
                    RetrievedDocument(text=text, metadata=metadata, score=score)
                )

        logger.debug(
            f"Retrieved {len(docs)}/{len(results)} documents "
            f"(threshold={thresh}, top score={docs[0].score:.4f})" if docs else "No documents above threshold"
        )
        return docs

    def format_context(self, documents: list[RetrievedDocument]) -> str:
        """Format retrieved documents as a context string for the LLM.

        Args:
            documents: Retrieved documents to format.

        Returns:
            Formatted context string with source markers.
        """
        if not documents:
            return "No relevant documents found."

        parts: list[str] = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("file_name", "unknown")
            page = doc.metadata.get("page_number", "")
            page_info = f", Page {page}" if page else ""

            parts.append(
                f"[Document {i} — Source: {source}{page_info} "
                f"(relevance: {doc.score:.2f})]\n{doc.text}"
            )

        return "\n\n---\n\n".join(parts)
