"""FastAPI dependency injection for shared components."""

from __future__ import annotations

from rag_agent.embeddings.nvidia_embeddings import EmbeddingClient
from rag_agent.generation.generator import Generator
from rag_agent.llm.nvidia_client import LLMClient
from rag_agent.memory.conversation import ConversationMemory
from rag_agent.retrieval.retriever import Retriever
from rag_agent.vector_store.chroma_store import ChromaVectorStore
from rag_agent.vector_store.indexer import IndexingPipeline

# ── Lazy-initialized singletons ───────────────────────────────────

_embedding_client: EmbeddingClient | None = None
_llm_client: LLMClient | None = None
_vector_store: ChromaVectorStore | None = None
_retriever: Retriever | None = None
_generator: Generator | None = None
_indexer: IndexingPipeline | None = None
_memory: ConversationMemory | None = None


def get_embedding_client() -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def get_vector_store() -> ChromaVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore(get_embedding_client())
    return _vector_store


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(get_vector_store(), get_embedding_client())
    return _retriever


def get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator(get_retriever(), get_llm_client())
    return _generator


def get_indexer() -> IndexingPipeline:
    global _indexer
    if _indexer is None:
        _indexer = IndexingPipeline(get_vector_store(), get_embedding_client())
    return _indexer


def get_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory
