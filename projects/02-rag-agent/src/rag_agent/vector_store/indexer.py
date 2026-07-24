"""End-to-end indexing pipeline: load → chunk → embed → store."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rag_agent.chunking.metadata import enrich_chunk_metadata
from rag_agent.chunking.splitter import Chunk, DocumentSplitter
from rag_agent.config import settings
from rag_agent.embeddings.nvidia_embeddings import EmbeddingClient
from rag_agent.ingestion.base import Document, LoaderRegistry
from rag_agent.ingestion.docx_loader import DocxLoader
from rag_agent.ingestion.pdf_loader import PDFLoader
from rag_agent.ingestion.text_loader import MarkdownLoader, TextLoader
from rag_agent.ingestion.web_loader import WebLoader
from rag_agent.utils.hashing import compute_file_hash
from rag_agent.utils.logger import get_logger
from rag_agent.vector_store.chroma_store import ChromaVectorStore

logger = get_logger()


@dataclass
class IndexResult:
    """Result of an indexing operation."""

    source: str
    chunks_created: int
    status: str  # "indexed", "updated", "unchanged", "error"
    message: str = ""
    error: str | None = None


def _create_loader_registry() -> LoaderRegistry:
    """Create and populate the loader registry with all supported loaders."""
    registry = LoaderRegistry()
    registry.register(PDFLoader)
    registry.register(DocxLoader)
    registry.register(TextLoader)
    registry.register(MarkdownLoader)
    registry.register(WebLoader)
    return registry


class IndexingPipeline:
    """Orchestrates the full document indexing pipeline.

    Pipeline: Load → Chunk → Enrich metadata → Store in ChromaDB
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_client: EmbeddingClient,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_client = embedding_client
        self._loader_registry = _create_loader_registry()
        self._splitter = DocumentSplitter()

    def index_file(self, file_path: str, force: bool = False) -> IndexResult:
        """Index a single file into the vector store.

        Args:
            file_path: Path to the file to index.
            force: If True, re-index even if file hasn't changed.

        Returns:
            IndexResult with details about the operation.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            return IndexResult(
                source=str(path),
                chunks_created=0,
                status="error",
                error=f"File not found: {path}",
            )

        if not self._loader_registry.is_supported(str(path)):
            return IndexResult(
                source=str(path),
                chunks_created=0,
                status="error",
                error=f"Unsupported file type: {path.suffix}",
            )

        # Check for changes (skip if unchanged, unless --force)
        if not force:
            current_hash = compute_file_hash(path)
            source_hashes = self._vector_store.get_source_hashes()
            stored_hash = source_hashes.get(str(path), "")
            if stored_hash and stored_hash == current_hash:
                return IndexResult(
                    source=str(path),
                    chunks_created=0,
                    status="unchanged",
                    message="File unchanged since last indexing",
                )

        try:
            # Delete old chunks if re-indexing
            self._vector_store.delete_by_source(str(path))

            # Load
            loader = self._loader_registry.get_loader(str(path))
            documents = loader.load(str(path))
            logger.info(f"Loaded {len(documents)} document(s) from {path.name}")

            # Chunk
            chunks = self._splitter.split(documents)
            logger.info(f"Split into {len(chunks)} chunks")

            # Enrich metadata and store
            texts: list[str] = []
            metadatas: list[dict] = []
            for i, chunk in enumerate(chunks):
                meta = enrich_chunk_metadata(chunk.metadata, chunk.text, i)
                texts.append(chunk.text)
                metadatas.append(meta)

            # ChromaDB handles embedding internally via its embedding function
            ids = self._vector_store.add_documents(texts, metadatas)

            logger.info(f"Indexed {path.name}: {len(ids)} chunks stored")
            return IndexResult(
                source=str(path),
                chunks_created=len(ids),
                status="indexed",
                message=f"Successfully indexed {len(ids)} chunks",
            )

        except Exception as e:
            logger.error(f"Failed to index {path.name}: {e}")
            return IndexResult(
                source=str(path),
                chunks_created=0,
                status="error",
                error=str(e),
            )

    def index_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        force: bool = False,
    ) -> list[IndexResult]:
        """Index all supported files in a directory.

        Args:
            dir_path: Path to the directory.
            recursive: If True, scan subdirectories.
            force: If True, re-index all files.

        Returns:
            List of IndexResult for each file processed.
        """
        path = Path(dir_path).resolve()
        if not path.is_dir():
            return [
                IndexResult(
                    source=str(path),
                    chunks_created=0,
                    status="error",
                    error="Not a directory",
                )
            ]

        pattern = "**/*" if recursive else "*"
        results: list[IndexResult] = []

        for file_path in sorted(path.glob(pattern)):
            if file_path.is_file() and self._loader_registry.is_supported(str(file_path)):
                result = self.index_file(str(file_path), force=force)
                results.append(result)

        total_chunks = sum(r.chunks_created for r in results)
        logger.info(
            f"Directory indexing complete: {len(results)} files, "
            f"{total_chunks} chunks total"
        )
        return results

    def list_supported_extensions(self) -> list[str]:
        """Return all supported file extensions."""
        return self._loader_registry.list_extensions()

    def get_stats(self) -> dict:
        """Return current indexing statistics."""
        return self._vector_store.get_collection_stats()

    def remove_document(self, source_path: str) -> int:
        """Remove all chunks for a given source document.

        Args:
            source_path: The source file path.

        Returns:
            Number of chunks deleted.
        """
        return self._vector_store.delete_by_source(source_path)

    def clear_all(self) -> None:
        """Remove all documents from the index."""
        self._vector_store.clear_collection()
