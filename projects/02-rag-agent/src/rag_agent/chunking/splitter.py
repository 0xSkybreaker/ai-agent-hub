"""Text chunking with configurable splitting strategy."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_agent.config import settings
from rag_agent.ingestion.base import Document
from rag_agent.utils.logger import get_logger

logger = get_logger()


@dataclass
class Chunk:
    """A chunk of text with preserved metadata, ready for embedding."""

    text: str
    metadata: dict


class DocumentSplitter:
    """Splits documents into overlapping chunks using recursive character splitting.

    Uses LangChain's RecursiveCharacterTextSplitter under the hood,
    which tries to split on natural boundaries (paragraphs, sentences)
    before falling back to character-level splits.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "。", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def split(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into chunks, preserving and enriching metadata.

        Args:
            documents: List of Documents from loaders.

        Returns:
            List of Chunks ready for embedding.
        """
        chunks: list[Chunk] = []

        for doc in documents:
            texts = self._splitter.split_text(doc.content)
            for i, text in enumerate(texts):
                chunk_meta = dict(doc.metadata)
                chunk_meta["chunk_index"] = i
                chunk_meta["chunk_count"] = len(texts)

                chunks.append(Chunk(text=text, metadata=chunk_meta))

        logger.debug(
            f"Split {len(documents)} documents into {len(chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def split_single(self, document: Document) -> list[Chunk]:
        """Split a single document into chunks."""
        return self.split([document])
