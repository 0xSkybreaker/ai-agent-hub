"""Plain text and Markdown document loaders."""

from __future__ import annotations

from pathlib import Path

from rag_agent.ingestion.base import BaseLoader, Document
from rag_agent.utils.logger import get_logger

logger = get_logger()


class TextLoader(BaseLoader):
    """Loads plain text (.txt) files with encoding detection."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt", ".text", ".csv"]

    def load(self, source: str) -> list[Document]:
        path = Path(source).resolve()

        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                content = path.read_text(encoding=encoding)
                logger.debug(f"Loaded {source} with {encoding} encoding")
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            content = path.read_text(encoding="utf-8", errors="replace")
            logger.warning(f"Loaded {source} with replacement characters")

        return [
            Document(
                content=content,
                metadata={
                    "source_path": str(path),
                    "file_name": path.name,
                    "file_type": "txt",
                },
            )
        ]


class MarkdownLoader(BaseLoader):
    """Loads Markdown (.md) files, preserving heading structure in metadata."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown", ".mdx"]

    def load(self, source: str) -> list[Document]:
        path = Path(source).resolve()
        content = path.read_text(encoding="utf-8")

        # Extract title from first H1 heading
        title = path.stem
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                title = stripped[2:].strip()
                break

        return [
            Document(
                content=content,
                metadata={
                    "source_path": str(path),
                    "file_name": path.name,
                    "file_type": "md",
                    "title": title,
                },
            )
        ]
