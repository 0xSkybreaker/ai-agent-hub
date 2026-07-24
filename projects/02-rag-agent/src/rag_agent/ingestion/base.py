"""Document abstractions and loader registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Document:
    """Universal document representation used across the ingestion pipeline."""

    content: str
    metadata: dict = field(default_factory=dict)
    # Common metadata keys:
    #   source_path  — absolute path to source file
    #   file_name    — original filename
    #   file_type    — pdf, docx, txt, md, html
    #   page_number  — page number (PDF only)
    #   content_hash — SHA-256 hash of the chunk content
    #   chunk_index  — position within the document
    #   ingested_at  — ISO timestamp of ingestion


class BaseLoader(ABC):
    """Abstract base for all document loaders."""

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """Load one or more Documents from a source path.

        Args:
            source: File path or URL to load.

        Returns:
            List of Document objects (one per page for multi-page formats).
        """
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """File extensions this loader handles (e.g., ['.pdf'])."""
        ...


class LoaderRegistry:
    """Maps file extensions to loader classes for auto-dispatch."""

    def __init__(self) -> None:
        self._loaders: dict[str, type[BaseLoader]] = {}

    def register(self, loader_cls: type[BaseLoader]) -> None:
        """Register a loader class for all its supported extensions."""
        loader = loader_cls()
        for ext in loader.supported_extensions:
            self._loaders[ext.lower()] = loader_cls

    def get_loader(self, source: str) -> BaseLoader:
        """Get the appropriate loader for a source file.

        Args:
            source: File path or URL.

        Returns:
            A BaseLoader instance.

        Raises:
            ValueError: If no loader is registered for the file extension.
        """
        from pathlib import Path

        ext = Path(source).suffix.lower()
        if ext not in self._loaders:
            supported = ", ".join(sorted(self._loaders.keys()))
            raise ValueError(
                f"No loader for extension '{ext}'. Supported: {supported}"
            )
        return self._loaders[ext]()

    def list_extensions(self) -> list[str]:
        """Return all supported file extensions."""
        return sorted(self._loaders.keys())

    def is_supported(self, source: str) -> bool:
        """Check if a source file's extension is supported."""
        from pathlib import Path

        return Path(source).suffix.lower() in self._loaders
