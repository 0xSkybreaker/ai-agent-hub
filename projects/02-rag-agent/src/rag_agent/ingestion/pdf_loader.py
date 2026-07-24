"""PDF document loader using pdfplumber with pypdf fallback."""

from __future__ import annotations

from pathlib import Path

from rag_agent.ingestion.base import BaseLoader, Document
from rag_agent.utils.logger import get_logger

logger = get_logger()


class PDFLoader(BaseLoader):
    """Loads PDF files, extracting text page by page.

    Uses pdfplumber as the primary engine (better text extraction and
    table awareness), falling back to pypdf if pdfplumber fails.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def load(self, source: str) -> list[Document]:
        path = Path(source).resolve()
        documents: list[Document] = []

        # Try pdfplumber first
        try:
            import pdfplumber

            with pdfplumber.open(str(path)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        documents.append(
                            Document(
                                content=text.strip(),
                                metadata={
                                    "source_path": str(path),
                                    "file_name": path.name,
                                    "file_type": "pdf",
                                    "page_number": i,
                                    "total_pages": len(pdf.pages),
                                },
                            )
                        )
            if documents:
                logger.debug(f"pdfplumber extracted {len(documents)} pages from {source}")
                return documents
        except Exception as e:
            logger.warning(f"pdfplumber failed for {source}: {e}, trying pypdf")

        # Fallback to pypdf
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            for i, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    documents.append(
                        Document(
                            content=text.strip(),
                            metadata={
                                "source_path": str(path),
                                "file_name": path.name,
                                "file_type": "pdf",
                                "page_number": i,
                                "total_pages": len(reader.pages),
                            },
                        )
                    )
            logger.debug(f"pypdf extracted {len(documents)} pages from {source}")
        except Exception as e:
            logger.error(f"Failed to load PDF {source}: {e}")
            raise

        # If still empty, load as single document
        if not documents:
            logger.warning(f"No text extracted from {source}, loading raw")
            documents.append(
                Document(
                    content=f"[No extractable text from {path.name}]",
                    metadata={
                        "source_path": str(path),
                        "file_name": path.name,
                        "file_type": "pdf",
                        "page_number": 1,
                    },
                )
            )

        return documents
