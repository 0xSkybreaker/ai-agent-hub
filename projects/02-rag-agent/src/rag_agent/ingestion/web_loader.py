"""HTML / Web page document loader."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from rag_agent.ingestion.base import BaseLoader, Document
from rag_agent.utils.logger import get_logger

logger = get_logger()


class WebLoader(BaseLoader):
    """Loads HTML files (local) or web pages (URL).

    Uses BeautifulSoup to extract text content, stripping scripts,
    styles, and navigation elements.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]

    def load(self, source: str) -> list[Document]:
        from bs4 import BeautifulSoup

        # Determine if source is a URL or local file
        parsed = urlparse(source)
        is_url = bool(parsed.scheme and parsed.netloc)

        if is_url:
            import requests

            logger.debug(f"Fetching URL: {source}")
            try:
                resp = requests.get(source, timeout=30, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; RAG-Agent/1.0)"
                })
                resp.raise_for_status()
                html_content = resp.text
                file_name = parsed.netloc + parsed.path.rstrip("/") or "index.html"
                source_path = source
            except Exception as e:
                logger.error(f"Failed to fetch URL {source}: {e}")
                raise
        else:
            path = Path(source).resolve()
            html_content = path.read_text(encoding="utf-8")
            file_name = path.name
            source_path = str(path)

        soup = BeautifulSoup(html_content, "lxml")

        # Extract title
        title = file_name
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Remove non-content elements
        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Get text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up: remove excessive blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        content = "\n".join(lines)

        if not content:
            logger.warning(f"No text extracted from {source}")
            content = f"[No extractable text from {file_name}]"

        logger.debug(f"Loaded {len(lines)} lines from {file_name}")

        return [
            Document(
                content=content,
                metadata={
                    "source_path": source_path,
                    "file_name": file_name,
                    "file_type": "html",
                    "title": title,
                },
            )
        ]
