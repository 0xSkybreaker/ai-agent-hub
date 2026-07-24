"""Source citation extraction and formatting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SourceCitation:
    """A formatted source citation."""

    file_name: str
    source_path: str
    page_number: int | None
    chunk_index: int | None
    excerpt: str  # First ~150 chars of the chunk
    relevance_score: float


def extract_citations(
    documents: list,  # list[RetrievedDocument]
) -> list[SourceCitation]:
    """Convert retrieved documents into formatted citations.

    Args:
        documents: RetrievedDocument objects from the retriever.

    Returns:
        List of SourceCitation objects.
    """
    citations: list[SourceCitation] = []
    seen: set[str] = set()

    for doc in documents:
        source_path = doc.metadata.get("source_path", "unknown")
        file_name = doc.metadata.get("file_name", "unknown")

        # Deduplicate by source_path (unique sources only)
        if source_path in seen:
            continue
        seen.add(source_path)

        excerpt = doc.text[:200].replace("\n", " ").strip()
        if len(doc.text) > 200:
            excerpt += "..."

        citations.append(
            SourceCitation(
                file_name=file_name,
                source_path=source_path,
                page_number=doc.metadata.get("page_number"),
                chunk_index=doc.metadata.get("chunk_index"),
                excerpt=excerpt,
                relevance_score=doc.score,
            )
        )

    return citations


def format_citations_for_display(citations: list[SourceCitation]) -> str:
    """Format citations as a human-readable string for terminal/display.

    Args:
        citations: List of SourceCitation objects.

    Returns:
        Formatted multi-line string.
    """
    if not citations:
        return ""

    lines = ["\n---", "Sources:"]
    for i, c in enumerate(citations, start=1):
        parts = [f"  [{i}] {c.file_name}"]
        if c.page_number:
            parts.append(f" (Page {c.page_number})")
        parts.append(f" - relevance: {c.relevance_score:.2f}")
        lines.append("".join(parts))

    lines.append("---")
    return "\n".join(lines)


def format_citations_for_api(citations: list[SourceCitation]) -> list[dict]:
    """Format citations as JSON-serializable dicts for API responses.

    Args:
        citations: List of SourceCitation objects.

    Returns:
        List of dicts ready for JSON serialization.
    """
    return [
        {
            "file_name": c.file_name,
            "source_path": c.source_path,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "excerpt": c.excerpt,
            "relevance_score": round(c.relevance_score, 4),
        }
        for c in citations
    ]
