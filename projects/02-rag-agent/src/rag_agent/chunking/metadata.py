"""Metadata enrichment for document chunks."""

from __future__ import annotations

from datetime import datetime, timezone

from rag_agent.utils.hashing import compute_text_hash


def enrich_chunk_metadata(
    metadata: dict,
    chunk_text: str,
    chunk_index: int,
) -> dict:
    """Add computed metadata fields to a chunk's metadata.

    Args:
        metadata: Original document-level metadata.
        chunk_text: The chunk's text content.
        chunk_index: The chunk's position within the document.

    Returns:
        Enriched metadata dictionary.
    """
    enriched = dict(metadata)
    enriched.update({
        "content_hash": compute_text_hash(chunk_text),
        "chunk_index": chunk_index,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "char_count": len(chunk_text),
    })
    return enriched
