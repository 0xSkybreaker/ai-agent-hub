"""Agentic RAG — a ReAct Agent that autonomously drives RAG retrieval.

Unlike a basic RAG pipeline (query → retrieve → generate), this agent
DECIDES what to search for, EVALUATES whether results are sufficient,
REFORMULATES queries when needed, and SYNTHESIZES from multiple searches.

It reuses the rag-agent's infrastructure (vector store, retriever, embeddings)
but replaces the fixed pipeline with a ReAct decision loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Import rag-agent from sibling project ─────────────────────────
# This allows agentic-rag to reuse ChromaVectorStore, Retriever,
# EmbeddingClient, and ConversationMemory without copying code.
_RAG_AGENT_SRC = Path(__file__).resolve().parents[2] / "02-rag-agent" / "src"
if str(_RAG_AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(_RAG_AGENT_SRC))
