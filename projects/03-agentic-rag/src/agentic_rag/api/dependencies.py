"""FastAPI dependency injection for shared components.

Uses lazy-initialized singletons — same pattern as rag-agent's dependencies.py.
Components are created once and reused across requests.
"""

from __future__ import annotations

from agentic_rag.agent import AgenticRAGAgent
from agentic_rag.tools import create_registry

# ── Lazy-initialized singletons ───────────────────────────────────

_agent: AgenticRAGAgent | None = None


def get_agent() -> AgenticRAGAgent:
    """Get or create the shared AgenticRAGAgent singleton.

    The agent owns the ToolRegistry which owns the Retriever
    and ChromaVectorStore — so this one function provides everything.
    """
    global _agent
    if _agent is None:
        registry = create_registry()
        _agent = AgenticRAGAgent(tools=registry)
    return _agent
