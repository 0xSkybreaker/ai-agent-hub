"""Conversation memory with sliding-window buffer."""

from __future__ import annotations

import uuid

from rag_agent.config import settings
from rag_agent.utils.logger import get_logger

logger = get_logger()


class ConversationMemory:
    """Manages conversation history per session using a sliding window.

    Each session maintains its own list of messages (user/assistant pairs).
    Old messages are evicted when the max history turns limit is exceeded.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict]] = {}
        self._max_turns = settings.max_conversation_turns

    def create_session(self) -> str:
        """Create a new conversation session.

        Returns:
            The new session ID.
        """
        session_id = str(uuid.uuid4())[:8]
        self._sessions[session_id] = []
        logger.debug(f"Created session: {session_id}")
        return session_id

    def get_history(self, session_id: str) -> list[dict]:
        """Get the message history for a session.

        Args:
            session_id: The session identifier.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        return self._sessions.get(session_id, [])

    def add_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Add a Q&A exchange to the session history.

        Args:
            session_id: The session identifier.
            user_message: The user's question.
            assistant_message: The assistant's answer.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        history = self._sessions[session_id]
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_message})

        # Enforce sliding window
        max_messages = self._max_turns * 2  # user + assistant per turn
        if len(history) > max_messages:
            removed = history[: len(history) - max_messages]
            self._sessions[session_id] = history[-max_messages:]
            logger.debug(f"Evicted {len(removed)} messages from session {session_id}")

    def clear_session(self, session_id: str) -> None:
        """Delete a session's history.

        Args:
            session_id: The session identifier.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Cleared session: {session_id}")

    def list_sessions(self) -> list[str]:
        """Return all active session IDs."""
        return list(self._sessions.keys())

    def session_count(self) -> int:
        """Return the number of active sessions."""
        return len(self._sessions)
