"""RAG generator: orchestrates retrieval → prompt → LLM → citations."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from rag_agent.config import settings
from rag_agent.generation.citations import (
    SourceCitation,
    extract_citations,
    format_citations_for_api,
    format_citations_for_display,
)
from rag_agent.generation.prompts import RAG_SYSTEM_PROMPT
from rag_agent.llm.nvidia_client import LLMClient
from rag_agent.llm.streaming import collect_stream
from rag_agent.retrieval.retriever import RetrievedDocument, Retriever
from rag_agent.utils.logger import get_logger

logger = get_logger()


@dataclass
class GenerationResult:
    """Result of a RAG generation."""

    answer: str
    sources: list[SourceCitation] = field(default_factory=list)
    retrieved_count: int = 0
    model: str = ""


class Generator:
    """Orchestrates the full RAG generation pipeline.

    Pipeline: retrieve → build prompt → generate → extract citations
    """

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
    ) -> None:
        self._retriever = retriever
        self._llm_client = llm_client
        self.model_name = llm_client.model_name

    def generate(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int | None = None,
    ) -> GenerationResult:
        """Run the full RAG pipeline and return an answer.

        Args:
            question: The user's question.
            history: Optional conversation history.
            top_k: Number of documents to retrieve.

        Returns:
            GenerationResult with answer and source citations.
        """
        # Step 1: Retrieve relevant documents
        documents = self._retriever.retrieve(query=question, top_k=top_k)

        if not documents:
            logger.warning("No relevant documents found for query")
            return GenerationResult(
                answer="I don't have enough information in the provided documents to answer this question.",
                sources=[],
                retrieved_count=0,
                model=self.model_name,
            )

        # Step 2: Format context
        context = self._retriever.format_context(documents)

        # Step 3: Format chat history
        history_str = self._format_history(history)

        # Step 4: Build prompt
        prompt = RAG_SYSTEM_PROMPT.format(
            context=context,
            chat_history=history_str,
            question=question,
        )

        # Step 5: Generate response
        messages = [{"role": "user", "content": prompt}]
        answer = self._llm_client.generate(messages)

        # Step 6: Extract citations
        citations = extract_citations(documents)

        logger.info(
            f"Generated answer ({len(answer)} chars) from "
            f"{len(documents)} retrieved documents"
        )

        return GenerationResult(
            answer=answer,
            sources=citations,
            retrieved_count=len(documents),
            model=self.model_name,
        )

    def generate_stream(
        self,
        question: str,
        history: list[dict] | None = None,
        top_k: int | None = None,
    ) -> Iterator[str]:
        """Run the RAG pipeline with streaming output.

        Yields tokens of the answer, followed by formatted citations.

        Args:
            question: The user's question.
            history: Optional conversation history.
            top_k: Number of documents to retrieve.

        Yields:
            Text tokens of the answer.
        """
        # Step 1: Retrieve
        documents = self._retriever.retrieve(query=question, top_k=top_k)

        if not documents:
            yield "I don't have enough information in the provided documents to answer this question."
            return

        # Step 2 & 3: Format context and history
        context = self._retriever.format_context(documents)
        history_str = self._format_history(history)

        # Step 4: Build prompt
        prompt = RAG_SYSTEM_PROMPT.format(
            context=context,
            chat_history=history_str,
            question=question,
        )

        # Step 5: Stream
        messages = [{"role": "user", "content": prompt}]
        for token in self._llm_client.stream(messages):
            yield token

    def query(
        self,
        question: str,
        history: list[dict] | None = None,
        stream: bool = False,
        top_k: int | None = None,
    ) -> GenerationResult | Iterator[str]:
        """Convenience method dispatching to generate or generate_stream.

        Args:
            question: The user's question.
            history: Optional conversation history.
            stream: If True, stream tokens.
            top_k: Number of documents to retrieve.

        Returns:
            GenerationResult or token iterator.
        """
        if stream:
            return self.generate_stream(
                question=question, history=history, top_k=top_k
            )
        return self.generate(
            question=question, history=history, top_k=top_k
        )

    def _format_history(self, history: list[dict] | None) -> str:
        """Format conversation history for the prompt.

        Args:
            history: List of message dicts.

        Returns:
            Formatted history string.
        """
        if not history:
            return "No previous conversation."

        lines: list[str] = []
        for msg in history[-settings.max_conversation_turns * 2 :]:  # Keep last N exchanges
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")

        return "\n".join(lines)
