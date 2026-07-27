"""AgenticRAGAgent — a ReAct Agent driving RAG retrieval autonomously.

This is THE file to understand. It combines the ReAct loop from 01-mini-agent
with the RAG infrastructure from 02-rag-agent.

The agent DOES NOT follow a fixed pipeline. Instead, it:
  1. Thinks about what to search for
  2. Searches using tools
  3. Evaluates results
  4. Reformulates and re-searches if needed
  5. Synthesizes from all gathered context
  6. Cites sources explicitly

This is what makes it "agentic" — the LLM is in control of the retrieval
strategy, not a pre-programmed pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

from agentic_rag.config import settings
from agentic_rag.llm import get_client
from agentic_rag.prompts import AGENTIC_RAG_SYSTEM_PROMPT
from agentic_rag.tools import ToolRegistry, create_registry


# ── Result dataclasses ─────────────────────────────────────────────


@dataclass
class Step:
    """A single step in the Agent's reasoning trace.

    Each step represents one tool call or the final answer.
    """

    step_number: int
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    tool_result: str = ""
    final_answer: str = ""

    @property
    def is_final(self) -> bool:
        return bool(self.final_answer)

    @property
    def used_tool(self) -> bool:
        return bool(self.tool_name)


@dataclass
class AgentResult:
    """The result of running the Agentic RAG Agent."""

    answer: str
    steps: list[Step]
    total_steps: int
    model: str
    sources: list[dict] = field(default_factory=list)


# ── Agent ──────────────────────────────────────────────────────────


class AgenticRAGAgent:
    """A ReAct Agent that autonomously drives RAG retrieval.

    Unlike the one-shot RAG pipeline (retrieve → generate), this agent:
    - Decides WHAT to search for
    - EVALUATES whether retrieved results are sufficient
    - REFORMULATES queries when initial results are poor
    - Breaks complex questions into sub-searches
    - SYNTHESIZES from multiple search rounds
    - CITES sources and VERIFIES claims

    Usage:
        agent = AgenticRAGAgent()
        result = agent.run("What are the key benefits of RAG?")
        print(result.answer)
        for step in result.steps:
            print(f"  Step {step.step_number}: {step.tool_name}")
    """

    def __init__(
        self,
        tools: ToolRegistry | None = None,
        client: "OpenAI | None" = None,
    ) -> None:
        """Initialize the agent.

        Args:
            tools: ToolRegistry with RAG tools. If None, uses create_registry()
                   which includes search_documents, get_document, list_documents.
            client: Optional pre-configured OpenAI client. If None, uses get_client().
                   Useful for testing with mock clients.
        """
        self.tools = tools or create_registry()
        self.model = settings.chat_model
        self._client = client if client is not None else get_client()

    def run(
        self,
        task: str,
        *,
        max_steps: int | None = None,
        verbose: bool = False,
    ) -> AgentResult:
        """Run the Agent on a question.

        The agent will autonomously search, evaluate, reformulate,
        and synthesize until it has an answer (or reaches max_steps).

        Args:
            task: The user's question.
            max_steps: Max tool-calling iterations (default: settings.max_steps).
            verbose: If True, print each step as it happens.

        Returns:
            AgentResult with the final answer, step trace, and sources.
        """
        max_steps = max_steps if max_steps is not None else settings.max_steps
        steps: list[Step] = []
        tool_schemas = self.tools.get_schemas()

        messages: list[dict] = [
            {"role": "system", "content": AGENTIC_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for i in range(max_steps):
            step = Step(step_number=i + 1)
            steps.append(step)

            # ── Call LLM with tools ────────────────────────
            try:
                raw = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                    tools=tool_schemas,
                    tool_choice="auto",
                )
            except Exception as e:
                step.final_answer = f"LLM call failed: {e}"
                if verbose:
                    print(f"  ❌ Step {step.step_number}: {e}")
                break

            choice = raw.choices[0]

            # ── Case 1: Final answer (no tool calls) ──────
            if choice.finish_reason == "stop" and not choice.message.tool_calls:
                step.final_answer = choice.message.content or ""
                if verbose:
                    print(f"  ✅ Step {step.step_number}: Final answer")
                break

            # ── Case 2: Tool call ─────────────────────────
            if choice.message.tool_calls:
                tool_call = choice.message.tool_calls[0]
                step.tool_name = tool_call.function.name
                step.tool_args = json.loads(tool_call.function.arguments)

                if verbose:
                    args_short = str(step.tool_args)
                    if len(args_short) > 80:
                        args_short = args_short[:80] + "..."
                    print(f"  🔧 Step {step.step_number}: {step.tool_name}({args_short})")

                # Execute the tool
                step.tool_result = self.tools.execute(step.tool_name, step.tool_args)

                if verbose:
                    preview = step.tool_result[:200] + "..." if len(step.tool_result) > 200 else step.tool_result
                    preview = preview.replace("\n", "\n     ")
                    print(f"     → {preview}")

                # Append tool call + result to conversation
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": step.tool_result,
                })
                continue

            # ── Case 3: Unexpected ────────────────────────
            step.final_answer = f"Unexpected finish_reason: {choice.finish_reason}"
            break

        else:
            # Exhausted max steps — agent couldn't find an answer
            final = (
                f"I was unable to answer this question within {max_steps} steps. "
                f"Here is what I found so far:\n\n"
            )
            # Summarize what was retrieved
            for s in steps:
                if s.used_tool:
                    final += f"- Used {s.tool_name} at step {s.step_number}\n"
            final += "\nThe documents may not contain enough information, or the question may need rephrasing."
            steps[-1].final_answer = final

        # Extract sources from search results
        sources = self._extract_sources(steps)

        return AgentResult(
            answer=steps[-1].final_answer or "No answer produced.",
            steps=steps,
            total_steps=len(steps),
            model=self.model,
            sources=sources,
        )

    def run_stream(
        self,
        task: str,
        *,
        max_steps: int | None = None,
        verbose: bool = False,
    ):
        """Streaming variant — yields events as the Agent works.

        Yields dicts:
            {"type": "step", "step": Step}
            {"type": "done", "result": AgentResult}

        Useful for showing progress in a UI or SSE API response.
        """
        max_steps = max_steps if max_steps is not None else settings.max_steps
        steps: list[Step] = []
        tool_schemas = self.tools.get_schemas()

        messages: list[dict] = [
            {"role": "system", "content": AGENTIC_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for i in range(max_steps):
            step = Step(step_number=i + 1)
            steps.append(step)

            try:
                raw = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                    tools=tool_schemas,
                    tool_choice="auto",
                )
            except Exception as e:
                step.final_answer = f"LLM call failed: {e}"
                yield {"type": "step", "step": step}
                break

            choice = raw.choices[0]

            if choice.finish_reason == "stop" and not choice.message.tool_calls:
                step.final_answer = choice.message.content or ""
                yield {"type": "step", "step": step}
                break

            if choice.message.tool_calls:
                tool_call = choice.message.tool_calls[0]
                step.tool_name = tool_call.function.name
                step.tool_args = json.loads(tool_call.function.arguments)
                step.tool_result = self.tools.execute(step.tool_name, step.tool_args)

                if verbose:
                    print(f"  🔧 Step {step.step_number}: {step.tool_name}({step.tool_args})")

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": step.tool_result,
                })

                yield {"type": "step", "step": step}
                continue

            step.final_answer = f"Unexpected finish: {choice.finish_reason}"
            yield {"type": "step", "step": step}
            break
        else:
            final = (
                f"I was unable to answer this question within {max_steps} steps."
            )
            steps[-1].final_answer = final

        sources = self._extract_sources(steps)

        yield {
            "type": "done",
            "result": AgentResult(
                answer=steps[-1].final_answer or "No answer.",
                steps=steps,
                total_steps=len(steps),
                model=self.model,
                sources=sources,
            ),
        }

    def _extract_sources(self, steps: list[Step]) -> list[dict]:
        """Extract document sources from search_documents results.

        Parses [Document N — Source: filename] markers from tool results
        to build a deduplicated list of sources used.

        Args:
            steps: All steps from the agent run.

        Returns:
            List of dicts with file_name, source_path, and chunks_found.
        """
        import re

        sources: dict[str, dict] = {}
        doc_pattern = re.compile(r'\[Document \d+\s*—\s*Source:\s*([^\](]+)')

        for step in steps:
            if step.tool_name == "search_documents" and step.tool_result:
                matches = doc_pattern.findall(step.tool_result)
                for source in matches:
                    source = source.strip()
                    if source not in sources:
                        sources[source] = {
                            "file_name": source.split("/")[-1].split("\\")[-1],
                            "source_path": source,
                        }

        return list(sources.values())
