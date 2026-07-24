"""ReAct Agent — the core loop.

The Agent receives a task, decides step-by-step whether to call tools or give a
final answer, and feeds tool results back into its reasoning.

This is THE file to understand. Everything else (tools, llm, cli) is plumbing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from mini_agent.config import settings
from mini_agent.llm import get_client
from mini_agent.tools import ToolRegistry, create_registry


# ── System prompt ──────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

Work step by step:
1. Think about what the user is asking.
2. If you need information or computation beyond your knowledge, use a tool.
3. After seeing the tool result, decide whether you need more tools or can answer.
4. When ready, give a clear, direct answer to the user.

Important rules:
- Use the calculator for any math. Do NOT do mental arithmetic — always calculate.
- Use get_current_time when asked about dates or times.
- Use read_file when asked to inspect files.
- Use list_directory to explore the file system.
- Use text_stats to count words, characters, or lines.
- If a tool fails, try to understand why and adapt. Don't give up after one error.
- If the task doesn't need any tools, answer directly without calling tools.
- When you have all the information needed, give the final answer — don't keep calling tools unnecessarily."""


# ── Result dataclasses ─────────────────────────────────────────────


@dataclass
class Step:
    """A single step in the Agent's reasoning trace."""

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
    """The result of running the Agent."""

    answer: str
    steps: list[Step]
    total_steps: int
    model: str


# ── Agent ──────────────────────────────────────────────────────────


class ReActAgent:
    """A minimal ReAct Agent using native function calling.

    Usage:
        agent = ReActAgent()
        result = agent.run("What is (30 + 45) * 2?")
        print(result.answer)
        for step in result.steps:
            print(step)
    """

    def __init__(self, tools: ToolRegistry | None = None) -> None:
        self.tools = tools or create_registry()
        self.model = settings.llm_model
        self._client = get_client()

    def run(
        self,
        task: str,
        *,
        max_steps: int | None = None,
        verbose: bool = False,
    ) -> AgentResult:
        """Run the Agent on a task.

        Args:
            task: The user's request.
            max_steps: Max tool-calling iterations (defaults to config).
            verbose: If True, print each step as it happens.

        Returns:
            AgentResult with the final answer and step trace.
        """
        max_steps = max_steps if max_steps is not None else settings.max_steps
        steps: list[Step] = []
        tool_schemas = self.tools.get_schemas()

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
                    if len(args_short) > 60:
                        args_short = args_short[:60] + "..."
                    print(f"  🔧 Step {step.step_number}: {step.tool_name}({args_short})")

                # Execute
                step.tool_result = self.tools.execute(step.tool_name, step.tool_args)

                if verbose:
                    preview = step.tool_result[:150] + "..." if len(step.tool_result) > 150 else step.tool_result
                    # Indent multi-line results
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
            # Exhausted max steps
            steps[-1].final_answer = (
                f"Reached maximum steps ({max_steps}) without completing the task."
            )

        return AgentResult(
            answer=steps[-1].final_answer or "No answer produced.",
            steps=steps,
            total_steps=len(steps),
            model=self.model,
        )

    def run_stream(
        self,
        task: str,
        *,
        max_steps: int | None = None,
        verbose: bool = False,
    ):
        """Streaming variant — yields events as the Agent works.

        Yields dicts: {"type": "step", "step": Step} | {"type": "done", "result": AgentResult}

        Useful for showing progress in a UI or API response.
        """
        max_steps = max_steps if max_steps is not None else settings.max_steps
        steps: list[Step] = []
        tool_schemas = self.tools.get_schemas()

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        for i in range(max_steps):
            step = Step(step_number=i + 1)
            steps.append(step)

            raw = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                tools=tool_schemas,
                tool_choice="auto",
            )
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
            steps[-1].final_answer = f"Reached max steps ({max_steps})."

        yield {
            "type": "done",
            "result": AgentResult(
                answer=steps[-1].final_answer or "No answer.",
                steps=steps,
                total_steps=len(steps),
                model=self.model,
            ),
        }
