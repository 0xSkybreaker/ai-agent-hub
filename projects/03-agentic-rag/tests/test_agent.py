"""Tests for the AgenticRAGAgent core loop."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from agentic_rag.agent import AgentResult, AgenticRAGAgent, Step
from agentic_rag.tools import Tool, ToolRegistry


class TestStep:
    """Tests for the Step dataclass."""

    def test_is_final_true(self):
        step = Step(step_number=1, final_answer="The answer is 42.")
        assert step.is_final is True
        assert step.used_tool is False

    def test_used_tool_true(self):
        step = Step(step_number=1, tool_name="search_documents", tool_args={"query": "test"})
        assert step.used_tool is True
        assert step.is_final is False

    def test_defaults(self):
        step = Step(step_number=1)
        assert step.tool_name == ""
        assert step.tool_args == {}
        assert step.tool_result == ""
        assert step.final_answer == ""


class TestAgenticRAGAgent:
    """Tests for the agent loop."""

    @pytest.fixture
    def agent(self, mock_llm_client):
        """Create an agent with controlled tool registry and mock client."""
        registry = ToolRegistry()
        registry.register(Tool(
            name="echo",
            description="Echoes back the input",
            parameters={
                "message": {"type": "string", "description": "Message to echo"}
            },
            func=lambda message: f"Echo: {message}",
        ))
        return AgenticRAGAgent(tools=registry, client=mock_llm_client)

    def test_run_immediate_answer(self, agent, mock_llm_client):
        """When LLM returns text immediately, agent should stop."""
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.content = "The answer is 42."
        mock_choice.message.tool_calls = None

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_llm_client.chat.completions.create.return_value = mock_response

        result = agent.run("What is the answer?")

        assert result.answer == "The answer is 42."
        assert result.total_steps == 1
        assert len(result.steps) == 1
        assert result.steps[0].is_final is True
        assert result.steps[0].used_tool is False
        assert result.model == agent.model

    def test_run_single_tool_call(self, agent, mock_llm_client):
        """Agent should execute a tool and then give final answer."""
        # First call: tool call
        func_mock1 = MagicMock()
        func_mock1.name = "echo"
        func_mock1.arguments = json.dumps({"message": "hello"})

        tool_call1 = MagicMock()
        tool_call1.id = "call_1"
        tool_call1.function = func_mock1

        call1 = MagicMock()
        call1.finish_reason = "tool_calls"
        call1.message.content = None
        call1.message.tool_calls = [tool_call1]

        # Second call: final answer
        call2 = MagicMock()
        call2.finish_reason = "stop"
        call2.message.content = "I echoed 'hello' and got the response."
        call2.message.tool_calls = None

        mock_llm_client.chat.completions.create.side_effect = [
            MagicMock(choices=[call1]),
            MagicMock(choices=[call2]),
        ]

        result = agent.run("Echo hello")

        assert result.total_steps == 2
        assert result.steps[0].used_tool is True
        assert result.steps[0].tool_name == "echo"
        assert result.steps[0].tool_args == {"message": "hello"}
        assert "Echo: hello" in result.steps[0].tool_result
        assert result.steps[1].is_final is True
        assert "echoed" in result.answer

    def test_run_max_steps_exhausted(self, agent, mock_llm_client):
        """When max steps is reached, agent should stop gracefully."""
        call = MagicMock()
        call.finish_reason = "tool_calls"
        call.message.content = None
        call.message.tool_calls = [
            MagicMock(
                id="call_1",
                function=MagicMock(
                    name="echo",
                    arguments=json.dumps({"message": "loop"}),
                ),
            )
        ]

        # Always return tool call (never final answer)
        mock_llm_client.chat.completions.create.return_value = MagicMock(choices=[call])

        result = agent.run("Loop forever", max_steps=3)

        assert result.total_steps == 3
        assert "unable to answer" in result.answer.lower() or "within 3 steps" in result.answer

    def test_run_llm_error(self, agent, mock_llm_client):
        """LLM API errors should be caught and reported."""
        mock_llm_client.chat.completions.create.side_effect = ConnectionError("API down")

        result = agent.run("test")

        assert "LLM call failed" in result.answer
        assert result.total_steps == 1

    def test_run_stream_events(self, agent, mock_llm_client):
        """Streaming variant should yield step and done events."""
        call1 = MagicMock()
        func_mock1 = MagicMock()
        func_mock1.name = "echo"
        func_mock1.arguments = json.dumps({"message": "test"})
        tool_call1 = MagicMock()
        tool_call1.id = "call_1"
        tool_call1.function = func_mock1
        call1.finish_reason = "tool_calls"
        call1.message.content = None
        call1.message.tool_calls = [tool_call1]

        call2 = MagicMock()
        call2.finish_reason = "stop"
        call2.message.content = "Done."
        call2.message.tool_calls = None

        mock_llm_client.chat.completions.create.side_effect = [
            MagicMock(choices=[call1]),
            MagicMock(choices=[call2]),
        ]

        events = list(agent.run_stream("test"))

        assert len(events) == 3  # step, step, done
        assert events[0]["type"] == "step"
        assert events[0]["step"].tool_name == "echo"
        assert events[1]["type"] == "step"
        assert events[1]["step"].is_final is True
        assert events[2]["type"] == "done"
        assert events[2]["result"].answer == "Done."

    def test_default_tools(self, mock_llm_client):
        """Agent created without explicit tools should use create_registry."""
        agent = AgenticRAGAgent(client=mock_llm_client)
        assert agent.tools.has("search_documents")
        assert agent.tools.has("get_document")
        assert agent.tools.has("list_documents")

    def test_extract_sources(self, agent):
        """Should extract source paths from search results."""
        steps = [
            Step(
                step_number=1,
                tool_name="search_documents",
                tool_result=(
                    "[Document 1 — Source: about_rag.md (relevance: 0.94)]\nText here\n\n"
                    "---\n\n"
                    "[Document 2 — Source: training.md (relevance: 0.78)]\nMore text"
                ),
            ),
            Step(step_number=2, final_answer="Answer."),
        ]
        sources = agent._extract_sources(steps)
        assert len(sources) == 2

        file_names = {s["file_name"] for s in sources}
        assert "about_rag.md" in file_names
        assert "training.md" in file_names
