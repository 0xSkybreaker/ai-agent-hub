"""CLI for mini-agent — single query and interactive modes."""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mini_agent.agent import AgentResult, ReActAgent, Step
from mini_agent.config import settings
from mini_agent.tools import create_registry

app = typer.Typer(
    name="mini-agent",
    help="A minimal ReAct Agent — built from scratch, zero framework dependencies.",
    add_completion=False,
)

# Force UTF-8 on Windows to avoid emoji encoding crashes in legacy terminals
console = Console(force_terminal=True)


def _build_agent() -> ReActAgent:
    """Create the agent with built-in tools."""
    registry = create_registry()
    return ReActAgent(tools=registry)


def _display_step(step: Step) -> None:
    """Pretty-print a single step."""
    if step.used_tool:
        args_str = ", ".join(f"{k}={v}" for k, v in step.tool_args.items())
        tool_label = Text(f"[{step.step_number}] {step.tool_name}({args_str})", style="bold yellow")
        console.print(tool_label)

        # Show tool result, truncating if long
        result = step.tool_result
        if len(result) > 300:
            result = result[:300] + f"\n... [dim]({len(step.tool_result)} chars total)[/dim]"
        console.print(f"   {result}", style="dim")

    if step.is_final:
        console.print()
        console.print(Panel(
            step.final_answer,
            title=f"[bold green]Answer[/bold green] (step {step.step_number})",
            border_style="green",
        ))


def _display_summary(result: AgentResult) -> None:
    """Show a summary table after the agent finishes."""
    table = Table(title="Run Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", result.model)
    table.add_row("Steps", str(result.total_steps))
    table.add_row("Tool calls", str(sum(1 for s in result.steps if s.used_tool)))
    table.add_row("Max steps", str(settings.max_steps))

    console.print(table)


@app.command()
def run(
    task: str = typer.Argument(..., help="What do you want the Agent to do?"),
    max_steps: int = typer.Option(
        None, "--max-steps", "-m", help="Maximum tool-calling iterations"
    ),
    show_tools: bool = typer.Option(
        False, "--tools", "-t", help="Show available tools before running"
    ),
):
    """Run the Agent on a single task."""
    agent = _build_agent()

    if show_tools:
        console.print()
        console.print("[bold]Available tools:[/bold]")
        console.print(agent.tools.list_tools())
        console.print()

    console.print()
    console.print(f"[bold]Task:[/bold] {task}")
    console.print()

    result = agent.run(task, max_steps=max_steps, verbose=False)

    # Display steps
    for step in result.steps:
        _display_step(step)

    # Summary
    console.print()
    _display_summary(result)


@app.command()
def chat():
    """Interactive chat mode — have a conversation with the Agent."""
    agent = _build_agent()

    console.print()
    console.print(Panel(
        "Mini-Agent Chat\n\n"
        "Ask me to do anything — I'll use tools when needed.\n\n"
        "[dim]Commands:[/dim]\n"
        "  [bold]/tools[/bold]  — list available tools\n"
        "  [bold]/trace[/bold]  — toggle step-by-step trace on/off\n"
        "  [bold]/model[/bold]  — show current model\n"
        "  [bold]/clear[/bold] — start a fresh conversation\n"
        "  [bold]/exit[/bold]  — quit",
        title="Mini ReAct Agent",
        border_style="blue",
    ))

    show_trace = True

    while True:
        try:
            user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ("/exit", "/quit"):
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "/tools":
            console.print()
            console.print("[bold]Available tools:[/bold]")
            console.print(agent.tools.list_tools())
            continue

        if user_input.lower() == "/trace":
            show_trace = not show_trace
            state = "ON" if show_trace else "OFF"
            console.print(f"[dim]Step trace: {state}[/dim]")
            continue

        if user_input.lower() == "/model":
            console.print(f"[dim]Model: {agent.model}[/dim]")
            continue

        if user_input.lower() == "/clear":
            agent = _build_agent()
            console.print("[dim]Conversation reset.[/dim]")
            continue

        # Run the agent
        console.print()
        result = agent.run(user_input, verbose=False)

        if show_trace:
            for step in result.steps:
                _display_step(step)
        else:
            # Just show the final answer
            console.print(Panel(
                result.answer,
                title="Answer",
                border_style="green",
            ))

        console.print(
            f"[dim]({result.total_steps} steps, "
            f"{sum(1 for s in result.steps if s.used_tool)} tool calls)[/dim]"
        )


@app.command()
def tools():
    """List all available tools and their descriptions."""
    registry = create_registry()
    console.print()
    console.print("[bold]Available tools:[/bold]")
    console.print()

    table = Table(show_header=True)
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Parameters")
    table.add_column("Description")

    for name, tool in registry._tools.items():
        params = ", ".join(tool.parameters.keys()) if tool.parameters else "(none)"
        table.add_row(name, params, tool.description)

    console.print(table)
    console.print()
    console.print("[dim]These are exposed to the LLM as function-calling schemas.[/dim]")


@app.command()
def version():
    """Show version and configuration info."""
    console.print()
    console.print("[bold]mini-agent[/bold] — A minimal ReAct Agent from scratch")
    console.print()
    table = Table(show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("LLM Base URL", settings.llm_base_url)
    table.add_row("Model", settings.llm_model)
    table.add_row("Max Steps", str(settings.max_steps))
    table.add_row("Temperature", str(settings.temperature))
    table.add_row("Max Tokens", str(settings.max_tokens))
    console.print(table)


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
