"""CLI for agentic-rag — single query, interactive chat, and server modes."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentic_rag.agent import AgentResult, AgenticRAGAgent, Step
from agentic_rag.config import settings
from agentic_rag.tools import create_registry

app = typer.Typer(
    name="agentic-rag",
    help="Agentic RAG — a ReAct Agent that autonomously drives document search and synthesis.",
    add_completion=False,
)

console = Console(force_terminal=True)


def _build_agent() -> AgenticRAGAgent:
    """Create the agent with RAG tools."""
    registry = create_registry()
    return AgenticRAGAgent(tools=registry)


def _display_step(step: Step) -> None:
    """Pretty-print a single agent step."""
    if step.used_tool:
        args_str = ", ".join(f"{k}={v}" for k, v in step.tool_args.items())
        tool_label = Text(f"[{step.step_number}] {step.tool_name}({args_str})", style="bold yellow")
        console.print(tool_label)

        # Show tool result, truncating if long
        result = step.tool_result
        if len(result) > 400:
            result = result[:400] + f"\n... [dim]({len(step.tool_result)} chars total)[/dim]"
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

    searches = sum(1 for s in result.steps if s.tool_name == "search_documents")
    docs_listed = sum(1 for s in result.steps if s.tool_name == "list_documents")
    deep_dives = sum(1 for s in result.steps if s.tool_name == "get_document")

    table.add_row("Model", result.model)
    table.add_row("Total steps", str(result.total_steps))
    table.add_row("Searches performed", str(searches))
    table.add_row("Documents listed", str(docs_listed))
    table.add_row("Deep dives", str(deep_dives))
    table.add_row("Sources cited", str(len(result.sources)))
    table.add_row("Max steps", str(settings.max_steps))

    console.print(table)

    if result.sources:
        console.print()
        console.print("[bold]Sources used:[/bold]")
        for i, src in enumerate(result.sources, 1):
            console.print(f"  [{i}] {src['file_name']} ({src['source_path']})")


@app.command()
def run(
    task: str = typer.Argument(..., help="What question do you want to ask?"),
    max_steps: int = typer.Option(
        None, "--max-steps", "-m", help="Maximum agent iterations"
    ),
    show_tools: bool = typer.Option(
        False, "--tools", "-t", help="Show available tools before running"
    ),
):
    """Ask a question — the Agent will search, evaluate, and synthesize."""
    agent = _build_agent()

    if show_tools:
        console.print()
        console.print("[bold]Available tools:[/bold]")
        console.print(agent.tools.list_tools())
        console.print()

    console.print()
    console.print(f"[bold]Question:[/bold] {task}")
    console.print()

    result = agent.run(task, max_steps=max_steps, verbose=False)

    for step in result.steps:
        _display_step(step)

    console.print()
    _display_summary(result)


@app.command()
def chat():
    """Interactive chat mode — have a conversation with the Agent."""
    agent = _build_agent()

    console.print()
    console.print(Panel(
        "Agentic RAG Chat\n\n"
        "Ask me anything about your documents. I'll search, evaluate,\n"
        "reformulate, and synthesize until I find the answer.\n\n"
        "[dim]Commands:[/dim]\n"
        "  [bold]/tools[/bold]  — list available tools\n"
        "  [bold]/trace[/bold]  — toggle step-by-step trace on/off\n"
        "  [bold]/model[/bold]  — show current model\n"
        "  [bold]/clear[/bold] — start a fresh conversation\n"
        "  [bold]/exit[/bold]  — quit\n\n"
        "[dim]Tip: Ask complex questions! The agent shines with\n"
        "multi-part questions that require synthesizing from\n"
        "multiple document searches.[/dim]",
        title="Agentic RAG",
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
            console.print(f"[dim]Max steps: {settings.max_steps}[/dim]")
            continue

        if user_input.lower() == "/clear":
            agent = _build_agent()
            console.print("[dim]Conversation reset. New agent created.[/dim]")
            continue

        console.print()
        result = agent.run(user_input, verbose=False)

        if show_trace:
            for step in result.steps:
                _display_step(step)
        else:
            console.print(Panel(
                result.answer,
                title="Answer",
                border_style="green",
            ))

        console.print(
            f"[dim]({result.total_steps} steps, "
            f"{sum(1 for s in result.steps if s.used_tool)} tool calls, "
            f"{len(result.sources)} sources)[/dim]"
        )


@app.command()
def tools():
    """List all available tools and their capabilities."""
    registry = create_registry()
    console.print()
    console.print("[bold]Agentic RAG Tools[/bold]")
    console.print()
    console.print(
        "These tools are exposed to the LLM as function-calling schemas.\n"
        "The Agent decides when and how to use each one."
    )
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
    console.print("[dim]The agent's system prompt teaches it WHEN to use each tool.[/dim]")


@app.command()
def version():
    """Show version and configuration info."""
    from agentic_rag import __version__ as _v

    console.print()
    console.print("[bold]agentic-rag[/bold] — ReAct Agent driving RAG retrieval autonomously")
    console.print(f"[dim]Version: {_v}[/dim]")
    console.print()
    console.print("[bold]What makes it 'agentic':[/bold]")
    console.print("  • Agent decides what to search for (not a fixed query)")
    console.print("  • Agent evaluates if results are sufficient")
    console.print("  • Agent reformulates queries when needed")
    console.print("  • Agent synthesizes from multiple search rounds")
    console.print("  • Agent verifies claims against retrieved context")
    console.print()

    table = Table(show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("LLM Base URL", settings.nvidia_base_url)
    table.add_row("Model", settings.chat_model)
    table.add_row("Max Steps", str(settings.max_steps))
    table.add_row("Temperature", str(settings.temperature))
    table.add_row("Max Tokens", str(settings.max_tokens))
    table.add_row("Top-K", str(settings.top_k))
    table.add_row("ChromaDB", settings.chroma_persist_dir)
    table.add_row("Collection", settings.collection_name)
    table.add_row("API Port", str(settings.port))
    console.print(table)


@app.command()
def serve():
    """Start the FastAPI server for API access."""
    import uvicorn

    console.print("[bold]Starting Agentic RAG API server...[/bold]")
    console.print(f"  Host: {settings.host}")
    console.print(f"  Port: {settings.port}")
    console.print(f"  Docs: http://{settings.host}:{settings.port}/docs")
    console.print()

    uvicorn.run(
        "agentic_rag.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
