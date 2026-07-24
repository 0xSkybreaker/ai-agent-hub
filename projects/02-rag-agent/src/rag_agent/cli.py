"""CLI interface for the RAG Agent using Typer and Rich."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from rag_agent.config import settings
from rag_agent.embeddings.nvidia_embeddings import EmbeddingClient
from rag_agent.generation.generator import Generator
from rag_agent.llm.nvidia_client import LLMClient
from rag_agent.memory.conversation import ConversationMemory
from rag_agent.retrieval.retriever import Retriever
from rag_agent.utils.logger import get_logger
from rag_agent.vector_store.chroma_store import ChromaVectorStore
from rag_agent.vector_store.indexer import IndexingPipeline

app = typer.Typer(
    name="rag-agent",
    help="RAG Agent — Document Q&A using NVIDIA NIM",
    add_completion=False,
)

console = Console()
logger = get_logger()

# ── Lazy component initialization ─────────────────────────────────

_embedding_client: EmbeddingClient | None = None
_llm_client: LLMClient | None = None
_vector_store: ChromaVectorStore | None = None
_indexer: IndexingPipeline | None = None
_retriever: Retriever | None = None
_generator: Generator | None = None
_memory: ConversationMemory | None = None


def _get_embedding_client() -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


def _get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def _get_vector_store() -> ChromaVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore(_get_embedding_client())
    return _vector_store


def _get_indexer() -> IndexingPipeline:
    global _indexer
    if _indexer is None:
        _indexer = IndexingPipeline(_get_vector_store(), _get_embedding_client())
    return _indexer


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(_get_vector_store(), _get_embedding_client())
    return _retriever


def _get_generator() -> Generator:
    global _generator
    if _generator is None:
        _generator = Generator(_get_retriever(), _get_llm_client())
    return _generator


def _get_memory() -> ConversationMemory:
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory


# ── Commands ───────────────────────────────────────────────────────

@app.command()
def index(
    source: str = typer.Argument(..., help="File or directory path to index"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-index even if unchanged"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Recurse into subdirectories"),
):
    """Index documents into the vector store."""
    indexer = _get_indexer()
    path = Path(source).resolve()

    if not path.exists():
        console.print(f"[red]Error:[/red] Path not found: {path}")
        raise typer.Exit(1)

    if path.is_file():
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description=f"Indexing {path.name}...", total=None)
            result = indexer.index_file(str(path), force=force)

        if result.status == "error":
            console.print(f"[red]Error:[/red] {result.error}")
        elif result.status == "unchanged":
            console.print(f"[yellow]Unchanged:[/yellow] {path.name}")
        else:
            console.print(f"[green]✓[/green] Indexed {path.name}: {result.chunks_created} chunks")

    elif path.is_dir():
        console.print(f"Indexing directory: {path}")
        with Progress(
            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description="Scanning and indexing...", total=None)
            results = indexer.index_directory(str(path), recursive=recursive, force=force)

        # Summary table
        table = Table(title="Indexing Results")
        table.add_column("File", style="cyan")
        table.add_column("Status")
        table.add_column("Chunks")

        total_chunks = 0
        for r in results:
            status_color = {"indexed": "green", "unchanged": "yellow", "error": "red"}.get(r.status, "white")
            status_text = f"[{status_color}]{r.status}[/{status_color}]"
            table.add_row(Path(r.source).name, status_text, str(r.chunks_created))
            total_chunks += r.chunks_created

        console.print(table)
        console.print(f"\nTotal: {total_chunks} chunks across {len(results)} files")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(None, "--top-k", "-k", help="Number of documents to retrieve"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream the answer token by token"),
):
    """Ask a question about your documents."""
    generator = _get_generator()

    if stream:
        console.print("[bold]Answer:[/bold]")
        console.print()
        for token in generator.generate_stream(question=question, top_k=top_k):
            console.print(token, end="", highlight=False)
        console.print("\n")
    else:
        with Progress(
            SpinnerColumn(), TextColumn("Thinking..."),
            transient=True,
        ) as progress:
            progress.add_task(description="Thinking...", total=None)
            result = generator.generate(question=question, top_k=top_k)

        console.print(Panel(result.answer, title="Answer", border_style="green"))

        if result.sources:
            from rag_agent.generation.citations import format_citations_for_display
            console.print(format_citations_for_display(result.sources))


@app.command()
def chat():
    """Start an interactive chat session with conversation memory."""
    generator = _get_generator()
    memory = _get_memory()
    session_id = memory.create_session()

    console.print(Panel(
        "RAG Agent Chat — ask questions about your documents.\n"
        "Type [bold]/clear[/bold] to reset, [bold]/exit[/bold] to quit.",
        title="Chat Mode",
        border_style="blue",
    ))

    while True:
        try:
            user_input = console.input("\n[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit"):
            console.print("Goodbye!")
            break

        if user_input.lower() == "/clear":
            memory.clear_session(session_id)
            session_id = memory.create_session()
            console.print("[dim]Conversation cleared.[/dim]")
            continue

        history = memory.get_history(session_id)

        try:
            with Progress(
                SpinnerColumn(), TextColumn("Thinking..."),
                transient=True,
            ) as progress:
                progress.add_task(description="Thinking...", total=None)
                result = generator.generate(
                    question=user_input,
                    history=history,
                )

            console.print()
            console.print(Panel(result.answer, title="Assistant", border_style="green"))

            if result.sources:
                from rag_agent.generation.citations import format_citations_for_display
                console.print(format_citations_for_display(result.sources))

            memory.add_exchange(session_id, user_input, result.answer)

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            logger.exception("Chat error")


@app.command()
def serve(
    host: str = typer.Option(None, "--host", "-h", help="Host to bind"),
    port: int = typer.Option(None, "--port", "-p", help="Port to bind"),
):
    """Start the FastAPI server."""
    import uvicorn

    h = host or settings.host
    p = port or settings.port

    console.print(f"[green]Starting server at http://{h}:{p}[/green]")
    console.print(f"[dim]API docs: http://{h}:{p}/docs[/dim]")

    uvicorn.run(
        "rag_agent.api.server:app",
        host=h,
        port=p,
        log_level=settings.log_level.lower(),
    )


@app.command()
def stats():
    """Show indexing statistics."""
    indexer = _get_indexer()
    memory = _get_memory()
    s = indexer.get_stats()

    table = Table(title="Index Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Chunks", str(s["chunk_count"]))
    table.add_row("Unique Sources", str(s["unique_sources"]))
    table.add_row("Collection", s["collection_name"])
    table.add_row("Active Conversations", str(memory.session_count()))

    console.print(table)


@app.command()
def remove(
    source: str = typer.Argument(..., help="Source path to remove from index"),
):
    """Remove a document from the index."""
    indexer = _get_indexer()
    count = indexer.remove_document(source)
    if count > 0:
        console.print(f"[green]✓[/green] Removed {count} chunks for: {source}")
    else:
        console.print(f"[yellow]No chunks found for: {source}[/yellow]")


@app.command()
def supported():
    """List supported file types."""
    indexer = _get_indexer()
    exts = indexer.list_supported_extensions()
    console.print("Supported file types: " + ", ".join(exts))


def main():
    """Entry point for console script."""
    app()


if __name__ == "__main__":
    main()
