"""
Tool Orchestra CLI - Research Agent

A CLI interface for interacting with the research agent system.
Uses Orchestrator-8B for intelligent tool orchestration.
"""

import asyncio

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.config import get_settings, setup_langsmith
from src.orchestrator.loop import run
from src.tools import registry

app = typer.Typer(
    name="orchestra",
    help="Tool Orchestra - Research Agent",
    add_completion=False,
)
console = Console()


@app.command()
def query(
    question: str = typer.Argument(..., help="The question or task to process"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed execution trace"
    ),
) -> None:
    """
    Process a single query through the orchestration system.
    """
    setup_langsmith()

    console.print(
        Panel(f"[bold blue]Query:[/bold blue] {question}", title="Tool Orchestra")
    )

    with console.status("[bold green]Processing...", spinner="dots"):
        try:
            result = asyncio.run(run(question, verbose=verbose))

            # Display result
            console.print()
            console.print(
                Panel(
                    Markdown(result.get("answer", "No response generated")),
                    title="[bold green]Response[/bold green]",
                    border_style="green",
                )
            )

            # Display sources if any
            sources = result.get("sources", [])
            if sources:
                console.print()
                console.print("[bold]Sources:[/bold]")
                for source in sources:
                    console.print(f"  • {source}")

            # Display cost summary
            cost = result.get("cost", 0.0)
            turns = result.get("turns", 0)

            console.print()
            console.print(f"[dim]Turns: {turns} | Estimated cost: ${cost:.4f}[/dim]")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(code=1) from e


@app.command()
def chat() -> None:
    """
    Start an interactive chat session with the orchestrator.
    """
    setup_langsmith()

    console.print(
        Panel(
            "[bold]Welcome to Tool Orchestra Interactive Mode[/bold]\n\n"
            "Type your questions and press Enter. Commands:\n"
            "  [green]/cost[/green]   - Show session cost summary\n"
            "  [green]/clear[/green]  - Clear screen\n"
            "  [green]/quit[/green]   - Exit\n",
            title="Tool Orchestra Chat",
            border_style="blue",
        )
    )

    session_cost = 0.0
    session_queries = 0

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if not user_input.strip():
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower().strip()

            if cmd in ("/quit", "/exit", "/q"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            elif cmd == "/cost":
                console.print("\n[bold]Session Summary:[/bold]")
                console.print(f"  Total queries: {session_queries}")
                console.print(f"  Total cost:    ${session_cost:.4f}\n")
                continue

            elif cmd == "/clear":
                console.clear()
                console.print("[dim]Screen cleared.[/dim]\n")
                continue

            else:
                console.print(f"[red]Unknown command: {cmd}[/red]\n")
                continue

        # Process query
        with console.status("[bold green]Thinking...", spinner="dots"):
            try:
                result = asyncio.run(run(user_input))

                session_cost += result.get("cost", 0.0)
                session_queries += 1

                console.print()
                console.print(
                    Panel(
                        Markdown(result.get("answer", "No response generated")),
                        title="[bold green]Assistant[/bold green]",
                        border_style="green",
                    )
                )
                console.print(
                    f"[dim]Turns: {result.get('turns', 0)} | Cost: ${result.get('cost', 0.0):.4f}[/dim]\n"
                )

            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}\n")


@app.command()
def tools() -> None:
    """
    List all available tools and their schemas.
    """
    tool_list = registry.list_tools()

    table = Table(title="Available Tools")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    table.add_column("Parameters", style="yellow")

    for tool in tool_list:
        params = tool.get("parameters", {}).get("properties", {})
        param_names = ", ".join(params.keys()) if params else "None"
        table.add_row(tool["name"], tool["description"], param_names)

    console.print(table)


@app.command()
def config() -> None:
    """
    Display current configuration.
    """
    settings = get_settings()

    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Log Level", settings.log_level)
    table.add_row("Max Turns", str(settings.max_iterations))
    table.add_row("LM Studio URL", settings.lm_studio.base_url)
    table.add_row("Orchestrator Model", settings.models.orchestrator_model)
    table.add_row("Phi-4 Model", settings.models.phi4_model)
    table.add_row("Gemini Model", settings.models.gemini_model)
    table.add_row(
        "Gemini API Key", "***" if settings.models.gemini_api_key else "(not set)"
    )
    table.add_row("Cache Enabled", str(settings.cache.enabled))
    table.add_row("LangSmith Tracing", str(settings.langsmith.tracing_v2))

    console.print(table)


@app.command()
def version() -> None:
    """
    Show version information.
    """
    console.print(
        Panel(
            "[bold]Tool Orchestra[/bold]\n"
            "Research Agent powered by Orchestrator-8B\n\n"
            "Version: 0.2.0\n"
            "Python: 3.11+",
            title="Version Info",
            border_style="blue",
        )
    )


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
