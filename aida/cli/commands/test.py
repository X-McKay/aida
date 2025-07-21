"""Test commands for AIDA integration testing."""

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

from aida.tests.base import test_registry
from aida.tests.runner import run_tests

# Initialize CLI components
console = Console()
test_app = typer.Typer(help="Integration testing for refactored components")


@test_app.command()
def list():
    """List available test suites."""
    console.print(Panel.fit("📋 Available Test Suites", style="blue"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Suite Name", style="cyan")
    table.add_column("Description", style="green")

    suite_descriptions = {
        "llm": "LLM system with PydanticAI integration",
        "orchestrator": "Todo Orchestrator with real LLM calls",
        "hybrid_files": "Hybrid FileOperationsTool with multi-framework support",
        "hybrid_system": "Hybrid SystemTool with secure command execution",
        "chat_cli": "Chat CLI with various tool usage scenarios",
    }

    for suite_name in test_registry.list_suites():
        description = suite_descriptions.get(suite_name, "Integration tests")
        table.add_row(suite_name, description)

    console.print(table)
    console.print("\n[dim]Use 'aida test run --suite <name>' to run a specific suite[/dim]")


@test_app.command()
def run(
    suite: str | None = typer.Option(
        None, "--suite", "-s", help="Specific test suite to run (use 'list' to see available)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    persist_files: bool = typer.Option(
        False, "--persist-files", "-p", help="Keep generated test files instead of cleaning them up"
    ),
):
    """Run integration tests for refactored components."""
    if suite and suite not in test_registry.list_suites():
        console.print(f"[red]❌ Unknown test suite: {suite}[/red]")
        console.print("\nAvailable suites:")
        for available_suite in test_registry.list_suites():
            console.print(f"  - {available_suite}")
        raise typer.Exit(1)

    # Show what we're testing
    if suite:
        console.print(Panel.fit(f"🧪 Running {suite} tests", style="blue"))
    else:
        console.print(Panel.fit("🧪 Running all integration tests", style="blue"))

    console.print("[dim]Testing refactored components with real functionality (no mocks)[/dim]\n")

    # Run the tests
    try:
        success = asyncio.run(run_tests(suite, verbose, persist_files=persist_files))

        if success:
            console.print("\n[green]🎉 All tests passed![/green]")
            raise typer.Exit(0)
        else:
            console.print("\n[red]💥 Some tests failed![/red]")
            console.print("[dim]Use --verbose for more details[/dim]")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]⏹️ Tests interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]💥 Test runner crashed: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise typer.Exit(1)


@test_app.command()
def quick():
    """Quick smoke test of core functionality."""
    console.print(Panel.fit("⚡ Quick Smoke Test", style="yellow"))
    console.print("[dim]Testing basic LLM and orchestrator functionality[/dim]\n")

    # Run just the basic tests from each suite
    success = asyncio.run(run_tests(verbose=False))

    if success:
        console.print("\n[green]✅ Quick test passed - core functionality working![/green]")
        raise typer.Exit(0)
    else:
        console.print("\n[red]❌ Quick test failed - check individual suites[/red]")
        console.print("[dim]Use 'aida test run --verbose' for detailed output[/dim]")
        raise typer.Exit(1)


@test_app.callback()
def test_callback():
    """Integration testing for AIDA's refactored components.

    Tests the LLM system and Todo Orchestrator with real functionality
    to ensure refactoring hasn't broken core features.
    """
    pass


if __name__ == "__main__":
    test_app()
