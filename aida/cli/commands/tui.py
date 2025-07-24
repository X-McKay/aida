"""TUI command for AIDA CLI."""

import typer

from aida.cli.tui.app import run_tui

tui_app = typer.Typer(name="tui", help="Text User Interface for AIDA")


@tui_app.command()
def start():
    """Start the AIDA Text User Interface."""
    try:
        run_tui()
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C
        pass
    except Exception as e:
        typer.echo(f"Error starting TUI: {e}", err=True)
        raise typer.Exit(1)


# Default command when called without subcommand
@tui_app.callback(invoke_without_command=True)
def tui_main(ctx: typer.Context):
    """Start AIDA TUI mode."""
    if ctx.invoked_subcommand is None:
        start()
