"""Agent management commands for AIDA CLI."""

import asyncio
import json
from typing import Optional, List

import typer
from rich.table import Table
from rich.panel import Panel

from aida.cli.ui.console import get_console

agent_app = typer.Typer(name="agent", help="Manage AIDA agents")
console = get_console()


@agent_app.command()
def create(
    name: str,
    description: Optional[str] = typer.Option(None, "--description", "-d", help="Agent description"),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Configuration file")
):
    """Create a new agent."""
    console.print(f"[yellow]Creating agent: {name}[/yellow]")
    
    if not description:
        description = f"Agent {name}"
    
    # Simulate agent creation
    console.print(f"[success]✅ Agent '{name}' created successfully[/success]")
    console.print(f"[dim]Description: {description}[/dim]")


@agent_app.command()
def list():
    """List all agents."""
    console.print("[yellow]Listing agents...[/yellow]")
    
    # Mock agent data
    agents = [
        {"name": "interactive_agent", "status": "running", "tasks": 3},
        {"name": "demo_agent", "status": "stopped", "tasks": 0},
        {"name": "test_agent", "status": "running", "tasks": 1},
    ]
    
    if not agents:
        console.print("[warning]No agents found[/warning]")
        return
    
    table = Table(title="AIDA Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Active Tasks", style="dim")
    
    for agent in agents:
        status_color = "green" if agent["status"] == "running" else "red"
        table.add_row(
            agent["name"],
            f"[{status_color}]{agent['status']}[/{status_color}]",
            str(agent["tasks"])
        )
    
    console.print(table)


@agent_app.command()
def start(name: str):
    """Start an agent."""
    console.print(f"[yellow]Starting agent: {name}[/yellow]")
    console.print(f"[success]✅ Agent '{name}' started successfully[/success]")


@agent_app.command()
def stop(name: str):
    """Stop an agent."""
    console.print(f"[yellow]Stopping agent: {name}[/yellow]")
    console.print(f"[success]✅ Agent '{name}' stopped successfully[/success]")


@agent_app.command()
def status(name: Optional[str] = typer.Argument(None, help="Agent name")):
    """Show agent status."""
    if name:
        console.print(f"[bold]Agent Status: {name}[/bold]")
        console.print(f"Status: [green]running[/green]")
        console.print(f"Uptime: 1h 23m")
        console.print(f"Active Tasks: 2")
        console.print(f"Completed Tasks: 15")
    else:
        console.print("[bold]System Agent Status[/bold]")
        console.print("Total Agents: 3")
        console.print("Running: 2")
        console.print("Stopped: 1")


if __name__ == "__main__":
    agent_app()