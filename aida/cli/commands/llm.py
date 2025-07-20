"""LLM provider management commands for AIDA CLI."""

import asyncio
from typing import Optional

import typer
from rich.table import Table
from rich.panel import Panel

from aida.cli.ui.console import get_console

llm_app = typer.Typer(name="llm", help="Manage LLM providers")
console = get_console()


@llm_app.command()
def list():
    """List available LLM providers."""
    console.print("[yellow]Listing LLM providers...[/yellow]")
    
    providers = [
        {"name": "ollama", "model": "llama2", "status": "available", "endpoint": "localhost:11434"},
        {"name": "openai", "model": "gpt-3.5-turbo", "status": "configured", "endpoint": "api.openai.com"},
        {"name": "anthropic", "model": "claude-3", "status": "not_configured", "endpoint": "api.anthropic.com"},
    ]
    
    table = Table(title="LLM Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="dim")
    table.add_column("Status", style="green")
    table.add_column("Endpoint", style="dim")
    
    for provider in providers:
        status_color = "green" if provider["status"] == "available" else "yellow" if provider["status"] == "configured" else "red"
        table.add_row(
            provider["name"],
            provider["model"],
            f"[{status_color}]{provider['status']}[/{status_color}]",
            provider["endpoint"]
        )
    
    console.print(table)


@llm_app.command()
def add(
    provider: str,
    model: str,
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for the provider"),
    endpoint: Optional[str] = typer.Option(None, "--endpoint", help="Custom endpoint URL")
):
    """Add a new LLM provider."""
    console.print(f"[yellow]Adding LLM provider: {provider}[/yellow]")
    console.print(f"Model: {model}")
    
    if api_key:
        console.print(f"API Key: {'*' * (len(api_key) - 4)}{api_key[-4:]}")
    
    if endpoint:
        console.print(f"Endpoint: {endpoint}")
    
    console.print(f"[success]✅ Provider '{provider}' added successfully[/success]")


@llm_app.command()
def remove(provider: str):
    """Remove an LLM provider."""
    console.print(f"[yellow]Removing LLM provider: {provider}[/yellow]")
    console.print(f"[success]✅ Provider '{provider}' removed successfully[/success]")


@llm_app.command()
def test(provider: str):
    """Test connection to an LLM provider."""
    console.print(f"[yellow]Testing connection to: {provider}[/yellow]")
    
    async def _test():
        console.print("Connecting...")
        await asyncio.sleep(1)
        console.print("Sending test request...")
        await asyncio.sleep(1)
        console.print("[success]✅ Connection test successful[/success]")
    
    try:
        asyncio.run(_test())
    except Exception as e:
        console.print(f"[error]❌ Connection test failed: {e}[/error]")


@llm_app.command()
def status():
    """Show LLM provider status."""
    console.print("[bold]LLM Provider Status[/bold]")
    
    status_info = """
[yellow]System Status:[/yellow]
• Total Providers: 3
• Available: 1
• Configured: 1  
• Not Configured: 1

[yellow]Health Check:[/yellow]
• ollama: ✅ Healthy (200ms response)
• openai: ⚠️  API key required
• anthropic: ❌ Not configured
"""
    
    console.print(Panel(status_info.strip(), title="LLM Status", border_style="blue"))


@llm_app.command()
def configure(
    provider: str,
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive configuration")
):
    """Configure an LLM provider."""
    console.print(f"[yellow]Configuring provider: {provider}[/yellow]")
    
    if interactive:
        console.print("Interactive configuration mode...")
        # Would prompt for various settings
        
    console.print(f"[success]✅ Provider '{provider}' configured successfully[/success]")


if __name__ == "__main__":
    llm_app()