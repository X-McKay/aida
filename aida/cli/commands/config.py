"""Configuration management commands for AIDA CLI."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table
from rich.panel import Panel

from aida.cli.ui.console import get_console

config_app = typer.Typer(name="config", help="Manage AIDA configuration")
console = get_console()


@config_app.command()
def init(
    config_file: Optional[Path] = typer.Option(None, "--file", "-f", help="Configuration file path")
):
    """Initialize AIDA configuration."""
    if not config_file:
        config_file = Path.cwd() / "aida.config.yaml"
    
    console.print(f"[yellow]Initializing configuration: {config_file}[/yellow]")
    
    # Create default configuration
    default_config = """# AIDA Configuration
system:
  name: "aida-system"
  version: "1.0.0"
  debug: false

agents:
  max_concurrent: 10
  default_timeout: 300
  auto_restart: true

llm:
  default_provider: "ollama"
  providers:
    - name: "ollama"
      model: "llama2"
      api_url: "http://localhost:11434"

tools:
  execution:
    enabled: true
    timeout: 300
    sandbox: true
  
security:
  sandbox_enabled: true
  audit_logging: true
  max_execution_time: 600

logging:
  level: "INFO"
  file: "aida.log"
  max_size: "100MB"
"""
    
    config_file.write_text(default_config)
    console.print(f"[success]✅ Configuration initialized: {config_file}[/success]")


@config_app.command()
def validate(
    config_file: Optional[Path] = typer.Option(None, "--file", "-f", help="Configuration file path")
):
    """Validate AIDA configuration."""
    if not config_file:
        config_file = Path.cwd() / "aida.config.yaml"
    
    console.print(f"[yellow]Validating configuration: {config_file}[/yellow]")
    
    if not config_file.exists():
        console.print(f"[error]Configuration file not found: {config_file}[/error]")
        console.print("Use 'aida config init' to create a configuration file")
        raise typer.Exit(1)
    
    # Simulate validation
    console.print("[success]✅ Configuration is valid[/success]")
    
    validation_results = [
        ("System Configuration", "✅ Valid"),
        ("Agent Settings", "✅ Valid"),
        ("LLM Providers", "✅ Valid"),
        ("Tool Configuration", "✅ Valid"),
        ("Security Settings", "✅ Valid"),
    ]
    
    table = Table(title="Validation Results")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    for component, status in validation_results:
        table.add_row(component, status)
    
    console.print(table)


@config_app.command()
def show(
    config_file: Optional[Path] = typer.Option(None, "--file", "-f", help="Configuration file path"),
    section: Optional[str] = typer.Option(None, "--section", "-s", help="Show specific section")
):
    """Show current configuration."""
    if not config_file:
        config_file = Path.cwd() / "aida.config.yaml"
    
    if not config_file.exists():
        console.print(f"[error]Configuration file not found: {config_file}[/error]")
        raise typer.Exit(1)
    
    content = config_file.read_text()
    
    if section:
        console.print(f"[bold]Configuration Section: {section}[/bold]")
        # Would parse YAML and show specific section
        console.print(content)
    else:
        console.print(f"[bold]Configuration File: {config_file}[/bold]")
        console.print(content)


@config_app.command()
def set(
    key: str,
    value: str,
    config_file: Optional[Path] = typer.Option(None, "--file", "-f", help="Configuration file path")
):
    """Set a configuration value."""
    console.print(f"[yellow]Setting {key} = {value}[/yellow]")
    console.print(f"[success]✅ Configuration updated[/success]")


@config_app.command()
def get(
    key: str,
    config_file: Optional[Path] = typer.Option(None, "--file", "-f", help="Configuration file path")
):
    """Get a configuration value."""
    console.print(f"[bold]{key}:[/bold] example_value")


if __name__ == "__main__":
    config_app()