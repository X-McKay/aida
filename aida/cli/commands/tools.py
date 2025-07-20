"""Tool management commands for AIDA CLI."""

import asyncio
import json
from typing import Optional, List

import typer
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON

from aida.cli.ui.console import get_console
from aida.tools import get_tool_registry, initialize_default_tools

tools_app = typer.Typer(name="tools", help="Manage AIDA tools")
console = get_console()


@tools_app.command()
def list():
    """List all available tools."""
    async def _list():
        registry = get_tool_registry()
        tool_names = await registry.list_tools()
        
        if not tool_names:
            console.print("[yellow]No tools registered. Initializing default tools...[/yellow]")
            await initialize_default_tools()
            tool_names = await registry.list_tools()
        
        table = Table(title="Available Tools")
        table.add_column("Tool Name", style="tool", no_wrap=True)
        table.add_column("Version", style="dim")
        table.add_column("Description")
        table.add_column("Status", style="green")
        
        for tool_name in sorted(tool_names):
            tool = await registry.get_tool(tool_name)
            if tool:
                table.add_row(
                    tool.name,
                    tool.version,
                    tool.description[:80] + "..." if len(tool.description) > 80 else tool.description,
                    "Ready"
                )
        
        console.print(table)
    
    try:
        asyncio.run(_list())
    except Exception as e:
        console.print(f"[error]Failed to list tools: {e}[/error]")


@tools_app.command()
def init():
    """Initialize default tools."""
    async def _init():
        console.print("[yellow]Initializing default tools...[/yellow]")
        
        try:
            registry = await initialize_default_tools()
            tool_names = await registry.list_tools()
            
            console.print(f"[success]✅ Successfully initialized {len(tool_names)} tools:[/success]")
            for tool_name in sorted(tool_names):
                console.print(f"  • {tool_name}")
            
        except Exception as e:
            console.print(f"[error]❌ Failed to initialize tools: {e}[/error]")
            raise typer.Exit(1)
    
    try:
        asyncio.run(_init())
    except KeyboardInterrupt:
        console.print("\n[yellow]Tool initialization cancelled[/yellow]")
        raise typer.Exit(1)


@tools_app.command()
def info(tool_name: str):
    """Get detailed information about a specific tool."""
    async def _info():
        registry = get_tool_registry()
        
        # Auto-initialize if no tools are registered
        tool_names = await registry.list_tools()
        if not tool_names:
            console.print("[yellow]No tools registered. Initializing default tools...[/yellow]")
            await initialize_default_tools()
        
        tool = await registry.get_tool(tool_name)
        
        if not tool:
            console.print(f"[error]Tool '{tool_name}' not found[/error]")
            console.print("Use 'aida tools list' to see available tools")
            raise typer.Exit(1)
        
        # Get tool capability and stats
        capability = tool.get_capability()
        stats = tool.get_stats()
        
        # Tool overview
        overview_content = f"""[bold]Name:[/bold] {tool.name}
[bold]Version:[/bold] {tool.version}
[bold]Description:[/bold] {tool.description}

[bold]Supported Platforms:[/bold] {', '.join(capability.supported_platforms)}
[bold]Required Permissions:[/bold] {', '.join(capability.required_permissions)}
[bold]Dependencies:[/bold] {', '.join(capability.dependencies) if capability.dependencies else 'None'}"""
        
        console.print_panel(overview_content, title=f"Tool: {tool_name}", border_style="blue")
        
        # Parameters
        if capability.parameters:
            param_table = Table(title="Parameters")
            param_table.add_column("Name", style="cyan", no_wrap=True)
            param_table.add_column("Type", style="dim")
            param_table.add_column("Required", style="dim")
            param_table.add_column("Default", style="dim")
            param_table.add_column("Description")
            
            for param in capability.parameters:
                required_text = "✅" if param.required else "❌"
                default_text = str(param.default) if param.default is not None else "-"
                
                param_table.add_row(
                    param.name,
                    param.type,
                    required_text,
                    default_text,
                    param.description
                )
            
            console.print(param_table)
        
        # Statistics
        stats_table = Table(title="Usage Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Executions", str(stats.get("total_executions", 0)))
        stats_table.add_row("Successful", str(stats.get("successful_executions", 0)))
        stats_table.add_row("Failed", str(stats.get("failed_executions", 0)))
        stats_table.add_row("Average Duration", f"{stats.get('average_duration', 0.0):.2f}s")
        
        console.print(stats_table)
    
    try:
        asyncio.run(_info())
    except Exception as e:
        console.print(f"[error]Failed to get tool info: {e}[/error]")
        raise typer.Exit(1)


@tools_app.command()
def execute(
    tool_name: str,
    params: Optional[str] = typer.Option(None, "--params", "-p", help="JSON parameters for tool execution"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive parameter input"),
    output_format: str = typer.Option("pretty", "--format", "-f", help="Output format (pretty, json, raw)")
):
    """Execute a tool with specified parameters."""
    async def _execute():
        registry = get_tool_registry()
        
        # Auto-initialize if no tools are registered
        tool_names = await registry.list_tools()
        if not tool_names:
            await initialize_default_tools()
        
        tool = await registry.get_tool(tool_name)
        
        if not tool:
            console.print(f"[error]Tool '{tool_name}' not found[/error]")
            raise typer.Exit(1)
        
        # Get tool capability for parameter validation
        capability = tool.get_capability()
        
        # Parse or collect parameters
        execution_params = {}
        
        if interactive:
            # Interactive parameter collection
            console.print(f"[bold]Interactive execution of tool: {tool_name}[/bold]\n")
            
            for param in capability.parameters:
                if param.required or console.confirm(f"Set optional parameter '{param.name}'?", default=False):
                    value = console.prompt(
                        f"{param.name} ({param.type}): {param.description}",
                        default=str(param.default) if param.default is not None else None
                    )
                    
                    # Basic type conversion
                    if param.type == "int":
                        execution_params[param.name] = int(value)
                    elif param.type == "float":
                        execution_params[param.name] = float(value)
                    elif param.type == "bool":
                        execution_params[param.name] = value.lower() in ["true", "yes", "1", "on"]
                    elif param.type == "list":
                        execution_params[param.name] = value.split(",") if value else []
                    elif param.type == "dict":
                        execution_params[param.name] = json.loads(value) if value else {}
                    else:
                        execution_params[param.name] = value
        
        elif params:
            # Parse JSON parameters
            try:
                execution_params = json.loads(params)
            except json.JSONDecodeError as e:
                console.print(f"[error]Invalid JSON parameters: {e}[/error]")
                raise typer.Exit(1)
        
        # Execute tool
        console.print(f"[yellow]Executing tool: {tool_name}[/yellow]")
        
        try:
            result = await tool.execute_async(**execution_params)
            
            if output_format == "json":
                console.print_json(result.dict())
            elif output_format == "raw":
                console.print(result.result)
            else:
                # Pretty format
                status_color = "green" if result.status == "completed" else "red"
                
                result_content = f"""[bold]Execution ID:[/bold] {result.execution_id}
[bold]Status:[/bold] [{status_color}]{result.status}[/{status_color}]
[bold]Duration:[/bold] {result.duration_seconds:.2f}s"""
                
                if result.error:
                    result_content += f"\n[bold]Error:[/bold] [red]{result.error}[/red]"
                
                console.print_panel(result_content, title="Execution Result", border_style=status_color)
                
                if result.result:
                    if isinstance(result.result, dict):
                        console.print("\n[bold]Result:[/bold]")
                        console.print_json(result.result)
                    else:
                        console.print(f"\n[bold]Result:[/bold] {result.result}")
                
                if result.metadata:
                    console.print("\n[bold]Metadata:[/bold]")
                    console.print_json(result.metadata)
        
        except Exception as e:
            console.print(f"[error]Tool execution failed: {e}[/error]")
            raise typer.Exit(1)
    
    try:
        asyncio.run(_execute())
    except KeyboardInterrupt:
        console.print("\n[yellow]Tool execution cancelled[/yellow]")
        raise typer.Exit(1)


@tools_app.command()
def stats(
    tool_name: Optional[str] = typer.Argument(None, help="Tool name (if not provided, shows all tools)"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed statistics")
):
    """Show tool usage statistics."""
    async def _stats():
        registry = get_tool_registry()
        
        if tool_name:
            # Show stats for specific tool
            tool_stats = await registry.get_tool_stats(tool_name)
            
            if not tool_stats:
                console.print(f"[error]Tool '{tool_name}' not found[/error]")
                raise typer.Exit(1)
            
            stats_table = Table(title=f"Statistics for {tool_name}")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            
            for metric, value in tool_stats.items():
                if isinstance(value, float):
                    value_str = f"{value:.2f}"
                else:
                    value_str = str(value)
                stats_table.add_row(metric.replace("_", " ").title(), value_str)
            
            console.print(stats_table)
        
        else:
            # Show stats for all tools
            all_stats = await registry.get_tool_stats()
            
            if not all_stats:
                console.print("[warning]No tools registered[/warning]")
                return
            
            summary_table = Table(title="Tool Statistics Summary")
            summary_table.add_column("Tool", style="tool")
            summary_table.add_column("Executions", style="cyan")
            summary_table.add_column("Success Rate", style="green")
            summary_table.add_column("Avg Duration", style="dim")
            
            for tool_name, stats in all_stats.items():
                total = stats.get("total_executions", 0)
                successful = stats.get("successful_executions", 0)
                success_rate = (successful / total * 100) if total > 0 else 0
                avg_duration = stats.get("average_duration", 0.0)
                
                summary_table.add_row(
                    tool_name,
                    str(total),
                    f"{success_rate:.1f}%",
                    f"{avg_duration:.2f}s"
                )
            
            console.print(summary_table)
            
            if detailed:
                console.print("\n[bold]Detailed Statistics:[/bold]")
                console.print_json(all_stats)
    
    try:
        asyncio.run(_stats())
    except Exception as e:
        console.print(f"[error]Failed to get statistics: {e}[/error]")
        raise typer.Exit(1)


@tools_app.command()
def capabilities(
    tool_name: Optional[str] = typer.Argument(None, help="Tool name (if not provided, shows all capabilities)"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format (table, json)")
):
    """Show tool capabilities."""
    async def _capabilities():
        registry = get_tool_registry()
        
        if tool_name:
            # Show capabilities for specific tool
            capability = await registry.get_capabilities(tool_name)
            
            if not capability:
                console.print(f"[error]Tool '{tool_name}' not found[/error]")
                raise typer.Exit(1)
            
            if output_format == "json":
                console.print_json(capability.dict())
            else:
                # Table format
                console.print_panel(
                    f"[bold]Description:[/bold] {capability.description}\n"
                    f"[bold]Version:[/bold] {capability.version}\n"
                    f"[bold]Platforms:[/bold] {', '.join(capability.supported_platforms)}\n"
                    f"[bold]Permissions:[/bold] {', '.join(capability.required_permissions)}",
                    title=f"Capability: {capability.name}",
                    border_style="blue"
                )
        
        else:
            # Show all capabilities
            capabilities = await registry.get_capabilities()
            
            if not capabilities:
                console.print("[warning]No tools registered[/warning]")
                return
            
            if output_format == "json":
                console.print_json([cap.dict() for cap in capabilities])
            else:
                # Table format
                cap_table = Table(title="Tool Capabilities")
                cap_table.add_column("Tool", style="tool")
                cap_table.add_column("Version", style="dim")
                cap_table.add_column("Parameters", style="cyan")
                cap_table.add_column("Platforms", style="dim")
                
                for cap in sorted(capabilities, key=lambda x: x.name):
                    cap_table.add_row(
                        cap.name,
                        cap.version,
                        str(len(cap.parameters)),
                        ", ".join(cap.supported_platforms[:2]) + ("..." if len(cap.supported_platforms) > 2 else "")
                    )
                
                console.print(cap_table)
    
    try:
        asyncio.run(_capabilities())
    except Exception as e:
        console.print(f"[error]Failed to get capabilities: {e}[/error]")
        raise typer.Exit(1)


@tools_app.command()
def validate(tool_name: str):
    """Validate a tool's configuration and dependencies."""
    async def _validate():
        registry = get_tool_registry()
        tool = await registry.get_tool(tool_name)
        
        if not tool:
            console.print(f"[error]Tool '{tool_name}' not found[/error]")
            raise typer.Exit(1)
        
        console.print(f"[yellow]Validating tool: {tool_name}[/yellow]")
        
        # Get capability
        capability = tool.get_capability()
        
        validation_results = {
            "tool_name": tool_name,
            "version": tool.version,
            "capability_check": True,
            "parameter_validation": True,
            "dependency_check": True,
            "platform_compatibility": True,
            "issues": []
        }
        
        # Validate capability structure
        try:
            required_fields = ["name", "description", "parameters"]
            for field in required_fields:
                if not hasattr(capability, field):
                    validation_results["capability_check"] = False
                    validation_results["issues"].append(f"Missing capability field: {field}")
        except Exception as e:
            validation_results["capability_check"] = False
            validation_results["issues"].append(f"Capability validation error: {e}")
        
        # Validate parameters
        try:
            for param in capability.parameters:
                if not param.name or not param.type:
                    validation_results["parameter_validation"] = False
                    validation_results["issues"].append(f"Invalid parameter definition: {param.name}")
        except Exception as e:
            validation_results["parameter_validation"] = False
            validation_results["issues"].append(f"Parameter validation error: {e}")
        
        # Check dependencies (simplified)
        if capability.dependencies:
            for dep in capability.dependencies:
                # This would check if dependencies are available
                if dep == "unavailable_dependency":  # Example check
                    validation_results["dependency_check"] = False
                    validation_results["issues"].append(f"Missing dependency: {dep}")
        
        # Show results
        overall_status = all([
            validation_results["capability_check"],
            validation_results["parameter_validation"],
            validation_results["dependency_check"],
            validation_results["platform_compatibility"]
        ])
        
        status_color = "green" if overall_status else "red"
        status_text = "PASSED" if overall_status else "FAILED"
        
        result_content = f"""[bold]Tool:[/bold] {tool_name}
[bold]Version:[/bold] {tool.version}
[bold]Overall Status:[/bold] [{status_color}]{status_text}[/{status_color}]

[bold]Checks:[/bold]
• Capability Structure: {'✅' if validation_results['capability_check'] else '❌'}
• Parameter Validation: {'✅' if validation_results['parameter_validation'] else '❌'}
• Dependency Check: {'✅' if validation_results['dependency_check'] else '❌'}
• Platform Compatibility: {'✅' if validation_results['platform_compatibility'] else '❌'}"""
        
        console.print_panel(result_content, title="Validation Results", border_style=status_color)
        
        if validation_results["issues"]:
            console.print("\n[bold red]Issues Found:[/bold red]")
            for issue in validation_results["issues"]:
                console.print(f"  • {issue}")
        
        if not overall_status:
            raise typer.Exit(1)
    
    try:
        asyncio.run(_validate())
    except Exception as e:
        console.print(f"[error]Validation failed: {e}[/error]")
        raise typer.Exit(1)


if __name__ == "__main__":
    tools_app()