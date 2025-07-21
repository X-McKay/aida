"""Table creation utilities for AIDA CLI."""

from typing import Any

from rich.table import Table


def create_agent_table(agents: list[dict[str, Any]], title: str = "Agents") -> Table:
    """Create a table for displaying agent information."""
    table = Table(title=title, show_header=True, header_style="bold magenta")

    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Type", style="dim")
    table.add_column("Tasks", justify="right", style="dim")
    table.add_column("Uptime", style="dim")
    table.add_column("CPU", justify="right", style="dim")
    table.add_column("Memory", justify="right", style="dim")

    for agent in agents:
        name = agent.get("name", "unknown")
        status = agent.get("status", "unknown")
        agent_type = agent.get("type", "general")
        tasks = str(agent.get("active_tasks", 0))
        uptime = agent.get("uptime", "0s")
        cpu = f"{agent.get('cpu_usage', 0):.1f}%"
        memory = f"{agent.get('memory_usage', 0):.1f}MB"

        # Color code status
        if status == "running":
            status_display = "[green]🟢 Running[/green]"
        elif status == "stopped":
            status_display = "[red]🔴 Stopped[/red]"
        elif status == "starting":
            status_display = "[yellow]🟡 Starting[/yellow]"
        elif status == "error":
            status_display = "[red]💥 Error[/red]"
        else:
            status_display = "[dim]⚪ Unknown[/dim]"

        table.add_row(name, status_display, agent_type, tasks, uptime, cpu, memory)

    return table


def create_provider_table(providers: list[dict[str, Any]], title: str = "LLM Providers") -> Table:
    """Create a table for displaying LLM provider information."""
    table = Table(title=title, show_header=True, header_style="bold blue")

    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Model", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Response Time", justify="right", style="dim")
    table.add_column("Requests", justify="right", style="dim")
    table.add_column("Error Rate", justify="right", style="dim")
    table.add_column("Endpoint", style="dim")

    for provider in providers:
        name = provider.get("name", "unknown")
        model = provider.get("model", "unknown")
        status = provider.get("status", "unknown")
        response_time = f"{provider.get('response_time', 0):.0f}ms"
        requests = str(provider.get("requests", 0))
        error_rate = f"{provider.get('error_rate', 0):.2%}"
        endpoint = provider.get("endpoint", "")

        # Color code status
        if status == "healthy":
            status_display = "[green]✅ Healthy[/green]"
        elif status == "warning":
            status_display = "[yellow]⚠️  Warning[/yellow]"
        elif status == "error":
            status_display = "[red]❌ Error[/red]"
        elif status == "configured":
            status_display = "[blue]🔧 Configured[/blue]"
        else:
            status_display = "[dim]❓ Unknown[/dim]"

        table.add_row(name, model, status_display, response_time, requests, error_rate, endpoint)

    return table


def create_stats_table(stats: dict[str, Any], title: str = "Statistics") -> Table:
    """Create a table for displaying statistics."""
    table = Table(title=title, show_header=True, header_style="bold green")

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="green")
    table.add_column("Unit", style="dim")
    table.add_column("Description", style="dim")

    # Define metric formatting
    metric_formats = {
        "uptime": {"unit": "seconds", "format": lambda x: f"{x:.1f}"},
        "requests": {"unit": "count", "format": lambda x: f"{x:,}"},
        "response_time": {"unit": "ms", "format": lambda x: f"{x:.1f}"},
        "error_rate": {"unit": "%", "format": lambda x: f"{x:.2%}"},
        "cpu_usage": {"unit": "%", "format": lambda x: f"{x:.1f}"},
        "memory_usage": {"unit": "MB", "format": lambda x: f"{x:.1f}"},
        "disk_usage": {"unit": "GB", "format": lambda x: f"{x:.2f}"},
        "success_rate": {"unit": "%", "format": lambda x: f"{x:.1%}"},
    }

    for key, value in stats.items():
        if key.startswith("_"):  # Skip private fields
            continue

        metric_name = key.replace("_", " ").title()

        # Format value based on metric type
        if key in metric_formats:
            formatter = metric_formats[key]
            formatted_value = formatter["format"](value)
            unit = formatter["unit"]
        else:
            # Default formatting
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
            unit = ""

        # Generate description
        descriptions = {
            "uptime": "System uptime",
            "requests": "Total requests processed",
            "response_time": "Average response time",
            "error_rate": "Error percentage",
            "cpu_usage": "CPU utilization",
            "memory_usage": "Memory utilization",
            "disk_usage": "Disk space used",
            "success_rate": "Success percentage",
            "active_agents": "Currently running agents",
            "total_tasks": "Total tasks completed",
            "active_tasks": "Currently active tasks",
        }

        description = descriptions.get(key, "")

        table.add_row(metric_name, formatted_value, unit, description)

    return table


def create_tool_table(tools: list[dict[str, Any]], title: str = "Tools") -> Table:
    """Create a table for displaying tool information."""
    table = Table(title=title, show_header=True, header_style="bold purple")

    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Version", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Executions", justify="right", style="dim")
    table.add_column("Success Rate", justify="right", style="green")
    table.add_column("Avg Duration", justify="right", style="dim")
    table.add_column("Description", style="dim")

    for tool in tools:
        name = tool.get("name", "unknown")
        version = tool.get("version", "1.0.0")
        status = tool.get("status", "unknown")
        executions = tool.get("total_executions", 0)
        success_rate = tool.get("success_rate", 0)
        avg_duration = tool.get("avg_duration", 0)
        description = tool.get("description", "")

        # Color code status
        if status == "ready":
            status_display = "[green]✅ Ready[/green]"
        elif status == "busy":
            status_display = "[yellow]⏳ Busy[/yellow]"
        elif status == "error":
            status_display = "[red]❌ Error[/red]"
        else:
            status_display = "[dim]❓ Unknown[/dim]"

        # Format values
        executions_str = f"{executions:,}"
        success_rate_str = f"{success_rate:.1%}" if success_rate > 0 else "N/A"
        duration_str = f"{avg_duration:.2f}s" if avg_duration > 0 else "N/A"

        # Truncate description if too long
        if len(description) > 40:
            description = description[:37] + "..."

        table.add_row(
            name,
            version,
            status_display,
            executions_str,
            success_rate_str,
            duration_str,
            description,
        )

    return table


def create_health_table(checks: dict[str, dict[str, Any]], title: str = "Health Checks") -> Table:
    """Create a table for displaying health check results."""
    table = Table(title=title, show_header=True, header_style="bold cyan")

    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Last Check", style="dim")
    table.add_column("Response Time", justify="right", style="dim")
    table.add_column("Details", style="dim")

    for component, check_result in checks.items():
        status = check_result.get("status", "unknown")
        last_check = check_result.get("last_check", "Never")
        response_time = check_result.get("response_time", 0)
        details = check_result.get("details", "")

        # Color code status
        if status == "healthy":
            status_display = "[green]✅ Healthy[/green]"
        elif status == "warning":
            status_display = "[yellow]⚠️  Warning[/yellow]"
        elif status == "critical":
            status_display = "[red]💥 Critical[/red]"
        elif status == "unknown":
            status_display = "[dim]❓ Unknown[/dim]"
        else:
            status_display = f"[dim]{status}[/dim]"

        # Format response time
        response_time_str = f"{response_time:.0f}ms" if response_time > 0 else "N/A"

        # Truncate details if too long
        if len(details) > 50:
            details = details[:47] + "..."

        component_name = component.replace("_", " ").title()

        table.add_row(component_name, status_display, last_check, response_time_str, details)

    return table
