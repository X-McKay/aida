"""Status display components for AIDA CLI."""

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


class StatusDisplay:
    """Display system status information."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def show_system_status(self, status_data: dict[str, Any]) -> None:
        """Display comprehensive system status."""
        # Overall status panel
        overall_status = status_data.get("overall_status", "unknown")
        status_color = (
            "green"
            if overall_status == "healthy"
            else "yellow"
            if overall_status == "warning"
            else "red"
        )

        status_text = f"[bold {status_color}]{overall_status.upper()}[/bold {status_color}]"

        # Component status table
        components = status_data.get("components", {})

        table = Table(title="Component Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")

        for component, info in components.items():
            status = info.get("status", "unknown")
            details = info.get("details", "")

            if status == "healthy":
                status_display = "[green]✅ Healthy[/green]"
            elif status == "warning":
                status_display = "[yellow]⚠️  Warning[/yellow]"
            elif status == "error":
                status_display = "[red]❌ Error[/red]"
            else:
                status_display = "[dim]❓ Unknown[/dim]"

            table.add_row(component.replace("_", " ").title(), status_display, details)

        # Display panels
        status_panel = Panel(
            f"System Status: {status_text}\nLast Updated: {status_data.get('timestamp', 'Unknown')}",
            title="AIDA System",
            border_style=status_color,
        )

        self.console.print(status_panel)
        self.console.print(table)

        # Show any alerts or issues
        issues = status_data.get("issues", [])
        if issues:
            self.console.print("\n[bold red]Issues Detected:[/bold red]")
            for issue in issues:
                self.console.print(f"  • {issue}")

    def show_agent_status(self, agents: list[dict[str, Any]]) -> None:
        """Display agent status information."""
        if not agents:
            self.console.print("[yellow]No agents found[/yellow]")
            return

        table = Table(title="Agent Status", show_header=True)
        table.add_column("Agent", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Uptime", style="dim")
        table.add_column("Tasks", style="dim")
        table.add_column("CPU", style="dim")
        table.add_column("Memory", style="dim")

        for agent in agents:
            name = agent.get("name", "unknown")
            status = agent.get("status", "unknown")
            uptime = agent.get("uptime", "0s")
            tasks = str(agent.get("active_tasks", 0))
            cpu = f"{agent.get('cpu_usage', 0):.1f}%"
            memory = f"{agent.get('memory_usage', 0):.1f}MB"

            if status == "running":
                status_display = "[green]🟢 Running[/green]"
            elif status == "stopped":
                status_display = "[red]🔴 Stopped[/red]"
            elif status == "starting":
                status_display = "[yellow]🟡 Starting[/yellow]"
            else:
                status_display = "[dim]⚪ Unknown[/dim]"

            table.add_row(name, status_display, uptime, tasks, cpu, memory)

        self.console.print(table)

    def show_resource_usage(self, resources: dict[str, Any]) -> None:
        """Display system resource usage."""
        cpu_usage = resources.get("cpu_usage", 0)
        memory_usage = resources.get("memory_usage", 0)
        disk_usage = resources.get("disk_usage", 0)
        network_in = resources.get("network_in", 0)
        network_out = resources.get("network_out", 0)

        # Create progress bars for resource usage
        cpu_bar = self._create_usage_bar("CPU", cpu_usage, 80, 90)
        memory_bar = self._create_usage_bar("Memory", memory_usage, 70, 85)
        disk_bar = self._create_usage_bar("Disk", disk_usage, 80, 90)

        resource_text = f"""
{cpu_bar}
{memory_bar}
{disk_bar}

[bold]Network:[/bold]
  Inbound:  {network_in:.1f} MB/s
  Outbound: {network_out:.1f} MB/s
"""

        self.console.print(
            Panel(resource_text.strip(), title="Resource Usage", border_style="blue")
        )

    def _create_usage_bar(
        self, name: str, percentage: float, warning_threshold: float, critical_threshold: float
    ) -> str:
        """Create a usage bar with color coding."""
        bar_length = 20
        filled_length = int(bar_length * percentage / 100)

        if percentage >= critical_threshold:
            bar_color = "red"
        elif percentage >= warning_threshold:
            bar_color = "yellow"
        else:
            bar_color = "green"

        bar = "█" * filled_length + "░" * (bar_length - filled_length)

        return f"[bold]{name:6}:[/bold] [{bar_color}]{bar}[/{bar_color}] {percentage:5.1f}%"

    def show_health_summary(self, health_data: dict[str, Any]) -> None:
        """Display health check summary."""
        overall_health = health_data.get("overall_health", "unknown")
        checks_passed = health_data.get("checks_passed", 0)
        total_checks = health_data.get("total_checks", 0)

        if overall_health == "healthy":
            health_color = "green"
            health_icon = "✅"
        elif overall_health == "warning":
            health_color = "yellow"
            health_icon = "⚠️"
        else:
            health_color = "red"
            health_icon = "❌"

        summary_text = f"""
[bold {health_color}]{health_icon} Overall Health: {overall_health.upper()}[/bold {health_color}]

Health Checks: {checks_passed}/{total_checks} passed

Last Check: {health_data.get('last_check', 'Never')}
"""

        self.console.print(
            Panel(summary_text.strip(), title="Health Summary", border_style=health_color)
        )

        # Show detailed check results
        checks = health_data.get("detailed_checks", {})
        if checks:
            check_table = Table(title="Detailed Health Checks", show_header=True)
            check_table.add_column("Check", style="cyan")
            check_table.add_column("Result", justify="center")
            check_table.add_column("Details", style="dim")

            for check_name, result in checks.items():
                status = result.get("status", "unknown")
                details = result.get("details", "")

                if status == "pass":
                    status_display = "[green]✅ Pass[/green]"
                elif status == "warning":
                    status_display = "[yellow]⚠️  Warning[/yellow]"
                else:
                    status_display = "[red]❌ Fail[/red]"

                check_table.add_row(check_name.replace("_", " ").title(), status_display, details)

            self.console.print(check_table)
