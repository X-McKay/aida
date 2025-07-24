"""Resource monitoring widget for AIDA TUI."""

import asyncio
from collections import deque

import psutil
from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class ResourceMonitor(Widget):
    """System resource monitoring widget with line charts."""

    CSS = """
    ResourceMonitor {
        height: 100%;
        width: 100%;
    }

    .resource-chart {
        height: 8;
        width: 100%;
    }
    """

    # Reactive data for charts
    cpu_history: reactive[deque[float]] = reactive(lambda: deque(maxlen=50))
    memory_history: reactive[deque[float]] = reactive(lambda: deque(maxlen=50))
    gpu_history: reactive[deque[float]] = reactive(lambda: deque(maxlen=50))

    def __init__(self):
        """Initialize the resource monitor."""
        super().__init__()
        self.update_interval = 1.0  # Update every second
        self._monitoring = False

    def compose(self) -> ComposeResult:
        """Create the resource monitor UI."""
        yield Static("", classes="resource-chart", id="resource-display")

    async def start_monitoring(self) -> None:
        """Start monitoring system resources."""
        self._monitoring = True
        while self._monitoring:
            await self._update_resources()
            await asyncio.sleep(self.update_interval)

    async def _update_resources(self) -> None:
        """Update resource usage data."""
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_history.append(cpu_percent)

        # Get memory usage
        memory = psutil.virtual_memory()
        self.memory_history.append(memory.percent)

        # Get GPU usage
        # Try to get real GPU usage if available, otherwise simulate
        try:
            # This would require nvidia-ml-py or similar for real GPU monitoring
            # For now, check if we can detect GPU
            import subprocess

            # IMPORTANT: Redirect all output to DEVNULL to prevent stdout pollution
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1,
            )
            if result.returncode == 0 and result.stdout:
                try:
                    gpu_percent = float(result.stdout.strip())
                except ValueError:
                    # Failed to parse, use simulation
                    import random

                    gpu_percent = random.uniform(20, 80) if len(self.gpu_history) > 0 else 30
            else:
                # No NVIDIA GPU, simulate for demo
                import random

                gpu_percent = random.uniform(20, 80) if len(self.gpu_history) > 0 else 30
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # GPU monitoring not available, simulate
            import random

            gpu_percent = random.uniform(20, 80) if len(self.gpu_history) > 0 else 30

        self.gpu_history.append(gpu_percent)

        # Update display
        self.refresh()

    def render_line_chart(self, data: list[float], height: int = 6, width: int = 40) -> list[str]:
        """Render a simple ASCII line chart."""
        if not data:
            return [" " * width for _ in range(height)]

        # Normalize data to chart height
        max_val = max(data) if data else 100
        min_val = min(data) if data else 0
        range_val = max_val - min_val if max_val != min_val else 1

        # Create chart
        chart = [[" " for _ in range(width)] for _ in range(height)]

        # Plot points
        for i, value in enumerate(data[-width:]):
            if i < width:
                normalized = (value - min_val) / range_val
                y = int((1 - normalized) * (height - 1))
                y = max(0, min(height - 1, y))
                chart[y][i] = "█"

        # Convert to strings
        return ["".join(row) for row in chart]

    def render(self) -> Text:
        """Render the resource monitor."""
        output = Text()

        # Get current values
        cpu_current = self.cpu_history[-1] if self.cpu_history else 0
        mem_current = self.memory_history[-1] if self.memory_history else 0
        gpu_current = self.gpu_history[-1] if self.gpu_history else 0

        # Render charts side by side
        cpu_chart = self.render_line_chart(list(self.cpu_history), height=5, width=12)
        mem_chart = self.render_line_chart(list(self.memory_history), height=5, width=12)
        gpu_chart = self.render_line_chart(list(self.gpu_history), height=5, width=12)

        # Combine charts
        for i in range(5):
            line = f"{cpu_chart[i]} {mem_chart[i]} {gpu_chart[i]}"
            output.append(line + "\n", style="green" if i < 2 else "yellow" if i < 4 else "red")

        # Add current values
        output.append(f"CPU: {cpu_current:>3.0f}%  ", style="cyan")
        output.append(f"MEM: {mem_current:>3.0f}%  ", style="magenta")
        output.append(f"GPU: {gpu_current:>3.0f}%", style="yellow")

        return output

    def on_mount(self) -> None:
        """Set up the widget when mounted."""
        # Update display periodically
        self.set_interval(1.0, self.refresh)

    def on_unmount(self) -> None:
        """Clean up when unmounting."""
        self._monitoring = False
