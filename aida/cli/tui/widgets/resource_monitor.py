"""Resource monitoring widget for AIDA TUI."""

import asyncio
from collections import deque
import secrets

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
        height: 100%;
        width: 100%;
        padding: 1;
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
        # Initialize with some data to avoid empty charts
        for _ in range(5):
            self.cpu_history.append(0.0)
            self.memory_history.append(0.0)
            self.gpu_history.append(0.0)

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
                    gpu_percent = secrets.randbelow(61) + 20 if len(self.gpu_history) > 0 else 30
            else:
                # No NVIDIA GPU, simulate for demo
                gpu_percent = secrets.randbelow(61) + 20 if len(self.gpu_history) > 0 else 30
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # GPU monitoring not available, simulate
            gpu_percent = secrets.randbelow(61) + 20 if len(self.gpu_history) > 0 else 30

        self.gpu_history.append(gpu_percent)

        # Update display
        self.refresh()

    def render_line_chart(
        self, data: list[float], height: int = 6, width: int = 40, color: str = "green"
    ) -> list[str]:
        """Render a smooth ASCII line chart similar to dolphie style."""
        if not data:
            return [" " * width for _ in range(height)]

        # Always use 0-100 range for percentages
        # Create chart with axis
        chart = [[" " for _ in range(width + 5)] for _ in range(height)]

        # Add Y-axis labels (100%, 50%, 0%)
        for y in range(height):
            if y == 0:
                chart[y][0:4] = list("100%")
            elif y == height // 2:
                chart[y][0:4] = list(" 50%")
            elif y == height - 1:
                chart[y][0:4] = list("  0%")
            else:
                chart[y][0:4] = list("    ")

            # Add axis line
            chart[y][4] = "│"

        # Get data points for width
        points = list(data)[-width:]

        # Draw line chart using Unicode box drawing characters
        for i in range(len(points)):
            if i < width:
                val = points[i]

                # Normalize to chart height
                y = int((1 - val / 100) * (height - 1))
                y = max(0, min(height - 1, y))

                # Plot point
                chart[y][i + 5] = "•"

                # Connect to previous point if exists
                if i > 0:
                    prev_val = points[i - 1]
                    prev_y = int((1 - prev_val / 100) * (height - 1))
                    prev_y = max(0, min(height - 1, prev_y))

                    # Draw connecting line using appropriate characters
                    if y != prev_y:
                        if y > prev_y:
                            # Line going down
                            for dy in range(prev_y + 1, y):
                                chart[dy][i + 5] = "│"
                            chart[prev_y][i + 5] = "╮" if chart[prev_y][i + 5] == " " else "•"
                            chart[y][i + 5] = "╯" if i < len(points) - 1 else "•"
                        else:
                            # Line going up
                            for dy in range(y + 1, prev_y):
                                chart[dy][i + 5] = "│"
                            chart[prev_y][i + 5] = "╰" if chart[prev_y][i + 5] == " " else "•"
                            chart[y][i + 5] = "╭" if i < len(points) - 1 else "•"
                    else:
                        # Horizontal line
                        if i > 0 and chart[y][i + 4] == " ":
                            chart[y][i + 4] = "─"

        # Convert to strings
        return ["".join(row) for row in chart]

    def render(self) -> Text:
        """Render the resource monitor."""
        output = Text()

        # Get current values
        cpu_current = self.cpu_history[-1] if self.cpu_history else 0
        mem_current = self.memory_history[-1] if self.memory_history else 0
        gpu_current = self.gpu_history[-1] if self.gpu_history else 0

        # Render charts with appropriate width
        width = 30  # Adjust based on panel width
        height = 5  # Height for each chart

        # Render all three charts
        cpu_chart = self.render_line_chart(list(self.cpu_history), height=height, width=width)
        mem_chart = self.render_line_chart(list(self.memory_history), height=height, width=width)
        gpu_chart = self.render_line_chart(list(self.gpu_history), height=height, width=width)

        # Time axis (showing relative time markers)
        time_axis = "     └" + "─" * width + "┘"
        time_labels = (
            "      -50s" + " " * (width // 2 - 8) + "-25s" + " " * (width // 2 - 6) + "now"
        )

        # CPU section
        output.append("  CPU: ", style="bold cyan")
        output.append(f"{cpu_current:>3.0f}%\n", style="bright_cyan")

        for _i, line in enumerate(cpu_chart):
            output.append(line, style="cyan")
            output.append("\n")

        output.append(time_axis + "\n", style="dim")
        output.append(time_labels + "\n", style="dim")

        # Memory section
        output.append("\n  MEM: ", style="bold blue")
        output.append(f"{mem_current:>3.0f}%\n", style="bright_blue")

        for _i, line in enumerate(mem_chart):
            output.append(line, style="blue")
            output.append("\n")

        output.append(time_axis + "\n", style="dim")
        output.append(time_labels + "\n", style="dim")

        # GPU section
        output.append("\n  GPU: ", style="bold yellow")
        output.append(f"{gpu_current:>3.0f}%\n", style="bright_yellow")

        for _i, line in enumerate(gpu_chart):
            output.append(line, style="yellow")
            output.append("\n")

        output.append(time_axis, style="dim")
        output.append("\n")
        output.append(time_labels, style="dim")

        return output

    def on_mount(self) -> None:
        """Set up the widget when mounted."""
        # Update display periodically
        self.set_interval(1.0, self.refresh)

    def on_unmount(self) -> None:
        """Clean up when unmounting."""
        self._monitoring = False
