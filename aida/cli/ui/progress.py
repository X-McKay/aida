"""Progress tracking for AIDA CLI."""

import time

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn


class ProgressTracker:
    """Progress tracker for long-running operations."""

    def __init__(self, console: Console | None = None):
        """Initialize progress tracker with optional console and task storage."""
        self.console = console or Console()
        self.progress: Progress | None = None
        self.tasks: dict[str, TaskID] = {}

    def start(self) -> None:
        """Start the progress display."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        self.progress.start()

    def stop(self) -> None:
        """Stop the progress display."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.tasks.clear()

    def add_task(self, task_id: str, description: str, total: float | None = None) -> None:
        """Add a new task to track."""
        if self.progress:
            task = self.progress.add_task(description, total=total)
            self.tasks[task_id] = task

    def update_task(
        self,
        task_id: str,
        advance: float | None = None,
        completed: float | None = None,
        description: str | None = None,
    ) -> None:
        """Update task progress."""
        if self.progress and task_id in self.tasks:
            kwargs = {}
            if advance is not None:
                kwargs["advance"] = advance
            if completed is not None:
                kwargs["completed"] = completed
            if description is not None:
                kwargs["description"] = description

            self.progress.update(self.tasks[task_id], **kwargs)

    def complete_task(self, task_id: str) -> None:
        """Mark a task as completed."""
        if self.progress and task_id in self.tasks:
            self.progress.update(self.tasks[task_id], completed=100)

    def remove_task(self, task_id: str) -> None:
        """Remove a task from tracking."""
        if self.progress and task_id in self.tasks:
            self.progress.remove_task(self.tasks[task_id])
            del self.tasks[task_id]

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class SimpleProgress:
    """Simple progress indicator for basic operations."""

    def __init__(self, console: Console | None = None):
        """Initialize simple progress indicator with optional console."""
        self.console = console or Console()
        self.start_time = None

    def start(self, message: str) -> None:
        """Start progress with a message."""
        self.start_time = time.time()
        self.console.print(f"[yellow]{message}...[/yellow]")

    def update(self, message: str) -> None:
        """Update progress message."""
        self.console.print(f"[dim]  {message}[/dim]")

    def complete(self, message: str) -> None:
        """Complete progress with success message."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.console.print(f"[green]✅ {message} (completed in {elapsed:.1f}s)[/green]")
        else:
            self.console.print(f"[green]✅ {message}[/green]")
        self.start_time = None

    def error(self, message: str) -> None:
        """Complete progress with error message."""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.console.print(f"[red]❌ {message} (failed after {elapsed:.1f}s)[/red]")
        else:
            self.console.print(f"[red]❌ {message}[/red]")
        self.start_time = None
