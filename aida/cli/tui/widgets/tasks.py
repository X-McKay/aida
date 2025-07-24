"""Tasks widget for AIDA TUI."""

from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class TaskItem(Static):
    """A single task item."""

    def __init__(self, task_id: str, description: str, status: str = "pending"):
        """Initialize a task item."""
        self.task_id = task_id
        self.status = status

        # Format task display
        status_symbol = "○" if status == "pending" else "●" if status == "running" else "✓"
        status_color = (
            "yellow" if status == "pending" else "green" if status == "running" else "dim green"
        )

        content = Text()
        content.append(f"{status_symbol} ", style=status_color)
        content.append(description, style="white")

        super().__init__(content)


class TasksWidget(Widget):
    """Widget for displaying ongoing tasks."""

    CSS = """
    TasksWidget {
        height: 100%;
        overflow-y: auto;
    }

    TaskItem {
        margin-bottom: 1;
    }

    .no-tasks {
        color: $text-disabled;
        text-align: center;
        margin-top: 2;
    }
    """

    tasks: reactive[dict[str, dict]] = reactive({})

    def compose(self) -> ComposeResult:
        """Create the tasks UI."""
        # Start with empty message
        yield Static("No active tasks", classes="no-tasks", id="no-tasks-message")

    def on_mount(self) -> None:
        """Initialize when mounted."""
        # Start monitoring tasks
        self.set_interval(2.0, self.update_tasks)

    async def update_tasks(self) -> None:
        """Update task list from coordinator."""
        try:
            # Try to get active tasks from coordinator

            # This is a simplified version - in real implementation,
            # we'd need a way to access the global coordinator instance
            # For now, we'll simulate some tasks
            await self.simulate_tasks()

        except Exception:
            # If coordinator not available, show demo tasks
            await self.simulate_tasks()

    async def simulate_tasks(self) -> None:
        """Simulate some tasks for demo purposes."""
        # Demo tasks that cycle through states
        demo_tasks = {
            "task_001": {"description": "Data analysis", "status": "running"},
            "task_002": {"description": "Web scraping", "status": "pending"},
            "task_003": {"description": "Document summarization", "status": "completed"},
        }

        # Update if changed
        if self.tasks != demo_tasks:
            self.tasks = demo_tasks
            self.refresh_task_display()

    def add_task(self, task_id: str, description: str, status: str = "pending") -> None:
        """Add a new task to the display."""
        self.tasks[task_id] = {"description": description, "status": status}
        self.refresh_task_display()

    def update_task_status(self, task_id: str, status: str) -> None:
        """Update the status of a task."""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = status
            self.refresh_task_display()

    def remove_task(self, task_id: str) -> None:
        """Remove a task from the display."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self.refresh_task_display()

    def refresh_task_display(self) -> None:
        """Refresh the task display."""
        # Clear current display
        for widget in self.query(TaskItem):
            widget.remove()

        no_tasks_msg = self.query_one("#no-tasks-message", Static)

        if self.tasks:
            # Hide no tasks message
            no_tasks_msg.display = False

            # Add task items
            for task_id, task_info in self.tasks.items():
                task_item = TaskItem(
                    task_id, task_info["description"], task_info.get("status", "pending")
                )
                self.mount(task_item)
        else:
            # Show no tasks message
            no_tasks_msg.display = True

    def get_active_task_count(self) -> int:
        """Get count of active (non-completed) tasks."""
        return sum(1 for task in self.tasks.values() if task.get("status") != "completed")
