"""AIDA Text User Interface (TUI) main application."""

import contextlib
import io
import sys

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Static

from .widgets.agents import AgentsWidget
from .widgets.chat import ChatWidget
from .widgets.resource_monitor import ResourceMonitor
from .widgets.tasks import TasksWidget


@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout/stderr during TUI execution."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class AIDATui(App):
    """AIDA Text User Interface Application."""

    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        height: 100%;
        width: 100%;
    }

    #left-panel {
        width: 66%;
        height: 100%;
        border-right: solid $primary;
    }

    #right-panel {
        width: 34%;
        height: 100%;
    }

    #resource-monitor {
        height: 33%;
        border-bottom: solid $primary;
        padding: 1;
    }

    #tasks-widget {
        height: 33%;
        border-bottom: solid $primary;
        padding: 1;
    }

    #agents-widget {
        height: 34%;
        padding: 1;
    }

    .widget-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+l", "clear_chat", "Clear Chat"),
        ("ctrl+t", "toggle_theme", "Toggle Theme"),
    ]

    def __init__(self):
        """Initialize the TUI application."""
        super().__init__()
        self.title = "AIDA - Advanced Intelligent Distributed Agent System"
        self.chat_widget: ChatWidget | None = None
        self.resource_monitor: ResourceMonitor | None = None
        self.tasks_widget: TasksWidget | None = None
        self.agents_widget: AgentsWidget | None = None

    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header()

        with Horizontal(id="main-container"):
            # Left panel - Chat interface (2/3 width)
            with Vertical(id="left-panel"):
                self.chat_widget = ChatWidget()
                yield self.chat_widget

            # Right panel - Monitoring widgets (1/3 width)
            with Vertical(id="right-panel"):
                # Resource monitor (top third)
                with Container(id="resource-monitor"):
                    yield Static("MEMORY  CPU  GPU", classes="widget-title")
                    self.resource_monitor = ResourceMonitor()
                    yield self.resource_monitor

                # Ongoing tasks (middle third)
                with Container(id="tasks-widget"):
                    yield Static("ONGOING TASKS", classes="widget-title")
                    self.tasks_widget = TasksWidget()
                    yield self.tasks_widget

                # Available agents (bottom third)
                with Container(id="agents-widget"):
                    yield Static("AVAILABLE AGENTS", classes="widget-title")
                    self.agents_widget = AgentsWidget()
                    yield self.agents_widget

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize components when the app mounts."""
        # Initialize chat session
        if self.chat_widget:
            await self.chat_widget.initialize()

        # Load available agents
        if self.agents_widget:
            await self.agents_widget.load_agents()

        # Start resource monitoring in background
        if self.resource_monitor:
            self.run_worker(self.resource_monitor.start_monitoring(), exclusive=True)

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_clear_chat(self) -> None:
        """Clear the chat history."""
        if self.chat_widget:
            self.chat_widget.clear_chat()

    def action_toggle_theme(self) -> None:
        """Toggle between light and dark themes."""
        self.dark = not self.dark


def run_tui():
    """Run the AIDA TUI application."""
    app = AIDATui()
    app.run()


if __name__ == "__main__":
    run_tui()
