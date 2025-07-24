"""AIDA Text User Interface (TUI) main application."""

import contextlib
import io
import sys

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header

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
        padding: 0 1 1 1;
    }

    #right-panel {
        width: 34%;
        height: 100%;
        padding: 0 1 1 1;
    }

    #chat-container {
        height: 100%;
        border: solid $primary;
        border-title-color: $primary;
        border-title-background: $surface;
        border-title-align: center;
        padding: 1;
        background: $panel;
    }

    #resource-monitor {
        height: 33%;
        border: solid $primary;
        border-title-color: $primary;
        border-title-background: $surface;
        border-title-align: center;
        padding: 1;
        margin-bottom: 1;
        background: $panel;
    }

    #tasks-widget {
        height: 33%;
        border: solid $primary;
        border-title-color: $primary;
        border-title-background: $surface;
        border-title-align: center;
        padding: 1;
        margin-bottom: 1;
        background: $panel;
    }

    #agents-widget {
        height: 33%;
        border: solid $primary;
        border-title-color: $primary;
        border-title-background: $surface;
        border-title-align: center;
        padding: 1;
        background: $panel;
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
                chat_container = Container(id="chat-container")
                chat_container.border_title = "Interactive Chat"
                self.chat_widget = ChatWidget()
                yield chat_container

            # Right panel - Monitoring widgets (1/3 width)
            with Vertical(id="right-panel"):
                # System Information (top third)
                resource_container = Container(id="resource-monitor")
                resource_container.border_title = "System Information"
                self.resource_monitor = ResourceMonitor()
                yield resource_container

                # Ongoing tasks (middle third)
                tasks_container = Container(id="tasks-widget")
                tasks_container.border_title = "Ongoing Tasks"
                self.tasks_widget = TasksWidget()
                yield tasks_container

                # Available agents (bottom third)
                agents_container = Container(id="agents-widget")
                agents_container.border_title = "Available Agents"
                self.agents_widget = AgentsWidget()
                yield agents_container

        yield Footer()

    async def on_mount(self) -> None:
        """Mount widgets to containers and initialize components."""
        # Mount widgets to their containers
        self.query_one("#chat-container").mount(self.chat_widget)
        self.query_one("#resource-monitor").mount(self.resource_monitor)
        self.query_one("#tasks-widget").mount(self.tasks_widget)
        self.query_one("#agents-widget").mount(self.agents_widget)

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
