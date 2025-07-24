"""Agents widget for AIDA TUI."""

from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class AgentItem(Static):
    """A single agent item."""

    def __init__(self, agent_id: str, agent_type: str, status: str = "ready"):
        """Initialize an agent item."""
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.status = status

        # Format agent display
        status_symbol = "●" if status == "ready" else "○" if status == "busy" else "×"
        status_color = "green" if status == "ready" else "yellow" if status == "busy" else "red"

        content = Text()
        content.append(f"{status_symbol} ", style=status_color)
        content.append(f"{agent_type}: ", style="cyan")
        content.append(agent_id, style="white")

        super().__init__(content)


class AgentsWidget(Widget):
    """Widget for displaying available agents."""

    CSS = """
    AgentsWidget {
        height: 100%;
        overflow-y: auto;
    }

    AgentItem {
        margin-bottom: 1;
    }

    .no-agents {
        color: $text-disabled;
        text-align: center;
        margin-top: 2;
    }
    """

    agents: reactive[dict[str, dict]] = reactive({})

    def compose(self) -> ComposeResult:
        """Create the agents UI."""
        # Start with loading message
        yield Static("Loading agents...", classes="no-agents", id="no-agents-message")

    async def load_agents(self) -> None:
        """Load available agents."""
        # In a real implementation, we'd query the coordinator for registered agents
        # For now, we'll show the standard agents

        self.agents = {
            "coordinator": {
                "type": "Coordinator",
                "status": "ready",
                "capabilities": ["planning", "task_delegation"],
            },
            "coding_worker_1": {
                "type": "Code Worker",
                "status": "ready",
                "capabilities": ["code_analysis", "code_generation"],
            },
            "test_analyzer": {
                "type": "Test Analyzer",
                "status": "ready",
                "capabilities": ["test_analysis", "coverage_report"],
            },
            "doc_generator": {
                "type": "Doc Generator",
                "status": "busy",
                "capabilities": ["documentation", "api_docs"],
            },
        }

        self.refresh_agent_display()

    def add_agent(self, agent_id: str, agent_type: str, status: str = "ready") -> None:
        """Add a new agent to the display."""
        self.agents[agent_id] = {"type": agent_type, "status": status}
        self.refresh_agent_display()

    def update_agent_status(self, agent_id: str, status: str) -> None:
        """Update the status of an agent."""
        if agent_id in self.agents:
            self.agents[agent_id]["status"] = status
            self.refresh_agent_display()

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the display."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.refresh_agent_display()

    def refresh_agent_display(self) -> None:
        """Refresh the agent display."""
        # Clear current display
        for widget in self.query(AgentItem):
            widget.remove()

        no_agents_msg = self.query_one("#no-agents-message", Static)

        if self.agents:
            # Hide no agents message
            no_agents_msg.display = False

            # Add agent items sorted by type
            sorted_agents = sorted(self.agents.items(), key=lambda x: (x[1]["type"], x[0]))

            for agent_id, agent_info in sorted_agents:
                agent_item = AgentItem(
                    agent_id, agent_info["type"], agent_info.get("status", "ready")
                )
                self.mount(agent_item)
        else:
            # Show no agents message
            no_agents_msg.display = True
            no_agents_msg.update("No agents available")

    def get_ready_agents(self) -> list[str]:
        """Get list of ready agents."""
        return [agent_id for agent_id, info in self.agents.items() if info.get("status") == "ready"]

    def on_mount(self) -> None:
        """Set up periodic updates."""
        # Update agent status periodically
        self.set_interval(5.0, self.update_agent_statuses)

    async def update_agent_statuses(self) -> None:
        """Update agent statuses periodically."""
        # In a real implementation, this would query actual agent status
        # For demo, we'll just ensure the display is current
        pass
