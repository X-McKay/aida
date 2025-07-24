"""Tasks widget for AIDA TUI - displays ongoing plans with clickable details."""

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Static

from aida.core.orchestrator import get_orchestrator


class PlanItem(Button):
    """A clickable plan item that shows plan details when clicked."""

    class Selected(Message):
        """Message sent when a plan is selected."""

        def __init__(self, plan_id: str, plan_data: dict):
            """Initialize the Selected message."""
            self.plan_id = plan_id
            self.plan_data = plan_data
            super().__init__()

    def __init__(self, plan_id: str, status: str, description: str, progress: dict):
        """Initialize a plan item."""
        self.plan_id = plan_id
        self.status = status
        self.description = description
        self.progress = progress

        # Format status icon and color
        status_icons = {
            "pending": ("⏳", "yellow"),
            "in_progress": ("🔄", "cyan"),
            "completed": ("✅", "green"),
            "failed": ("❌", "red"),
            "cancelled": ("⏹", "dim"),
        }

        icon, color = status_icons.get(status, ("○", "white"))

        # Create label with progress info
        completed = progress.get("completed", 0)
        total = progress.get("total", 0)
        progress_text = f"[{completed}/{total}]" if total > 0 else ""

        label = f"{icon} {plan_id} {progress_text} - {description[:40]}..."

        super().__init__(label, variant="primary" if status == "in_progress" else "default")
        self.tooltip = f"Click to view plan details for {plan_id}"

    def on_button_pressed(self) -> None:
        """Handle button press."""
        # Post message to parent
        self.post_message(
            self.Selected(
                self.plan_id,
                {
                    "status": self.status,
                    "description": self.description,
                    "progress": self.progress,
                },
            )
        )


class PlanDetailsView(ScrollableContainer):
    """Detailed view of a selected plan."""

    def __init__(self, plan_id: str = "", plan_data: dict | None = None):
        """Initialize the details view."""
        super().__init__()
        self.plan_id = plan_id
        self.plan_data = plan_data or {}
        self.can_focus = True

    def compose(self) -> ComposeResult:
        """Create the details view."""
        if not self.plan_id:
            yield Static("No plan selected", classes="no-selection")
            return

        with Vertical():
            # Header
            yield Static(f"Plan: {self.plan_id}", classes="plan-header")
            yield Static(
                f"Status: {self.plan_data.get('status', 'unknown')}", classes="plan-status"
            )

            # Progress
            progress = self.plan_data.get("progress", {})
            yield Static(
                f"Progress: {progress.get('completed', 0)}/{progress.get('total', 0)} steps",
                classes="plan-progress",
            )

            # Description
            yield Static("\nDescription:", classes="section-header")
            yield Static(
                self.plan_data.get("description", "No description"), classes="plan-description"
            )

            # Steps
            yield Static("\nSteps:", classes="section-header")

            # Show actual plan steps if available
            if "steps" in self.plan_data:
                for i, step in enumerate(self.plan_data["steps"], 1):
                    step_status = step.get("status", "pending")
                    step_icon = (
                        "✅"
                        if step_status == "completed"
                        else "🔄"
                        if step_status == "in_progress"
                        else "⏳"
                    )
                    yield Static(
                        f"{i}. {step_icon} {step.get('description', 'Unknown step')}",
                        classes="plan-step",
                    )
            else:
                yield Static("Step details not available", classes="no-steps")

            # Close button
            yield Button("Close Details [ESC]", variant="primary", id="close-details")


class TasksWidget(Widget):
    """Widget for displaying ongoing plans with interactive details."""

    CSS = """
    TasksWidget {
        height: 100%;
        layout: vertical;
    }

    #tasks-list {
        height: 50%;
        overflow-y: auto;
        border-bottom: thick $primary;
        padding-bottom: 1;
        margin-bottom: 1;
    }

    #plan-details {
        height: 50%;
        overflow-y: auto;
        padding-top: 1;
        background: $surface-darken-1;
        border: solid $primary-darken-2;
        padding: 1;
    }

    PlanItem {
        margin-bottom: 1;
        width: 100%;
    }

    .no-tasks {
        color: $text-disabled;
        text-align: center;
        margin-top: 2;
    }

    .no-selection {
        color: $text-disabled;
        text-align: center;
        margin-top: 2;
    }

    .plan-header {
        text-style: bold;
        color: $primary;
    }

    .plan-status {
        color: $text;
        margin-bottom: 1;
    }

    .plan-progress {
        color: $accent;
    }

    .section-header {
        text-style: bold;
        color: $primary-lighten-2;
        margin-top: 1;
    }

    .plan-description {
        color: $text;
        margin-left: 2;
    }

    .plan-step {
        color: $text;
        margin-left: 2;
        margin-bottom: 1;
    }

    .no-steps {
        color: $text-disabled;
        margin-left: 2;
    }

    #close-details {
        margin-top: 2;
        width: 20;
    }
    """

    plans: reactive[dict[str, dict]] = reactive({})
    selected_plan: reactive[str | None] = reactive(None)

    def compose(self) -> ComposeResult:
        """Create the tasks UI."""
        # List of plans
        with ScrollableContainer(id="tasks-list"):
            yield Static("No active plans", classes="no-tasks", id="no-tasks-message")

        # Plan details view
        with ScrollableContainer(id="plan-details"):
            self.details_view = PlanDetailsView()
            yield self.details_view

    def on_mount(self) -> None:
        """Initialize when mounted."""
        # Start monitoring plans
        self.set_interval(2.0, self.update_plans)

    async def update_plans(self) -> None:
        """Update plan list from coordinator."""
        try:
            # Get orchestrator instance
            orchestrator = get_orchestrator()

            # Get active plans
            if hasattr(orchestrator, "coordinator") and orchestrator.coordinator:
                coordinator = orchestrator.coordinator

                # Get all active plans
                active_plans = {}

                # Check stored plans
                if hasattr(coordinator, "_storage"):
                    stored_plans = coordinator._storage.list_plans(status="active")
                    for plan_summary in stored_plans:
                        plan_id = plan_summary["id"]
                        plan = coordinator.load_plan(plan_id)
                        if plan:
                            active_plans[plan_id] = {
                                "status": plan.status,
                                "description": plan.user_request,
                                "progress": plan.get_progress(),
                                "steps": [
                                    {
                                        "description": step.description,
                                        "status": step.status.value,
                                    }
                                    for step in plan.steps
                                ],
                            }

                # Update if changed
                if self.plans != active_plans:
                    self.plans = active_plans
                    self.refresh_plan_display()

        except Exception:
            # If coordinator not available, show demo plans
            await self.simulate_plans()

    async def simulate_plans(self) -> None:
        """Simulate some plans for demo purposes."""
        # Demo plans
        demo_plans = {
            "plan_001_demo": {
                "status": "in_progress",
                "description": "Analyze Python codebase and generate documentation",
                "progress": {"completed": 2, "total": 4, "failed": 0},
                "steps": [
                    {"description": "Scan codebase structure", "status": "completed"},
                    {"description": "Analyze code complexity", "status": "completed"},
                    {"description": "Generate API documentation", "status": "in_progress"},
                    {"description": "Create usage examples", "status": "pending"},
                ],
            },
            "plan_002_demo": {
                "status": "pending",
                "description": "Research and summarize recent AI papers",
                "progress": {"completed": 0, "total": 3, "failed": 0},
                "steps": [
                    {"description": "Search arxiv for papers", "status": "pending"},
                    {"description": "Download and parse PDFs", "status": "pending"},
                    {"description": "Generate summary report", "status": "pending"},
                ],
            },
        }

        # Update if changed
        if self.plans != demo_plans:
            self.plans = demo_plans
            self.refresh_plan_display()

    def refresh_plan_display(self) -> None:
        """Refresh the plan display."""
        # Get the list container
        list_container = self.query_one("#tasks-list", ScrollableContainer)

        # Clear current display (except no-tasks message)
        for widget in list_container.query(PlanItem):
            widget.remove()

        no_tasks_msg = self.query_one("#no-tasks-message", Static)

        if self.plans:
            # Hide no tasks message
            no_tasks_msg.display = False

            # Add plan items
            for plan_id, plan_info in self.plans.items():
                plan_item = PlanItem(
                    plan_id,
                    plan_info["status"],
                    plan_info["description"],
                    plan_info.get("progress", {}),
                )
                list_container.mount(plan_item)
        else:
            # Show no tasks message
            no_tasks_msg.display = True

    def on_plan_item_selected(self, message: PlanItem.Selected) -> None:
        """Handle plan selection."""
        self.selected_plan = message.plan_id

        # Get full plan data
        plan_data = self.plans.get(message.plan_id, message.plan_data)

        # Update details view
        details_container = self.query_one("#plan-details", ScrollableContainer)
        details_container.remove_children()

        # Create new details view
        self.details_view = PlanDetailsView(message.plan_id, plan_data)
        details_container.mount(self.details_view)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "close-details":
            # Clear selection and show empty details
            self.selected_plan = None
            details_container = self.query_one("#plan-details", ScrollableContainer)
            details_container.remove_children()
            self.details_view = PlanDetailsView()
            details_container.mount(self.details_view)

    def get_active_plan_count(self) -> int:
        """Get count of active (non-completed) plans."""
        return sum(
            1
            for plan in self.plans.values()
            if plan.get("status") not in ["completed", "cancelled"]
        )
