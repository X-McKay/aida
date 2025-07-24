"""Unit tests for TUI components."""

import pytest

from aida.cli.tui.app import AIDATui
from aida.cli.tui.widgets.agents import AgentsWidget
from aida.cli.tui.widgets.chat import ChatWidget
from aida.cli.tui.widgets.resource_monitor import ResourceMonitor
from aida.cli.tui.widgets.tasks import TasksWidget


class TestTUI:
    """Test TUI components."""

    @pytest.mark.asyncio
    async def test_tui_app_creation(self):
        """Test TUI app can be created."""
        app = AIDATui()
        assert app is not None
        assert app.title == "AIDA - Advanced Intelligent Distributed Agent System"

    def test_chat_widget_creation(self):
        """Test chat widget creation."""
        widget = ChatWidget()
        assert widget is not None

    def test_resource_monitor_creation(self):
        """Test resource monitor creation."""
        widget = ResourceMonitor()
        assert widget is not None
        assert widget.update_interval == 1.0

    def test_tasks_widget_creation(self):
        """Test tasks widget creation."""
        widget = TasksWidget()
        assert widget is not None
        assert len(widget.plans) == 0

    def test_agents_widget_creation(self):
        """Test agents widget creation."""
        widget = AgentsWidget()
        assert widget is not None
        assert len(widget.agents) == 0

    def test_agents_widget_data_operations(self):
        """Test agent widget data operations."""
        widget = AgentsWidget()

        # Direct data manipulation (without UI refresh)
        widget.agents["test_agent"] = {"type": "Test Agent", "status": "ready"}
        assert len(widget.agents) == 1
        assert widget.agents["test_agent"]["type"] == "Test Agent"
        assert widget.agents["test_agent"]["status"] == "ready"

        # Update status
        widget.agents["test_agent"]["status"] = "busy"
        assert widget.agents["test_agent"]["status"] == "busy"

        # Get ready agents
        widget.agents["agent2"] = {"type": "Agent2", "status": "ready"}
        ready_agents = widget.get_ready_agents()
        assert len(ready_agents) == 1
        assert "agent2" in ready_agents

        # Remove agent
        del widget.agents["test_agent"]
        assert len(widget.agents) == 1

    def test_tasks_widget_data_operations(self):
        """Test tasks widget data operations."""
        widget = TasksWidget()

        # Direct data manipulation (without UI refresh)
        widget.plans["task_001"] = {"description": "Test task", "status": "pending"}
        assert len(widget.plans) == 1
        assert widget.plans["task_001"]["description"] == "Test task"
        assert widget.plans["task_001"]["status"] == "pending"

        # Update task status
        widget.plans["task_001"]["status"] = "running"
        assert widget.plans["task_001"]["status"] == "running"

        # Get active plan count
        count = widget.get_active_plan_count()
        assert count == 1

        # Complete task
        widget.plans["task_001"]["status"] = "completed"
        count = widget.get_active_plan_count()
        assert count == 0

        # Remove task
        del widget.plans["task_001"]
        assert len(widget.plans) == 0
