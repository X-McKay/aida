"""Tests for agent module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aida.core.agent import Agent, AgentCapability, AgentConfig, BaseAgent
from aida.core.events import EventBus
from aida.core.memory import MemoryManager


class TestAgentCapability:
    """Test AgentCapability model."""

    def test_capability_creation(self):
        """Test creating agent capability."""
        cap = AgentCapability(
            name="test_capability",
            version="1.0.0",
            description="Test capability",
            parameters={"param1": "value1"},
            required_tools=["tool1", "tool2"],
        )

        assert cap.name == "test_capability"
        assert cap.version == "1.0.0"
        assert cap.description == "Test capability"
        assert cap.parameters == {"param1": "value1"}
        assert cap.required_tools == ["tool1", "tool2"]

    def test_capability_defaults(self):
        """Test capability with defaults."""
        cap = AgentCapability(name="test")

        assert cap.name == "test"
        assert cap.version == "1.0.0"
        assert cap.description is None
        assert cap.parameters == {}
        assert cap.required_tools == []


class TestAgentConfig:
    """Test AgentConfig model."""

    def test_config_creation(self):
        """Test creating agent config."""
        config = AgentConfig(
            agent_id="agent_123",
            name="Test Agent",
            description="Test agent description",
            capabilities=[AgentCapability(name="cap1"), AgentCapability(name="cap2")],
            protocols={"a2a": {"enabled": True}},
            tools=["tool1", "tool2"],
            memory_config={"max_entries": 1000},
            security_config={"auth_enabled": True},
            max_concurrent_tasks=5,
            heartbeat_interval=15.0,
        )

        assert config.agent_id == "agent_123"
        assert config.name == "Test Agent"
        assert len(config.capabilities) == 2
        assert config.max_concurrent_tasks == 5
        assert config.heartbeat_interval == 15.0

    def test_config_defaults(self):
        """Test config with defaults."""
        config = AgentConfig(name="Test Agent")

        assert config.agent_id is None
        assert config.name == "Test Agent"
        assert config.description is None
        assert config.capabilities == []
        assert config.protocols == {}
        assert config.tools == []
        assert config.memory_config == {}
        assert config.security_config == {}
        assert config.max_concurrent_tasks == 10
        assert config.heartbeat_interval == 30.0


class TestBaseAgent:
    """Test BaseAgent abstract class."""

    @pytest.fixture
    def mock_memory_manager(self):
        """Create a mock memory manager."""
        manager = MagicMock(spec=MemoryManager)
        manager.start = AsyncMock()
        manager.stop = AsyncMock()
        manager.cleanup = AsyncMock()
        manager.get_basic_stats = Mock(return_value={})
        return manager

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        bus = MagicMock(spec=EventBus)
        bus.emit = AsyncMock()
        bus.subscribe = Mock()
        return bus

    @pytest.fixture
    def agent_config(self):
        """Create test agent config."""
        return AgentConfig(
            agent_id="test_agent_123",
            name="Test Agent",
            capabilities=[AgentCapability(name="planning"), AgentCapability(name="execution")],
            max_concurrent_tasks=5,
        )

    @pytest.fixture
    def test_agent_class(self):
        """Create a test implementation of BaseAgent."""

        class TestAgent(BaseAgent):
            async def process_message(self, message):
                return {"processed": True}

            async def execute_task(self, task):
                return {"result": "completed"}

        return TestAgent

    @pytest.fixture
    def agent(self, test_agent_class, agent_config, mock_memory_manager, mock_event_bus):
        """Create a test agent instance."""
        with (
            patch("aida.core.agent.MemoryManager", return_value=mock_memory_manager),
            patch("aida.core.agent.EventBus", return_value=mock_event_bus),
        ):
            return test_agent_class(agent_config)

    def test_agent_initialization(self, agent, agent_config):
        """Test agent initialization."""
        assert agent.agent_id == "test_agent_123"
        assert agent.name == "Test Agent"
        assert agent.config == agent_config
        assert len(agent.capabilities) == 2
        assert "planning" in agent.capabilities
        assert "execution" in agent.capabilities
        assert agent._started is False
        assert len(agent._tasks) == 0
        assert len(agent._running_tasks) == 0

    @pytest.mark.asyncio
    async def test_start_agent(self, agent, mock_memory_manager, mock_event_bus):
        """Test starting the agent."""
        # Mock the internal loops to prevent infinite execution
        agent._message_loop = AsyncMock()
        agent._heartbeat_loop = AsyncMock()
        agent._maintenance_loop = AsyncMock()

        await agent.start()

        assert agent._started is True
        assert agent.state.status == "running"
        assert agent.state.started_at is not None

        # Check memory manager started
        mock_memory_manager.start.assert_called_once()

        # Check started event emitted
        mock_event_bus.emit.assert_called()
        event = mock_event_bus.emit.call_args[0][0]
        assert event.type == "agent.started"
        assert event.data["agent_id"] == "test_agent_123"

    @pytest.mark.asyncio
    async def test_stop_agent(self, agent, mock_memory_manager, mock_event_bus):
        """Test stopping the agent."""
        # Start the agent first
        agent._started = True
        agent.state.status = "running"

        # Create a real async task that we can cancel
        async def dummy_task():
            await asyncio.sleep(10)

        task1 = asyncio.create_task(dummy_task())
        agent._tasks.add(task1)

        await agent.stop()

        assert agent._started is False
        assert agent.state.status == "stopped"

        # Check task was cancelled
        assert task1.cancelled()

        # Check memory manager stopped
        mock_memory_manager.stop.assert_called_once()

        # Check stopped event emitted
        mock_event_bus.emit.assert_called()
        event = mock_event_bus.emit.call_args[0][0]
        assert event.type == "agent.stopped"

    @pytest.mark.asyncio
    async def test_send_message_no_protocol(self, agent):
        """Test sending message when protocol doesn't exist."""
        result = await agent.send_message("unknown_protocol", {"test": "message"})
        assert result is False

    @pytest.mark.asyncio
    async def test_add_capability(self, agent, mock_event_bus):
        """Test adding a new capability."""
        new_cap = AgentCapability(name="new_capability", version="2.0.0")

        await agent.add_capability(new_cap)

        assert "new_capability" in agent.capabilities
        assert agent.capabilities["new_capability"] == new_cap

        # Check capability added event
        mock_event_bus.emit.assert_called()
        event = mock_event_bus.emit.call_args[0][0]
        assert event.type == "agent.capability_added"
        assert event.data["capability"]["name"] == "new_capability"

    @pytest.mark.asyncio
    async def test_remove_capability(self, agent, mock_event_bus):
        """Test removing a capability."""
        # Add a capability first
        agent.capabilities["test_cap"] = AgentCapability(name="test_cap")

        await agent.remove_capability("test_cap")

        assert "test_cap" not in agent.capabilities

        # Check capability removed event
        mock_event_bus.emit.assert_called()
        event = mock_event_bus.emit.call_args[0][0]
        assert event.type == "agent.capability_removed"
        assert event.data["capability_name"] == "test_cap"

    def test_add_tool(self, agent):
        """Test adding a tool."""
        mock_tool = Mock()

        agent.add_tool("test_tool", mock_tool)

        assert "test_tool" in agent.tools
        assert agent.tools["test_tool"] == mock_tool

    def test_get_tool(self, agent):
        """Test getting a tool."""
        mock_tool = Mock()
        agent.tools["existing_tool"] = mock_tool

        # Get existing tool
        result = agent.get_tool("existing_tool")
        assert result == mock_tool

        # Get non-existent tool
        result = agent.get_tool("non_existent")
        assert result is None

    @pytest.mark.asyncio
    async def test_submit_task(self, agent):
        """Test submitting a task."""
        task = {"type": "test_task", "data": "test"}

        # Mock the execution method
        agent._execute_task_with_tracking = AsyncMock()

        task_id = await agent.submit_task(task)

        assert task_id is not None
        assert task["id"] == task_id
        assert task_id in agent._running_tasks

    @pytest.mark.asyncio
    async def test_get_task_status(self, agent):
        """Test getting task status."""
        # No task
        status = await agent.get_task_status("unknown_id")
        assert status is None

        # Running task
        mock_task = MagicMock()
        mock_task.done.return_value = False
        agent._running_tasks["task_123"] = mock_task

        status = await agent.get_task_status("task_123")
        assert status == "running"

        # Completed task
        mock_task.done.return_value = True
        mock_task.exception.return_value = None

        status = await agent.get_task_status("task_123")
        assert status == "completed"

        # Failed task
        mock_task.exception.return_value = Exception("Task failed")

        status = await agent.get_task_status("task_123")
        assert status == "failed"

    def test_get_stats(self, agent):
        """Test getting agent statistics."""
        agent.state.tasks_completed = 10
        agent._running_tasks = {"task1": Mock(), "task2": Mock()}

        stats = agent.get_stats()

        assert stats["agent_id"] == "test_agent_123"
        assert stats["name"] == "Test Agent"
        assert stats["status"] == agent.state.status
        assert stats["capabilities"] == ["planning", "execution"]
        assert stats["tools"] == []
        assert stats["active_tasks"] == 2
        assert stats["total_tasks"] == 12  # 10 completed + 2 active
        assert "memory_usage" in stats
        assert "protocols" in stats


class TestAgent:
    """Test concrete Agent implementation."""

    @pytest.fixture
    def agent_config(self):
        """Create test agent config."""
        return AgentConfig(agent_id="concrete_agent_123", name="Concrete Test Agent")

    @pytest.fixture
    def agent(self, agent_config):
        """Create concrete agent instance."""
        with patch("aida.core.agent.MemoryManager"), patch("aida.core.agent.EventBus"):
            return Agent(agent_config)

    @pytest.mark.asyncio
    async def test_process_message_task_request(self, agent):
        """Test processing task request message."""
        message = MagicMock()
        message.message_type = "task_request"
        message.payload = {"type": "test_task"}
        message.sender_id = "sender_123"

        # Mock submit_task
        agent.submit_task = AsyncMock(return_value="task_456")

        result = await agent.process_message(message)

        assert result is not None
        assert result.message_type == "task_response"
        assert result.payload["task_id"] == "task_456"
        assert result.recipient_id == "sender_123"

    @pytest.mark.asyncio
    async def test_process_message_capability_discovery(self, agent):
        """Test processing capability discovery message."""
        message = MagicMock()
        message.message_type = "capability_discovery"
        message.sender_id = "sender_123"

        agent.capabilities = {
            "cap1": AgentCapability(name="cap1", version="1.0"),
            "cap2": AgentCapability(name="cap2", version="2.0"),
        }

        result = await agent.process_message(message)

        assert result is not None
        assert result.message_type == "capability_response"
        assert len(result.payload["capabilities"]) == 2
        assert result.recipient_id == "sender_123"

    @pytest.mark.asyncio
    async def test_process_message_unknown_type(self, agent):
        """Test processing unknown message type."""
        message = MagicMock()
        message.message_type = "unknown_type"

        result = await agent.process_message(message)

        assert result is None

    @pytest.mark.asyncio
    async def test_execute_task_echo(self, agent):
        """Test executing echo task."""
        task = {"id": "task_123", "type": "echo", "data": "Hello, AIDA!"}

        result = await agent.execute_task(task)

        assert result == {"result": "Hello, AIDA!"}

    @pytest.mark.asyncio
    async def test_execute_task_compute(self, agent):
        """Test executing compute task."""
        task = {"id": "task_124", "type": "compute"}

        result = await agent.execute_task(task)

        assert result == {"result": "computation completed"}

    @pytest.mark.asyncio
    async def test_execute_task_unknown(self, agent):
        """Test executing unknown task type."""
        task = {"id": "task_125", "type": "unknown_task"}

        with pytest.raises(ValueError, match="Unknown task type: unknown_task"):
            await agent.execute_task(task)


if __name__ == "__main__":
    pytest.main([__file__])
