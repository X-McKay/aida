"""Tests for state management module."""

import asyncio
from datetime import datetime
import json

import pytest

from aida.core.state import (
    AgentState,
    AgentStatus,
    MemoryStateStore,
    StateManager,
    SystemState,
)


class TestAgentStatus:
    """Test AgentStatus enum."""

    def test_agent_status_values(self):
        """Test agent status enum values."""
        assert AgentStatus.INITIALIZING.value == "initializing"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.PAUSED.value == "paused"
        assert AgentStatus.STOPPING.value == "stopping"
        assert AgentStatus.STOPPED.value == "stopped"
        assert AgentStatus.ERROR.value == "error"


class TestAgentState:
    """Test AgentState model."""

    def test_agent_state_creation(self):
        """Test creating agent state."""
        now = datetime.utcnow()
        state = AgentState(
            agent_id="agent_123",
            status=AgentStatus.RUNNING,
            started_at=now,
            last_heartbeat=now,
            tasks_completed=10,
            tasks_failed=2,
            memory_usage={"heap": "100MB", "stack": "10MB"},
            error_count=1,
            last_error="Connection timeout",
            metadata={"version": "1.0.0"},
        )

        assert state.agent_id == "agent_123"
        assert state.status == AgentStatus.RUNNING
        assert state.started_at == now
        assert state.last_heartbeat == now
        assert state.tasks_completed == 10
        assert state.tasks_failed == 2
        assert state.memory_usage["heap"] == "100MB"
        assert state.error_count == 1
        assert state.last_error == "Connection timeout"
        assert state.metadata["version"] == "1.0.0"

    def test_agent_state_defaults(self):
        """Test agent state default values."""
        state = AgentState(agent_id="agent_123")

        assert state.status == AgentStatus.INITIALIZING
        assert state.started_at is None
        assert state.last_heartbeat is None
        assert state.tasks_completed == 0
        assert state.tasks_failed == 0
        assert state.memory_usage == {}
        assert state.error_count == 0
        assert state.last_error is None
        assert state.metadata == {}

    def test_agent_state_json_encoding(self):
        """Test agent state JSON encoding."""
        now = datetime.utcnow()
        state = AgentState(
            agent_id="agent_123", status=AgentStatus.RUNNING, started_at=now, last_heartbeat=now
        )

        # Should be able to convert to JSON
        state_json = state.model_dump_json()
        assert isinstance(state_json, str)
        assert "agent_123" in state_json
        assert "running" in state_json

        # Parse back and check datetime is properly encoded
        parsed = json.loads(state_json)
        assert parsed["started_at"] is not None
        assert parsed["last_heartbeat"] is not None


class TestSystemState:
    """Test SystemState model."""

    def test_system_state_creation(self):
        """Test creating system state."""
        agent1 = AgentState(agent_id="agent_1", status=AgentStatus.RUNNING)
        agent2 = AgentState(agent_id="agent_2", status=AgentStatus.PAUSED)

        now = datetime.utcnow()
        state = SystemState(
            system_id="system_123",
            agents={"agent_1": agent1, "agent_2": agent2},
            started_at=now,
            total_tasks_completed=100,
            total_tasks_failed=5,
            active_connections=3,
            system_metrics={"cpu": "50%", "memory": "2GB"},
        )

        assert state.system_id == "system_123"
        assert len(state.agents) == 2
        assert state.agents["agent_1"].status == AgentStatus.RUNNING
        assert state.agents["agent_2"].status == AgentStatus.PAUSED
        assert state.started_at == now
        assert state.total_tasks_completed == 100
        assert state.total_tasks_failed == 5
        assert state.active_connections == 3
        assert state.system_metrics["cpu"] == "50%"

    def test_system_state_defaults(self):
        """Test system state default values."""
        state = SystemState(system_id="system_123")

        assert state.agents == {}
        assert state.started_at is not None
        assert state.total_tasks_completed == 0
        assert state.total_tasks_failed == 0
        assert state.active_connections == 0
        assert state.system_metrics == {}

    def test_system_state_json_encoding(self):
        """Test system state JSON encoding."""
        agent = AgentState(agent_id="agent_1", status=AgentStatus.RUNNING)
        state = SystemState(system_id="system_123", agents={"agent_1": agent})

        # Should be able to convert to JSON
        state_json = state.model_dump_json()
        assert isinstance(state_json, str)
        assert "system_123" in state_json
        assert "agent_1" in state_json
        assert "running" in state_json


class TestMemoryStateStore:
    """Test MemoryStateStore class."""

    @pytest.fixture
    def store(self):
        """Create a memory state store."""
        return MemoryStateStore()

    @pytest.mark.asyncio
    async def test_set_and_get(self, store):
        """Test setting and getting values."""
        # Set string value
        await store.set("key1", "value1")
        value = await store.get("key1")
        assert value == "value1"

        # Set dict value
        await store.set("key2", {"nested": "value"})
        value = await store.get("key2")
        assert value == {"nested": "value"}

        # Get non-existent key
        value = await store.get("non_existent")
        assert value is None

    @pytest.mark.asyncio
    async def test_delete(self, store):
        """Test deleting values."""
        await store.set("key1", "value1")

        # Delete existing key
        success = await store.delete("key1")
        assert success is True
        assert await store.get("key1") is None

        # Delete non-existent key
        success = await store.delete("non_existent")
        assert success is False

    @pytest.mark.asyncio
    async def test_exists(self, store):
        """Test checking key existence."""
        await store.set("key1", "value1")

        assert await store.exists("key1") is True
        assert await store.exists("non_existent") is False

    @pytest.mark.asyncio
    async def test_keys(self, store):
        """Test getting keys."""
        # Add multiple keys
        await store.set("app:config:db", "value1")
        await store.set("app:config:cache", "value2")
        await store.set("app:state:agent1", "value3")
        await store.set("other:key", "value4")

        # Get all keys
        all_keys = await store.keys()
        assert len(all_keys) == 4

        # Get keys with pattern
        app_keys = await store.keys("app:*")
        assert len(app_keys) == 3
        assert all(k.startswith("app:") for k in app_keys)

        config_keys = await store.keys("app:config:*")
        assert len(config_keys) == 2
        assert all(k.startswith("app:config:") for k in config_keys)

    @pytest.mark.asyncio
    async def test_clear(self, store):
        """Test clearing all data."""
        # Add multiple keys
        await store.set("key1", "value1")
        await store.set("key2", "value2")
        await store.set("key3", "value3")

        # Clear
        await store.clear()

        # All should be gone
        assert await store.get("key1") is None
        assert await store.get("key2") is None
        assert await store.get("key3") is None
        assert len(await store.keys()) == 0


class TestStateManager:
    """Test StateManager class."""

    async def create_manager(self):
        """Create and initialize a state manager."""
        manager = StateManager(store=MemoryStateStore())
        await manager.initialize_system("test_system")
        return manager

    @pytest.mark.asyncio
    async def test_register_agent(self):
        """Test registering an agent."""
        manager = await self.create_manager()

        # Create an agent state
        agent_state = AgentState(agent_id="agent_123", metadata={"version": "1.0.0"})

        # Register the agent
        await manager.register_agent(agent_state)

        # Check agent state was created
        state = await manager.get_agent_state("agent_123")
        assert state is not None
        assert state.agent_id == "agent_123"
        assert state.status == AgentStatus.INITIALIZING
        assert state.metadata["version"] == "1.0.0"

        # Check system state was updated
        system_state = await manager.get_system_state()
        assert "agent_123" in system_state.agents

    @pytest.mark.asyncio
    async def test_update_agent_status(self):
        """Test updating agent status."""
        manager = await self.create_manager()
        # Register an agent
        agent_state = AgentState(agent_id="agent_123")
        await manager.register_agent(agent_state)

        # Update status
        await manager.update_agent_state(
            "agent_123", {"status": AgentStatus.RUNNING, "started_at": datetime.utcnow()}
        )

        # Check status was updated
        state = await manager.get_agent_state("agent_123")
        assert state.status == AgentStatus.RUNNING
        assert state.started_at is not None

        # Update non-existent agent should not raise error
        await manager.update_agent_state("non_existent", {"status": AgentStatus.RUNNING})

    @pytest.mark.asyncio
    async def test_heartbeat(self):
        """Test agent heartbeat."""
        manager = await self.create_manager()
        # Register an agent
        agent_state = AgentState(agent_id="agent_123")
        await manager.register_agent(agent_state)

        # Send heartbeat
        heartbeat_time = datetime.utcnow()
        await manager.update_agent_state("agent_123", {"last_heartbeat": heartbeat_time})

        # Check heartbeat was recorded
        state = await manager.get_agent_state("agent_123")
        assert state.last_heartbeat is not None
        assert state.last_heartbeat == heartbeat_time

    @pytest.mark.asyncio
    async def test_record_task_completion(self):
        """Test recording task completion."""
        manager = await self.create_manager()
        # Register an agent
        agent_state = AgentState(agent_id="agent_123")
        await manager.register_agent(agent_state)

        # Record successful task
        state = await manager.get_agent_state("agent_123")
        await manager.update_agent_state(
            "agent_123", {"tasks_completed": state.tasks_completed + 1}
        )

        state = await manager.get_agent_state("agent_123")
        assert state.tasks_completed == 1
        assert state.tasks_failed == 0

        # Record failed task
        await manager.update_agent_state("agent_123", {"tasks_failed": state.tasks_failed + 1})

        state = await manager.get_agent_state("agent_123")
        assert state.tasks_completed == 1
        assert state.tasks_failed == 1

        # Update system state totals manually
        await manager.get_system_state()
        await manager.update_system_metrics(
            {
                "total_tasks_completed": state.tasks_completed,
                "total_tasks_failed": state.tasks_failed,
            }
        )

    @pytest.mark.asyncio
    async def test_record_error(self):
        """Test recording agent errors."""
        manager = await self.create_manager()
        # Register an agent
        agent_state = AgentState(agent_id="agent_123")
        await manager.register_agent(agent_state)

        # Record error
        state = await manager.get_agent_state("agent_123")
        await manager.update_agent_state(
            "agent_123", {"error_count": state.error_count + 1, "last_error": "Connection timeout"}
        )

        state = await manager.get_agent_state("agent_123")
        assert state.error_count == 1
        assert state.last_error == "Connection timeout"

        # Record another error
        await manager.update_agent_state(
            "agent_123",
            {"error_count": state.error_count + 1, "last_error": "Memory limit exceeded"},
        )

        state = await manager.get_agent_state("agent_123")
        assert state.error_count == 2
        assert state.last_error == "Memory limit exceeded"

    @pytest.mark.asyncio
    async def test_update_memory_usage(self):
        """Test updating memory usage."""
        manager = await self.create_manager()
        # Register an agent
        agent_state = AgentState(agent_id="agent_123")
        await manager.register_agent(agent_state)

        # Update memory usage
        await manager.update_agent_state(
            "agent_123", {"memory_usage": {"heap": "150MB", "stack": "20MB", "total": "170MB"}}
        )

        state = await manager.get_agent_state("agent_123")
        assert state.memory_usage["heap"] == "150MB"
        assert state.memory_usage["stack"] == "20MB"
        assert state.memory_usage["total"] == "170MB"

    @pytest.mark.asyncio
    async def test_unregister_agent(self):
        """Test unregistering an agent."""
        manager = await self.create_manager()
        # Register an agent
        agent_state = AgentState(agent_id="agent_123")
        await manager.register_agent(agent_state)

        # Unregister (use remove_agent method)
        success = await manager.remove_agent("agent_123")
        assert success is True

        # Agent should be removed from system state
        system_state = await manager.get_system_state()
        assert "agent_123" not in system_state.agents

        # Individual state should be removed
        state = await manager.get_agent_state("agent_123")
        assert state is None

        # Unregister non-existent
        success = await manager.remove_agent("non_existent")
        assert success is False

    @pytest.mark.asyncio
    async def test_get_active_agents(self):
        """Test getting active agents."""
        manager = await self.create_manager()
        # Register multiple agents with different statuses
        agent1_state = AgentState(agent_id="agent_1")
        agent2_state = AgentState(agent_id="agent_2")
        agent3_state = AgentState(agent_id="agent_3")

        await manager.register_agent(agent1_state)
        await manager.register_agent(agent2_state)
        await manager.register_agent(agent3_state)

        await manager.update_agent_state("agent_1", {"status": AgentStatus.RUNNING})
        await manager.update_agent_state("agent_2", {"status": AgentStatus.RUNNING})
        await manager.update_agent_state("agent_3", {"status": AgentStatus.STOPPED})

        # Get all agents and filter active ones
        all_agents = await manager.get_all_agent_states()
        active = [a.agent_id for a in all_agents if a.status == AgentStatus.RUNNING]
        assert len(active) == 2
        assert "agent_1" in active
        assert "agent_2" in active
        assert "agent_3" not in active

    @pytest.mark.asyncio
    async def test_update_system_metrics(self):
        """Test updating system metrics."""
        manager = await self.create_manager()
        await manager.update_system_metrics(
            {
                "cpu_usage": "45%",
                "memory_total": "8GB",
                "memory_used": "3.5GB",
                "network_connections": 25,
            }
        )

        system_state = await manager.get_system_state()
        assert system_state.system_metrics["cpu_usage"] == "45%"
        assert system_state.system_metrics["memory_total"] == "8GB"
        assert system_state.system_metrics["memory_used"] == "3.5GB"
        assert system_state.system_metrics["network_connections"] == 25

    @pytest.mark.asyncio
    async def test_increment_connections(self):
        """Test incrementing/decrementing connections."""
        manager = await self.create_manager()
        # Increment
        system_state = await manager.get_system_state()
        await manager.update_system_metrics(
            {"active_connections": system_state.active_connections + 3}
        )

        system_state = await manager.get_system_state()
        assert system_state.system_metrics.get("active_connections", 0) == 3

        # Increment more
        await manager.update_system_metrics(
            {"active_connections": system_state.system_metrics.get("active_connections", 0) + 2}
        )

        system_state = await manager.get_system_state()
        assert system_state.system_metrics.get("active_connections", 0) == 5

        # Decrement
        await manager.update_system_metrics(
            {
                "active_connections": max(
                    0, system_state.system_metrics.get("active_connections", 0) - 2
                )
            }
        )

        system_state = await manager.get_system_state()
        assert system_state.system_metrics.get("active_connections", 0) == 3

    @pytest.mark.asyncio
    async def test_callbacks(self):
        """Test state change callbacks."""
        manager = await self.create_manager()
        callback_data = []

        def status_callback(agent_id: str, old_status: AgentStatus, new_status: AgentStatus):
            callback_data.append(
                {"type": "status", "agent_id": agent_id, "old": old_status, "new": new_status}
            )

        def error_callback(agent_id: str, error: str, count: int):
            callback_data.append(
                {"type": "error", "agent_id": agent_id, "error": error, "count": count}
            )

        # Subscribe to changes
        manager.subscribe_to_changes(
            lambda change_type, entity_id: callback_data.append(
                {"type": change_type, "entity_id": entity_id}
            )
        )

        # Trigger agent registration
        agent_state = AgentState(agent_id="agent_123")
        await manager.register_agent(agent_state)

        # Trigger status change
        await manager.update_agent_state("agent_123", {"status": AgentStatus.RUNNING})

        # Allow callbacks to execute
        await asyncio.sleep(0.01)

        # Check callbacks were called
        assert len(callback_data) >= 2

        # Check we got the expected change events
        change_types = [d["type"] for d in callback_data]
        assert "agent_registered" in change_types
        assert "agent_updated" in change_types

    @pytest.mark.asyncio
    async def test_state_persistence(self):
        """Test state persistence across restarts."""
        manager = await self.create_manager()
        # Register an agent
        agent_state = AgentState(agent_id="agent_123", metadata={"test": True})
        await manager.register_agent(agent_state)
        await manager.update_agent_state(
            "agent_123", {"status": AgentStatus.RUNNING, "tasks_completed": 1}
        )

        # Get current state
        original_state = await manager.get_agent_state("agent_123")
        original_system = await manager.get_system_state()

        # Create new manager with same store
        new_manager = StateManager(store=manager.store)

        # State should be preserved
        restored_state = await new_manager.get_agent_state("agent_123")
        restored_system = await new_manager.get_system_state()

        assert restored_state.agent_id == original_state.agent_id
        assert restored_state.status == original_state.status
        assert restored_state.tasks_completed == original_state.tasks_completed

        assert restored_system.system_id == original_system.system_id
        assert len(restored_system.agents) == len(original_system.agents)


class TestGlobalStateManager:
    """Test global state manager."""

    def test_state_manager_creation(self):
        """Test creating state manager instances."""
        manager1 = StateManager()
        manager2 = StateManager()
        # Each instance is independent
        assert manager1 is not manager2


if __name__ == "__main__":
    pytest.main([__file__])
