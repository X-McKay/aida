"""Unit tests for TaskDispatcher."""

import asyncio

import pytest

from aida.agents.base import AgentConfig, BaseAgent
from aida.agents.coordination.dispatcher import (
    DispatchStrategy,
    RetryStrategy,
    TaskDispatcher,
)
from aida.agents.coordination.plan_models import TodoStep


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, agent_id: str, capabilities: list[str], success_rate: float = 1.0):
        config = AgentConfig(
            agent_id=agent_id,
            agent_type="mock_worker",
            capabilities=capabilities,
        )
        super().__init__(config)
        self.success_rate = success_rate
        self.execute_count = 0
        self.last_task_data = None

    async def handle_message(self, message):
        """Mock message handler."""
        return None

    async def execute_task(self, task_data: dict) -> dict:
        """Mock task execution."""
        self.execute_count += 1
        self.last_task_data = task_data

        # Simulate some work
        await asyncio.sleep(0.1)

        # Return success or failure based on success rate
        import secrets

        if secrets.randbelow(100) / 100 < self.success_rate:
            return {
                "success": True,
                "result": {"output": f"Task {task_data['step_id']} completed"},
            }
        else:
            return {
                "success": False,
                "error": "Simulated failure",
                "retriable": True,
            }


@pytest.mark.asyncio
class TestTaskDispatcher:
    """Test suite for TaskDispatcher."""

    async def test_round_robin_selection(self):
        """Test round-robin agent selection."""
        agents = [
            MockAgent("agent1", ["coding"]),
            MockAgent("agent2", ["testing"]),
            MockAgent("agent3", ["documentation"]),
        ]

        dispatcher = TaskDispatcher(
            agents=agents,
            dispatch_strategy=DispatchStrategy.ROUND_ROBIN,
        )

        step = TodoStep(
            id="step1",
            description="Test step",
            tool_name="coding",
            parameters={},
        )

        # Select agents multiple times
        selected_agents = []
        for _ in range(6):
            agent = dispatcher.select_agent(step)
            selected_agents.append(agent.agent_id)

        # Should cycle through agents
        assert selected_agents == ["agent1", "agent2", "agent3", "agent1", "agent2", "agent3"]

    async def test_capability_based_selection(self):
        """Test capability-based agent selection."""
        agents = [
            MockAgent("agent1", ["coding", "testing"]),
            MockAgent("agent2", ["documentation"]),
            MockAgent("agent3", ["testing", "documentation"]),
        ]

        dispatcher = TaskDispatcher(
            agents=agents,
            dispatch_strategy=DispatchStrategy.CAPABILITY_BASED,
        )

        # Test selecting agent for coding task
        step1 = TodoStep(
            id="step1",
            description="Write code",
            tool_name="coding",
            parameters={},
        )
        agent = dispatcher.select_agent(step1)
        assert agent.agent_id == "agent1"  # Only agent1 has coding capability

        # Test selecting agent for documentation task
        step2 = TodoStep(
            id="step2",
            description="Write docs",
            tool_name="documentation",
            parameters={},
        )
        agent = dispatcher.select_agent(step2)
        assert agent.agent_id in ["agent2", "agent3"]  # Both have documentation capability

    async def test_successful_dispatch(self):
        """Test successful task dispatch."""
        agent = MockAgent("agent1", ["coding"], success_rate=1.0)
        dispatcher = TaskDispatcher(agents=[agent])

        step = TodoStep(
            id="step1",
            description="Test step",
            tool_name="coding",
            parameters={"param1": "value1"},
        )

        result = await dispatcher.dispatch(step, context={"test": "context"})

        assert result.success is True
        assert result.task_id is not None
        assert result.step_id == "step1"
        assert result.retry_count == 0
        assert result.execution_time > 0
        assert agent.execute_count == 1

    async def test_retry_on_failure(self):
        """Test retry logic on failure."""
        # Agent that fails first 2 times, then succeeds
        agent = MockAgent("agent1", ["coding"], success_rate=0.0)

        dispatcher = TaskDispatcher(
            agents=[agent],
            retry_strategy=RetryStrategy.IMMEDIATE,
            max_retries=3,
        )

        step = TodoStep(
            id="step1",
            description="Test step",
            tool_name="coding",
            parameters={},
        )

        # Mock to control success on third attempt
        call_count = 0

        async def mock_execute(task_data):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                return {"success": True, "result": {"output": "Success on retry"}}
            return {"success": False, "error": "Failed", "retriable": True}

        agent.execute_task = mock_execute

        result = await dispatcher.dispatch(step)

        assert result.success is True
        assert result.retry_count == 2  # 0-indexed, so 2 means 3rd attempt
        assert call_count == 3

    async def test_non_retriable_failure(self):
        """Test handling of non-retriable failures."""
        agent = MockAgent("agent1", ["coding"])

        # Mock non-retriable failure
        async def mock_execute(task_data):
            return {
                "success": False,
                "error": "Non-retriable error",
                "retriable": False,
            }

        agent.execute_task = mock_execute

        dispatcher = TaskDispatcher(
            agents=[agent],
            max_retries=3,
        )

        step = TodoStep(
            id="step1",
            description="Test step",
            tool_name="coding",
            parameters={},
        )

        result = await dispatcher.dispatch(step)

        assert result.success is False
        assert result.retry_count == 0  # Should not retry
        assert result.error == "Non-retriable error"
        assert agent.execute_count == 0  # Mock execute was called once

    async def test_timeout_handling(self):
        """Test task timeout handling."""
        agent = MockAgent("agent1", ["coding"])

        # Mock slow execution
        async def mock_execute(task_data):
            await asyncio.sleep(2.0)  # Longer than timeout
            return {"success": True, "result": {"output": "Too slow"}}

        agent.execute_task = mock_execute

        dispatcher = TaskDispatcher(
            agents=[agent],
            timeout=0.5,  # 500ms timeout
            max_retries=0,  # No retries for this test
        )

        step = TodoStep(
            id="step1",
            description="Test step",
            tool_name="coding",
            parameters={},
        )

        result = await dispatcher.dispatch(step)

        assert result.success is False
        assert "timed out" in result.error.lower()

        # Check metrics
        metrics = dispatcher.get_metrics()
        assert metrics["timeouts"] == 1

    async def test_exponential_backoff(self):
        """Test exponential backoff retry strategy."""
        dispatcher = TaskDispatcher(
            agents=[MockAgent("agent1", ["coding"])],
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            initial_retry_delay=0.1,
        )

        # Test delay calculation
        assert dispatcher.calculate_retry_delay(1) == 0.1  # 0.1 * 2^0
        assert dispatcher.calculate_retry_delay(2) == 0.2  # 0.1 * 2^1
        assert dispatcher.calculate_retry_delay(3) == 0.4  # 0.1 * 2^2
        assert dispatcher.calculate_retry_delay(4) == 0.8  # 0.1 * 2^3

    async def test_metrics_tracking(self):
        """Test metrics tracking."""
        agents = [
            MockAgent("agent1", ["coding"], success_rate=0.5),  # 50% success rate
        ]

        dispatcher = TaskDispatcher(
            agents=agents,
            max_retries=1,
            retry_strategy=RetryStrategy.IMMEDIATE,
        )

        # Run multiple tasks
        for i in range(10):
            step = TodoStep(
                id=f"step{i}",
                description=f"Test step {i}",
                tool_name="coding",
                parameters={},
            )
            await dispatcher.dispatch(step)

        metrics = dispatcher.get_metrics()

        assert metrics["total_dispatched"] == 10
        assert metrics["successful"] > 0
        assert metrics["failed"] > 0
        assert metrics["successful"] + metrics["failed"] == 10
        assert metrics["retried"] > 0  # Some tasks should have been retried
        assert 0 < metrics["success_rate"] < 1
        assert 0 < metrics["failure_rate"] < 1

    async def test_no_agents_available(self):
        """Test handling when no agents are available."""
        dispatcher = TaskDispatcher(agents=[])

        step = TodoStep(
            id="step1",
            description="Test step",
            tool_name="coding",
            parameters={},
        )

        result = await dispatcher.dispatch(step)

        assert result.success is False
        assert "No suitable agent available" in result.error
        assert result.retriable is False
