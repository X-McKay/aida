"""Tests for the TODO-based orchestrator."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from aida.core.orchestrator import (
    ReplanReason,
    TodoOrchestrator,
    TodoPlan,
    TodoStatus,
    TodoStep,
    get_todo_orchestrator,
)
from aida.tools.base import ParameterSpec, ToolCapability, ToolResult


@pytest.fixture
def mock_tool_registry():
    """Mock tool registry."""
    registry = Mock()
    registry.list_tools = AsyncMock(return_value=["thinking", "execution"])

    # Mock thinking tool
    thinking_tool = Mock()
    thinking_capability = ToolCapability(
        name="thinking",
        description="Thinking and analysis tool",
        parameters=[
            ParameterSpec(
                name="problem", type="string", description="Problem to analyze", required=True
            ),
            ParameterSpec(
                name="reasoning_type",
                type="string",
                description="Type of reasoning",
                required=False,
            ),
        ],
    )
    thinking_tool.get_capability.return_value = thinking_capability
    thinking_tool.execute_async = AsyncMock(
        return_value=ToolResult(
            status="success", result={"analysis": "Test analysis"}, duration_seconds=1.0
        )
    )

    # Mock execution tool
    execution_tool = Mock()
    execution_capability = ToolCapability(
        name="execution",
        description="Code execution tool",
        parameters=[
            ParameterSpec(name="code", type="string", description="Code to execute", required=True),
            ParameterSpec(
                name="language", type="string", description="Programming language", required=False
            ),
        ],
    )
    execution_tool.get_capability.return_value = execution_capability
    execution_tool.execute_async = AsyncMock(
        return_value=ToolResult(
            status="success", result={"output": "Hello World", "exit_code": 0}, duration_seconds=0.5
        )
    )

    registry.get_tool = AsyncMock(
        side_effect=lambda name: {"thinking": thinking_tool, "execution": execution_tool}.get(name)
    )

    return registry


@pytest.fixture
def mock_chat_response():
    """Mock chat response."""
    response = Mock()
    response.content = ""
    return response


@pytest.fixture
def orchestrator(mock_tool_registry, mock_chat_response):
    """Create orchestrator with mocked dependencies."""
    with patch("aida.tools.base.get_tool_registry", return_value=mock_tool_registry), patch(
        "aida.llm.chat", new_callable=AsyncMock, return_value=mock_chat_response
    ):
        orch = TodoOrchestrator()
        orch._tools_initialized = True  # Skip initialization
        return orch


class TestTodoStep:
    """Test TodoStep functionality."""

    def test_todo_step_creation(self):
        """Test creating a TodoStep."""
        step = TodoStep(
            id="test_001",
            description="Test step",
            tool_name="thinking",
            parameters={"problem": "test problem"},
        )

        assert step.id == "test_001"
        assert step.description == "Test step"
        assert step.status == TodoStatus.PENDING
        assert step.retry_count == 0
        assert step.can_execute(set())

    def test_todo_step_dependencies(self):
        """Test step dependency checking."""
        step = TodoStep(
            id="test_002",
            description="Dependent step",
            tool_name="execution",
            parameters={"code": "print('hello')"},
            dependencies=["test_001"],
        )

        # Can't execute without dependency
        assert not step.can_execute(set())

        # Can execute with dependency completed
        assert step.can_execute({"test_001"})

    def test_markdown_formatting(self):
        """Test markdown TODO line generation."""
        step = TodoStep(
            id="test_001",
            description="Test markdown formatting",
            tool_name="thinking",
            parameters={},
        )

        # Pending
        assert step.to_markdown_line() == "- ⬜ Test markdown formatting"

        # In progress
        step.status = TodoStatus.IN_PROGRESS
        assert step.to_markdown_line() == "- 🔄 Test markdown formatting (In Progress)"

        # Completed
        step.status = TodoStatus.COMPLETED
        assert step.to_markdown_line() == "- ✅ Test markdown formatting"

        # Failed
        step.status = TodoStatus.FAILED
        step.error = "Test error"
        assert "❌" in step.to_markdown_line()
        assert "Test error" in step.to_markdown_line()


class TestTodoPlan:
    """Test TodoPlan functionality."""

    def test_todo_plan_creation(self):
        """Test creating a TodoPlan."""
        steps = [
            TodoStep(id="step_001", description="First step", tool_name="thinking", parameters={}),
            TodoStep(
                id="step_002", description="Second step", tool_name="execution", parameters={}
            ),
        ]

        plan = TodoPlan(
            id="plan_001",
            user_request="Test request",
            analysis="Test analysis",
            expected_outcome="Test outcome",
            steps=steps,
        )

        assert plan.id == "plan_001"
        assert len(plan.steps) == 2
        assert plan.plan_version == 1

    def test_progress_tracking(self):
        """Test progress calculation."""
        steps = [
            TodoStep(id="step_001", description="Step 1", tool_name="thinking", parameters={}),
            TodoStep(id="step_002", description="Step 2", tool_name="execution", parameters={}),
            TodoStep(id="step_003", description="Step 3", tool_name="thinking", parameters={}),
        ]

        plan = TodoPlan(
            id="plan_001", user_request="Test", analysis="", expected_outcome="", steps=steps
        )

        progress = plan.get_progress()
        assert progress["total"] == 3
        assert progress["completed"] == 0
        assert progress["percentage"] == 0.0
        assert progress["status"] == "pending"

        # Complete one step
        steps[0].status = TodoStatus.COMPLETED
        progress = plan.get_progress()
        assert progress["completed"] == 1
        assert progress["percentage"] == 33.3

        # Complete all steps
        for step in steps:
            step.status = TodoStatus.COMPLETED
        progress = plan.get_progress()
        assert progress["completed"] == 3
        assert progress["percentage"] == 100.0
        assert progress["status"] == "completed"

    def test_next_executable_step(self):
        """Test finding next executable step."""
        steps = [
            TodoStep(id="step_001", description="Step 1", tool_name="thinking", parameters={}),
            TodoStep(
                id="step_002",
                description="Step 2",
                tool_name="execution",
                parameters={},
                dependencies=["step_001"],
            ),
            TodoStep(id="step_003", description="Step 3", tool_name="thinking", parameters={}),
        ]

        plan = TodoPlan(
            id="plan_001", user_request="Test", analysis="", expected_outcome="", steps=steps
        )

        # Should get step 1 or 3 (no dependencies)
        next_step = plan.get_next_executable_step()
        assert next_step is not None
        assert next_step.id in ["step_001", "step_003"]

        # Complete step 1
        steps[0].status = TodoStatus.COMPLETED

        # Now step 2 should be executable
        next_step = plan.get_next_executable_step()
        assert next_step.id in ["step_002", "step_003"]

    def test_should_replan(self):
        """Test replan detection."""
        plan = TodoPlan(
            id="plan_001", user_request="Test", analysis="", expected_outcome="", steps=[]
        )

        # No replan needed initially
        should_replan, reason = plan.should_replan()
        assert not should_replan

        # Add failed step
        failed_step = TodoStep(
            id="step_001", description="Failed step", tool_name="thinking", parameters={}
        )
        failed_step.status = TodoStatus.FAILED
        plan.steps.append(failed_step)

        should_replan, reason = plan.should_replan()
        assert should_replan
        assert reason == ReplanReason.STEP_FAILED

    def test_markdown_generation(self):
        """Test markdown generation."""
        steps = [
            TodoStep(
                id="step_001", description="Analyze problem", tool_name="thinking", parameters={}
            ),
            TodoStep(
                id="step_002", description="Execute solution", tool_name="execution", parameters={}
            ),
        ]
        steps[0].status = TodoStatus.COMPLETED

        plan = TodoPlan(
            id="plan_001",
            user_request="Solve a problem",
            analysis="Need to analyze and execute",
            expected_outcome="Problem solved",
            steps=steps,
        )

        markdown = plan.to_markdown()
        assert "# Workflow Plan: Solve a problem" in markdown
        assert "✅ Analyze problem" in markdown
        assert "⬜ Execute solution" in markdown
        assert "Progress: 1/2" in markdown


class TestTodoOrchestrator:
    """Test TodoOrchestrator functionality."""

    @pytest.mark.asyncio
    async def test_create_plan(self, orchestrator, mock_chat_response):
        """Test plan creation."""
        # Mock LLM response
        mock_chat_response.content = """```json
{
    "analysis": "Need to analyze the problem",
    "expected_outcome": "Problem will be solved",
    "execution_plan": [
        {
            "description": "Analyze the problem",
            "tool": "thinking",
            "parameters": {
                "problem": "test problem",
                "reasoning_type": "systematic_analysis"
            }
        }
    ]
}
```"""

        plan = await orchestrator.create_plan("Solve test problem")

        assert plan is not None
        assert plan.user_request == "Solve test problem"
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "thinking"
        assert plan.analysis == "Need to analyze the problem"

    @pytest.mark.asyncio
    async def test_execute_step(self, orchestrator):
        """Test step execution."""
        step = TodoStep(
            id="test_001",
            description="Test thinking",
            tool_name="thinking",
            parameters={"problem": "test problem"},
        )

        plan = TodoPlan(
            id="plan_001", user_request="Test", analysis="", expected_outcome="", steps=[step]
        )

        result = await orchestrator._execute_step(step, plan)

        assert result["success"] is True
        assert step.status == TodoStatus.COMPLETED
        assert step.result is not None

    @pytest.mark.asyncio
    async def test_execute_step_with_error(self, orchestrator, mock_tool_registry):
        """Test step execution with error."""
        # Make tool execution fail
        mock_tool_registry.get_tool.return_value.execute_async.side_effect = Exception("Test error")

        step = TodoStep(
            id="test_001",
            description="Failing step",
            tool_name="thinking",
            parameters={"problem": "test problem"},
        )

        plan = TodoPlan(
            id="plan_001", user_request="Test", analysis="", expected_outcome="", steps=[step]
        )

        result = await orchestrator._execute_step(step, plan)

        assert result["success"] is False
        assert step.status == TodoStatus.FAILED
        assert step.error == "Test error"

    @pytest.mark.asyncio
    async def test_execute_plan_success(self, orchestrator, mock_llm_manager):
        """Test full plan execution."""
        # Create a simple plan
        steps = [
            TodoStep(
                id="step_001",
                description="Think",
                tool_name="thinking",
                parameters={"problem": "test"},
            )
        ]

        plan = TodoPlan(
            id="plan_001",
            user_request="Test request",
            analysis="Test analysis",
            expected_outcome="Test outcome",
            steps=steps,
        )

        result = await orchestrator.execute_plan(plan)

        assert result["status"] == "completed"
        assert len(result["results"]) == 1
        assert result["results"][0]["success"] is True

    @pytest.mark.asyncio
    async def test_replan_functionality(self, orchestrator, mock_chat_response):
        """Test replanning functionality."""
        # Mock replan response
        mock_chat_response.content = """```json
{
    "analysis": "Replanned analysis",
    "expected_outcome": "Updated outcome",
    "execution_plan": [
        {
            "description": "Retry with different approach",
            "tool": "thinking",
            "parameters": {
                "problem": "retry problem"
            }
        }
    ]
}
```"""

        # Create plan with failed step
        failed_step = TodoStep(
            id="step_001", description="Failed step", tool_name="thinking", parameters={}
        )
        failed_step.status = TodoStatus.FAILED
        failed_step.error = "Test failure"

        plan = TodoPlan(
            id="plan_001",
            user_request="Test",
            analysis="Original analysis",
            expected_outcome="Original outcome",
            steps=[failed_step],
        )

        old_version = plan.plan_version
        await orchestrator._replan(plan, ReplanReason.STEP_FAILED)

        assert plan.plan_version == old_version + 1
        assert len(plan.replan_history) == 1
        assert plan.replan_history[0]["reason"] == "step_failed"
        assert len(plan.steps) >= 1  # Should have the failed step plus new ones

    def test_global_instance(self):
        """Test global orchestrator instance."""
        orch1 = get_todo_orchestrator()
        orch2 = get_todo_orchestrator()
        assert orch1 is orch2  # Should be same instance


class TestIntegration:
    """Integration tests for the todo orchestrator."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, orchestrator, mock_chat_response):
        """Test complete workflow from creation to execution."""
        # Mock LLM responses
        mock_chat_response.content = """```json
{
    "analysis": "Multi-step problem solving",
    "expected_outcome": "Problem analyzed and solution executed",
    "execution_plan": [
        {
            "description": "Analyze the problem",
            "tool": "thinking",
            "parameters": {
                "problem": "complex problem",
                "reasoning_type": "systematic_analysis"
            }
        },
        {
            "description": "Execute the solution",
            "tool": "execution",
            "parameters": {
                "code": "print('Solution executed')",
                "language": "python"
            },
            "dependencies": ["step_000"]
        }
    ]
}
```"""

        # Create and execute plan
        plan = await orchestrator.create_plan("Solve a complex problem")
        result = await orchestrator.execute_plan(plan)

        assert result["status"] == "completed"
        assert len(result["results"]) == 2
        assert all(r["success"] for r in result["results"])

        # Check final state
        progress = plan.get_progress()
        assert progress["completed"] == 2
        assert progress["percentage"] == 100.0

        # Check markdown output
        markdown = result["final_markdown"]
        assert "✅" in markdown  # Should have completed checkboxes
        assert "100.0%" in markdown


if __name__ == "__main__":
    pytest.main([__file__])
