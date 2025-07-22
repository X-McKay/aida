"""Tests for orchestrator storage and plan management functionality."""

import tempfile
from unittest.mock import Mock, patch

import pytest

from aida.core.orchestrator import TodoOrchestrator, TodoPlan, TodoStatus, TodoStep


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def orchestrator_with_storage(temp_storage_dir):
    """Create an orchestrator with a temporary storage directory."""
    return TodoOrchestrator(storage_dir=temp_storage_dir)


@pytest.fixture
def sample_plan():
    """Create a sample plan for testing."""
    return TodoPlan(
        id="test_plan_001",
        user_request="Test request",
        analysis="Test analysis",
        expected_outcome="Test outcome",
        steps=[
            TodoStep(
                id="step_001",
                description="Test step 1",
                tool_name="thinking",
                parameters={"problem": "test"},
            ),
            TodoStep(
                id="step_002",
                description="Test step 2",
                tool_name="execution",
                parameters={"code": "print('test')"},
            ),
        ],
    )


class TestPlanManagement:
    """Test plan management functionality."""

    def test_load_plan_from_memory(self, orchestrator_with_storage, sample_plan):
        """Test loading a plan from memory cache."""
        # Add plan to active plans
        orchestrator_with_storage.active_plans[sample_plan.id] = sample_plan

        # Load should return from memory
        loaded_plan = orchestrator_with_storage.load_plan(sample_plan.id)
        assert loaded_plan is sample_plan
        assert loaded_plan.id == "test_plan_001"

    def test_load_plan_from_storage(self, orchestrator_with_storage, sample_plan):
        """Test loading a plan from storage."""
        # Save plan to storage
        orchestrator_with_storage.storage_manager.save_plan(sample_plan)

        # Ensure not in memory
        assert sample_plan.id not in orchestrator_with_storage.active_plans

        # Load should fetch from storage
        loaded_plan = orchestrator_with_storage.load_plan(sample_plan.id)
        assert loaded_plan is not None
        assert loaded_plan.id == sample_plan.id
        assert loaded_plan.user_request == sample_plan.user_request

        # Should now be in memory cache
        assert sample_plan.id in orchestrator_with_storage.active_plans

    def test_load_nonexistent_plan(self, orchestrator_with_storage):
        """Test loading a plan that doesn't exist."""
        loaded_plan = orchestrator_with_storage.load_plan("nonexistent_plan")
        assert loaded_plan is None

    def test_list_stored_plans(self, orchestrator_with_storage, sample_plan):
        """Test listing stored plans."""
        # Initially empty
        plans = orchestrator_with_storage.list_stored_plans()
        assert len(plans) == 0

        # Save a plan
        orchestrator_with_storage.storage_manager.save_plan(sample_plan)

        # Should now have one plan
        plans = orchestrator_with_storage.list_stored_plans()
        assert len(plans) == 1
        assert plans[0]["id"] == sample_plan.id

    def test_delete_plan(self, orchestrator_with_storage, sample_plan):
        """Test deleting a plan from memory and storage."""
        # Add to memory and storage
        orchestrator_with_storage.active_plans[sample_plan.id] = sample_plan
        orchestrator_with_storage.storage_manager.save_plan(sample_plan)

        # Delete plan
        success = orchestrator_with_storage.delete_plan(sample_plan.id)
        assert success is True

        # Should be removed from memory
        assert sample_plan.id not in orchestrator_with_storage.active_plans

        # Should be removed from storage
        loaded_plan = orchestrator_with_storage.load_plan(sample_plan.id)
        assert loaded_plan is None

    def test_get_plan_summary(self, orchestrator_with_storage, sample_plan):
        """Test getting plan summary."""
        # Add plan to memory
        orchestrator_with_storage.active_plans[sample_plan.id] = sample_plan

        # Get summary
        summary = orchestrator_with_storage.get_plan_summary(sample_plan.id)
        assert summary is not None
        assert summary["plan_id"] == sample_plan.id
        assert summary["total"] == 2
        assert summary["completed"] == 0

        # Mark one step as completed
        sample_plan.steps[0].status = TodoStatus.COMPLETED
        summary = orchestrator_with_storage.get_plan_summary(sample_plan.id)
        assert summary["completed"] == 1

    def test_get_plan_summary_nonexistent(self, orchestrator_with_storage):
        """Test getting summary for nonexistent plan."""
        summary = orchestrator_with_storage.get_plan_summary("nonexistent_plan")
        assert summary is None

    def test_display_plan(self, orchestrator_with_storage, sample_plan):
        """Test getting terminal display format."""
        # Add plan to memory
        orchestrator_with_storage.active_plans[sample_plan.id] = sample_plan

        # Get display
        display = orchestrator_with_storage.display_plan(sample_plan.id)
        assert display is not None
        assert "Test request" in display
        assert "Test step 1" in display
        assert "Test step 2" in display

    def test_archive_completed_plans(self, orchestrator_with_storage):
        """Test archiving completed plans."""
        # Mock the storage manager method
        orchestrator_with_storage.storage_manager.archive_completed_plans = Mock(return_value=3)

        # Call archive
        archived_count = orchestrator_with_storage.archive_completed_plans()
        assert archived_count == 3
        orchestrator_with_storage.storage_manager.archive_completed_plans.assert_called_once()

    def test_cleanup_old_plans(self, orchestrator_with_storage):
        """Test cleaning up old plans."""
        # Mock the storage manager method
        orchestrator_with_storage.storage_manager.cleanup_old_plans = Mock(return_value=5)

        # Call cleanup with default days
        cleaned_count = orchestrator_with_storage.cleanup_old_plans()
        assert cleaned_count == 5
        orchestrator_with_storage.storage_manager.cleanup_old_plans.assert_called_once_with(30)

        # Call cleanup with custom days
        cleaned_count = orchestrator_with_storage.cleanup_old_plans(days_old=7)
        assert cleaned_count == 5
        orchestrator_with_storage.storage_manager.cleanup_old_plans.assert_called_with(7)

    def test_export_summary_report(self, orchestrator_with_storage):
        """Test exporting summary report."""
        # Mock the storage manager method
        orchestrator_with_storage.storage_manager.export_plan_summary = Mock(
            return_value="report_path.json"
        )

        # Call export without file
        report_path = orchestrator_with_storage.export_summary_report()
        assert report_path == "report_path.json"
        orchestrator_with_storage.storage_manager.export_plan_summary.assert_called_once_with(None)

        # Call export with file
        report_path = orchestrator_with_storage.export_summary_report("custom_report.json")
        assert report_path == "report_path.json"
        orchestrator_with_storage.storage_manager.export_plan_summary.assert_called_with(
            "custom_report.json"
        )


class TestExecutionHelpers:
    """Test execution helper methods."""

    def test_should_retry_on_timeout(self, orchestrator_with_storage):
        """Test retry logic for timeout errors."""
        step = TodoStep(
            id="test_step",
            description="Test step",
            tool_name="test_tool",
            parameters={},
            max_retries=3,
        )

        # Timeout error should trigger retry
        timeout_error = Exception("Connection timeout")
        assert orchestrator_with_storage._should_retry(step, timeout_error) is True

        # Network error should trigger retry
        network_error = Exception("Connection error occurred")
        assert orchestrator_with_storage._should_retry(step, network_error) is True

        # Rate limit should trigger retry
        rate_limit_error = Exception("Rate limit exceeded")
        assert orchestrator_with_storage._should_retry(step, rate_limit_error) is True

        # Other errors should not trigger retry
        logic_error = Exception("Invalid parameters")
        assert orchestrator_with_storage._should_retry(step, logic_error) is False

    @pytest.mark.asyncio
    async def test_ensure_tools_initialized(self, orchestrator_with_storage):
        """Test tool initialization."""
        # Initially not initialized
        assert orchestrator_with_storage._tools_initialized is False

        # Mock initialize_default_tools
        with patch("aida.tools.base.initialize_default_tools") as mock_init:
            await orchestrator_with_storage._ensure_tools_initialized()

            # Should call initialization
            mock_init.assert_called_once()

            # Should mark as initialized
            assert orchestrator_with_storage._tools_initialized is True

            # Second call should not reinitialize
            await orchestrator_with_storage._ensure_tools_initialized()
            mock_init.assert_called_once()  # Still only called once


class TestJSONExtraction:
    """Test JSON extraction from LLM responses."""

    def test_extract_json_from_code_block(self, orchestrator_with_storage):
        """Test extracting JSON from markdown code blocks."""
        response = """Here's the plan:
```json
{
    "analysis": "Test analysis",
    "expected_outcome": "Test outcome",
    "execution_plan": []
}
```
That's the plan!"""

        json_str = orchestrator_with_storage._extract_json_from_response(response)
        assert (
            json_str
            == """{
    "analysis": "Test analysis",
    "expected_outcome": "Test outcome",
    "execution_plan": []
}"""
        )

    def test_extract_json_from_plain_code_block(self, orchestrator_with_storage):
        """Test extracting JSON from plain code blocks."""
        response = """Here's the plan:
```
{
    "analysis": "Test analysis",
    "expected_outcome": "Test outcome",
    "execution_plan": []
}
```"""

        json_str = orchestrator_with_storage._extract_json_from_response(response)
        assert '"analysis": "Test analysis"' in json_str

    def test_extract_json_from_plain_text(self, orchestrator_with_storage):
        """Test extracting JSON from plain text."""
        response = """{
    "analysis": "Test analysis",
    "expected_outcome": "Test outcome",
    "execution_plan": []
}"""

        json_str = orchestrator_with_storage._extract_json_from_response(response)
        assert json_str == response

    def test_extract_json_from_nested_objects(self, orchestrator_with_storage):
        """Test extracting JSON with nested objects."""
        response = (
            """Some text before {"analysis": "Test", "plan": {"steps": [{"id": 1}]}} and after"""
        )

        json_str = orchestrator_with_storage._extract_json_from_response(response)
        assert '"analysis": "Test"' in json_str
        assert '"steps"' in json_str

    def test_extract_json_invalid_response(self, orchestrator_with_storage):
        """Test handling invalid JSON responses."""
        response = "This is just plain text with no JSON"

        with pytest.raises(ValueError, match="No valid JSON found"):
            orchestrator_with_storage._extract_json_from_response(response)

    def test_extract_json_from_dict_response(self, orchestrator_with_storage):
        """Test handling responses that are already dicts."""
        response = {"analysis": "Test", "expected_outcome": "Test", "execution_plan": []}

        json_str = orchestrator_with_storage._extract_json_from_response(response)
        import json

        parsed = json.loads(json_str)
        assert parsed == response


class TestPlanValidation:
    """Test plan validation logic."""

    def test_validate_plan_data_valid(self, orchestrator_with_storage):
        """Test validating valid plan data."""
        plan_data = {
            "analysis": "Test analysis",
            "expected_outcome": "Test outcome",
            "execution_plan": [
                {"description": "Step 1", "tool": "thinking", "parameters": {"problem": "test"}}
            ],
        }

        # Should not raise
        orchestrator_with_storage._validate_plan_data(plan_data)

    def test_validate_plan_data_not_dict(self, orchestrator_with_storage):
        """Test validating non-dict plan data."""
        with pytest.raises(ValueError, match="not a valid dictionary"):
            orchestrator_with_storage._validate_plan_data("not a dict")

    def test_validate_plan_data_missing_fields(self, orchestrator_with_storage):
        """Test validating plan data with missing fields."""
        # Missing analysis
        with pytest.raises(ValueError, match="missing required 'analysis'"):
            orchestrator_with_storage._validate_plan_data(
                {"expected_outcome": "Test", "execution_plan": []}
            )

        # Missing execution_plan
        with pytest.raises(ValueError, match="missing required 'execution_plan'"):
            orchestrator_with_storage._validate_plan_data(
                {"analysis": "Test", "expected_outcome": "Test"}
            )

    def test_validate_plan_data_empty_plan(self, orchestrator_with_storage):
        """Test validating empty execution plan."""
        plan_data = {"analysis": "Test", "expected_outcome": "Test", "execution_plan": []}

        with pytest.raises(ValueError, match="Plan data missing required|no execution steps"):
            orchestrator_with_storage._validate_plan_data(plan_data)

    def test_validate_plan_data_too_many_steps(self, orchestrator_with_storage):
        """Test validating plan with too many steps."""
        plan_data = {
            "analysis": "Test",
            "expected_outcome": "Test",
            "execution_plan": [{"description": f"Step {i}", "tool": "test"} for i in range(51)],
        }

        with pytest.raises(ValueError, match="too many steps"):
            orchestrator_with_storage._validate_plan_data(plan_data)


if __name__ == "__main__":
    pytest.main([__file__])
