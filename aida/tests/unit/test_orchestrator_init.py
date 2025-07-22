"""Tests for orchestrator __init__ module."""

from unittest.mock import Mock, patch

from aida.core.orchestrator import (
    OrchestratorConfig,
    PlanStorageManager,
    ReplanReason,
    TodoOrchestrator,
    TodoPlan,
    TodoStatus,
    TodoStep,
    get_orchestrator,
    get_todo_orchestrator,
)


class TestOrchestratorInit:
    """Test orchestrator __init__ module functions."""

    def test_imports_available(self):
        """Test that all expected imports are available."""
        # Test that we can import all the expected classes/enums
        assert TodoOrchestrator is not None
        assert TodoPlan is not None
        assert TodoStep is not None
        assert TodoStatus is not None
        assert ReplanReason is not None
        assert PlanStorageManager is not None
        assert OrchestratorConfig is not None
        assert get_todo_orchestrator is not None
        assert get_orchestrator is not None

    @patch("aida.core.orchestrator.TodoOrchestrator")
    def test_get_todo_orchestrator_first_call(self, mock_orchestrator_class):
        """Test get_todo_orchestrator on first call creates new instance."""
        # Reset the global orchestrator to simulate first call
        import aida.core.orchestrator

        aida.core.orchestrator._global_orchestrator = None

        mock_instance = Mock()
        mock_orchestrator_class.return_value = mock_instance

        result = get_todo_orchestrator("/test/storage/dir")

        # Should create new orchestrator with provided storage_dir
        mock_orchestrator_class.assert_called_once_with("/test/storage/dir")
        assert result == mock_instance

    @patch("aida.core.orchestrator.TodoOrchestrator")
    def test_get_todo_orchestrator_subsequent_calls(self, mock_orchestrator_class):
        """Test get_todo_orchestrator on subsequent calls returns same instance."""
        # Reset the global orchestrator
        import aida.core.orchestrator

        aida.core.orchestrator._global_orchestrator = None

        mock_instance = Mock()
        mock_orchestrator_class.return_value = mock_instance

        # First call
        result1 = get_todo_orchestrator("/test/storage/dir")
        # Second call with different storage_dir (should be ignored)
        result2 = get_todo_orchestrator("/different/dir")

        # Should only create orchestrator once
        mock_orchestrator_class.assert_called_once_with("/test/storage/dir")
        assert result1 == mock_instance
        assert result2 == mock_instance
        assert result1 is result2

    @patch("aida.core.orchestrator.TodoOrchestrator")
    def test_get_todo_orchestrator_no_storage_dir(self, mock_orchestrator_class):
        """Test get_todo_orchestrator with None storage_dir."""
        # Reset the global orchestrator
        import aida.core.orchestrator

        aida.core.orchestrator._global_orchestrator = None

        mock_instance = Mock()
        mock_orchestrator_class.return_value = mock_instance

        result = get_todo_orchestrator(None)

        # Should create orchestrator with None
        mock_orchestrator_class.assert_called_once_with(None)
        assert result == mock_instance

    @patch("aida.core.orchestrator.TodoOrchestrator")
    def test_get_orchestrator_alias(self, mock_orchestrator_class):
        """Test that get_orchestrator is an alias for get_todo_orchestrator."""
        # Reset the global orchestrator
        import aida.core.orchestrator

        aida.core.orchestrator._global_orchestrator = None

        mock_instance = Mock()
        mock_orchestrator_class.return_value = mock_instance

        result = get_orchestrator("/test/dir")

        # Should behave exactly like get_todo_orchestrator
        mock_orchestrator_class.assert_called_once_with("/test/dir")
        assert result == mock_instance

    def test_get_orchestrator_same_as_get_todo_orchestrator(self):
        """Test that get_orchestrator and get_todo_orchestrator are the same function."""
        assert get_orchestrator is get_todo_orchestrator

    def test_global_orchestrator_persistence(self):
        """Test that global orchestrator persists between calls."""
        # This test is more integration-like but tests the global state behavior
        import aida.core.orchestrator

        # Store original state
        original_global = aida.core.orchestrator._global_orchestrator

        try:
            # Reset to None
            aida.core.orchestrator._global_orchestrator = None

            with patch("aida.core.orchestrator.TodoOrchestrator") as mock_class:
                mock_instance = Mock()
                mock_class.return_value = mock_instance

                # Multiple calls should return same instance
                result1 = get_todo_orchestrator()
                result2 = get_todo_orchestrator()
                result3 = get_orchestrator()

                # Only one creation call
                mock_class.assert_called_once()
                assert result1 is result2
                assert result2 is result3

        finally:
            # Restore original state
            aida.core.orchestrator._global_orchestrator = original_global
