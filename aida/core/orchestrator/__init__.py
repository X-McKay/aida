"""TODO orchestrator module for AIDA."""

from .config import OrchestratorConfig
from .models import ReplanReason, TodoPlan, TodoStatus, TodoStep
from .orchestrator import TodoOrchestrator
from .storage import PlanStorageManager

# Global orchestrator instance for backwards compatibility
_global_orchestrator = None


def get_todo_orchestrator(storage_dir: str = None) -> TodoOrchestrator:
    """Get a global TodoOrchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = TodoOrchestrator(storage_dir)
    return _global_orchestrator


# Alias for consistency
get_orchestrator = get_todo_orchestrator

__all__ = [
    "TodoOrchestrator",
    "TodoPlan",
    "TodoStep",
    "TodoStatus",
    "ReplanReason",
    "PlanStorageManager",
    "OrchestratorConfig",
    "get_todo_orchestrator",
    "get_orchestrator",
]
