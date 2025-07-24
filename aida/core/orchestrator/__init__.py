"""Compatibility layer for legacy orchestrator imports.

This module provides a compatibility shim for code that still imports
from the old orchestrator. It redirects to the new agent-based system.

TODO: Update all code to use the new agent system directly and remove this.
"""

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# For backward compatibility - import from new location
from aida.agents.coordination.plan_models import ReplanReason, TodoPlan, TodoStatus, TodoStep

# Import old classes for compatibility
try:
    from .config import OrchestratorConfig
    from .storage import PlanStorageManager
except ImportError:
    # These may be removed already
    OrchestratorConfig = None
    PlanStorageManager = None

# Global instances for compatibility
_coordinator_instance = None
_compatibility_warned = False


class TodoOrchestrator:
    """Compatibility wrapper to make CoordinatorAgent look like TodoOrchestrator."""

    def __init__(self, storage_dir: str | None = None):
        """Initialize compatibility wrapper."""
        from aida.agents.coordination.coordinator_agent import CoordinatorAgent
        from aida.agents.worker.coding_worker import CodingWorker

        self.storage_dir = storage_dir or ".aida/orchestrator"
        self.coordinator = None
        self.worker = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure coordinator and worker are initialized."""
        if not self._initialized:
            from aida.agents.coordination.coordinator_agent import CoordinatorAgent
            from aida.agents.coordination.models import CoordinatorConfig
            from aida.agents.worker.coding_worker import CodingWorker

            # Create coordinator with storage path
            config = CoordinatorConfig(agent_id="legacy_coordinator", storage_path=self.storage_dir)
            self.coordinator = CoordinatorAgent(config=config)
            await self.coordinator.start()

            # Create a default worker
            self.worker = CodingWorker("legacy_worker")
            await self.worker.start()

            # Wait for registration
            await asyncio.sleep(1)
            self._initialized = True

    async def execute_request(
        self, request: Any, context: dict[str, Any] = None, **kwargs
    ) -> dict[str, Any]:
        """Execute request using coordinator - handles both string and dict formats."""
        await self._ensure_initialized()

        # Handle string request (from chat.py)
        if isinstance(request, str):
            user_request = request
            context = context or {}
            return await self.coordinator.execute_user_request(user_request, context)

        # Handle dict request (legacy format)
        if isinstance(request, dict):
            # Convert old request format if needed
            if "prompt" in request and "task_type" not in request:
                # Old format - convert to new
                request = {
                    "task_type": "code_generation",
                    "specification": request["prompt"],
                    "context": request.get("context", {}),
                }
            return await self.coordinator.execute_request(request)

        raise ValueError(f"Invalid request type: {type(request)}")

    def execute_request_sync(self, request: dict[str, Any]) -> dict[str, Any]:
        """Synchronous wrapper for execute_request."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.execute_request(request))

    async def create_plan(
        self, user_request: str, context: dict[str, Any] = None
    ) -> dict[str, Any]:
        """Create a plan for a user request - compatibility method."""
        logger.warning(
            "create_plan() is deprecated - planning is now integrated into execute_request()"
        )
        # Return a dummy plan for compatibility
        return {
            "status": "success",
            "message": "Planning is now integrated into execute_request",
            "plan": TodoPlan(plan_id="dummy", user_request=user_request, steps=[]),
        }

    async def shutdown(self):
        """Clean up resources."""
        if self.worker:
            await self.worker.stop()
        if self.coordinator:
            await self.coordinator.stop()


def get_todo_orchestrator(storage_dir: str | None = None) -> TodoOrchestrator:
    """Get a global TodoOrchestrator instance (compatibility)."""
    global _coordinator_instance, _compatibility_warned

    if not _compatibility_warned:
        logger.warning(
            "Using legacy get_todo_orchestrator() - please update to use "
            "CoordinatorAgent directly from aida.agents.coordination"
        )
        _compatibility_warned = True

    if _coordinator_instance is None:
        _coordinator_instance = TodoOrchestrator(storage_dir)

    return _coordinator_instance


# Alias for consistency
get_orchestrator = get_todo_orchestrator

__all__ = [
    "TodoOrchestrator",
    "TodoPlan",
    "TodoStep",
    "TodoStatus",
    "ReplanReason",
    "get_todo_orchestrator",
    "get_orchestrator",
]

# Keep these for compatibility if they exist
if OrchestratorConfig is not None:
    __all__.append("OrchestratorConfig")
if PlanStorageManager is not None:
    __all__.append("PlanStorageManager")
