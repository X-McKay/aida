"""Compatibility layer for legacy orchestrator imports.

This module provides a compatibility shim for code that still imports
from the old orchestrator. It redirects to the new agent-based system.

TODO: Update all code to use the new agent system directly and remove this.
"""

import asyncio
import logging
from typing import Any, cast

logger = logging.getLogger(__name__)

# For backward compatibility - import from new location
from aida.agents.coordination.plan_models import ReplanReason, TodoPlan, TodoStatus, TodoStep

# Legacy classes removed - set to None for compatibility
OrchestratorConfig = None
PlanStorageManager = None

# Global instances for compatibility
_coordinator_instance = None
_compatibility_warned = False


class TodoOrchestrator:
    """Compatibility wrapper to make CoordinatorAgent look like TodoOrchestrator."""

    def __init__(self, storage_dir: str | None = None):
        """Initialize compatibility wrapper."""
        self.storage_dir = storage_dir or ".aida/orchestrator"
        self.coordinator = None
        self.worker = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure coordinator and worker are initialized."""
        if not self._initialized:
            from aida.agents.base import AgentConfig
            from aida.agents.coordination.coordinator_agent import CoordinatorAgent
            from aida.agents.coordination.storage import CoordinatorPlanStorage
            from aida.agents.worker.coding_worker import CodingWorker

            # Create coordinator with storage path
            config = AgentConfig(
                agent_id="legacy_coordinator",
                agent_type="coordinator",
                capabilities=["planning", "task_delegation", "agent_coordination"],
            )
            self.coordinator = CoordinatorAgent(config=config)
            # Override storage to use the same directory
            self.coordinator._storage = CoordinatorPlanStorage(self.storage_dir)
            await self.coordinator.start()

            # Create a default worker
            self.worker = CodingWorker("legacy_worker")
            await self.worker.start()

            # For local operation, create a WorkerProxy for the worker
            from aida.agents.coordination.worker_proxy import WorkerProxy

            proxy = WorkerProxy(
                worker_id=self.worker.agent_id,
                capabilities=self.worker.config.capabilities,
                coordinator_agent=self.coordinator,
            )
            # For local operation, set the actual worker on the proxy
            proxy._worker = self.worker

            # Register the proxy with the coordinator
            self.coordinator._known_workers[self.worker.agent_id] = proxy
            await self.coordinator._update_dispatcher()

            # Wait for registration
            await asyncio.sleep(0.5)
            self._initialized = True

    async def execute_request(
        self, request: Any, context: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """Execute request using coordinator - handles both string and dict formats."""
        await self._ensure_initialized()

        # Handle string request (from chat.py)
        if isinstance(request, str):
            user_request = request
            context = context or {}

            try:
                # Create plan from user request
                plan = await self.coordinator.create_plan(user_request, context)

                # Execute the plan
                result = await self.coordinator.execute_plan(plan)

                # Return formatted result
                return {
                    "status": "completed",
                    "plan": plan,
                    "result": result,
                    "message": result.get("summary", "Task completed successfully"),
                }
            except Exception as e:
                logger.error(f"Request execution failed: {e}")
                return {
                    "status": "failed",
                    "error": str(e),
                    "message": f"Failed to execute request: {e}",
                }

        # Handle dict request (legacy format)
        if isinstance(request, dict):
            # Convert old request format to string if needed
            if "prompt" in request:
                return await self.execute_request(request["prompt"], request.get("context", {}))

            # Try to extract a user request string
            user_request = request.get("user_request") or request.get("specification", "")
            if user_request:
                return await self.execute_request(user_request, request.get("context", {}))

        raise ValueError(f"Invalid request type: {type(request)}")

    def execute_request_sync(self, request: dict[str, Any]) -> dict[str, Any]:
        """Synchronous wrapper for execute_request."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.execute_request(request))

    async def create_plan(
        self, user_request: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create a plan for a user request - compatibility method."""
        logger.warning(
            "create_plan() is deprecated - planning is now integrated into execute_request()"
        )
        # Return a dummy plan for compatibility
        return {
            "status": "success",
            "message": "Planning is now integrated into execute_request",
            "plan": TodoPlan(
                id="dummy",
                user_request=user_request,
                analysis="Deprecated method - planning integrated into execute_request",
                expected_outcome="N/A",
                steps=[],
            ),
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

    # Type checker needs explicit cast
    return cast(TodoOrchestrator, _coordinator_instance)


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
