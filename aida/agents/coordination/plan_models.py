"""Models for coordinator planning and execution.

These models were extracted from the legacy orchestrator system
and are now used by the coordinator-worker agent framework.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TodoStatus(str, Enum):
    """Status of a todo/task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class ReplanReason(str, Enum):
    """Reasons for replanning."""

    STEP_FAILED = "step_failed"
    DEPENDENCY_FAILED = "dependency_failed"
    USER_REQUEST = "user_request"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    TIMEOUT = "timeout"
    ERROR = "error"


class TodoStep(BaseModel):
    """A single step in the execution plan."""

    id: str
    description: str
    tool_name: str
    parameters: dict[str, Any]
    dependencies: list[str] = Field(default_factory=list)
    status: TodoStatus = TodoStatus.PENDING
    result: Any | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        # Convert datetime objects to ISO format strings
        if data.get("started_at"):
            data["started_at"] = data["started_at"].isoformat()
        if data.get("completed_at"):
            data["completed_at"] = data["completed_at"].isoformat()
        # Convert enum to string
        data["status"] = (
            data["status"].value if isinstance(data["status"], Enum) else data["status"]
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TodoStep":
        """Create from dictionary."""
        # Convert ISO strings back to datetime
        if data.get("started_at") and isinstance(data["started_at"], str):
            data["started_at"] = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at") and isinstance(data["completed_at"], str):
            data["completed_at"] = datetime.fromisoformat(data["completed_at"])
        # Convert string to enum
        if data.get("status") and isinstance(data["status"], str):
            data["status"] = TodoStatus(data["status"])
        return cls(**data)


class TodoPlan(BaseModel):
    """Complete execution plan with steps."""

    id: str
    user_request: str
    analysis: str
    expected_outcome: str
    context: dict[str, Any] = Field(default_factory=dict)
    steps: list[TodoStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    plan_version: int = 1

    @property
    def status(self) -> str:
        """Get overall plan status."""
        if not self.steps:
            return "empty"

        if all(step.status == TodoStatus.COMPLETED for step in self.steps):
            return "completed"
        elif any(step.status == TodoStatus.FAILED for step in self.steps):
            return "failed"
        elif any(step.status == TodoStatus.IN_PROGRESS for step in self.steps):
            return "in_progress"
        elif any(step.status == TodoStatus.BLOCKED for step in self.steps):
            return "blocked"
        else:
            return "pending"

    def get_progress(self) -> dict[str, int]:
        """Get plan progress statistics."""
        if not self.steps:
            return {"total": 0, "completed": 0, "failed": 0, "in_progress": 0}

        progress = {
            "total": len(self.steps),
            "completed": sum(1 for step in self.steps if step.status == TodoStatus.COMPLETED),
            "failed": sum(1 for step in self.steps if step.status == TodoStatus.FAILED),
            "in_progress": sum(1 for step in self.steps if step.status == TodoStatus.IN_PROGRESS),
        }
        return progress

    def get_next_executable_step(self) -> TodoStep | None:
        """Get the next step that can be executed.

        A step can be executed if:
        1. It's in PENDING status
        2. All its dependencies are COMPLETED

        Returns:
            Next executable step or None if no step can be executed
        """
        if not self.steps:
            return None

        # Build a map of step IDs to their status
        step_status = {step.id: step.status for step in self.steps}

        for step in self.steps:
            if step.status != TodoStatus.PENDING:
                continue

            # Check if all dependencies are completed
            dependencies_met = all(
                step_status.get(dep_id) == TodoStatus.COMPLETED for dep_id in step.dependencies
            )

            if dependencies_met:
                return step

        return None

    def should_replan(self) -> tuple[bool, ReplanReason | None]:
        """Check if the plan needs replanning.

        Returns:
            Tuple of (should_replan, reason)
        """
        # Check for failures
        failed_count = sum(1 for step in self.steps if step.status == TodoStatus.FAILED)
        if failed_count > 0:
            return True, ReplanReason.STEP_FAILED

        # Check for blocked steps
        blocked_count = sum(1 for step in self.steps if step.status == TodoStatus.BLOCKED)
        if blocked_count > 0:
            return True, ReplanReason.DEPENDENCY_FAILED

        return False, None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        data["created_at"] = data["created_at"].isoformat()
        data["last_updated"] = data["last_updated"].isoformat()
        data["steps"] = [
            step.to_dict() if hasattr(step, "to_dict") else step for step in self.steps
        ]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TodoPlan":
        """Create from dictionary."""
        if data.get("created_at") and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("last_updated") and isinstance(data["last_updated"], str):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])

        # Convert step dictionaries to TodoStep objects
        if data.get("steps"):
            data["steps"] = [
                TodoStep.from_dict(step) if isinstance(step, dict) else step
                for step in data["steps"]
            ]

        return cls(**data)

    def to_markdown(self) -> str:
        """Convert plan to markdown format for display."""
        lines = []
        lines.append(f"# Plan: {self.id}")
        lines.append(f"\n**Request:** {self.user_request}")
        lines.append(f"\n**Analysis:** {self.analysis}")
        lines.append(f"\n**Expected Outcome:** {self.expected_outcome}")
        lines.append(f"\n**Status:** {self.status}")

        if self.steps:
            lines.append("\n## Steps:")
            for i, step in enumerate(self.steps, 1):
                status_str = f"[{step.status.value.upper()}]"

                lines.append(f"\n{i}. {status_str} **{step.description}**")
                lines.append(f"   - Tool: {step.tool_name}")
                if step.parameters:
                    lines.append(f"   - Parameters: {step.parameters}")
                if step.dependencies:
                    lines.append(f"   - Dependencies: {step.dependencies}")
                if step.error:
                    lines.append(f"   - Error: {step.error}")

        return "\n".join(lines)
