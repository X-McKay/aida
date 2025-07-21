"""Pydantic data models for the TODO orchestrator."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from aida.tools.base import ToolResult


class TodoStatus(Enum):
    """Status of a TODO item."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReplanReason(Enum):
    """Reasons for replanning."""

    STEP_FAILED = "step_failed"
    USER_CLARIFICATION = "user_clarification"
    NEW_INFORMATION = "new_information"
    DEPENDENCY_CHANGED = "dependency_changed"
    PERIODIC_CHECK = "periodic_check"


class TodoStep(BaseModel):
    """A single TODO step in the workflow."""

    id: str
    description: str
    tool_name: str
    parameters: dict[str, Any]
    status: TodoStatus = TodoStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: ToolResult | None = None
    error: str | None = None
    dependencies: list[str] = Field(default_factory=list)  # IDs of steps this depends on
    retry_count: int = 0
    max_retries: int = 2

    class Config:
        """Pydantic configuration for TodoStep model."""

        arbitrary_types_allowed = True  # Allow ToolResult type

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, v):
        """Validate that description is not empty."""
        if not v or not v.strip():
            raise ValueError("Description cannot be empty")
        return v.strip()

    @field_validator("tool_name")
    @classmethod
    def tool_name_must_be_valid(cls, v):
        """Validate that tool name is not empty."""
        if not v or not v.strip():
            raise ValueError("Tool name cannot be empty")
        return v.strip()

    @field_validator("retry_count")
    @classmethod
    def retry_count_must_be_non_negative(cls, v):
        """Validate that retry count is non-negative."""
        if v < 0:
            raise ValueError("Retry count must be non-negative")
        return v

    @field_validator("max_retries")
    @classmethod
    def max_retries_must_be_non_negative(cls, v):
        """Validate that max retries is non-negative."""
        if v < 0:
            raise ValueError("Max retries must be non-negative")
        return v

    def to_markdown_line(self) -> str:
        """Convert to markdown TODO line."""
        checkbox = (
            "[x]"
            if self.status == TodoStatus.COMPLETED
            else (
                "[!]"
                if self.status == TodoStatus.FAILED
                else (
                    "[~]"
                    if self.status == TodoStatus.IN_PROGRESS
                    else "[-]"
                    if self.status == TodoStatus.SKIPPED
                    else "[ ]"
                )
            )
        )

        suffix = ""
        if self.status == TodoStatus.FAILED and self.error:
            suffix = (
                f" (Failed: {self.error[:50]}...)"
                if len(self.error) > 50
                else f" (Failed: {self.error})"
            )
        elif self.status == TodoStatus.IN_PROGRESS:
            suffix = " (In Progress)"
        elif self.retry_count > 0:
            suffix = f" (Retry {self.retry_count}/{self.max_retries})"

        return f"- {checkbox} {self.description}{suffix}"

    def can_execute(self, completed_steps: set[str]) -> bool:
        """Check if this step can be executed based on dependencies."""
        return all(dep_id in completed_steps for dep_id in self.dependencies)

    def is_terminal(self) -> bool:
        """Check if this step is in a terminal state."""
        return self.status in {TodoStatus.COMPLETED, TodoStatus.FAILED, TodoStatus.SKIPPED}

    def to_dict(self) -> dict[str, Any]:
        """Convert step to dictionary for storage."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert step to JSON for storage."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TodoStep":
        """Create step from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "TodoStep":
        """Create step from JSON string."""
        return cls.model_validate_json(json_str)

    def get_duration(self) -> float | None:
        """Get step execution duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def get_status_display(self) -> str:
        """Get a user-friendly status display."""
        if self.status == TodoStatus.COMPLETED:
            duration = self.get_duration()
            duration_str = f" ({duration:.1f}s)" if duration else ""
            return f"COMPLETED{duration_str}"
        elif self.status == TodoStatus.FAILED:
            retry_str = (
                f" (Retry {self.retry_count}/{self.max_retries})" if self.retry_count > 0 else ""
            )
            return f"FAILED{retry_str}"
        elif self.status == TodoStatus.IN_PROGRESS:
            return "IN_PROGRESS"
        elif self.status == TodoStatus.SKIPPED:
            return "SKIPPED"
        else:
            return "PENDING"


class TodoPlan(BaseModel):
    """A TODO-style workflow plan with progressive checking."""

    id: str
    user_request: str
    analysis: str
    expected_outcome: str
    steps: list[TodoStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    last_evaluated: datetime = Field(default_factory=datetime.utcnow)
    plan_version: int = 1
    replan_history: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration for TodoPlan model."""

        arbitrary_types_allowed = True

    @field_validator("user_request")
    @classmethod
    def user_request_must_not_be_empty(cls, v):
        """Validate that user request is not empty."""
        if not v or not v.strip():
            raise ValueError("User request cannot be empty")
        return v.strip()

    @field_validator("analysis")
    @classmethod
    def analysis_must_not_be_empty(cls, v):
        """Validate that analysis is not empty."""
        if not v or not v.strip():
            raise ValueError("Analysis cannot be empty")
        return v.strip()

    @field_validator("expected_outcome")
    @classmethod
    def expected_outcome_must_not_be_empty(cls, v):
        """Validate that expected outcome is not empty."""
        if not v or not v.strip():
            raise ValueError("Expected outcome cannot be empty")
        return v.strip()

    @field_validator("plan_version")
    @classmethod
    def plan_version_must_be_positive(cls, v):
        """Validate that plan version is positive."""
        if v < 1:
            raise ValueError("Plan version must be positive")
        return v

    def to_markdown(self) -> str:
        """Convert plan to markdown TODO format."""
        lines = [
            f"# Workflow Plan: {self.user_request}",
            "",
            f"**Analysis:** {self.analysis}",
            f"**Expected Outcome:** {self.expected_outcome}",
            f"**Created:** {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Version:** {self.plan_version}",
            "",
            "## TODO List",
            "",
        ]

        for step in self.steps:
            lines.append(step.to_markdown_line())

        # Add progress summary
        progress = self.get_progress()
        lines.extend(
            [
                "",
                "## Progress Summary",
                f"- **Status:** {progress['status'].title()}",
                f"- **Completed:** {progress['completed']}/{progress['total']} steps",
                f"- **Progress:** {progress['percentage']:.1f}%",
            ]
        )

        if progress["failed"] > 0:
            lines.append(f"- **Failed:** {progress['failed']} steps")

        return "\n".join(lines)

    def get_progress(self) -> dict[str, Any]:
        """Get current progress statistics."""
        total = len(self.steps)
        completed = sum(1 for step in self.steps if step.status == TodoStatus.COMPLETED)
        failed = sum(1 for step in self.steps if step.status == TodoStatus.FAILED)
        in_progress = sum(1 for step in self.steps if step.status == TodoStatus.IN_PROGRESS)
        pending = sum(1 for step in self.steps if step.status == TodoStatus.PENDING)

        if total == 0:
            percentage = 100
            status = "completed"
        else:
            percentage = (completed / total) * 100

            if failed > 0 and completed + failed == total:
                status = "failed" if completed == 0 else "partial_failure"
            elif completed == total:
                status = "completed"
            elif in_progress > 0:
                status = "in_progress"
            else:
                status = "pending"

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": pending,
            "percentage": percentage,
            "status": status,
        }

    def get_next_executable_step(self) -> TodoStep | None:
        """Get the next step that can be executed."""
        completed_step_ids = {step.id for step in self.steps if step.status == TodoStatus.COMPLETED}

        for step in self.steps:
            if step.status == TodoStatus.PENDING and step.can_execute(completed_step_ids):
                return step

        return None

    def get_failed_steps(self) -> list[TodoStep]:
        """Get list of failed steps."""
        return [step for step in self.steps if step.status == TodoStatus.FAILED]

    def should_replan(self) -> tuple[bool, ReplanReason | None]:
        """Check if the plan should be re-evaluated."""
        now = datetime.utcnow()

        # Check for failed steps
        failed_steps = self.get_failed_steps()
        if failed_steps:
            return True, ReplanReason.STEP_FAILED

        # Check for periodic re-evaluation
        completed_since_last = sum(
            1
            for step in self.steps
            if (
                step.status == TodoStatus.COMPLETED
                and step.completed_at
                and step.completed_at > self.last_evaluated
            )
        )

        time_since_evaluation = (now - self.last_evaluated).total_seconds()

        # Use configuration values
        from .config import OrchestratorConfig

        if (
            completed_since_last >= OrchestratorConfig.REPLAN_AFTER_STEPS
            or time_since_evaluation > OrchestratorConfig.REPLAN_AFTER_SECONDS
        ):
            return True, ReplanReason.PERIODIC_CHECK

        return False, None

    def to_dict(self) -> dict[str, Any]:
        """Convert plan to dictionary for storage."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert plan to JSON for storage."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TodoPlan":
        """Create plan from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "TodoPlan":
        """Create plan from JSON string."""
        return cls.model_validate_json(json_str)

    def to_terminal_display(self) -> str:
        """Convert plan to terminal-friendly display format."""
        progress = self.get_progress()

        lines = [
            f"Plan: {self.user_request}",
            f"Progress: {progress['completed']}/{progress['total']} ({progress['percentage']:.1f}%)",
            f"Expected: {self.expected_outcome}",
            "",
            "Tasks:",
        ]

        for i, step in enumerate(self.steps, 1):
            status_icon = {
                TodoStatus.PENDING: "[ ]",
                TodoStatus.IN_PROGRESS: "[~]",
                TodoStatus.COMPLETED: "[x]",
                TodoStatus.FAILED: "[!]",
                TodoStatus.SKIPPED: "[-]",
            }.get(step.status, "[?]")

            task_line = f"  {i:2d}. {status_icon} {step.description}"
            if step.status == TodoStatus.FAILED and step.error:
                task_line += (
                    f" (Error: {step.error[:30]}...)"
                    if len(step.error) > 30
                    else f" (Error: {step.error})"
                )
            elif step.retry_count > 0:
                task_line += f" (Retry {step.retry_count}/{step.max_retries})"

            lines.append(task_line)

        if progress["status"] == "completed":
            lines.append("\nAll tasks completed!")
        elif progress["status"] == "failed":
            lines.append(f"\n{progress['failed']} task(s) failed")
        elif progress["status"] == "in_progress":
            lines.append("\nCurrently in progress...")

        return "\n".join(lines)

    def get_summary_stats(self) -> dict[str, Any]:
        """Get detailed summary statistics."""
        progress = self.get_progress()

        # Calculate timing information
        duration = None
        if self.steps:
            first_start = min((s.started_at for s in self.steps if s.started_at), default=None)
            last_complete = max(
                (s.completed_at for s in self.steps if s.completed_at), default=None
            )
            if first_start and last_complete:
                duration = (last_complete - first_start).total_seconds()

        # Get failed steps details
        failed_steps = [
            {"id": step.id, "description": step.description, "error": step.error}
            for step in self.steps
            if step.status == TodoStatus.FAILED
        ]

        return {
            **progress,
            "plan_id": self.id,
            "user_request": self.user_request,
            "created_at": self.created_at.isoformat(),
            "plan_version": self.plan_version,
            "duration_seconds": duration,
            "failed_steps": failed_steps,
            "replan_count": len(self.replan_history),
        }
