"""A2A message type definitions for coordinator-worker communication.

This module defines the message schemas and types used in the
coordinator-worker architecture, extending the base A2A protocol.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WorkerMessageTypes:
    """Extended message types for coordinator-worker communication."""

    # Task delegation
    TASK_ASSIGNMENT = "task_assignment"
    TASK_ACCEPTANCE = "task_acceptance"
    TASK_REJECTION = "task_rejection"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETION = "task_completion"

    # Worker management
    WORKER_REGISTRATION = "worker_registration"
    WORKER_HEARTBEAT = "worker_heartbeat"
    WORKER_STATUS = "worker_status"
    WORKER_SHUTDOWN = "worker_shutdown"

    # Resource management
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_GRANT = "resource_grant"
    RESOURCE_RELEASE = "resource_release"

    # Coordination
    PLAN_UPDATE = "plan_update"
    REPLAN_REQUEST = "replan_request"
    RESULT_AGGREGATION = "result_aggregation"


class TaskPriority(Enum):
    """Task priority levels."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 5
    LOW = 8
    BACKGROUND = 10


class TaskAssignmentPayload(BaseModel):
    """Payload for task assignment messages."""

    task_id: str
    plan_id: str
    step_description: str
    capability_required: str
    parameters: dict[str, Any]
    dependencies: list[str] = Field(default_factory=list)
    timeout_seconds: int = 300
    priority: TaskPriority = TaskPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3


class TaskAcceptancePayload(BaseModel):
    """Payload for task acceptance messages."""

    task_id: str
    worker_id: str
    estimated_completion_seconds: int | None = None
    resource_requirements: dict[str, Any] | None = None


class TaskRejectionPayload(BaseModel):
    """Payload for task rejection messages."""

    task_id: str
    worker_id: str
    reason: str
    suggested_alternative: str | None = None


class TaskProgressPayload(BaseModel):
    """Payload for task progress updates."""

    task_id: str
    worker_id: str
    progress_percentage: int  # 0-100
    status_message: str
    interim_results: dict[str, Any] | None = None


class TaskCompletionPayload(BaseModel):
    """Payload for task completion messages."""

    task_id: str
    worker_id: str
    success: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    execution_time_seconds: float
    resources_used: dict[str, Any] | None = None


class WorkerRegistrationPayload(BaseModel):
    """Payload for worker registration."""

    worker_id: str
    worker_type: str
    capabilities: list[str]
    max_concurrent_tasks: int = 1
    resource_limits: dict[str, Any] | None = None
    sandbox_config: dict[str, Any] | None = None


class WorkerStatusPayload(BaseModel):
    """Payload for worker status updates."""

    worker_id: str
    state: str  # idle, busy, overloaded, error
    active_tasks: list[str]
    completed_tasks_count: int
    failed_tasks_count: int
    average_task_time: float
    available_resources: dict[str, Any] | None = None
    health_metrics: dict[str, Any] | None = None


class ResourceRequestPayload(BaseModel):
    """Payload for resource requests."""

    task_id: str
    worker_id: str
    resource_type: str  # mcp_server, memory, cpu, etc.
    resource_spec: dict[str, Any]
    duration_seconds: int | None = None
    reason: str


class PlanUpdatePayload(BaseModel):
    """Payload for plan update notifications."""

    plan_id: str
    update_type: str  # created, modified, cancelled, completed
    plan_version: int
    changes: dict[str, Any] | None = None
    affected_tasks: list[str] = Field(default_factory=list)


@dataclass
class MessageRoutingRule:
    """Rule for routing messages between agents."""

    message_type: str
    source_pattern: str | None = None  # Regex pattern for source agent ID
    destination_pattern: str | None = None  # Regex pattern for destination
    priority_override: int | None = None
    requires_acknowledgment: bool = False
    timeout_seconds: int = 30


class MessageRouter:
    """Routes messages between coordinator and workers."""

    def __init__(self):
        self.routing_rules: list[MessageRoutingRule] = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Set up default routing rules."""
        # Task assignments require acknowledgment
        self.routing_rules.append(
            MessageRoutingRule(
                message_type=WorkerMessageTypes.TASK_ASSIGNMENT,
                requires_acknowledgment=True,
                timeout_seconds=10,
                priority_override=2,
            )
        )

        # Progress updates are lower priority
        self.routing_rules.append(
            MessageRoutingRule(
                message_type=WorkerMessageTypes.TASK_PROGRESS,
                requires_acknowledgment=False,
                priority_override=7,
            )
        )

        # Completion messages are high priority
        self.routing_rules.append(
            MessageRoutingRule(
                message_type=WorkerMessageTypes.TASK_COMPLETION,
                requires_acknowledgment=True,
                timeout_seconds=5,
                priority_override=3,
            )
        )

        # Worker registration is critical
        self.routing_rules.append(
            MessageRoutingRule(
                message_type=WorkerMessageTypes.WORKER_REGISTRATION,
                requires_acknowledgment=True,
                timeout_seconds=5,
                priority_override=1,
            )
        )

    def get_routing_rule(self, message_type: str) -> MessageRoutingRule | None:
        """Get routing rule for a message type."""
        for rule in self.routing_rules:
            if rule.message_type == message_type:
                return rule
        return None

    def should_acknowledge(self, message_type: str) -> bool:
        """Check if message type requires acknowledgment."""
        rule = self.get_routing_rule(message_type)
        return rule.requires_acknowledgment if rule else False

    def get_priority(self, message_type: str, default: int = 5) -> int:
        """Get priority for message type."""
        rule = self.get_routing_rule(message_type)
        if rule and rule.priority_override is not None:
            return rule.priority_override
        return default

    def get_timeout(self, message_type: str, default: int = 30) -> int:
        """Get timeout for message type."""
        rule = self.get_routing_rule(message_type)
        return rule.timeout_seconds if rule else default


def create_task_assignment_message(
    coordinator_id: str,
    worker_id: str,
    task_id: str,
    plan_id: str,
    step_description: str,
    capability: str,
    parameters: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """Helper to create a task assignment message.

    Returns:
        Message data ready to be wrapped in A2AMessage
    """
    payload = TaskAssignmentPayload(
        task_id=task_id,
        plan_id=plan_id,
        step_description=step_description,
        capability_required=capability,
        parameters=parameters,
        **kwargs,
    )

    router = MessageRouter()

    return {
        "sender_id": coordinator_id,
        "recipient_id": worker_id,
        "message_type": WorkerMessageTypes.TASK_ASSIGNMENT,
        "payload": payload.model_dump(),
        "priority": router.get_priority(WorkerMessageTypes.TASK_ASSIGNMENT),
        "requires_ack": router.should_acknowledge(WorkerMessageTypes.TASK_ASSIGNMENT),
    }


def create_task_completion_message(
    worker_id: str,
    coordinator_id: str,
    task_id: str,
    success: bool,
    result: dict[str, Any] | None = None,
    error: str | None = None,
    execution_time: float = 0.0,
) -> dict[str, Any]:
    """Helper to create a task completion message.

    Returns:
        Message data ready to be wrapped in A2AMessage
    """
    payload = TaskCompletionPayload(
        task_id=task_id,
        worker_id=worker_id,
        success=success,
        result=result,
        error=error,
        execution_time_seconds=execution_time,
    )

    router = MessageRouter()

    return {
        "sender_id": worker_id,
        "recipient_id": coordinator_id,
        "message_type": WorkerMessageTypes.TASK_COMPLETION,
        "payload": payload.model_dump(),
        "priority": router.get_priority(WorkerMessageTypes.TASK_COMPLETION),
        "requires_ack": router.should_acknowledge(WorkerMessageTypes.TASK_COMPLETION),
    }


def create_worker_registration_message(
    worker_id: str, worker_type: str, capabilities: list[str], **kwargs
) -> dict[str, Any]:
    """Helper to create a worker registration message.

    Returns:
        Message data ready to be wrapped in A2AMessage
    """
    payload = WorkerRegistrationPayload(
        worker_id=worker_id, worker_type=worker_type, capabilities=capabilities, **kwargs
    )

    router = MessageRouter()

    return {
        "sender_id": worker_id,
        "message_type": WorkerMessageTypes.WORKER_REGISTRATION,
        "payload": payload.model_dump(),
        "priority": router.get_priority(WorkerMessageTypes.WORKER_REGISTRATION),
        "requires_ack": router.should_acknowledge(WorkerMessageTypes.WORKER_REGISTRATION),
    }
