"""Base agent implementation for the coordinator-worker architecture."""

from aida.agents.base.base_agent import AgentConfig, BaseAgent
from aida.agents.base.messages import (
    MessageRouter,
    TaskPriority,
    WorkerMessageTypes,
    create_task_assignment_message,
    create_task_completion_message,
    create_worker_registration_message,
)
from aida.agents.base.sandbox import (
    ResourceLimits,
    SandboxConfig,
    SandboxIsolationLevel,
    SandboxManager,
    create_default_sandbox_config,
)
from aida.agents.base.worker_agent import WorkerAgent, WorkerConfig

__all__ = [
    # Base agents
    "BaseAgent",
    "AgentConfig",
    "WorkerAgent",
    "WorkerConfig",
    # Messages
    "WorkerMessageTypes",
    "TaskPriority",
    "MessageRouter",
    "create_task_assignment_message",
    "create_task_completion_message",
    "create_worker_registration_message",
    # Sandbox
    "SandboxConfig",
    "SandboxIsolationLevel",
    "ResourceLimits",
    "SandboxManager",
    "create_default_sandbox_config",
]
