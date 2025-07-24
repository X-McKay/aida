"""TaskDispatcher for managing agent selection, task execution, and retry logic.

This module provides a clean interface for dispatching tasks to appropriate agents,
handling retries, timeouts, and observability.
"""

import asyncio
from datetime import datetime
from enum import Enum
import logging
import time
from typing import Any, Protocol
import uuid

from pydantic import BaseModel, Field

from aida.agents.base import BaseAgent
from aida.agents.coordination.plan_models import TodoStep

logger = logging.getLogger(__name__)


class TaskResult(BaseModel):
    """Result from task execution."""

    task_id: str
    step_id: str
    success: bool
    result: Any = None
    error: str | None = None
    retriable: bool = True
    retry_count: int = 0
    execution_time: float = 0.0
    agent_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        data["timestamp"] = data["timestamp"].isoformat()
        return data


class RetryStrategy(str, Enum):
    """Retry strategies for failed tasks."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE = "immediate"
    NO_RETRY = "no_retry"


class DispatchStrategy(str, Enum):
    """Strategies for selecting agents."""

    ROUND_ROBIN = "round_robin"
    CAPABILITY_BASED = "capability_based"
    LOAD_BALANCED = "load_balanced"
    RANDOM = "random"


class TaskExecutor(Protocol):
    """Protocol for task executors (agents)."""

    agent_id: str
    capabilities: list[str]

    async def execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a task and return results."""
        ...


class TaskDispatcher:
    """
    Handles task assignment, execution, retry logic, and observability.

    This class encapsulates the logic for:
    - Selecting appropriate agents for tasks
    - Managing retry logic with configurable strategies
    - Tracking task execution metrics
    - Providing observability through structured logging
    """

    def __init__(
        self,
        agents: list[BaseAgent],
        dispatch_strategy: DispatchStrategy = DispatchStrategy.CAPABILITY_BASED,
        retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        timeout: float = 300.0,  # 5 minutes default
    ):
        """Initialize the TaskDispatcher.

        Args:
            agents: List of available agents
            dispatch_strategy: Strategy for agent selection
            retry_strategy: Strategy for retrying failed tasks
            max_retries: Maximum retry attempts
            initial_retry_delay: Initial delay between retries (seconds)
            timeout: Task execution timeout (seconds)
        """
        self.agents = {agent.agent_id: agent for agent in agents}
        self.dispatch_strategy = dispatch_strategy
        self.retry_strategy = retry_strategy
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.timeout = timeout

        # For round-robin strategy
        self._round_robin_idx = 0

        # Metrics tracking
        self.metrics = {
            "total_dispatched": 0,
            "successful": 0,
            "failed": 0,
            "retried": 0,
            "timeouts": 0,
        }

    def select_agent(self, step: TodoStep) -> BaseAgent | None:
        """Select an agent based on the configured strategy.

        Args:
            step: The task step to execute

        Returns:
            Selected agent or None if no suitable agent found
        """
        if not self.agents:
            logger.error("No agents available for task dispatch")
            return None

        if self.dispatch_strategy == DispatchStrategy.ROUND_ROBIN:
            agent_ids = list(self.agents.keys())
            agent_id = agent_ids[self._round_robin_idx % len(agent_ids)]
            self._round_robin_idx += 1
            return self.agents[agent_id]

        elif self.dispatch_strategy == DispatchStrategy.CAPABILITY_BASED:
            # Match agent capabilities with task requirements
            required_capability = step.tool_name
            for agent in self.agents.values():
                if hasattr(agent, "capabilities") and required_capability in agent.capabilities:
                    logger.debug(
                        f"Selected agent {agent.agent_id} for capability {required_capability}"
                    )
                    return agent

            # Fallback to first available agent
            logger.warning(f"No agent found with capability {required_capability}, using fallback")
            return next(iter(self.agents.values()))

        elif self.dispatch_strategy == DispatchStrategy.RANDOM:
            import random

            return random.choice(list(self.agents.values()))

        else:
            # Default to first agent
            return next(iter(self.agents.values()))

    def calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay before retry based on strategy.

        Args:
            attempt: Current retry attempt number

        Returns:
            Delay in seconds
        """
        if (
            self.retry_strategy == RetryStrategy.NO_RETRY
            or self.retry_strategy == RetryStrategy.IMMEDIATE
        ):
            return 0
        elif self.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            return self.initial_retry_delay * attempt
        elif self.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return self.initial_retry_delay * (2 ** (attempt - 1))
        else:
            return self.initial_retry_delay

    async def dispatch(
        self,
        step: TodoStep,
        context: dict[str, Any] | None = None,
        task_id: str | None = None,
        enable_revision: bool = True,
    ) -> TaskResult:
        """Dispatch a task step to an agent with retry logic.

        Args:
            step: The task step to execute
            context: Optional execution context
            task_id: Optional task ID for tracking

        Returns:
            TaskResult with execution details
        """
        # Generate task ID if not provided
        if not task_id:
            task_id = str(uuid.uuid4())

        # Update metrics
        self.metrics["total_dispatched"] += 1

        logger.info(
            f"Dispatching task {task_id} step {step.id}: {step.description}",
            extra={
                "task_id": task_id,
                "step_id": step.id,
                "tool_name": step.tool_name,
            },
        )

        last_error = None
        last_result = None

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                # Calculate and apply retry delay
                delay = self.calculate_retry_delay(attempt)
                if delay > 0:
                    logger.info(
                        f"Retrying task {task_id} after {delay}s delay (attempt {attempt + 1})"
                    )
                    await asyncio.sleep(delay)
                self.metrics["retried"] += 1

            # Select agent
            agent = self.select_agent(step)
            if not agent:
                return TaskResult(
                    task_id=task_id,
                    step_id=step.id,
                    success=False,
                    error="No suitable agent available",
                    retriable=False,
                    retry_count=attempt,
                )

            # Prepare task data
            task_data = {
                "task_id": task_id,
                "step_id": step.id,
                "tool_name": step.tool_name,
                "parameters": step.parameters,
                "description": step.description,
                "context": context or {},
                "retry_attempt": attempt,
            }

            # Execute with timeout
            start_time = time.time()
            try:
                # Create timeout task
                result = await asyncio.wait_for(
                    self._execute_task(agent, task_data), timeout=self.timeout
                )

                execution_time = time.time() - start_time

                # Check if successful
                if result.get("success", False):
                    logger.info(
                        f"Task {task_id} completed successfully by {agent.agent_id} in {execution_time:.2f}s",
                        extra={
                            "task_id": task_id,
                            "agent_id": agent.agent_id,
                            "execution_time": execution_time,
                        },
                    )

                    self.metrics["successful"] += 1

                    return TaskResult(
                        task_id=task_id,
                        step_id=step.id,
                        success=True,
                        result=result.get("result"),
                        retry_count=attempt,
                        execution_time=execution_time,
                        agent_id=agent.agent_id,
                    )

                # Task failed but might be retriable
                last_error = result.get("error", "Unknown error")
                last_result = result

                if not result.get("retriable", True):
                    # Non-retriable failure
                    logger.error(
                        f"Task {task_id} failed with non-retriable error: {last_error}",
                        extra={"task_id": task_id, "error": last_error},
                    )
                    break

                logger.warning(
                    f"Task {task_id} failed (attempt {attempt + 1}): {last_error}",
                    extra={"task_id": task_id, "attempt": attempt + 1},
                )

            except TimeoutError:
                execution_time = time.time() - start_time
                last_error = f"Task timed out after {self.timeout}s"
                self.metrics["timeouts"] += 1

                logger.error(
                    f"Task {task_id} timed out on agent {agent.agent_id}",
                    extra={
                        "task_id": task_id,
                        "agent_id": agent.agent_id,
                        "timeout": self.timeout,
                    },
                )

            except Exception as e:
                execution_time = time.time() - start_time
                last_error = f"Unexpected error: {str(e)}"

                logger.exception(
                    f"Unexpected error executing task {task_id}",
                    extra={"task_id": task_id, "agent_id": agent.agent_id},
                )

        # All retries exhausted
        self.metrics["failed"] += 1

        # Set retriable=True if revision is enabled so coordinator can try revision
        return TaskResult(
            task_id=task_id,
            step_id=step.id,
            success=False,
            error=last_error,
            retriable=enable_revision,
            retry_count=self.max_retries,
            result=last_result,
        )

    async def _execute_task(self, agent: BaseAgent, task_data: dict[str, Any]) -> dict[str, Any]:
        """Execute task on agent.

        Args:
            agent: The agent to execute on
            task_data: Task data to pass to agent

        Returns:
            Execution result dictionary
        """
        # Call the agent's execute method
        # This will be adapted based on the actual agent interface
        logger.debug(
            f"Executing task on agent {agent.agent_id}, has execute_task: {hasattr(agent, 'execute_task')}, has handle_message: {hasattr(agent, 'handle_message')}"
        )

        if hasattr(agent, "execute_task"):
            return await agent.execute_task(task_data)
        elif hasattr(agent, "handle_message"):
            # Adapt to A2A message format
            from aida.core.protocols.a2a import A2AMessage

            # Extract step info from task_data
            step_id = task_data.get("step_id", "unknown")
            plan_id = (
                task_data.get("task_id", "").split("_")[0]
                if "_" in task_data.get("task_id", "")
                else "unknown"
            )

            # Adapt task data to worker format - must match worker expectations
            worker_payload = {
                "task_id": task_data.get("task_id", f"{plan_id}_{step_id}"),
                "plan_id": plan_id,
                "step_description": task_data.get(
                    "description", f"Execute {task_data.get('tool_name', 'task')}"
                ),
                "capability_required": task_data.get("tool_name", "unknown"),
                "parameters": task_data.get("parameters", {}),
                "timeout_seconds": self.timeout,
                "priority": task_data.get("priority", 5),
            }

            logger.debug(
                f"Sending task_assignment to {agent.agent_id} with payload: {worker_payload}"
            )

            message = A2AMessage(
                sender_id="coordinator",
                recipient_id=agent.agent_id,
                message_type="task_assignment",
                payload=worker_payload,
            )

            response = await agent.handle_message(message)
            logger.debug(f"Received response from {agent.agent_id}: {response}")

            if response:
                return response.payload
            else:
                return {"success": False, "error": "No response from agent"}
        else:
            # Log what methods the agent actually has
            methods = [m for m in dir(agent) if not m.startswith("_")]
            logger.error(
                f"Agent {agent.agent_id} type {type(agent)} does not support task execution. Available methods: {methods[:10]}..."
            )
            return {
                "success": False,
                "error": f"Agent {agent.agent_id} does not support task execution",
                "retriable": False,
            }

    def get_metrics(self) -> dict[str, Any]:
        """Get dispatcher metrics.

        Returns:
            Dictionary of metrics
        """
        total = self.metrics["total_dispatched"]
        if total > 0:
            success_rate = self.metrics["successful"] / total
            failure_rate = self.metrics["failed"] / total
            retry_rate = self.metrics["retried"] / total
            timeout_rate = self.metrics["timeouts"] / total
        else:
            success_rate = failure_rate = retry_rate = timeout_rate = 0

        return {
            **self.metrics,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "retry_rate": retry_rate,
            "timeout_rate": timeout_rate,
        }

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {
            "total_dispatched": 0,
            "successful": 0,
            "failed": 0,
            "retried": 0,
            "timeouts": 0,
        }
