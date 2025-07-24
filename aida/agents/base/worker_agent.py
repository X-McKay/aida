"""Base class for worker agents that execute tasks in sandboxed environments.

WorkerAgent extends BaseAgent with:
- Automatic worker registration with coordinator
- Task queue management and execution
- Sandboxed execution via Dagger containers
- Progress reporting and result handling
- MCP tool execution within sandbox
"""

from abc import abstractmethod
import asyncio
from datetime import datetime
import logging
from typing import Any

from aida.agents.base import (
    AgentConfig,
    BaseAgent,
    SandboxConfig,
    SandboxManager,
    WorkerMessageTypes,
    create_task_completion_message,
    create_worker_registration_message,
)
from aida.core.protocols.a2a import A2AMessage

logger = logging.getLogger(__name__)


from dataclasses import dataclass


@dataclass
class WorkerConfig:
    """Configuration for worker agents.

    Contains all fields from AgentConfig plus worker-specific settings.
    """

    # Base AgentConfig fields
    agent_id: str
    agent_type: str
    capabilities: list[str]
    host: str = "localhost"
    port: int = 0
    log_level: str = "INFO"
    max_concurrent_tasks: int = 10
    task_timeout_seconds: int = 300
    heartbeat_interval: int = 15  # More frequent than base
    allowed_mcp_servers: list[str] | None = None

    # Worker-specific fields
    coordinator_endpoint: str = "ws://localhost:8100"
    sandbox_config: SandboxConfig | None = None
    max_task_retries: int = 3
    task_timeout_multiplier: float = 1.2
    auto_register: bool = True
    startup_delay: int = 2

    def to_agent_config(self) -> AgentConfig:
        """Convert to base AgentConfig for BaseAgent initialization."""
        return AgentConfig(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            capabilities=self.capabilities,
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            max_concurrent_tasks=self.max_concurrent_tasks,
            task_timeout_seconds=self.task_timeout_seconds,
            heartbeat_interval=self.heartbeat_interval,
            allowed_mcp_servers=self.allowed_mcp_servers,
        )


class WorkerAgent(BaseAgent):
    """Base class for all worker agents.

    This class handles:
    - Registration with coordinator
    - Task acceptance/rejection logic
    - Sandboxed task execution
    - Progress reporting
    - Result transmission
    """

    def __init__(self, config: WorkerConfig):
        """Initialize the worker agent.

        Args:
            config: Worker configuration
        """
        # Initialize base with converted config
        super().__init__(config.to_agent_config())

        # Store full worker config
        self.worker_config = config
        self.coordinator_id: str | None = None
        self.sandbox_manager = SandboxManager()

        # Task tracking
        self._active_tasks: dict[str, dict[str, Any]] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._task_executor: asyncio.Task | None = None

        # Container reference
        self._container: Any | None = None  # dagger.Container when available

        # Performance metrics
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._total_execution_time = 0.0

    async def _on_start(self) -> None:
        """Additional initialization after base start."""
        # Initialize sandbox manager
        await self.sandbox_manager.initialize()

        # Create and start container if sandbox configured
        if self.worker_config.sandbox_config:
            try:
                self._container = await self.sandbox_manager.create_sandbox(
                    self.agent_id, self.worker_config.sandbox_config
                )
                logger.info(f"Created sandbox for worker {self.agent_id}")
            except Exception as e:
                logger.error(f"Failed to create sandbox: {e}")
                # Continue without sandbox for now

        # Start task executor
        self._task_executor = asyncio.create_task(self._task_execution_loop())
        self._tasks.add(self._task_executor)

        # Auto-register with coordinator
        if self.worker_config.auto_register:
            # Wait a bit for coordinator to be ready
            await asyncio.sleep(self.worker_config.startup_delay)
            await self._register_with_coordinator()

    async def _on_stop(self) -> None:
        """Cleanup before stopping."""
        # Stop task executor
        if self._task_executor:
            self._task_executor.cancel()

        # Destroy sandbox
        if self._container:
            await self.sandbox_manager.destroy_sandbox(self.agent_id)

        # Cleanup sandbox manager
        await self.sandbox_manager.cleanup()

        # Notify coordinator we're shutting down
        if self.coordinator_id:
            await self._send_shutdown_notification()

    async def handle_message(self, message: A2AMessage) -> A2AMessage | None:
        """Handle incoming A2A messages."""
        logger.debug(
            f"Worker {self.agent_id} received message type: {message.message_type} from {message.sender_id}"
        )

        handlers = {
            WorkerMessageTypes.TASK_ASSIGNMENT: self._handle_task_assignment,
            WorkerMessageTypes.RESOURCE_GRANT: self._handle_resource_grant,
            WorkerMessageTypes.PLAN_UPDATE: self._handle_plan_update,
            A2AMessage.MessageTypes.HEARTBEAT: self._handle_heartbeat,
            "handshake": self._handle_handshake,
        }

        handler = handlers.get(message.message_type)
        if handler:
            return await handler(message)

        # Unknown message type
        logger.warning(
            f"Worker {self.agent_id} received unknown message type: {message.message_type}"
        )
        return None

    async def _handle_handshake(self, message: A2AMessage) -> A2AMessage:
        """Handle A2A protocol handshake."""
        return A2AMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type="handshake",
            payload={"capabilities": self.capabilities, "agent_type": self.agent_type},
        )

    async def _register_with_coordinator(self) -> None:
        """Register with the coordinator."""
        # Try to connect to coordinator
        connected = await self.a2a_protocol.connect_to_agent(
            self.worker_config.coordinator_endpoint
        )

        if not connected:
            logger.error("Failed to connect to coordinator")
            return

        # Create registration message
        registration_data = create_worker_registration_message(
            worker_id=self.agent_id,
            worker_type=self.agent_type,
            capabilities=self.capabilities,
            max_concurrent_tasks=self.worker_config.max_concurrent_tasks,
            resource_limits=self.worker_config.sandbox_config.resource_limits.to_dagger_opts()
            if self.worker_config.sandbox_config
            else None,
            sandbox_config={
                "isolation_level": self.worker_config.sandbox_config.isolation_level.value,
                "has_network": self.worker_config.sandbox_config.resource_limits.network_enabled,
            }
            if self.worker_config.sandbox_config
            else None,
        )

        message = A2AMessage(
            sender_id=registration_data["sender_id"],
            message_type=registration_data["message_type"],
            payload=registration_data["payload"],
            priority=registration_data.get("priority", 5),
            requires_ack=registration_data.get("requires_ack", False),
        )
        response = await self.send_message(message)

        if response:
            logger.info("Successfully registered with coordinator")
            # Store the coordinator ID from the response
            self.coordinator_id = (
                message.recipient_id
            )  # Use the actual coordinator we're registering with
        else:
            logger.error("Failed to register with coordinator")

    async def _handle_task_assignment(self, message: A2AMessage) -> A2AMessage:
        """Handle task assignment from coordinator."""
        logger.info(f"Worker {self.agent_id} handling task assignment")
        payload = message.payload
        task_id = payload["task_id"]
        capability = payload["capability_required"]

        logger.info(f"Task {task_id} requires capability: {capability}")

        # Check if we support this capability
        if capability not in self.capabilities:
            # Reject the task
            return self.create_response(
                message,
                {
                    "task_id": task_id,
                    "worker_id": self.agent_id,
                    "reason": f"Capability '{capability}' not supported",
                    "suggested_alternative": None,
                },
                WorkerMessageTypes.TASK_REJECTION,
            )

        # Check if we're overloaded
        if len(self._active_tasks) >= self.worker_config.max_concurrent_tasks:
            return self.create_response(
                message,
                {
                    "task_id": task_id,
                    "worker_id": self.agent_id,
                    "reason": "Worker at maximum capacity",
                    "suggested_alternative": None,
                },
                WorkerMessageTypes.TASK_REJECTION,
            )

        # Accept the task
        task_data = {
            "task_id": task_id,
            "plan_id": payload["plan_id"],
            "description": payload["step_description"],
            "capability": capability,
            "parameters": payload["parameters"],
            "timeout": payload["timeout_seconds"],
            "priority": payload.get("priority", 5),
            "received_at": datetime.utcnow(),
            "coordinator_id": message.sender_id,
            "original_message": message,  # Store the original message to respond to
            "correlation_id": message.correlation_id,  # Store correlation ID if present
        }

        # Queue the task
        await self._task_queue.put(task_data)
        self._active_tasks[task_id] = task_data

        # Send acceptance
        return self.create_response(
            message,
            {
                "task_id": task_id,
                "worker_id": self.agent_id,
                "estimated_completion_seconds": int(payload["timeout_seconds"] * 0.8),
                "resource_requirements": None,
            },
            WorkerMessageTypes.TASK_ACCEPTANCE,
        )

    async def _handle_resource_grant(self, message: A2AMessage) -> A2AMessage | None:
        """Handle resource grant from coordinator."""
        # In full implementation, would update available resources
        logger.info(f"Received resource grant: {message.payload}")
        return None

    async def _handle_plan_update(self, message: A2AMessage) -> A2AMessage | None:
        """Handle plan update notification."""
        payload = message.payload
        plan_id = payload["plan_id"]
        update_type = payload["update_type"]

        if update_type == "cancelled":
            # Cancel any tasks for this plan
            for task_id, task_data in list(self._active_tasks.items()):
                if task_data.get("plan_id") == plan_id:
                    logger.info(f"Cancelling task {task_id} due to plan cancellation")
                    # In full implementation, would interrupt execution
                    del self._active_tasks[task_id]

        return None

    async def _handle_heartbeat(self, message: A2AMessage) -> A2AMessage | None:
        """Handle heartbeat from coordinator."""
        # Update coordinator last seen
        return None

    async def _task_execution_loop(self) -> None:
        """Main task execution loop."""
        logger.info(f"Starting task execution loop for {self.agent_id}")

        while self._running:
            try:
                # Get task from queue with timeout
                task_data = await asyncio.wait_for(self._task_queue.get(), timeout=1.0)

                # Execute the task
                await self._execute_task(task_data)

            except TimeoutError:
                # No tasks available
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in task execution loop: {e}")

    async def _execute_task(self, task_data: dict[str, Any]) -> None:
        """Execute a single task."""
        task_id = task_data["task_id"]
        start_time = datetime.utcnow()

        logger.info(f"Executing task {task_id}: {task_data['description']}")

        # Send initial progress
        await self._send_progress(task_id, 0, "Starting task execution")

        try:
            # Apply timeout with multiplier
            timeout = task_data["timeout"] * self.worker_config.task_timeout_multiplier

            # Execute in sandbox if available
            if self._container:
                result = await asyncio.wait_for(
                    self._execute_in_sandbox(task_data), timeout=timeout
                )
            else:
                # Execute directly (less secure)
                result = await asyncio.wait_for(self.execute_task_logic(task_data), timeout=timeout)

            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()

            # Update metrics
            self._tasks_completed += 1
            self._total_execution_time += execution_time

            # Send completion
            await self._send_completion(
                task_id,
                task_data["coordinator_id"],
                success=True,
                result=result,
                execution_time=execution_time,
            )

            logger.info(f"Task {task_id} completed successfully in {execution_time:.1f}s")

        except TimeoutError:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._tasks_failed += 1

            await self._send_completion(
                task_id,
                task_data["coordinator_id"],
                success=False,
                error=f"Task timed out after {execution_time:.1f}s",
                execution_time=execution_time,
            )

            logger.error(f"Task {task_id} timed out")

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._tasks_failed += 1

            await self._send_completion(
                task_id,
                task_data["coordinator_id"],
                success=False,
                error=str(e),
                execution_time=execution_time,
            )

            logger.error(f"Task {task_id} failed: {e}")

        finally:
            # Remove from active tasks
            if task_id in self._active_tasks:
                del self._active_tasks[task_id]

    async def _execute_in_sandbox(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Execute task in sandboxed environment.

        This is a simplified version. In production would:
        - Mount necessary files
        - Set up MCP servers
        - Execute with proper isolation
        """
        # For now, just delegate to task logic
        return await self.execute_task_logic(task_data)

    @abstractmethod
    async def execute_task_logic(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the actual task logic.

        Subclasses must implement this to perform the work.

        Args:
            task_data: Task information including parameters

        Returns:
            Task execution result
        """
        pass

    async def _send_progress(
        self,
        task_id: str,
        progress: int,
        message: str,
        interim_results: dict[str, Any] | None = None,
    ) -> None:
        """Send progress update to coordinator."""
        if not self.coordinator_id:
            return

        progress_message = A2AMessage(
            sender_id=self.agent_id,
            recipient_id=self.coordinator_id,
            message_type=WorkerMessageTypes.TASK_PROGRESS,
            payload={
                "task_id": task_id,
                "worker_id": self.agent_id,
                "progress_percentage": progress,
                "status_message": message,
                "interim_results": interim_results,
            },
        )

        await self.send_message(progress_message)

    async def _send_completion(
        self,
        task_id: str,
        coordinator_id: str,
        success: bool,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        execution_time: float = 0.0,
    ) -> None:
        """Send task completion to coordinator."""
        # Check if we have the original message to respond to
        task_data = self._active_tasks.get(task_id)
        if task_data and "original_message" in task_data:
            # Send response to the original sender (could be proxy)
            original_msg = task_data["original_message"]
            response = A2AMessage(
                sender_id=self.agent_id,
                recipient_id=original_msg.sender_id,
                message_type="task_response",
                correlation_id=task_data.get("correlation_id") or original_msg.correlation_id,
                payload={
                    "task_id": task_id,
                    "success": success,
                    "result": result,
                    "error": error,
                    "execution_time": execution_time,
                    "worker_id": self.agent_id,
                },
            )
            await self.send_message(response)

        # Also send completion to coordinator for tracking
        completion_data = create_task_completion_message(
            worker_id=self.agent_id,
            coordinator_id=coordinator_id,
            task_id=task_id,
            success=success,
            result=result,
            error=error,
            execution_time=execution_time,
        )

        message = A2AMessage(
            sender_id=completion_data["sender_id"],
            message_type=completion_data["message_type"],
            recipient_id=completion_data.get("recipient_id"),
            payload=completion_data["payload"],
            priority=completion_data.get("priority", 5),
            requires_ack=completion_data.get("requires_ack", False),
        )
        await self.send_message(message)

    async def _send_shutdown_notification(self) -> None:
        """Notify coordinator of shutdown."""
        if not self.coordinator_id:
            return

        shutdown_message = A2AMessage(
            sender_id=self.agent_id,
            recipient_id=self.coordinator_id,
            message_type=WorkerMessageTypes.WORKER_SHUTDOWN,
            payload={
                "worker_id": self.agent_id,
                "reason": "Graceful shutdown",
                "completed_tasks": self._tasks_completed,
                "failed_tasks": self._tasks_failed,
            },
        )

        await self.send_message(shutdown_message)

    async def health_check(self) -> dict[str, Any]:
        """Perform health check and return status."""
        base_health = await super().health_check()

        # Add worker-specific health info
        worker_health = {
            **base_health,
            "worker_state": {
                "active_tasks": len(self._active_tasks),
                "queued_tasks": self._task_queue.qsize(),
                "completed_tasks": self._tasks_completed,
                "failed_tasks": self._tasks_failed,
                "success_rate": self._tasks_completed
                / max(1, self._tasks_completed + self._tasks_failed),
                "avg_execution_time": self._total_execution_time / max(1, self._tasks_completed),
                "has_sandbox": self._container is not None,
                "coordinator_connected": self.coordinator_id is not None,
            },
        }

        return worker_health

    async def get_status(self) -> dict[str, Any]:
        """Get detailed worker status."""
        health = await self.health_check()

        # Determine overall state
        if len(self._active_tasks) == 0:
            state = "idle"
        elif len(self._active_tasks) < self.worker_config.max_concurrent_tasks:
            state = "busy"
        else:
            state = "overloaded"

        return {
            "worker_id": self.agent_id,
            "state": state,
            "active_tasks": list(self._active_tasks.keys()),
            "completed_tasks_count": self._tasks_completed,
            "failed_tasks_count": self._tasks_failed,
            "average_task_time": health["worker_state"]["avg_execution_time"],
            "health_metrics": health,
        }
