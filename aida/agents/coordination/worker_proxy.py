"""Proxy agent for remote workers connected via A2A protocol.

This proxy allows the dispatcher to interact with remote workers
as if they were local agent objects.
"""

import asyncio
from datetime import datetime
import logging
from typing import Any

from aida.agents.base import AgentConfig, BaseAgent
from aida.core.protocols.a2a import A2AMessage

logger = logging.getLogger(__name__)


class WorkerProxy(BaseAgent):
    """Proxy for remote worker agents.

    This class implements the agent interface but forwards all
    operations to a remote worker via A2A messages.
    """

    def __init__(
        self,
        worker_id: str,
        capabilities: list[str],
        coordinator_agent: Any,  # Avoid circular import
    ):
        """Initialize worker proxy.

        Args:
            worker_id: ID of the remote worker
            capabilities: Worker's capabilities
            coordinator_agent: Reference to coordinator for message sending
        """
        # Create config for proxy
        config = AgentConfig(
            agent_id=f"proxy_{worker_id}",
            agent_type="worker_proxy",
            capabilities=capabilities,
        )
        super().__init__(config)

        self.remote_worker_id = worker_id
        self.coordinator = coordinator_agent
        self._pending_responses: dict[str, asyncio.Future] = {}

    async def handle_message(self, message: A2AMessage) -> A2AMessage | None:
        """Forward message to remote worker and wait for response.

        This is called by the dispatcher to send tasks to workers.
        """
        # The message is already formatted for the worker by the dispatcher
        # We just need to forward it to the actual worker

        # Generate a correlation ID for tracking the response
        correlation_id = f"{message.message_type}_{datetime.utcnow().timestamp()}"

        # Update the message to go to the remote worker
        forwarded_message = A2AMessage(
            sender_id=self.coordinator.agent_id,
            recipient_id=self.remote_worker_id,
            message_type=message.message_type,
            payload=message.payload,
            correlation_id=correlation_id,
        )

        # Create a future to wait for response
        future = asyncio.Future()
        self._pending_responses[correlation_id] = future

        # Send via coordinator's A2A protocol
        success = await self.coordinator.send_message(forwarded_message)

        if not success:
            # Clean up and return error
            self._pending_responses.pop(correlation_id, None)
            return A2AMessage(
                sender_id=self.remote_worker_id,
                recipient_id=message.sender_id,
                message_type="task_response",
                payload={
                    "success": False,
                    "error": "Failed to send message to worker",
                    "retriable": True,
                },
            )

        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        except TimeoutError:
            logger.error(f"Timeout waiting for response from {self.remote_worker_id}")
            # Create error response
            return A2AMessage(
                sender_id=self.remote_worker_id,
                recipient_id=message.sender_id,
                message_type="task_response",
                payload={
                    "success": False,
                    "error": "Worker response timeout",
                    "retriable": True,
                },
            )
        finally:
            # Clean up future
            self._pending_responses.pop(correlation_id, None)

    def handle_worker_response(self, message: A2AMessage) -> None:
        """Handle response from remote worker.

        Called by coordinator when it receives a response from the worker.
        """
        # Find the waiting future based on correlation ID
        correlation_id = message.correlation_id

        if not correlation_id:
            # Try to match by other means (e.g., task_id in payload)
            payload = message.payload or {}
            task_id = payload.get("task_id")

            # Look for a matching future
            for cid, _future in self._pending_responses.items():
                if task_id and task_id in cid:
                    correlation_id = cid
                    break

        if correlation_id and correlation_id in self._pending_responses:
            future = self._pending_responses[correlation_id]
            if not future.done():
                future.set_result(message)
        else:
            logger.warning(
                f"Received unexpected response from {self.remote_worker_id}: {message.id}"
            )

    async def execute_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Execute task via remote worker.

        This provides compatibility with agents that expect execute_task.
        """
        # Convert task_data to worker format (matching what dispatcher does)
        step_id = task_data.get("step_id", "unknown")
        plan_id = (
            task_data.get("task_id", "").split("_")[0]
            if "_" in task_data.get("task_id", "")
            else "unknown"
        )

        worker_payload = {
            "task_id": task_data.get("task_id", f"{plan_id}_{step_id}"),
            "plan_id": plan_id,
            "step_description": task_data.get(
                "description", f"Execute {task_data.get('tool_name', 'task')}"
            ),
            "capability_required": task_data.get("tool_name", "unknown"),
            "parameters": task_data.get("parameters", {}),
            "timeout_seconds": 30,  # Default timeout
            "priority": task_data.get("priority", 5),
        }

        # Create A2A message for task assignment
        message = A2AMessage(
            sender_id="coordinator",
            recipient_id=self.remote_worker_id,
            message_type="task_assignment",
            payload=worker_payload,
        )

        # Use handle_message to forward
        response = await self.handle_message(message)

        if response and response.payload:
            return response.payload
        else:
            return {
                "success": False,
                "error": "No response from remote worker",
                "retriable": True,
            }
