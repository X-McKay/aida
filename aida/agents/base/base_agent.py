"""Simplified BaseAgent class for the coordinator-worker architecture.

This is a clean reimplementation focused on the essentials:
- A2A protocol integration
- MCP client management
- Event bus integration
- Lifecycle management
"""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any

from aida.core.events import EventBus
from aida.core.mcp_executor import MCPExecutor
from aida.core.protocols.a2a import A2AMessage, A2AProtocol

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for a base agent."""

    agent_id: str
    agent_type: str
    capabilities: list[str]
    host: str = "localhost"
    port: int = 0  # 0 means auto-assign
    log_level: str = "INFO"
    max_concurrent_tasks: int = 10
    task_timeout_seconds: int = 300
    heartbeat_interval: int = 30
    allowed_mcp_servers: list[str] | None = None


class BaseAgent(ABC):
    """Simplified base class for all agents in the coordinator-worker architecture.

    This class provides:
    - A2A protocol initialization and management
    - MCP client pool for tool execution
    - Event bus integration
    - Basic lifecycle methods
    - Health monitoring

    Subclasses must implement:
    - handle_message() for processing A2A messages
    - get_capabilities() to advertise agent capabilities
    """

    def __init__(self, config: AgentConfig):
        """Initialize the base agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.agent_id = config.agent_id
        self.agent_type = config.agent_type
        self.capabilities = config.capabilities

        # Set up logging
        logging.getLogger().setLevel(config.log_level)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Core components
        self.a2a_protocol = A2AProtocol(agent_id=self.agent_id, host=config.host, port=config.port)
        self.event_bus = EventBus()
        self.mcp_clients: dict[str, MCPExecutor] = {}

        # State management
        self.state = "initialized"
        self._running = False
        self._tasks: set[asyncio.Task] = set()
        self._start_time: datetime | None = None

        # Message handlers registry
        self._message_handlers: dict[str, Any] = {}

        # Statistics
        self._stats = {
            "messages_received": 0,
            "messages_sent": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "uptime_seconds": 0,
        }

    async def start(self) -> None:
        """Start the agent and all its components."""
        self.logger.info(f"Starting agent {self.agent_id} ({self.agent_type})")

        try:
            # Connect A2A protocol
            if not await self.a2a_protocol.connect():
                raise RuntimeError("Failed to start A2A protocol")

            # Register capabilities with A2A
            for capability in self.capabilities:
                self.a2a_protocol.add_capability(capability)

            # Initialize MCP clients if configured
            if self.config.allowed_mcp_servers:
                await self._initialize_mcp_clients()

            # Set running flag before starting loops
            self._running = True

            # Start message processing loop
            self._tasks.add(asyncio.create_task(self._message_loop()))

            # Start health check loop
            self._tasks.add(asyncio.create_task(self._health_check_loop()))

            # Allow subclasses to perform additional initialization
            await self._on_start()

            self.state = "running"
            self._start_time = datetime.utcnow()

            self.logger.info(f"Agent {self.agent_id} started successfully")

        except Exception as e:
            self.logger.error(f"Failed to start agent: {e}")
            self.state = "error"
            raise

    async def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        self.logger.info(f"Stopping agent {self.agent_id}")

        self._running = False
        self.state = "stopping"

        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Allow subclasses to perform cleanup
        await self._on_stop()

        # Disconnect A2A protocol
        await self.a2a_protocol.disconnect()

        # Cleanup MCP clients
        for _client in self.mcp_clients.values():
            try:
                # TODO: Add cleanup method to MCPExecutor
                pass
            except Exception as e:
                self.logger.error(f"Error cleaning up MCP client: {e}")

        self.state = "stopped"
        self.logger.info(f"Agent {self.agent_id} stopped")

    async def health_check(self) -> dict[str, Any]:
        """Perform health check and return status."""
        uptime = 0
        if self._start_time:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()

        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "state": self.state,
            "uptime_seconds": uptime,
            "capabilities": self.capabilities,
            "stats": self._stats.copy(),
            "a2a_connected": self.a2a_protocol._server is not None,
            "mcp_clients": len(self.mcp_clients),
            "active_tasks": len([t for t in self._tasks if not t.done()]),
        }

    @abstractmethod
    async def handle_message(self, message: A2AMessage) -> A2AMessage | None:
        """Handle incoming A2A message.

        Subclasses must implement this method to process messages.

        Args:
            message: Incoming A2A message

        Returns:
            Optional response message
        """
        pass

    def get_capabilities(self) -> list[str]:
        """Get agent capabilities.

        Subclasses can override to dynamically determine capabilities.

        Returns:
            List of capability names
        """
        return self.capabilities

    async def send_message(self, message: A2AMessage) -> bool:
        """Send an A2A message.

        Args:
            message: Message to send

        Returns:
            True if sent successfully
        """
        success = await self.a2a_protocol.send(message)
        if success:
            self._stats["messages_sent"] += 1
        return success

    async def execute_mcp_tool(
        self, server: str, tool: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute an MCP tool.

        Args:
            server: MCP server name
            tool: Tool name
            args: Tool arguments

        Returns:
            Tool execution result

        Raises:
            ValueError: If server not allowed or not initialized
            RuntimeError: If tool execution fails
        """
        if server not in self.mcp_clients:
            raise ValueError(f"MCP server '{server}' not initialized")

        client = self.mcp_clients[server]
        try:
            result = await client.execute_tool(tool, args)
            return result
        except Exception as e:
            self.logger.error(f"MCP tool execution failed: {server}.{tool} - {e}")
            raise RuntimeError(f"Tool execution failed: {e}") from e

    async def _initialize_mcp_clients(self) -> None:
        """Initialize MCP clients for allowed servers."""
        if not self.config.allowed_mcp_servers:
            return

        for server in self.config.allowed_mcp_servers:
            try:
                client = MCPExecutor()
                await client.add_provider(server)
                self.mcp_clients[server] = client
                self.logger.info(f"Initialized MCP client for: {server}")
            except Exception as e:
                self.logger.error(f"Failed to initialize MCP client for {server}: {e}")

    async def _message_loop(self) -> None:
        """Main message processing loop."""
        self.logger.info(f"Starting message loop for {self.agent_id}")
        self.logger.debug(f"Message loop using A2A protocol instance: {id(self.a2a_protocol)}")

        while self._running:
            try:
                # Receive message with timeout
                self.logger.debug(f"Agent {self.agent_id} calling receive()...")
                message = await self.a2a_protocol.receive()

                if message:
                    self._stats["messages_received"] += 1
                    self.logger.debug(
                        f"Agent {self.agent_id} received message type: {message.message_type} from {message.sender_id}"
                    )

                    # Handle system messages
                    if message.message_type == A2AMessage.MessageTypes.HEARTBEAT:
                        # Heartbeats are handled by protocol layer
                        continue
                    elif message.message_type == A2AMessage.MessageTypes.CAPABILITY_DISCOVERY:
                        await self._handle_capability_discovery(message)
                    else:
                        # Let subclass handle the message
                        try:
                            response = await self.handle_message(message)
                            if response:
                                await self.send_message(response)
                        except Exception as e:
                            self.logger.error(f"Error handling message: {e}", exc_info=True)
                            # Send error response
                            error_response = A2AMessage(
                                sender_id=self.agent_id,
                                recipient_id=message.sender_id,
                                message_type=A2AMessage.MessageTypes.ERROR,
                                correlation_id=message.id,
                                payload={
                                    "error": str(e),
                                    "original_message_type": message.message_type,
                                },
                            )
                            await self.send_message(error_response)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in message loop: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while self._running:
            try:
                # Update uptime
                if self._start_time:
                    self._stats["uptime_seconds"] = (
                        datetime.utcnow() - self._start_time
                    ).total_seconds()

                # Log health status periodically
                if int(self._stats["uptime_seconds"]) % 300 == 0:  # Every 5 minutes
                    health = await self.health_check()
                    self.logger.info(f"Health check: {health}")

                await asyncio.sleep(self.config.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)

    async def _handle_capability_discovery(self, message: A2AMessage) -> None:
        """Handle capability discovery request."""
        response = A2AMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type=A2AMessage.MessageTypes.CAPABILITY_RESPONSE,
            correlation_id=message.id,
            payload={
                "agent_type": self.agent_type,
                "capabilities": self.get_capabilities(),
                "state": self.state,
                "stats": self._stats.copy(),
            },
        )
        await self.send_message(response)

    async def _on_start(self) -> None:
        """Hook for subclasses to perform additional initialization.

        Called after core components are initialized but before
        the agent is marked as running.
        """
        pass

    async def _on_stop(self) -> None:
        """Hook for subclasses to perform cleanup.

        Called before core components are shut down.
        """
        pass

    def create_response(
        self, original_message: A2AMessage, payload: dict[str, Any], message_type: str = "response"
    ) -> A2AMessage:
        """Helper to create a response message.

        Args:
            original_message: The message being responded to
            payload: Response payload
            message_type: Type of response message

        Returns:
            Configured response message
        """
        return A2AMessage(
            sender_id=self.agent_id,
            recipient_id=original_message.sender_id,
            message_type=message_type,
            correlation_id=original_message.id,
            payload=payload,
            priority=original_message.priority,
        )

    def create_error_response(self, original_message: A2AMessage, error: str) -> A2AMessage:
        """Helper to create an error response.

        Args:
            original_message: The message that caused the error
            error: Error description

        Returns:
            Error response message
        """
        return self.create_response(
            original_message,
            {"status": "error", "error": error, "timestamp": datetime.utcnow().isoformat()},
            A2AMessage.MessageTypes.ERROR,
        )
