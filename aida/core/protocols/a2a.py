"""Agent-to-Agent (A2A) communication protocol."""

import asyncio
from dataclasses import dataclass
import logging
from typing import Any

import websockets
from websockets import WebSocketClientProtocol, WebSocketServerProtocol  # type: ignore[import]

from aida.core.protocols.base import Protocol, ProtocolMessage

logger = logging.getLogger(__name__)


class A2AMessage(ProtocolMessage):
    """A2A-specific message format."""

    priority: int = 5  # 1=highest, 10=lowest
    requires_ack: bool = False
    correlation_id: str | None = None
    routing_path: list[str] = []

    class MessageTypes:
        """Standard A2A message types."""

        HEARTBEAT = "heartbeat"
        TASK_REQUEST = "task_request"
        TASK_RESPONSE = "task_response"
        TASK_STATUS = "task_status"
        CAPABILITY_DISCOVERY = "capability_discovery"
        CAPABILITY_RESPONSE = "capability_response"
        COORDINATION_REQUEST = "coordination_request"
        COORDINATION_RESPONSE = "coordination_response"
        ERROR = "error"
        ACK = "ack"


@dataclass
class AgentInfo:
    """Information about a discovered agent."""

    agent_id: str
    capabilities: list[str]
    endpoint: str
    last_seen: float
    status: str = "active"


class A2AProtocol(Protocol):
    """Agent-to-Agent communication protocol implementation."""

    def __init__(
        self,
        agent_id: str,
        host: str = "localhost",
        port: int = 8080,
        discovery_enabled: bool = True,
    ):
        """Initialize the A2A protocol.

        Args:
            agent_id: Unique identifier for this agent
            host: Host address to bind the WebSocket server to
            port: Port number to bind the WebSocket server to
            discovery_enabled: Whether to enable automatic agent discovery
        """
        super().__init__(agent_id)
        self.host = host
        self.port = port
        self.discovery_enabled = discovery_enabled

        # Connection management
        self._server = None
        self._connections: dict[str, WebSocketServerProtocol] = {}
        self._client_connections: dict[str, WebSocketClientProtocol] = {}

        # Agent discovery
        self._known_agents: dict[str, AgentInfo] = {}
        self._capabilities: set[str] = set()

        # Message queues
        self._incoming_queue = asyncio.Queue()
        self._outgoing_queue = asyncio.Queue()

        # Tasks
        self._tasks: set[asyncio.Task] = set()

        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "connections_active": 0,
            "errors": 0,
        }

    async def connect(self) -> bool:
        """Start the A2A protocol server and client handlers."""
        try:
            # Start WebSocket server
            self._server = await websockets.serve(self._handle_connection, self.host, self.port)

            # Start background tasks
            self._tasks.add(asyncio.create_task(self._process_outgoing()))
            self._tasks.add(asyncio.create_task(self._heartbeat_loop()))

            if self.discovery_enabled:
                self._tasks.add(asyncio.create_task(self._discovery_loop()))

            logger.info(f"A2A server started on {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to start A2A protocol: {e}")
            return False

    async def disconnect(self) -> None:
        """Shutdown the A2A protocol."""
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Close all connections
        for conn in self._connections.values():
            await conn.close()

        for conn in self._client_connections.values():
            await conn.close()

        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        logger.info("A2A protocol disconnected")

    async def send(self, message: ProtocolMessage) -> bool:
        """Send a message to another agent."""
        if not isinstance(message, A2AMessage):
            # Convert to A2AMessage, preserving all fields
            msg_dict = message.dict()
            # Ensure required fields are present
            if "sender_id" not in msg_dict:
                msg_dict["sender_id"] = self.agent_id
            if "message_type" not in msg_dict:
                msg_dict["message_type"] = "unknown"
            # Create with explicit fields to satisfy type checker
            message = A2AMessage(
                sender_id=msg_dict.get("sender_id", self.agent_id),
                message_type=msg_dict.get("message_type", "unknown"),
                id=msg_dict.get("id", msg_dict.get("id")),  # type: ignore
                timestamp=msg_dict.get("timestamp", msg_dict.get("timestamp")),  # type: ignore
                recipient_id=msg_dict.get("recipient_id"),
                payload=msg_dict.get("payload", {}),
                metadata=msg_dict.get("metadata", {}),
                priority=msg_dict.get("priority", 5),
                requires_ack=msg_dict.get("requires_ack", False),
                correlation_id=msg_dict.get("correlation_id"),
                routing_path=msg_dict.get("routing_path", []),
            )

        try:
            await self._outgoing_queue.put(message)
            return True
        except Exception as e:
            logger.error(f"Failed to queue message: {e}")
            self._stats["errors"] += 1
            return False

    async def receive(self) -> A2AMessage | None:
        """Receive a message from another agent."""
        try:
            message = await asyncio.wait_for(self._incoming_queue.get(), timeout=1.0)
            return message
        except TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None

    async def discover_agents(self) -> list[AgentInfo]:
        """Discover other agents in the network."""
        # Send discovery request
        discovery_msg = A2AMessage(
            sender_id=self.agent_id,
            message_type=A2AMessage.MessageTypes.CAPABILITY_DISCOVERY,
            payload={"capabilities": list(self._capabilities)},
        )

        # Broadcast to all known agents
        for agent_info in self._known_agents.values():
            await self._send_to_agent(agent_info.agent_id, discovery_msg)

        return list(self._known_agents.values())

    async def connect_to_agent(self, agent_endpoint: str) -> bool:
        """Establish connection to a specific agent."""
        try:
            websocket = await websockets.connect(agent_endpoint)

            # Extract agent ID from handshake
            handshake_msg = A2AMessage(
                sender_id=self.agent_id,
                message_type="handshake",
                payload={"capabilities": list(self._capabilities)},
            )

            await websocket.send(handshake_msg.json())
            response = await websocket.recv()
            response_msg = A2AMessage.parse_raw(response)

            agent_id = response_msg.sender_id
            self._client_connections[agent_id] = websocket

            # Start message handler for this connection
            task = asyncio.create_task(self._handle_client_connection(agent_id, websocket))
            self._tasks.add(task)

            logger.info(f"Connected to agent {agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to agent at {agent_endpoint}: {e}")
            return False

    def add_capability(self, capability: str) -> None:
        """Add a capability to this agent."""
        self._capabilities.add(capability)

    def get_stats(self) -> dict[str, Any]:
        """Get protocol statistics."""
        self._stats["connections_active"] = len(self._connections) + len(self._client_connections)
        return self._stats.copy()

    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle incoming WebSocket connections."""
        agent_id = None
        try:
            # Wait for handshake
            handshake_data = await websocket.recv()
            handshake_msg = A2AMessage.parse_raw(handshake_data)

            agent_id = handshake_msg.sender_id
            self._connections[agent_id] = websocket

            # Send handshake response
            response = A2AMessage(
                sender_id=self.agent_id,
                recipient_id=agent_id,
                message_type="handshake_response",
                payload={"capabilities": list(self._capabilities)},
            )
            await websocket.send(response.json())

            # Update agent info
            self._known_agents[agent_id] = AgentInfo(
                agent_id=agent_id,
                capabilities=handshake_msg.payload.get("capabilities", []),
                endpoint=f"ws://{websocket.remote_address[0]}:{websocket.remote_address[1]}",
                last_seen=asyncio.get_event_loop().time(),
            )

            logger.info(f"Agent {agent_id} connected")

            # Handle messages from this connection
            async for message_data in websocket:
                try:
                    message = A2AMessage.parse_raw(message_data)
                    await self._incoming_queue.put(message)
                    self._stats["messages_received"] += 1

                    # Update last seen
                    if agent_id in self._known_agents:
                        self._known_agents[agent_id].last_seen = asyncio.get_event_loop().time()

                except Exception as e:
                    logger.error(f"Failed to process message from {agent_id}: {e}")

        except Exception as e:
            logger.error(f"Connection error: {e}")

        finally:
            # Cleanup
            if agent_id and agent_id in self._connections:
                del self._connections[agent_id]
                logger.info(f"Agent {agent_id} disconnected")

    async def _handle_client_connection(self, agent_id: str, websocket: WebSocketClientProtocol):
        """Handle outgoing client connections."""
        try:
            async for message_data in websocket:
                try:
                    message = A2AMessage.parse_raw(message_data)
                    await self._incoming_queue.put(message)
                    self._stats["messages_received"] += 1

                except Exception as e:
                    logger.error(f"Failed to process message from {agent_id}: {e}")

        except Exception as e:
            logger.error(f"Client connection error with {agent_id}: {e}")

        finally:
            if agent_id in self._client_connections:
                del self._client_connections[agent_id]

    async def _process_outgoing(self):
        """Process outgoing message queue."""
        while True:
            try:
                message = await self._outgoing_queue.get()

                if message.recipient_id:
                    # Send to specific agent
                    await self._send_to_agent(message.recipient_id, message)
                else:
                    # Broadcast to all connected agents
                    await self._broadcast_message(message)

                self._stats["messages_sent"] += 1

            except Exception as e:
                logger.error(f"Error processing outgoing message: {e}")
                self._stats["errors"] += 1

    async def _send_to_agent(self, agent_id: str, message: A2AMessage):
        """Send message to a specific agent."""
        # Try server connection first
        if agent_id in self._connections:
            try:
                await self._connections[agent_id].send(message.json())
                return
            except Exception as e:
                logger.error(f"Failed to send via server connection: {e}")
                del self._connections[agent_id]

        # Try client connection
        if agent_id in self._client_connections:
            try:
                await self._client_connections[agent_id].send(message.json())
                return
            except Exception as e:
                logger.error(f"Failed to send via client connection: {e}")
                del self._client_connections[agent_id]

        # If no direct connection, try to establish one
        if agent_id in self._known_agents:
            agent_info = self._known_agents[agent_id]
            if await self.connect_to_agent(agent_info.endpoint):
                await self._send_to_agent(agent_id, message)

    async def _broadcast_message(self, message: A2AMessage):
        """Broadcast message to all connected agents."""
        disconnected = []

        # Send to server connections
        for agent_id, conn in self._connections.items():
            try:
                await conn.send(message.json())
            except Exception as e:
                logger.error(f"Failed to broadcast to {agent_id}: {e}")
                disconnected.append(agent_id)

        # Send to client connections
        for agent_id, conn in self._client_connections.items():
            try:
                await conn.send(message.json())
            except Exception as e:
                logger.error(f"Failed to broadcast to {agent_id}: {e}")
                disconnected.append(agent_id)

        # Cleanup disconnected agents
        for agent_id in disconnected:
            self._connections.pop(agent_id, None)
            self._client_connections.pop(agent_id, None)

    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages."""
        while True:
            try:
                heartbeat = A2AMessage(
                    sender_id=self.agent_id,
                    message_type=A2AMessage.MessageTypes.HEARTBEAT,
                    payload={"timestamp": asyncio.get_event_loop().time()},
                )

                await self._broadcast_message(heartbeat)
                await asyncio.sleep(30)  # Heartbeat every 30 seconds

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(30)

    async def _discovery_loop(self):
        """Periodic agent discovery."""
        while True:
            try:
                await self.discover_agents()
                await asyncio.sleep(60)  # Discovery every minute

            except Exception as e:
                logger.error(f"Discovery error: {e}")
                await asyncio.sleep(60)
