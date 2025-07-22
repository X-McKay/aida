"""Tests for A2A protocol implementation."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from aida.core.protocols.a2a import A2AMessage, A2AProtocol, AgentInfo


class TestA2AMessage:
    """Test A2AMessage class."""

    def test_a2a_message_creation(self):
        """Test A2AMessage creation with all fields."""
        message = A2AMessage(
            sender_id="agent1",
            message_type="test",
            priority=3,
            requires_ack=True,
            correlation_id="corr123",
            routing_path=["agent1", "agent2"],
            payload={"test": "data"},
            recipient_id="agent2",
        )

        assert message.sender_id == "agent1"
        assert message.message_type == "test"
        assert message.priority == 3
        assert message.requires_ack is True
        assert message.correlation_id == "corr123"
        assert message.routing_path == ["agent1", "agent2"]
        assert message.payload == {"test": "data"}
        assert message.recipient_id == "agent2"

    def test_a2a_message_defaults(self):
        """Test A2AMessage default values."""
        message = A2AMessage(sender_id="agent1", message_type="test")

        assert message.priority == 5
        assert message.requires_ack is False
        assert message.correlation_id is None
        assert message.routing_path == []

    def test_message_types_constants(self):
        """Test A2AMessage.MessageTypes constants."""
        assert A2AMessage.MessageTypes.HEARTBEAT == "heartbeat"
        assert A2AMessage.MessageTypes.TASK_REQUEST == "task_request"
        assert A2AMessage.MessageTypes.TASK_RESPONSE == "task_response"
        assert A2AMessage.MessageTypes.TASK_STATUS == "task_status"
        assert A2AMessage.MessageTypes.CAPABILITY_DISCOVERY == "capability_discovery"
        assert A2AMessage.MessageTypes.CAPABILITY_RESPONSE == "capability_response"
        assert A2AMessage.MessageTypes.COORDINATION_REQUEST == "coordination_request"
        assert A2AMessage.MessageTypes.COORDINATION_RESPONSE == "coordination_response"
        assert A2AMessage.MessageTypes.ERROR == "error"
        assert A2AMessage.MessageTypes.ACK == "ack"


class TestAgentInfo:
    """Test AgentInfo class."""

    def test_agent_info_creation(self):
        """Test AgentInfo creation."""
        agent_info = AgentInfo(
            agent_id="agent123",
            capabilities=["task1", "task2"],
            endpoint="ws://localhost:8080",
            last_seen=1234567890.0,
            status="active",
        )

        assert agent_info.agent_id == "agent123"
        assert agent_info.capabilities == ["task1", "task2"]
        assert agent_info.endpoint == "ws://localhost:8080"
        assert agent_info.last_seen == 1234567890.0
        assert agent_info.status == "active"

    def test_agent_info_defaults(self):
        """Test AgentInfo default values."""
        agent_info = AgentInfo(
            agent_id="agent123",
            capabilities=["task1"],
            endpoint="ws://localhost:8080",
            last_seen=1234567890.0,
        )

        assert agent_info.status == "active"


class TestA2AProtocol:
    """Test A2AProtocol class."""

    @pytest.fixture
    def protocol(self):
        """Create A2A protocol instance."""
        return A2AProtocol(
            agent_id="test_agent", host="localhost", port=8080, discovery_enabled=True
        )

    def test_protocol_initialization(self, protocol):
        """Test A2AProtocol initialization."""
        assert protocol.agent_id == "test_agent"
        assert protocol.host == "localhost"
        assert protocol.port == 8080
        assert protocol.discovery_enabled is True

        # Check internal state
        assert protocol._server is None
        assert protocol._connections == {}
        assert protocol._client_connections == {}
        assert protocol._known_agents == {}
        assert protocol._capabilities == set()
        assert isinstance(protocol._incoming_queue, asyncio.Queue)
        assert isinstance(protocol._outgoing_queue, asyncio.Queue)
        assert protocol._tasks == set()

        # Check stats
        expected_stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "connections_active": 0,
            "errors": 0,
        }
        assert protocol._stats == expected_stats

    def test_protocol_initialization_defaults(self):
        """Test A2AProtocol initialization with defaults."""
        protocol = A2AProtocol("test_agent")

        assert protocol.agent_id == "test_agent"
        assert protocol.host == "localhost"
        assert protocol.port == 8080
        assert protocol.discovery_enabled is True

    def test_add_capability(self, protocol):
        """Test adding capabilities."""
        protocol.add_capability("task_execution")
        protocol.add_capability("file_processing")

        assert "task_execution" in protocol._capabilities
        assert "file_processing" in protocol._capabilities
        assert len(protocol._capabilities) == 2

    def test_add_duplicate_capability(self, protocol):
        """Test adding duplicate capability."""
        protocol.add_capability("task_execution")
        protocol.add_capability("task_execution")

        # Should only have one instance (set behavior)
        assert len(protocol._capabilities) == 1
        assert "task_execution" in protocol._capabilities

    def test_get_stats(self, protocol):
        """Test getting statistics."""
        # Modify some stats
        protocol._stats["messages_sent"] = 5
        protocol._stats["messages_received"] = 3
        protocol._stats["errors"] = 1

        # Add some mock connections
        protocol._connections["agent1"] = Mock()
        protocol._client_connections["agent2"] = Mock()

        stats = protocol.get_stats()

        assert stats["messages_sent"] == 5
        assert stats["messages_received"] == 3
        assert stats["errors"] == 1
        assert stats["connections_active"] == 2  # 1 server + 1 client connection

        # Ensure we get a copy, not the original
        stats["messages_sent"] = 100
        assert protocol._stats["messages_sent"] == 5

    @pytest.mark.asyncio
    async def test_send_a2a_message(self, protocol):
        """Test sending A2A message."""
        message = A2AMessage(sender_id="test_agent", message_type="test", payload={"data": "test"})

        result = await protocol.send(message)

        assert result is True
        # Message should be in outgoing queue
        assert not protocol._outgoing_queue.empty()
        queued_message = await protocol._outgoing_queue.get()
        assert queued_message.sender_id == "test_agent"
        assert queued_message.message_type == "test"
        assert queued_message.payload == {"data": "test"}

    @pytest.mark.asyncio
    async def test_send_protocol_message_conversion(self, protocol):
        """Test sending generic ProtocolMessage gets converted to A2AMessage."""
        from aida.core.protocols.base import ProtocolMessage

        generic_message = ProtocolMessage(
            sender_id="test_agent", message_type="test", payload={"data": "test"}
        )

        result = await protocol.send(generic_message)

        assert result is True
        # Should be converted to A2AMessage
        queued_message = await protocol._outgoing_queue.get()
        assert isinstance(queued_message, A2AMessage)
        assert queued_message.sender_id == "test_agent"
        assert queued_message.message_type == "test"
        assert queued_message.priority == 5  # Default A2A priority
        assert queued_message.requires_ack is False  # Default A2A value

    @pytest.mark.asyncio
    async def test_receive_timeout(self, protocol):
        """Test receive with timeout."""
        # Queue is empty, should timeout and return None
        result = await protocol.receive()
        assert result is None

    @pytest.mark.asyncio
    async def test_receive_message(self, protocol):
        """Test receiving message from queue."""
        message = A2AMessage(
            sender_id="other_agent", message_type="response", payload={"result": "success"}
        )

        # Put message in incoming queue
        await protocol._incoming_queue.put(message)

        # Should receive the message
        received = await protocol.receive()
        assert received is not None
        assert received.sender_id == "other_agent"
        assert received.message_type == "response"
        assert received.payload == {"result": "success"}

    @pytest.mark.asyncio
    async def test_discover_agents(self, protocol):
        """Test agent discovery."""
        # Add some known agents
        protocol._known_agents["agent1"] = AgentInfo(
            agent_id="agent1",
            capabilities=["task1"],
            endpoint="ws://host1:8080",
            last_seen=1234567890.0,
        )
        protocol._known_agents["agent2"] = AgentInfo(
            agent_id="agent2",
            capabilities=["task2"],
            endpoint="ws://host2:8080",
            last_seen=1234567891.0,
        )

        # Add capabilities to this agent
        protocol.add_capability("discovery")

        with patch.object(protocol, "_send_to_agent") as mock_send:
            agents = await protocol.discover_agents()

            # Should return list of known agents
            assert len(agents) == 2
            assert any(agent.agent_id == "agent1" for agent in agents)
            assert any(agent.agent_id == "agent2" for agent in agents)

            # Should have sent discovery messages to both agents
            assert mock_send.call_count == 2

    @pytest.mark.asyncio
    async def test_discover_agents_empty(self, protocol):
        """Test agent discovery with no known agents."""
        with patch.object(protocol, "_send_to_agent") as mock_send:
            agents = await protocol.discover_agents()

            assert agents == []
            assert mock_send.call_count == 0
