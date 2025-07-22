"""Tests for base protocol module."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from aida.core.protocols.base import Protocol, ProtocolMessage


class TestProtocolMessage:
    """Test ProtocolMessage class."""

    def test_message_creation(self):
        """Test creating a message."""
        msg = ProtocolMessage(
            id="msg_123",
            sender_id="agent_1",
            recipient_id="agent_2",
            message_type="test_message",
            payload={"action": "test"},
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )

        assert msg.id == "msg_123"
        assert msg.sender_id == "agent_1"
        assert msg.recipient_id == "agent_2"
        assert msg.message_type == "test_message"
        assert msg.payload == {"action": "test"}
        assert msg.timestamp == datetime(2024, 1, 1, 12, 0, 0)

    def test_message_defaults(self):
        """Test message with default values."""
        msg = ProtocolMessage(sender_id="agent_1", message_type="test_message")

        assert msg.id is not None  # Should generate UUID
        assert isinstance(msg.timestamp, datetime)
        assert msg.recipient_id is None
        assert msg.payload == {}
        assert msg.metadata == {}

    def test_message_with_metadata(self):
        """Test message with metadata."""
        metadata = {"priority": "high", "retry_count": 0}
        msg = ProtocolMessage(sender_id="agent_1", message_type="test_message", metadata=metadata)

        assert msg.metadata == metadata

    def test_message_dict_conversion(self):
        """Test converting message to dict."""
        msg = ProtocolMessage(
            id="msg_123",
            sender_id="agent_1",
            recipient_id="agent_2",
            message_type="test_message",
            payload={"action": "test"},
        )

        msg_dict = msg.model_dump()
        assert msg_dict["id"] == "msg_123"
        assert msg_dict["sender_id"] == "agent_1"
        assert msg_dict["recipient_id"] == "agent_2"
        assert msg_dict["message_type"] == "test_message"
        assert msg_dict["payload"] == {"action": "test"}


class TestProtocol:
    """Test Protocol base class."""

    @pytest.fixture
    def protocol_impl(self):
        """Create a test protocol implementation."""

        class TestProtocol(Protocol):
            def __init__(self):
                super().__init__("test_agent_123")

            async def send(self, message: ProtocolMessage) -> bool:
                """Send a message."""
                return True

            async def receive(self) -> ProtocolMessage | None:
                """Receive a message."""
                return None

            async def connect(self) -> bool:
                """Establish connection."""
                return True

            async def disconnect(self) -> None:
                """Close connection."""
                pass

        return TestProtocol()

    def test_protocol_initialization(self, protocol_impl):
        """Test protocol initialization."""
        assert protocol_impl.agent_id == "test_agent_123"
        assert protocol_impl._handlers == {}
        assert protocol_impl._middleware == []

    def test_register_handler(self, protocol_impl):
        """Test registering a message handler."""
        handler = Mock()
        protocol_impl.register_handler("test_action", handler)

        assert "test_action" in protocol_impl._handlers
        assert protocol_impl._handlers["test_action"] is handler

    def test_add_middleware(self, protocol_impl):
        """Test adding middleware."""
        middleware = Mock()
        protocol_impl.add_middleware(middleware)

        assert middleware in protocol_impl._middleware
        assert len(protocol_impl._middleware) == 1

    @pytest.mark.asyncio
    async def test_handle_message_with_handler(self, protocol_impl):
        """Test handling message with registered handler."""
        # Register a handler
        handler = AsyncMock(return_value={"result": "success"})
        protocol_impl.register_handler("test_action", handler)

        # Create a message
        msg = ProtocolMessage(
            sender_id="agent_1", message_type="test_action", payload={"data": "test"}
        )

        # Handle the message
        result = await protocol_impl.handle_message(msg)

        handler.assert_called_once_with(msg)
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_handle_message_without_handler(self, protocol_impl):
        """Test handling message without registered handler."""
        # Create a message with no handler
        msg = ProtocolMessage(
            sender_id="agent_1", message_type="unknown_action", payload={"data": "test"}
        )

        result = await protocol_impl.handle_message(msg)

        assert result is None

    @pytest.mark.asyncio
    async def test_handle_message_with_middleware(self, protocol_impl):
        """Test handling message with middleware."""

        # Add middleware that modifies the message
        async def test_middleware(message):
            message.metadata["processed"] = True
            return message

        protocol_impl.add_middleware(test_middleware)

        # Register a handler
        handler = AsyncMock(return_value={"result": "success"})
        protocol_impl.register_handler("test_action", handler)

        # Create a message
        msg = ProtocolMessage(sender_id="agent_1", message_type="test_action")

        # Handle the message
        await protocol_impl.handle_message(msg)

        # Check middleware was applied
        assert msg.metadata["processed"] is True

    @pytest.mark.asyncio
    async def test_handle_message_middleware_blocks(self, protocol_impl):
        """Test middleware can block message processing."""

        # Add middleware that returns None
        async def blocking_middleware(message):
            return None

        protocol_impl.add_middleware(blocking_middleware)

        # Register a handler that should not be called
        handler = AsyncMock()
        protocol_impl.register_handler("test_action", handler)

        # Create a message
        msg = ProtocolMessage(sender_id="agent_1", message_type="test_action")

        # Handle the message
        result = await protocol_impl.handle_message(msg)

        assert result is None
        handler.assert_not_called()

    def test_create_message(self, protocol_impl):
        """Test creating a new protocol message."""
        msg = protocol_impl.create_message(
            message_type="test_type",
            recipient_id="agent_456",
            payload={"data": "test"},
            metadata={"priority": "high"},
        )

        assert msg.sender_id == "test_agent_123"
        assert msg.recipient_id == "agent_456"
        assert msg.message_type == "test_type"
        assert msg.payload == {"data": "test"}
        assert msg.metadata == {"priority": "high"}

    def test_create_message_defaults(self, protocol_impl):
        """Test creating message with defaults."""
        msg = protocol_impl.create_message(message_type="test_type")

        assert msg.sender_id == "test_agent_123"
        assert msg.recipient_id is None
        assert msg.message_type == "test_type"
        assert msg.payload == {}
        assert msg.metadata == {}

    @pytest.mark.asyncio
    async def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented."""
        # Protocol without all methods implemented
        with pytest.raises(TypeError):

            class IncompleteProtocol(Protocol):
                async def send(self, message: ProtocolMessage) -> bool:
                    pass

                # Missing receive, connect, disconnect

            IncompleteProtocol("test")


if __name__ == "__main__":
    pytest.main([__file__])
