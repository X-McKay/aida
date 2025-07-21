"""Base protocol interface for AIDA communication."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar
import uuid

from pydantic import BaseModel, Field

T = TypeVar("T", bound="ProtocolMessage")


class ProtocolMessage(BaseModel):
    """Base class for all protocol messages."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sender_id: str
    recipient_id: str | None = None
    message_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class Protocol(ABC):
    """Abstract base class for communication protocols."""

    def __init__(self, agent_id: str):
        """Initialize the protocol with an agent ID.

        Args:
            agent_id: Unique identifier for the agent using this protocol.
                Used as the sender_id when creating messages.
        """
        self.agent_id = agent_id
        self._handlers: dict[str, Callable] = {}
        self._middleware: list[Callable] = []

    @abstractmethod
    async def send(self, message: ProtocolMessage) -> bool:
        """Send a message using this protocol."""
        pass

    @abstractmethod
    async def receive(self) -> ProtocolMessage | None:
        """Receive a message using this protocol."""
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection for this protocol."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection for this protocol."""
        pass

    def register_handler(self, message_type: str, handler: Callable) -> None:
        """Register a message handler for a specific message type."""
        self._handlers[message_type] = handler

    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware for message processing."""
        self._middleware.append(middleware)

    async def handle_message(self, message: ProtocolMessage) -> Any:
        """Process an incoming message through handlers and middleware."""
        # Apply middleware
        for middleware in self._middleware:
            message = await middleware(message)
            if message is None:
                return None

        # Find and execute handler
        handler = self._handlers.get(message.message_type)
        if handler:
            return await handler(message)

        return None

    def create_message(
        self,
        message_type: str,
        recipient_id: str | None = None,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProtocolMessage:
        """Create a new protocol message."""
        return ProtocolMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload or {},
            metadata=metadata or {},
        )
