"""Base protocol interface for AIDA communication."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar
from datetime import datetime
import uuid

from pydantic import BaseModel, Field


T = TypeVar("T", bound="ProtocolMessage")


class ProtocolMessage(BaseModel):
    """Base class for all protocol messages."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sender_id: str
    recipient_id: Optional[str] = None
    message_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class Protocol(ABC):
    """Abstract base class for communication protocols."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._handlers: Dict[str, callable] = {}
        self._middleware: list = []
    
    @abstractmethod
    async def send(self, message: ProtocolMessage) -> bool:
        """Send a message using this protocol."""
        pass
    
    @abstractmethod
    async def receive(self) -> Optional[ProtocolMessage]:
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
    
    def register_handler(self, message_type: str, handler: callable) -> None:
        """Register a message handler for a specific message type."""
        self._handlers[message_type] = handler
    
    def add_middleware(self, middleware: callable) -> None:
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
        recipient_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProtocolMessage:
        """Create a new protocol message."""
        return ProtocolMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload or {},
            metadata=metadata or {}
        )