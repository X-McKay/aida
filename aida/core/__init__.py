"""Core AIDA system components."""

from aida.core.events import Event, EventBus
from aida.core.protocols import A2AProtocol, MCPProtocol, Protocol

__all__ = [
    "A2AProtocol",
    "MCPProtocol",
    "Protocol",
    "Event",
    "EventBus",
]
