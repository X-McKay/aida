"""Communication protocols for AIDA agents."""

from aida.core.protocols.a2a import A2AMessage, A2AProtocol
from aida.core.protocols.base import Protocol, ProtocolMessage
from aida.core.protocols.mcp import MCPMessage, MCPProtocol

__all__ = [
    "Protocol",
    "ProtocolMessage",
    "A2AProtocol",
    "A2AMessage",
    "MCPProtocol",
    "MCPMessage",
]
