"""Communication protocols for AIDA agents."""

from aida.core.protocols.base import Protocol, ProtocolMessage
from aida.core.protocols.a2a import A2AProtocol, A2AMessage
from aida.core.protocols.mcp import MCPProtocol, MCPMessage

__all__ = [
    "Protocol",
    "ProtocolMessage", 
    "A2AProtocol",
    "A2AMessage",
    "MCPProtocol",
    "MCPMessage",
]