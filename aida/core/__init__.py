"""Core AIDA system components."""

from aida.core.agent import Agent, BaseAgent
from aida.core.events import Event, EventBus, EventHandler
from aida.core.memory import MemoryEntry, MemoryManager, MemoryStore
from aida.core.protocols import A2AProtocol, MCPProtocol, Protocol
from aida.core.state import AgentState, StateManager, SystemState

__all__ = [
    "Agent",
    "BaseAgent",
    "A2AProtocol",
    "MCPProtocol",
    "Protocol",
    "Event",
    "EventBus",
    "EventHandler",
    "AgentState",
    "SystemState",
    "StateManager",
    "MemoryManager",
    "MemoryEntry",
    "MemoryStore",
]
