"""AIDA - Advanced Intelligent Distributed Agent System."""

__version__ = "1.0.0"
__author__ = "AIDA Development Team"
__email__ = "dev@aida.ai"
__description__ = "A comprehensive, production-ready agentic system"

from aida.core.agent import Agent
from aida.core.protocols import A2AProtocol, MCPProtocol
from aida.core.state import AgentState, SystemState
from aida.core.events import Event, EventBus

__all__ = [
    "Agent",
    "A2AProtocol", 
    "MCPProtocol",
    "AgentState",
    "SystemState",
    "Event",
    "EventBus",
    "__version__",
]