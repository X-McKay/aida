"""AIDA - Advanced Intelligent Distributed Agent System."""

__version__ = "1.0.0"
__author__ = "AIDA Development Team"
__email__ = "dev@aida.ai"
__description__ = "A comprehensive, production-ready agentic system"

from aida.core.events import Event, EventBus
from aida.core.protocols import A2AProtocol, MCPProtocol

__all__ = [
    "A2AProtocol",
    "MCPProtocol",
    "Event",
    "EventBus",
    "__version__",
]
