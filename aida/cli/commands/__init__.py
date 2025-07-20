"""CLI commands for AIDA."""

from aida.cli.commands.agent import agent_app
from aida.cli.commands.config import config_app  
from aida.cli.commands.llm import llm_app
from aida.cli.commands.system import system_app
from aida.cli.commands.interactive import interactive_app
from aida.cli.commands.tools import tools_app

__all__ = [
    "agent_app",
    "config_app",
    "llm_app", 
    "system_app",
    "interactive_app",
    "tools_app",
]