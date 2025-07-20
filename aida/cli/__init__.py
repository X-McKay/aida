"""CLI interface for AIDA."""

from aida.cli.main import main, app
from aida.cli.commands import *
from aida.cli.ui import get_console, ProgressTracker, StatusDisplay

__all__ = [
    "main",
    "app", 
    "get_console",
    "ProgressTracker",
    "StatusDisplay",
]