"""UI components for AIDA CLI."""

from aida.cli.ui.console import get_console, setup_console
from aida.cli.ui.progress import ProgressTracker, SimpleProgress
from aida.cli.ui.status import StatusDisplay
from aida.cli.ui.tables import (
    create_agent_table,
    create_health_table,
    create_provider_table,
    create_stats_table,
    create_tool_table,
)

__all__ = [
    "get_console",
    "setup_console",
    "ProgressTracker",
    "SimpleProgress",
    "StatusDisplay",
    "create_agent_table",
    "create_provider_table",
    "create_stats_table",
    "create_tool_table",
    "create_health_table",
]
