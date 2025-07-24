"""Tool system for AIDA agents."""

from aida.tools.base import (
    Tool,
    ToolError,
    ToolRegistry,
    ToolResult,
    get_tool_registry,
    initialize_default_tools,
)
from aida.tools.context import ContextTool

# Modular tools
from aida.tools.execution import ExecutionTool

# Legacy tools (to be refactored)
from aida.tools.files import FileOperationsTool
from aida.tools.llm_response import LLMResponseTool
from aida.tools.system import SystemTool
from aida.tools.thinking import ThinkingTool

# Legacy tools removed - see docs/tools/README.md for current hybrid tools

__all__ = [
    "Tool",
    "ToolResult",
    "ToolError",
    "ToolRegistry",
    "get_tool_registry",
    "initialize_default_tools",
    "ExecutionTool",
    "FileOperationsTool",
    "SystemTool",
    "ContextTool",
    "LLMResponseTool",
    "ThinkingTool",
]
