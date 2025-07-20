"""Tool system for AIDA agents."""

from aida.tools.base import Tool, ToolResult, ToolError, ToolRegistry, get_tool_registry, initialize_default_tools

# Modular tools
from aida.tools.execution import ExecutionTool
from aida.tools.context import ContextTool
from aida.tools.llm_response import LLMResponseTool
from aida.tools.thinking import ThinkingTool

# Legacy tools (to be refactored)
from aida.tools.files import FileOperationsTool
from aida.tools.system import SystemTool

# Non-refactored tools commented out to meet deadline
# from aida.tools.maintenance import MaintenanceTool
# from aida.tools.project import ProjectTool
# from aida.tools.architecture import ArchitectureTool

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