"""MCP server implementation for execution tool."""

import logging
from typing import Any

from aida.tools.base_mcp import SimpleMCPServer

from .models import ExecutionLanguage

logger = logging.getLogger(__name__)


class ExecutionMCPServer(SimpleMCPServer):
    """MCP server wrapper for ExecutionTool."""

    def __init__(self, execution_tool):
        """Initialize the MCP server for execution tool.

        Args:
            execution_tool: The ExecutionTool instance to wrap with MCP server capabilities.
                Provides code execution functionality in various languages.
        """
        operations = {
            "execute": {
                "description": "Execute code in a secure container",
                "parameters": {
                    "language": {
                        "type": "string",
                        "enum": [lang.value for lang in ExecutionLanguage],
                        "description": "Programming language",
                    },
                    "code": {"type": "string", "description": "Code to execute"},
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Package dependencies",
                    },
                    "timeout": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 300,
                        "description": "Timeout in seconds",
                    },
                },
                "required": ["language", "code"],
            },
            "run_python": {
                "description": "Run Python code",
                "parameters": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Python packages to install",
                    },
                },
                "required": ["code"],
                "handler": self._handle_run_python,
            },
            "run_script": {
                "description": "Run a shell script",
                "parameters": {
                    "script": {"type": "string", "description": "Shell script to execute"}
                },
                "required": ["script"],
                "handler": self._handle_run_script,
            },
        }
        super().__init__(execution_tool, operations)

    async def _handle_run_python(self, arguments: dict[str, Any]):
        """Handle Python execution."""
        result = await self.tool.execute(
            language="python", code=arguments["code"], packages=arguments.get("packages", [])
        )
        if result.status == "completed":
            return result.result
        else:
            raise Exception(result.error or "Execution failed")

    async def _handle_run_script(self, arguments: dict[str, Any]):
        """Handle shell script execution."""
        result = await self.tool.execute(language="bash", code=arguments["script"])
        if result.status == "completed":
            return result.result
        else:
            raise Exception(result.error or "Script execution failed")
