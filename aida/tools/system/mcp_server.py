"""MCP server implementation for system tool."""

import logging
from typing import Any

from aida.tools.base_mcp import SimpleMCPServer

logger = logging.getLogger(__name__)


class SystemMCPServer(SimpleMCPServer):
    """MCP server wrapper for SystemTool."""

    def __init__(self, system_tool):
        """Initialize the SystemMCPServer with the system tool.

        Args:
            system_tool: Instance of SystemTool to wrap with MCP protocol
        """
        operations = {
            "execute": {
                "description": "Execute a system command",
                "parameters": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command arguments",
                    },
                    "timeout": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 300,
                        "description": "Timeout in seconds",
                    },
                    "cwd": {"type": "string", "description": "Working directory"},
                },
                "required": ["command"],
                "handler": self._handle_execute,
            },
            "system_info": {
                "description": "Get system information",
                "parameters": {},
                "required": [],
                "handler": self._handle_system_info,
            },
            "processes": {
                "description": "List running processes",
                "parameters": {},
                "required": [],
                "handler": self._handle_processes,
            },
            "which": {
                "description": "Find command in PATH",
                "parameters": {"command": {"type": "string", "description": "Command to find"}},
                "required": ["command"],
                "handler": self._handle_which,
            },
            "env": {
                "description": "Get environment variables",
                "parameters": {
                    "name": {"type": "string", "description": "Variable name (optional)"}
                },
                "required": [],
                "handler": self._handle_env,
            },
        }
        super().__init__(system_tool, operations)

    async def _handle_execute(self, arguments: dict[str, Any]):
        """Handle command execution."""
        result = await self.tool.execute(
            operation="execute",
            command=arguments["command"],
            args=arguments.get("args", []),
            timeout=arguments.get("timeout", 30),
            cwd=arguments.get("cwd"),
        )
        if result.status.value == "completed":
            cmd_result = result.result
            return {
                "exit_code": cmd_result.exit_code,
                "stdout": cmd_result.stdout,
                "stderr": cmd_result.stderr,
                "timed_out": cmd_result.timed_out,
            }
        else:
            raise Exception(result.error or "Command execution failed")

    async def _handle_system_info(self, arguments: dict[str, Any]):
        """Handle system info request."""
        result = await self.tool.execute(operation="system_info")
        if result.status.value == "completed":
            return result.result.dict()
        else:
            raise Exception(result.error or "Failed to get system info")

    async def _handle_processes(self, arguments: dict[str, Any]):
        """Handle process list request."""
        result = await self.tool.execute(operation="process_list")
        if result.status.value == "completed":
            # Return simplified process list
            processes = []
            for proc in result.result[:20]:  # Limit to 20 processes
                processes.append(
                    {
                        "pid": proc.pid,
                        "name": proc.name,
                        "status": proc.status,
                        "cpu_percent": proc.cpu_percent,
                        "memory_percent": proc.memory_percent,
                    }
                )
            return processes
        else:
            raise Exception(result.error or "Failed to list processes")

    async def _handle_which(self, arguments: dict[str, Any]):
        """Handle which command."""
        result = await self.tool.execute(operation="which", command=arguments["command"])
        if result.status.value == "completed":
            return {"path": result.result}
        else:
            return {"path": None, "error": result.error}

    async def _handle_env(self, arguments: dict[str, Any]):
        """Handle environment variable request."""
        result = await self.tool.execute(operation="env_get", var_name=arguments.get("name"))
        if result.status.value == "completed":
            return result.result
        else:
            raise Exception(result.error or "Failed to get environment")
