"""Base MCP server implementation for AIDA tools."""

from abc import ABC, abstractmethod
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class BaseMCPServer(ABC):
    """Base MCP server implementation for consistent tool interface."""

    def __init__(self, tool):
        """Initialize MCP server with tool reference."""
        self.tool = tool
        self.server_info = {
            "name": f"aida-{tool.name}",
            "version": tool.version,
            "description": tool.description,
        }
        self._tool_definitions = None

    def list_tools(self) -> list[dict[str, Any]]:
        """List available MCP tools."""
        if self._tool_definitions is None:
            self._tool_definitions = self._create_tool_definitions()
        return self._tool_definitions

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP tool call."""
        try:
            # Route to specific handler
            handler = self._get_tool_handler(name)
            if not handler:
                return self._format_error(f"Unknown tool: {name}")

            # Execute handler
            result = await handler(arguments)

            # Format response
            return self._format_response(result)

        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            return self._format_error(str(e))

    @abstractmethod
    def _create_tool_definitions(self) -> list[dict[str, Any]]:
        """Create MCP tool definitions."""
        pass

    @abstractmethod
    def _get_tool_handler(self, name: str) -> callable | None:
        """Get handler for specific tool name."""
        pass

    def _format_response(self, result: Any) -> dict[str, Any]:
        """Format successful response for MCP protocol."""
        # Handle different result types
        if isinstance(result, str):
            response_text = result
        elif isinstance(result, list):
            response_text = "\n".join(f"• {item}" for item in result)
        elif isinstance(result, dict):
            response_text = json.dumps(result, indent=2)
        else:
            response_text = str(result)

        return {"content": [{"type": "text", "text": response_text}]}

    def _format_error(self, error_message: str) -> dict[str, Any]:
        """Format error response for MCP protocol."""
        return {"content": [{"type": "text", "text": error_message}], "isError": True}

    def _create_tool_schema(
        self, name: str, description: str, properties: dict[str, Any], required: list[str] = None
    ) -> dict[str, Any]:
        """Helper to create consistent tool schemas."""
        return {
            "name": name,
            "description": description,
            "inputSchema": {"type": "object", "properties": properties, "required": required or []},
        }


class SimpleMCPServer(BaseMCPServer):
    """Simple MCP server for tools with basic operations."""

    def __init__(self, tool, operations: dict[str, dict[str, Any]]):
        """Initialize with tool and operation definitions.

        Args:
            tool: The tool instance
            operations: Dict mapping operation names to their MCP definitions
                Each definition should have:
                - description: Tool description
                - parameters: Dict of parameter definitions
                - required: List of required parameter names
                - handler: Async function to handle the operation
        """
        super().__init__(tool)
        self.operations = operations

    def _create_tool_definitions(self) -> list[dict[str, Any]]:
        """Create MCP tool definitions from operations."""
        tools = []

        for op_name, op_def in self.operations.items():
            tool_name = f"{self.tool.name}_{op_name}"
            tools.append(
                self._create_tool_schema(
                    name=tool_name,
                    description=op_def["description"],
                    properties=op_def["parameters"],
                    required=op_def.get("required", []),
                )
            )

        return tools

    def _get_tool_handler(self, name: str) -> callable | None:
        """Get handler for specific tool name."""
        # Extract operation from tool name
        prefix = f"{self.tool.name}_"
        if not name.startswith(prefix):
            return None

        op_name = name[len(prefix) :]
        op_def = self.operations.get(op_name)

        if op_def and "handler" in op_def:
            return op_def["handler"]

        # Default handler that calls tool.execute
        async def default_handler(arguments: dict[str, Any]):
            result = await self.tool.execute(**arguments)
            if result.status == "completed":
                return result.result
            else:
                raise Exception(result.error or "Operation failed")

        return default_handler
