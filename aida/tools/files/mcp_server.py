"""MCP server implementation for file operations tool."""

import json
import logging
from typing import Any

from aida.tools.base_mcp import SimpleMCPServer

logger = logging.getLogger(__name__)


class FilesMCPServer(SimpleMCPServer):
    """MCP server wrapper for FileOperationsTool."""

    def _format_response(self, result: Any) -> dict[str, Any]:
        """Format response ensuring proper JSON serialization for all types."""
        # Always JSON serialize the result to ensure consistent parsing
        response_text = json.dumps(result, indent=2)

        return {"content": [{"type": "text", "text": response_text}]}

    def __init__(self, files_tool):
        operations = {
            "read": {
                "description": "Read file contents",
                "parameters": {
                    "path": {"type": "string", "description": "File path to read"},
                    "encoding": {
                        "type": "string",
                        "description": "File encoding",
                        "default": "utf-8",
                    },
                },
                "required": ["path"],
                "handler": self._handle_read,
            },
            "write": {
                "description": "Write content to file",
                "parameters": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                    "encoding": {
                        "type": "string",
                        "description": "File encoding",
                        "default": "utf-8",
                    },
                },
                "required": ["path", "content"],
                "handler": self._handle_write,
            },
            "list": {
                "description": "List directory contents",
                "parameters": {
                    "path": {"type": "string", "description": "Directory path"},
                    "recursive": {
                        "type": "boolean",
                        "description": "List recursively",
                        "default": False,
                    },
                },
                "required": ["path"],
                "handler": self._handle_list,
            },
            "search": {
                "description": "Search for text in files",
                "parameters": {
                    "path": {"type": "string", "description": "Path to search in"},
                    "text": {"type": "string", "description": "Text to search for"},
                    "recursive": {
                        "type": "boolean",
                        "description": "Search recursively",
                        "default": True,
                    },
                },
                "required": ["path", "text"],
                "handler": self._handle_search,
            },
            "delete": {
                "description": "Delete file or directory",
                "parameters": {
                    "path": {"type": "string", "description": "Path to delete"},
                    "recursive": {
                        "type": "boolean",
                        "description": "Delete directories recursively",
                        "default": False,
                    },
                },
                "required": ["path"],
                "handler": self._handle_delete,
            },
        }
        super().__init__(files_tool, operations)

    async def _handle_read(self, arguments: dict[str, Any]):
        """Handle read operation."""
        result = await self.tool.execute(
            operation="read", path=arguments["path"], encoding=arguments.get("encoding", "utf-8")
        )
        if result.status.value == "completed":
            return result.result
        else:
            raise Exception(result.error or "Read failed")

    async def _handle_write(self, arguments: dict[str, Any]):
        """Handle write operation."""
        result = await self.tool.execute(
            operation="write",
            path=arguments["path"],
            content=arguments["content"],
            encoding=arguments.get("encoding", "utf-8"),
        )
        if result.status.value == "completed":
            return {"success": True, "path": arguments["path"]}
        else:
            raise Exception(result.error or "Write failed")

    async def _handle_list(self, arguments: dict[str, Any]):
        """Handle list directory operation."""
        result = await self.tool.execute(
            operation="list_dir",
            path=arguments["path"],
            recursive=arguments.get("recursive", False),
        )
        if result.status.value == "completed":
            return result.result
        else:
            raise Exception(result.error or "List failed")

    async def _handle_search(self, arguments: dict[str, Any]):
        """Handle search operation."""
        result = await self.tool.execute(
            operation="search",
            path=arguments["path"],
            search_text=arguments["text"],
            recursive=arguments.get("recursive", True),
        )
        if result.status.value == "completed":
            return result.result
        else:
            raise Exception(result.error or "Search failed")

    async def _handle_delete(self, arguments: dict[str, Any]):
        """Handle delete operation."""
        result = await self.tool.execute(
            operation="delete", path=arguments["path"], recursive=arguments.get("recursive", False)
        )
        if result.status.value == "completed":
            return {"success": True, "path": arguments["path"]}
        else:
            raise Exception(result.error or "Delete failed")
