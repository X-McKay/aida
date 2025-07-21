"""MCP server implementation for context tool."""

import json
import logging
from typing import Any

from .models import ContextPriority

logger = logging.getLogger(__name__)


class ContextMCPServer:
    """MCP server wrapper for ContextTool."""

    def __init__(self, context_tool):
        """Initialize MCP server with context tool.

        Args:
            context_tool: The context tool instance to wrap for MCP
        """
        self.context_tool = context_tool
        self.server_info = {
            "name": "aida-context",
            "version": context_tool.version,
            "description": context_tool.description,
        }

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available MCP tools."""
        tools = []

        # Compress tool
        tools.append(
            {
                "name": "context_compress",
                "description": "Compress context while preserving important information",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to compress"},
                        "compression_level": {
                            "type": "number",
                            "minimum": 0.1,
                            "maximum": 0.9,
                            "description": "Compression level (0.1 = keep 10%, 0.9 = keep 90%)",
                        },
                        "priority": {
                            "type": "string",
                            "enum": [p.value for p in ContextPriority],
                            "description": "Priority for content selection",
                        },
                    },
                    "required": ["content"],
                },
            }
        )

        # Summarize tool
        tools.append(
            {
                "name": "context_summarize",
                "description": "Create a summary of context content",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to summarize"},
                        "max_tokens": {
                            "type": "integer",
                            "minimum": 100,
                            "maximum": 10000,
                            "description": "Maximum tokens in summary",
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["structured", "narrative", "bullet_points"],
                            "description": "Output format",
                        },
                    },
                    "required": ["content"],
                },
            }
        )

        # Extract key points
        tools.append(
            {
                "name": "context_extract_points",
                "description": "Extract key points from context",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to analyze"},
                        "max_points": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20,
                            "description": "Maximum number of key points",
                        },
                    },
                    "required": ["content"],
                },
            }
        )

        # Search tool
        tools.append(
            {
                "name": "context_search",
                "description": "Search for information within context",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to search in"},
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "description": "Maximum search results",
                        },
                    },
                    "required": ["content", "query"],
                },
            }
        )

        # Export tool
        tools.append(
            {
                "name": "context_export",
                "description": "Export context to a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to export"},
                        "file_path": {"type": "string", "description": "Output file path"},
                        "format": {
                            "type": "string",
                            "enum": ["json", "markdown", "yaml", "text"],
                            "description": "Export format",
                        },
                    },
                    "required": ["content", "file_path"],
                },
            }
        )

        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP tool call."""
        try:
            if name == "context_compress":
                result = await self.context_tool.execute(
                    operation="compress",
                    content=arguments["content"],
                    compression_level=arguments.get("compression_level", 0.5),
                    priority=arguments.get("priority", "balanced"),
                )

            elif name == "context_summarize":
                result = await self.context_tool.execute(
                    operation="summarize",
                    content=arguments["content"],
                    max_tokens=arguments.get("max_tokens", 2000),
                    output_format=arguments.get("output_format", "structured"),
                )

            elif name == "context_extract_points":
                result = await self.context_tool.execute(
                    operation="extract_key_points",
                    content=arguments["content"],
                    max_results=arguments.get("max_points", 10),
                )

            elif name == "context_search":
                result = await self.context_tool.execute(
                    operation="search",
                    content=arguments["content"],
                    query=arguments["query"],
                    max_results=arguments.get("max_results", 10),
                )

            elif name == "context_export":
                result = await self.context_tool.execute(
                    operation="export",
                    content=arguments["content"],
                    file_path=arguments["file_path"],
                    format_type=arguments.get("format", "json"),
                )

            else:
                return {
                    "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
                    "isError": True,
                }

            # Format result based on type
            if isinstance(result.result, str):
                response_text = result.result
            elif isinstance(result.result, list):
                response_text = "\n".join(f"• {item}" for item in result.result)
            else:
                response_text = json.dumps(result.result, indent=2)

            return {"content": [{"type": "text", "text": response_text}]}

        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            return {"content": [{"type": "text", "text": f"Error: {str(e)}"}], "isError": True}
