"""Tests for context MCP server."""

from datetime import datetime
import json
from unittest.mock import AsyncMock, Mock, patch
import uuid

import pytest

from aida.tools.base import ToolResult, ToolStatus
from aida.tools.context.mcp_server import ContextMCPServer
from aida.tools.context.models import ContextPriority


class TestContextMCPServer:
    """Test ContextMCPServer class."""

    def _create_tool_result(self, status=ToolStatus.COMPLETED, result=None, error=None):
        """Helper to create a valid ToolResult."""
        return ToolResult(
            tool_name="context",
            execution_id=str(uuid.uuid4()),
            status=status,
            result=result,
            error=error,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
            if status in [ToolStatus.COMPLETED, ToolStatus.FAILED]
            else None,
            duration_seconds=0.1 if status in [ToolStatus.COMPLETED, ToolStatus.FAILED] else None,
        )

    @pytest.fixture
    def mock_tool(self):
        """Create a mock context tool."""
        tool = Mock()
        tool.execute = AsyncMock()
        tool.version = "2.0.0"
        tool.description = "Context management tool"
        return tool

    @pytest.fixture
    def server(self, mock_tool):
        """Create a context MCP server."""
        return ContextMCPServer(mock_tool)

    def test_initialization(self, server, mock_tool):
        """Test server initialization."""
        assert server.context_tool == mock_tool
        assert server.server_info["name"] == "aida-context"
        assert server.server_info["version"] == "2.0.0"
        assert server.server_info["description"] == "Context management tool"

    @pytest.mark.asyncio
    async def test_list_tools(self, server):
        """Test listing available tools."""
        tools = await server.list_tools()

        # Should have 5 tools
        assert len(tools) == 5

        tool_names = [tool["name"] for tool in tools]
        assert "context_compress" in tool_names
        assert "context_summarize" in tool_names
        assert "context_extract_points" in tool_names
        assert "context_search" in tool_names
        assert "context_export" in tool_names

        # Check compress tool schema
        compress_tool = next(t for t in tools if t["name"] == "context_compress")
        assert (
            compress_tool["description"]
            == "Compress context while preserving important information"
        )
        assert "content" in compress_tool["inputSchema"]["properties"]
        assert "compression_level" in compress_tool["inputSchema"]["properties"]
        assert "priority" in compress_tool["inputSchema"]["properties"]
        assert compress_tool["inputSchema"]["required"] == ["content"]

        # Check that priority enum includes all ContextPriority values
        priority_prop = compress_tool["inputSchema"]["properties"]["priority"]
        assert set(priority_prop["enum"]) == {p.value for p in ContextPriority}

        # Check summarize tool schema
        summarize_tool = next(t for t in tools if t["name"] == "context_summarize")
        assert "max_tokens" in summarize_tool["inputSchema"]["properties"]
        assert "output_format" in summarize_tool["inputSchema"]["properties"]
        output_formats = summarize_tool["inputSchema"]["properties"]["output_format"]["enum"]
        assert set(output_formats) == {"structured", "narrative", "bullet_points"}

        # Check search tool requires both content and query
        search_tool = next(t for t in tools if t["name"] == "context_search")
        assert search_tool["inputSchema"]["required"] == ["content", "query"]

    @pytest.mark.asyncio
    async def test_compress_tool_success(self, server, mock_tool):
        """Test successful compress operation."""
        mock_result = {
            "compressed_content": "Compressed text...",
            "original_length": 1000,
            "compressed_length": 500,
            "compression_ratio": 0.5,
        }

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=mock_result
        )

        arguments = {
            "content": "Long text to compress...",
            "compression_level": 0.5,
            "priority": "high",
        }

        result = await server.call_tool("context_compress", arguments)

        mock_tool.execute.assert_called_once_with(
            operation="compress",
            content="Long text to compress...",
            compression_level=0.5,
            priority="high",
        )

        assert "content" in result
        assert not result.get("isError", False)
        assert "compressed_content" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_compress_tool_defaults(self, server, mock_tool):
        """Test compress with default parameters."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result="Compressed content"
        )

        # Only provide required content
        arguments = {"content": "Text to compress"}

        result = await server.call_tool("context_compress", arguments)

        mock_tool.execute.assert_called_once_with(
            operation="compress",
            content="Text to compress",
            compression_level=0.5,  # Default
            priority="balanced",  # Default
        )

        assert result["content"][0]["text"] == "Compressed content"

    @pytest.mark.asyncio
    async def test_summarize_tool_success(self, server, mock_tool):
        """Test successful summarize operation."""
        mock_result = {
            "summary": "This is a summary",
            "key_points": ["Point 1", "Point 2"],
            "token_count": 150,
        }

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=mock_result
        )

        arguments = {
            "content": "Long content to summarize...",
            "max_tokens": 200,
            "output_format": "structured",
        }

        result = await server.call_tool("context_summarize", arguments)

        mock_tool.execute.assert_called_once_with(
            operation="summarize",
            content="Long content to summarize...",
            max_tokens=200,
            output_format="structured",
        )

        assert "summary" in result["content"][0]["text"]
        assert "key_points" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_extract_key_points_success(self, server, mock_tool):
        """Test successful extract key points operation."""
        mock_result = ["Key point 1", "Key point 2", "Key point 3"]

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=mock_result
        )

        arguments = {"content": "Content to analyze...", "max_points": 5}

        result = await server.call_tool("context_extract_points", arguments)

        mock_tool.execute.assert_called_once_with(
            operation="extract_key_points",
            content="Content to analyze...",
            max_results=5,  # Note: parameter name mapping
        )

        # Should format list as bullet points
        text = result["content"][0]["text"]
        assert "• Key point 1" in text
        assert "• Key point 2" in text
        assert "• Key point 3" in text

    @pytest.mark.asyncio
    async def test_search_tool_success(self, server, mock_tool):
        """Test successful search operation."""
        mock_result = [{"text": "Match 1", "score": 0.9}, {"text": "Match 2", "score": 0.8}]

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=mock_result
        )

        arguments = {
            "content": "Content to search in...",
            "query": "search query",
            "max_results": 10,
        }

        result = await server.call_tool("context_search", arguments)

        mock_tool.execute.assert_called_once_with(
            operation="search",
            content="Content to search in...",
            query="search query",
            max_results=10,
        )

        # Should return JSON formatted result
        assert "Match 1" in result["content"][0]["text"]
        assert "0.9" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_export_tool_success(self, server, mock_tool):
        """Test successful export operation."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result="Exported to /path/to/file.json"
        )

        arguments = {
            "content": "Content to export",
            "file_path": "/path/to/file.json",
            "format": "json",
        }

        result = await server.call_tool("context_export", arguments)

        mock_tool.execute.assert_called_once_with(
            operation="export",
            content="Content to export",
            file_path="/path/to/file.json",
            format_type="json",  # Note: parameter name mapping
        )

        assert result["content"][0]["text"] == "Exported to /path/to/file.json"

    @pytest.mark.asyncio
    async def test_export_tool_default_format(self, server, mock_tool):
        """Test export with default format."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result="Exported successfully"
        )

        # Only provide required parameters
        arguments = {"content": "Content to export", "file_path": "/path/to/file"}

        await server.call_tool("context_export", arguments)

        mock_tool.execute.assert_called_once_with(
            operation="export",
            content="Content to export",
            file_path="/path/to/file",
            format_type="json",  # Default format
        )

    @pytest.mark.asyncio
    async def test_unknown_tool(self, server, mock_tool):
        """Test calling unknown tool."""
        result = await server.call_tool("unknown_tool", {"content": "test"})

        assert result["isError"] is True
        assert "Unknown tool: unknown_tool" in result["content"][0]["text"]
        mock_tool.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_execution_error(self, server, mock_tool):
        """Test handling of tool execution errors."""
        mock_tool.execute.side_effect = Exception("Execution failed")

        arguments = {"content": "Test content"}

        result = await server.call_tool("context_compress", arguments)

        assert result["isError"] is True
        assert "Error: Execution failed" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_string_result_formatting(self, server, mock_tool):
        """Test formatting of string results."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result="Simple string result"
        )

        result = await server.call_tool("context_compress", {"content": "test"})

        assert result["content"][0]["text"] == "Simple string result"
        assert not result.get("isError", False)

    @pytest.mark.asyncio
    async def test_list_result_formatting(self, server, mock_tool):
        """Test formatting of list results."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=["Item 1", "Item 2", "Item 3"]
        )

        result = await server.call_tool("context_extract_points", {"content": "test"})

        text = result["content"][0]["text"]
        assert text == "• Item 1\n• Item 2\n• Item 3"

    @pytest.mark.asyncio
    async def test_dict_result_formatting(self, server, mock_tool):
        """Test formatting of dict results."""
        mock_result = {"key1": "value1", "key2": {"nested": "value2"}, "key3": [1, 2, 3]}

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=mock_result
        )

        result = await server.call_tool("context_compress", {"content": "test"})

        # Should be JSON formatted
        text = result["content"][0]["text"]
        parsed = json.loads(text)
        assert parsed == mock_result

    @pytest.mark.asyncio
    async def test_error_logging(self, server, mock_tool):
        """Test that errors are logged."""
        mock_tool.execute.side_effect = ValueError("Test error")

        with patch("aida.tools.context.mcp_server.logger") as mock_logger:
            await server.call_tool("context_compress", {"content": "test"})

            mock_logger.error.assert_called_once()
            error_msg = mock_logger.error.call_args[0][0]
            assert "MCP tool call failed" in error_msg


if __name__ == "__main__":
    pytest.main([__file__])
