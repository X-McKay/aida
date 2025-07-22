"""Tests for files MCP server."""

from datetime import datetime
import json
from unittest.mock import AsyncMock, Mock
import uuid

import pytest

from aida.tools.base import ToolResult, ToolStatus
from aida.tools.files.mcp_server import FilesMCPServer


class TestFilesMCPServer:
    """Test FilesMCPServer class."""

    def _create_tool_result(self, status=ToolStatus.COMPLETED, result=None, error=None):
        """Helper to create a valid ToolResult."""
        return ToolResult(
            tool_name="files",
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
        """Create a mock files tool."""
        tool = Mock()
        tool.execute = AsyncMock()
        tool.name = "files"
        tool.version = "2.0.0"
        tool.description = "File operations tool"
        return tool

    @pytest.fixture
    def server(self, mock_tool):
        """Create a files MCP server."""
        return FilesMCPServer(mock_tool)

    def test_initialization(self, server):
        """Test server initialization."""
        assert server.tool is not None
        assert "read" in server.operations
        assert "write" in server.operations
        assert "list" in server.operations
        assert "search" in server.operations
        assert "delete" in server.operations

    def test_operation_schemas(self, server):
        """Test operation schemas are properly defined."""
        # Check read operation
        read_op = server.operations["read"]
        assert read_op["description"] == "Read file contents"
        assert "path" in read_op["parameters"]
        assert "encoding" in read_op["parameters"]
        assert read_op["parameters"]["encoding"]["default"] == "utf-8"
        assert read_op["required"] == ["path"]
        assert "handler" in read_op

        # Check write operation
        write_op = server.operations["write"]
        assert write_op["description"] == "Write content to file"
        assert "content" in write_op["parameters"]
        assert write_op["required"] == ["path", "content"]

        # Check list operation
        list_op = server.operations["list"]
        assert list_op["description"] == "List directory contents"
        assert "recursive" in list_op["parameters"]
        assert list_op["parameters"]["recursive"]["default"] is False

        # Check search operation
        search_op = server.operations["search"]
        assert search_op["description"] == "Search for text in files"
        assert "text" in search_op["parameters"]
        assert search_op["parameters"]["recursive"]["default"] is True
        assert search_op["required"] == ["path", "text"]

        # Check delete operation
        delete_op = server.operations["delete"]
        assert delete_op["description"] == "Delete file or directory"
        assert delete_op["parameters"]["recursive"]["default"] is False

    @pytest.mark.asyncio
    async def test_handle_read_success(self, server, mock_tool):
        """Test successful read operation."""
        file_content = {"content": "Hello, World!", "encoding": "utf-8", "size": 13}

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=file_content
        )

        arguments = {"path": "/test/file.txt", "encoding": "utf-8"}

        result = await server._handle_read(arguments)

        mock_tool.execute.assert_called_once_with(
            operation="read", path="/test/file.txt", encoding="utf-8"
        )

        assert result == file_content

    @pytest.mark.asyncio
    async def test_handle_read_default_encoding(self, server, mock_tool):
        """Test read with default encoding."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result={"content": "Test content"}
        )

        arguments = {"path": "/test/file.txt"}

        await server._handle_read(arguments)

        mock_tool.execute.assert_called_once_with(
            operation="read",
            path="/test/file.txt",
            encoding="utf-8",  # Default
        )

    @pytest.mark.asyncio
    async def test_handle_read_failure(self, server, mock_tool):
        """Test failed read operation."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.FAILED, error="File not found"
        )

        arguments = {"path": "/nonexistent.txt"}

        with pytest.raises(Exception, match="File not found"):
            await server._handle_read(arguments)

    @pytest.mark.asyncio
    async def test_handle_write_success(self, server, mock_tool):
        """Test successful write operation."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result={"bytes_written": 100}
        )

        arguments = {"path": "/test/output.txt", "content": "New content", "encoding": "utf-8"}

        result = await server._handle_write(arguments)

        mock_tool.execute.assert_called_once_with(
            operation="write", path="/test/output.txt", content="New content", encoding="utf-8"
        )

        assert result == {"success": True, "path": "/test/output.txt"}

    @pytest.mark.asyncio
    async def test_handle_write_failure(self, server, mock_tool):
        """Test failed write operation."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.FAILED, error="Permission denied"
        )

        arguments = {"path": "/readonly/file.txt", "content": "test"}

        with pytest.raises(Exception, match="Permission denied"):
            await server._handle_write(arguments)

    @pytest.mark.asyncio
    async def test_handle_list_success(self, server, mock_tool):
        """Test successful list directory operation."""
        dir_contents = [
            {"name": "file1.txt", "type": "file", "size": 1024},
            {"name": "subdir", "type": "directory", "size": 0},
            {"name": "file2.py", "type": "file", "size": 2048},
        ]

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=dir_contents
        )

        arguments = {"path": "/test/dir", "recursive": True}

        result = await server._handle_list(arguments)

        mock_tool.execute.assert_called_once_with(
            operation="list_dir",  # Note: operation name mapping
            path="/test/dir",
            recursive=True,
        )

        assert result == dir_contents

    @pytest.mark.asyncio
    async def test_handle_list_default_not_recursive(self, server, mock_tool):
        """Test list with default recursive=False."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=[]
        )

        arguments = {"path": "/test"}

        await server._handle_list(arguments)

        mock_tool.execute.assert_called_once_with(
            operation="list_dir",
            path="/test",
            recursive=False,  # Default
        )

    @pytest.mark.asyncio
    async def test_handle_search_success(self, server, mock_tool):
        """Test successful search operation."""
        search_results = [
            {"file": "/test/file1.txt", "line": 10, "text": "Found text here"},
            {"file": "/test/file2.txt", "line": 25, "text": "Another match"},
        ]

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=search_results
        )

        arguments = {"path": "/test", "text": "search term", "recursive": True}

        result = await server._handle_search(arguments)

        mock_tool.execute.assert_called_once_with(
            operation="search",
            path="/test",
            search_text="search term",  # Note: parameter name mapping
            recursive=True,
        )

        assert result == search_results

    @pytest.mark.asyncio
    async def test_handle_search_default_recursive(self, server, mock_tool):
        """Test search with default recursive=True."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=[]
        )

        arguments = {"path": "/test", "text": "find me"}

        await server._handle_search(arguments)

        mock_tool.execute.assert_called_once_with(
            operation="search",
            path="/test",
            search_text="find me",
            recursive=True,  # Default for search
        )

    @pytest.mark.asyncio
    async def test_handle_delete_success(self, server, mock_tool):
        """Test successful delete operation."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result={"deleted": True}
        )

        arguments = {"path": "/test/file.txt", "recursive": False}

        result = await server._handle_delete(arguments)

        mock_tool.execute.assert_called_once_with(
            operation="delete", path="/test/file.txt", recursive=False
        )

        assert result == {"success": True, "path": "/test/file.txt"}

    @pytest.mark.asyncio
    async def test_handle_delete_recursive(self, server, mock_tool):
        """Test delete with recursive=True."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result={"deleted": True, "files_deleted": 10}
        )

        arguments = {"path": "/test/dir", "recursive": True}

        result = await server._handle_delete(arguments)

        mock_tool.execute.assert_called_once_with(
            operation="delete", path="/test/dir", recursive=True
        )

        assert result == {"success": True, "path": "/test/dir"}

    @pytest.mark.asyncio
    async def test_handle_delete_failure(self, server, mock_tool):
        """Test failed delete operation."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.FAILED, error="Directory not empty"
        )

        arguments = {"path": "/test/nonempty"}

        with pytest.raises(Exception, match="Directory not empty"):
            await server._handle_delete(arguments)

    def test_format_response_override(self, server):
        """Test that FilesMCPServer overrides _format_response for JSON consistency."""
        # Test with dict
        result = {"key": "value", "nested": {"item": 1}}
        response = server._format_response(result)

        assert "content" in response
        assert len(response["content"]) == 1
        assert response["content"][0]["type"] == "text"

        # The text should be JSON formatted
        text = response["content"][0]["text"]
        parsed = json.loads(text)
        assert parsed == result

        # Test with list
        list_result = ["item1", "item2", "item3"]
        response = server._format_response(list_result)
        text = response["content"][0]["text"]
        parsed = json.loads(text)
        assert parsed == list_result

        # Test with string
        string_result = "Simple string"
        response = server._format_response(string_result)
        text = response["content"][0]["text"]
        parsed = json.loads(text)
        assert parsed == string_result

    @pytest.mark.asyncio
    async def test_inherited_from_simple_mcp_server(self, server):
        """Test that server inherits from SimpleMCPServer correctly."""
        assert hasattr(server, "call_tool")
        assert hasattr(server, "list_tools")
        assert hasattr(server, "operations")
        assert hasattr(server, "tool")
        assert hasattr(server, "server_info")

    def test_list_tools(self, server):
        """Test listing available MCP tools."""
        tools = server.list_tools()

        # Should have tools for each operation
        tool_names = [tool["name"] for tool in tools]
        assert "files_read" in tool_names
        assert "files_write" in tool_names
        assert "files_list" in tool_names
        assert "files_search" in tool_names
        assert "files_delete" in tool_names

        # Check read tool definition
        read_tool = next(t for t in tools if t["name"] == "files_read")
        assert read_tool["description"] == "Read file contents"
        assert "inputSchema" in read_tool
        assert read_tool["inputSchema"]["type"] == "object"
        assert "path" in read_tool["inputSchema"]["properties"]
        assert read_tool["inputSchema"]["required"] == ["path"]

    @pytest.mark.asyncio
    async def test_call_tool_integration(self, server, mock_tool):
        """Test calling operations through call_tool method."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result={"content": "File contents"}
        )

        # Call read through call_tool
        result = await server.call_tool(name="files_read", arguments={"path": "/test.txt"})

        # Should return MCP formatted response with JSON
        assert "content" in result
        text = result["content"][0]["text"]
        parsed = json.loads(text)
        assert parsed == {"content": "File contents"}


if __name__ == "__main__":
    pytest.main([__file__])
