"""Tests for file operations tool using MCP."""

from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aida.providers.mcp.filesystem_client import MCPFilesystemClient
from aida.tools.base import ToolCapability, ToolStatus
from aida.tools.files.files import FileOperationsTool
from aida.tools.files.models import FileOperation


class TestFileOperationsTool:
    """Test FileOperationsTool class with MCP integration."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP client."""
        client = AsyncMock(spec=MCPFilesystemClient)
        client.connect = AsyncMock(return_value=True)
        client.disconnect = AsyncMock()
        client.call_tool = AsyncMock()
        return client

    @pytest.fixture
    def tool(self, mock_mcp_client):
        """Create a file operations tool with mocked MCP client."""
        tool = FileOperationsTool()
        # Mock the client initialization
        with patch.object(tool, "_ensure_initialized", new=AsyncMock()):
            tool._client = mock_mcp_client
            tool._initialized = True
        return tool

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_initialization(self):
        """Test tool initialization."""
        tool = FileOperationsTool()
        assert tool.name == "file_operations"
        assert tool.version == "3.0.0"  # Version bump for MCP
        assert tool.description == "File operations using official MCP filesystem server"
        assert tool.allowed_directories is not None
        assert not tool._initialized

    def test_initialization_with_directories(self):
        """Test tool initialization with custom directories."""
        custom_dirs = ["/tmp", "/home/user"]
        tool = FileOperationsTool(allowed_directories=custom_dirs)
        assert tool.allowed_directories == custom_dirs

    def test_get_capability(self):
        """Test getting tool capability."""
        tool = FileOperationsTool()
        capability = tool.get_capability()

        assert isinstance(capability, ToolCapability)
        assert capability.name == "file_operations"
        assert capability.version == "3.0.0"
        assert len(capability.parameters) == 7  # Fixed: 7 parameters, not 8

        # Check required parameter
        operation_param = next(p for p in capability.parameters if p.name == "operation")
        assert operation_param.required is True
        assert operation_param.type == "str"
        assert len(operation_param.choices) == len(FileOperation)

        # Check optional parameters
        content_param = next(p for p in capability.parameters if p.name == "content")
        assert content_param.required is False

        recursive_param = next(p for p in capability.parameters if p.name == "recursive")
        assert recursive_param.default is False

    @pytest.mark.asyncio
    async def test_ensure_initialized(self):
        """Test MCP client initialization."""
        tool = FileOperationsTool()
        mock_client = AsyncMock(spec=MCPFilesystemClient)
        mock_client.connect = AsyncMock(return_value=True)

        with patch("aida.tools.files.files.MCPFilesystemClient", return_value=mock_client):
            await tool._ensure_initialized()

        assert tool._initialized
        assert tool._client == mock_client
        mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_initialized_failure(self):
        """Test MCP client initialization failure."""
        tool = FileOperationsTool()
        mock_client = AsyncMock(spec=MCPFilesystemClient)
        mock_client.connect = AsyncMock(return_value=False)

        with patch("aida.tools.files.files.MCPFilesystemClient", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Failed to connect to MCP filesystem server"):
                await tool._ensure_initialized()

    @pytest.mark.asyncio
    async def test_execute_missing_operation(self, tool):
        """Test execute with missing operation parameter."""
        result = await tool.execute(path="/tmp/test.txt")

        assert result.status == ToolStatus.FAILED
        assert "Missing required parameter: operation" in result.error
        assert result.metadata["error_type"] == "validation_error"

    @pytest.mark.asyncio
    async def test_read_file_success(self, tool):
        """Test successful file read via MCP."""
        # Mock MCP response
        tool._client.call_tool.return_value = {
            "content": "Hello, World!\nThis is a test file.",
            "encoding": "utf-8",
            "size": 35,
        }

        result = await tool.execute(operation="read", path="/tmp/test.txt")

        assert result.status == ToolStatus.COMPLETED
        assert result.result["content"] == "Hello, World!\nThis is a test file."
        assert result.metadata["operation"] == "read"
        assert result.metadata["files_affected"] == 1

        # Verify MCP call
        tool._client.call_tool.assert_called_once_with("read_file", {"path": "/tmp/test.txt"})

    @pytest.mark.asyncio
    async def test_write_file_success(self, tool):
        """Test successful file write via MCP."""
        content = "Test content to write"
        tool._client.call_tool.return_value = {
            "bytes_written": len(content.encode("utf-8")),
            "path": "/tmp/output.txt",
        }

        result = await tool.execute(operation="write", path="/tmp/output.txt", content=content)

        assert result.status == ToolStatus.COMPLETED
        assert result.result["bytes_written"] == len(content.encode("utf-8"))
        assert result.metadata["files_affected"] == 1

        # Verify MCP call
        tool._client.call_tool.assert_called_once_with(
            "write_file", {"path": "/tmp/output.txt", "content": content}
        )

    @pytest.mark.asyncio
    async def test_delete_file_success(self, tool):
        """Test successful file deletion via MCP."""
        tool._client.call_tool.return_value = {"deleted": True}

        result = await tool.execute(operation="delete", path="/tmp/delete_me.txt")

        assert result.status == ToolStatus.COMPLETED
        assert result.metadata["files_affected"] == 1

        # Verify MCP call
        tool._client.call_tool.assert_called_once_with(
            "delete_file", {"path": "/tmp/delete_me.txt"}
        )

    @pytest.mark.asyncio
    async def test_delete_recursive(self, tool):
        """Test recursive deletion via MCP."""
        tool._client.call_tool.return_value = {"deleted": True, "files_deleted": 5}

        result = await tool.execute(operation="delete", path="/tmp/dir", recursive=True)

        assert result.status == ToolStatus.COMPLETED

        # Verify MCP call includes recursive flag
        tool._client.call_tool.assert_called_once_with(
            "delete_file", {"path": "/tmp/dir", "recursive": True}
        )

    @pytest.mark.asyncio
    async def test_move_file_success(self, tool):
        """Test successful file move via MCP."""
        tool._client.call_tool.return_value = {"moved": True}

        result = await tool.execute(
            operation="move", path="/tmp/source.txt", destination="/tmp/dest.txt"
        )

        assert result.status == ToolStatus.COMPLETED
        assert result.metadata["files_affected"] == 1

        # Verify MCP call
        tool._client.call_tool.assert_called_once_with(
            "move_file", {"path": "/tmp/source.txt", "destination": "/tmp/dest.txt"}
        )

    @pytest.mark.asyncio
    async def test_create_directory(self, tool):
        """Test directory creation via MCP."""
        tool._client.call_tool.return_value = {"created": True}

        result = await tool.execute(operation="create_dir", path="/tmp/new_dir")

        assert result.status == ToolStatus.COMPLETED

        # Verify MCP call
        tool._client.call_tool.assert_called_once_with("create_directory", {"path": "/tmp/new_dir"})

    @pytest.mark.asyncio
    async def test_list_directory(self, tool):
        """Test directory listing via MCP."""
        tool._client.call_tool.return_value = {
            "entries": [
                {"name": "file1.txt", "path": "/tmp/file1.txt", "type": "file", "size": 100},
                {"name": "subdir", "path": "/tmp/subdir", "type": "directory"},
            ]
        }

        result = await tool.execute(operation="list_dir", path="/tmp")

        assert result.status == ToolStatus.COMPLETED
        assert result.result["entries"] == tool._client.call_tool.return_value["entries"]

        # Verify MCP call
        tool._client.call_tool.assert_called_once_with("list_directory", {"path": "/tmp"})

    @pytest.mark.asyncio
    async def test_get_file_info(self, tool):
        """Test getting file info via MCP."""
        tool._client.call_tool.return_value = {
            "name": "test.txt",
            "path": "/tmp/test.txt",
            "size": 1024,
            "type": "file",
            "modified": "2024-01-01T00:00:00Z",
        }

        result = await tool.execute(operation="get_info", path="/tmp/test.txt")

        assert result.status == ToolStatus.COMPLETED
        assert result.result["name"] == "test.txt"
        assert result.result["size"] == 1024

        # Verify MCP call
        tool._client.call_tool.assert_called_once_with("get_file_info", {"path": "/tmp/test.txt"})

    @pytest.mark.asyncio
    async def test_edit_file(self, tool):
        """Test file editing via MCP."""
        tool._client.call_tool.return_value = {"edited": True, "replacements": 2}

        result = await tool.execute(
            operation="edit", path="/tmp/edit.txt", search_text="old", replace_text="new"
        )

        assert result.status == ToolStatus.COMPLETED
        assert result.result["replacements"] == 2

        # Verify MCP call
        tool._client.call_tool.assert_called_once_with(
            "edit_file", {"path": "/tmp/edit.txt", "find": "old", "replace": "new"}
        )

    @pytest.mark.asyncio
    async def test_append_operation(self, tool):
        """Test append operation (complex operation requiring multiple MCP calls)."""
        # Mock read and write operations
        tool._client.call_tool.side_effect = [
            {"content": "Existing content"},  # read_file response
            {"bytes_written": 25},  # write_file response
        ]

        result = await tool.execute(
            operation="append", path="/tmp/append.txt", content="\nNew line"
        )

        assert result.status == ToolStatus.COMPLETED
        assert result.result["bytes_appended"] == 9  # len("\nNew line")

        # Verify MCP calls
        assert tool._client.call_tool.call_count == 2
        tool._client.call_tool.assert_any_call("read_file", {"path": "/tmp/append.txt"})
        tool._client.call_tool.assert_any_call(
            "write_file", {"path": "/tmp/append.txt", "content": "Existing content\nNew line"}
        )

    @pytest.mark.asyncio
    async def test_append_to_nonexistent_file(self, tool):
        """Test appending to a non-existent file."""
        # Mock read failure (file doesn't exist) and write success
        tool._client.call_tool.side_effect = [
            Exception("File not found"),  # read_file fails
            {"bytes_written": 9},  # write_file succeeds
        ]

        result = await tool.execute(
            operation="append", path="/tmp/new_append.txt", content="New content"
        )

        assert result.status == ToolStatus.COMPLETED
        assert result.result["bytes_appended"] == 11  # len("New content")

    @pytest.mark.asyncio
    async def test_copy_operation(self, tool):
        """Test copy operation (complex operation)."""
        # Mock read and write operations
        tool._client.call_tool.side_effect = [
            {"content": "File content to copy"},  # read_file response
            {"bytes_written": 20},  # write_file response
        ]

        result = await tool.execute(
            operation="copy", path="/tmp/source.txt", destination="/tmp/copy.txt"
        )

        assert result.status == ToolStatus.COMPLETED
        assert result.result["source"] == "/tmp/source.txt"
        assert result.result["destination"] == "/tmp/copy.txt"
        assert result.result["bytes_copied"] == 20

    @pytest.mark.asyncio
    async def test_search_operation(self, tool):
        """Test search operation (complex operation)."""
        # Mock list and read operations
        tool._client.call_tool.side_effect = [
            {  # list_directory response
                "entries": [
                    {"name": "file1.txt", "path": "/tmp/file1.txt", "type": "file"},
                    {"name": "file2.txt", "path": "/tmp/file2.txt", "type": "file"},
                    {"name": "subdir", "path": "/tmp/subdir", "type": "directory"},
                ]
            },
            {"content": "Hello Python world"},  # read file1
            {"content": "Just some text"},  # read file2
        ]

        result = await tool.execute(operation="search", path="/tmp", search_text="Python")

        assert result.status == ToolStatus.COMPLETED
        assert len(result.result["results"]) == 1
        assert result.result["results"][0]["file"] == "/tmp/file1.txt"
        assert result.result["results"][0]["matches"] == 1
        assert result.result["files_searched"] == 3

    @pytest.mark.asyncio
    async def test_find_operation(self, tool):
        """Test find operation (complex operation)."""
        tool._client.call_tool.return_value = {
            "entries": [
                {"name": "test1.py", "path": "/tmp/test1.py", "type": "file"},
                {"name": "test2.py", "path": "/tmp/test2.py", "type": "file"},
                {"name": "data.txt", "path": "/tmp/data.txt", "type": "file"},
            ]
        }

        result = await tool.execute(operation="find", path="/tmp", pattern="*.py")

        assert result.status == ToolStatus.COMPLETED
        assert len(result.result["results"]) == 2
        assert all(path.endswith(".py") for path in result.result["results"])
        assert result.result["count"] == 2

    @pytest.mark.asyncio
    async def test_batch_operations(self, tool):
        """Test batch operations."""
        # For batch operations, the tool needs to actually execute the complex operation
        # So we need to set up the proper response structure
        batch_ops = [
            {"operation": "write", "path": "/tmp/batch1.txt", "content": "Content 1"},
            {"operation": "read", "path": "/tmp/batch1.txt"},
            {"operation": "read", "path": "/tmp/nonexistent.txt"},
        ]

        # The batch operation is handled by _handle_complex_operation
        # We need to mock the internal execute calls properly

        # Keep the original execute method
        original_execute = tool.execute

        # Counter to track which operation we're on
        call_count = 0

        async def mock_execute(**kwargs):
            nonlocal call_count
            # If this is the main batch call, let it proceed
            if kwargs.get("operation") == "batch":
                return await original_execute(**kwargs)

            # For sub-operations, return appropriate responses
            if call_count == 0:  # write operation
                call_count += 1
                return MagicMock(status=ToolStatus.COMPLETED, result={"bytes_written": 10})
            elif call_count == 1:  # first read operation
                call_count += 1
                return MagicMock(status=ToolStatus.COMPLETED, result={"content": "test"})
            else:  # second read operation (fails)
                call_count += 1
                return MagicMock(status=ToolStatus.FAILED, error="File not found")

        # Patch execute to intercept sub-operation calls
        with patch.object(tool, "execute", side_effect=mock_execute):
            result = await tool.execute(operation="batch", path="batch", batch_operations=batch_ops)

        assert result.status == ToolStatus.COMPLETED
        # The batch result structure contains a results array
        assert "results" in result.result
        assert len(result.result["results"]) == 3
        assert result.result["results"][0]["success"] is True
        assert result.result["results"][1]["success"] is True
        assert result.result["results"][2]["success"] is False
        assert "error" in result.result["results"][2]

    @pytest.mark.asyncio
    async def test_mcp_error_handling(self, tool):
        """Test MCP client error handling."""
        tool._client.call_tool.side_effect = Exception("MCP connection error")

        result = await tool.execute(operation="read", path="/tmp/test.txt")

        assert result.status == ToolStatus.FAILED
        assert "MCP connection error" in str(result.error)

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup of MCP client."""
        tool = FileOperationsTool()
        mock_client = AsyncMock(spec=MCPFilesystemClient)
        tool._client = mock_client
        tool._initialized = True

        await tool.cleanup()

        mock_client.disconnect.assert_called_once()
        assert not tool._initialized

    def test_create_pydantic_tools(self, tool):
        """Test creating PydanticAI-compatible tools."""
        pydantic_tools = tool._create_pydantic_tools()

        expected_tools = [
            "read_file",
            "write_file",
            "list_directory",
            "search_files",
            "find_files",
            "create_dir",
        ]

        for tool_name in expected_tools:
            assert tool_name in pydantic_tools
            assert callable(pydantic_tools[tool_name])

    @pytest.mark.asyncio
    async def test_pydantic_read_file(self, tool):
        """Test PydanticAI read_file function."""
        tool._client.call_tool.return_value = {
            "content": "PydanticAI test content",
            "encoding": "utf-8",
        }

        pydantic_tools = tool._create_pydantic_tools()
        read_file = pydantic_tools["read_file"]

        result = await read_file("/tmp/test.txt")

        assert result["content"] == "PydanticAI test content"
        assert result["encoding"] == "utf-8"

    @pytest.mark.asyncio
    async def test_pydantic_write_file(self, tool):
        """Test PydanticAI write_file function."""
        tool._client.call_tool.return_value = {"bytes_written": 20}

        pydantic_tools = tool._create_pydantic_tools()
        write_file = pydantic_tools["write_file"]

        result = await write_file("/tmp/test.txt", "Write via PydanticAI")

        assert result["bytes_written"] == 20

    @pytest.mark.asyncio
    async def test_pydantic_list_directory(self, tool):
        """Test PydanticAI list_directory function."""
        # Mock the execute method to return the expected result
        mock_result = MagicMock()
        mock_result.result = [
            {"name": "file1.txt", "type": "file"},
            {"name": "file2.txt", "type": "file"},
        ]

        with patch.object(tool, "execute", return_value=mock_result):
            pydantic_tools = tool._create_pydantic_tools()
            list_directory = pydantic_tools["list_directory"]

            result = await list_directory("/tmp")

            assert len(result) == 2
            assert all(isinstance(entry, dict) for entry in result)

    @pytest.mark.asyncio
    async def test_pydantic_search_files(self, tool):
        """Test PydanticAI search_files function."""
        # Mock the complex search operation
        with patch.object(tool, "execute") as mock_execute:
            mock_execute.return_value = MagicMock(
                result={
                    "results": [
                        {"file": "/tmp/file1.txt", "matches": 2},
                        {"file": "/tmp/file2.txt", "matches": 1},
                    ]
                }
            )

            pydantic_tools = tool._create_pydantic_tools()
            search_files = pydantic_tools["search_files"]

            result = await search_files("/tmp", "test")

            assert len(result) == 2
            assert all(isinstance(match, dict) for match in result)

    @pytest.mark.asyncio
    async def test_pydantic_find_files(self, tool):
        """Test PydanticAI find_files function."""
        with patch.object(tool, "execute") as mock_execute:
            mock_execute.return_value = MagicMock(
                result={"results": ["/tmp/test1.py", "/tmp/test2.py"]}
            )

            pydantic_tools = tool._create_pydantic_tools()
            find_files = pydantic_tools["find_files"]

            result = await find_files("/tmp", "*.py")

            assert len(result) == 2
            assert all(f.endswith(".py") for f in result)

    @pytest.mark.asyncio
    async def test_pydantic_create_dir(self, tool):
        """Test PydanticAI create_dir function."""
        tool._client.call_tool.return_value = {"created": True}

        pydantic_tools = tool._create_pydantic_tools()
        create_dir = pydantic_tools["create_dir"]

        result = await create_dir("/tmp/new_dir")

        assert result["created"] is True
        assert result["path"] == "/tmp/new_dir"

    def test_create_mcp_server(self, tool):
        """Test creating MCP server."""
        with patch("aida.tools.files.mcp_server.FilesMCPServer") as mock_mcp:
            tool._create_mcp_server()
            mock_mcp.assert_called_once_with(tool)

    def test_create_observability(self, tool):
        """Test creating observability."""
        config = {"trace_enabled": True}

        with patch("aida.tools.files.observability.FilesObservability") as mock_obs:
            tool._create_observability(config)
            mock_obs.assert_called_once_with(tool, config)

    @pytest.mark.asyncio
    async def test_unsupported_operation(self, tool):
        """Test handling of unsupported MCP operation."""
        # Test an operation that MCP doesn't support directly
        result = await tool.execute(operation="invalid_op", path="/tmp/test.txt")

        assert result.status == ToolStatus.FAILED
        # The error will be from validation when creating FileOperationRequest
        assert "validation error" in str(result.error) or "enum" in str(result.error)


if __name__ == "__main__":
    pytest.main([__file__])
