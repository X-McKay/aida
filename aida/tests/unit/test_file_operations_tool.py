"""Tests for file operations tool."""

from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from aida.tools.base import ToolCapability, ToolStatus
from aida.tools.files.config import FilesConfig
from aida.tools.files.files import FileOperationsTool
from aida.tools.files.models import FileOperation


class TestFileOperationsTool:
    """Test FileOperationsTool class."""

    @pytest.fixture
    def tool(self):
        """Create a file operations tool."""
        return FileOperationsTool()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_initialization(self, tool):
        """Test tool initialization."""
        assert tool.name == "file_operations"
        assert tool.version == "2.0.0"
        assert (
            tool.description
            == "Comprehensive file and directory operations with search and editing capabilities"
        )

    def test_get_capability(self, tool):
        """Test getting tool capability."""
        capability = tool.get_capability()

        assert isinstance(capability, ToolCapability)
        assert capability.name == "file_operations"
        assert capability.version == "2.0.0"
        assert len(capability.parameters) == 9

        # Check required parameter
        operation_param = next(p for p in capability.parameters if p.name == "operation")
        assert operation_param.required is True
        assert operation_param.type == "str"
        assert len(operation_param.choices) == len(FileOperation)

        # Check optional parameters
        content_param = next(p for p in capability.parameters if p.name == "content")
        assert content_param.required is False

        encoding_param = next(p for p in capability.parameters if p.name == "encoding")
        assert encoding_param.default == "utf-8"

    @pytest.mark.asyncio
    async def test_execute_missing_operation(self, tool):
        """Test execute with missing operation parameter."""
        result = await tool.execute(path="/tmp/test.txt")

        assert result.status == ToolStatus.FAILED
        assert "Missing required parameter: operation" in result.error
        assert result.metadata["error_type"] == "validation_error"

    @pytest.mark.asyncio
    async def test_execute_unsafe_path(self, tool):
        """Test execute with unsafe path."""
        with patch.object(FilesConfig, "is_safe_path", return_value=False):
            result = await tool.execute(operation="read", path="/etc/passwd")

            assert result.status == ToolStatus.FAILED
            assert "Access denied" in result.error

    @pytest.mark.asyncio
    async def test_read_file_success(self, tool, temp_dir):
        """Test successful file read."""
        # Create test file
        test_file = temp_dir / "test.txt"
        test_content = "Hello, World!\nThis is a test file."
        test_file.write_text(test_content)

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(operation="read", path=str(test_file))

        assert result.status == ToolStatus.COMPLETED
        assert result.result["content"] == test_content
        assert result.result["line_count"] == 2
        assert result.result["encoding"] == "utf-8"
        assert result.metadata["operation"] == "read"
        assert result.metadata["files_affected"] == 1

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tool):
        """Test reading non-existent file."""
        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(operation="read", path="/tmp/non_existent_file.txt")

        assert result.status == ToolStatus.FAILED
        assert "File not found" in result.error

    @pytest.mark.asyncio
    async def test_read_directory_error(self, tool, temp_dir):
        """Test reading a directory instead of file."""
        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(operation="read", path=str(temp_dir))

        assert result.status == ToolStatus.FAILED
        assert "Not a file" in result.error

    @pytest.mark.asyncio
    async def test_read_large_file(self, tool, temp_dir):
        """Test reading file that exceeds size limit."""
        test_file = temp_dir / "large.txt"
        test_file.write_text("x" * 100)  # Small content

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            with patch.object(FilesConfig, "MAX_FILE_SIZE", 50):
                result = await tool.execute(operation="read", path=str(test_file))

        assert result.status == ToolStatus.FAILED
        assert "File too large" in result.error

    @pytest.mark.asyncio
    async def test_write_file_success(self, tool, temp_dir):
        """Test successful file write."""
        test_file = temp_dir / "output.txt"
        content = "Test content to write"

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(operation="write", path=str(test_file), content=content)

        assert result.status == ToolStatus.COMPLETED
        assert result.result["bytes_written"] == len(content.encode("utf-8"))
        assert test_file.exists()
        assert test_file.read_text() == content
        assert result.metadata["files_affected"] == 1

    @pytest.mark.asyncio
    async def test_write_file_create_parents(self, tool, temp_dir):
        """Test writing file with parent directory creation."""
        test_file = temp_dir / "subdir" / "nested" / "output.txt"
        content = "Nested content"

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(
                operation="write", path=str(test_file), content=content, create_parents=True
            )

        assert result.status == ToolStatus.COMPLETED
        assert test_file.exists()
        assert test_file.read_text() == content

    @pytest.mark.asyncio
    async def test_append_file_existing(self, tool, temp_dir):
        """Test appending to existing file."""
        test_file = temp_dir / "append.txt"
        initial_content = "Initial content\n"
        append_content = "Appended content"
        test_file.write_text(initial_content)

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(
                operation="append", path=str(test_file), content=append_content
            )

        assert result.status == ToolStatus.COMPLETED
        assert test_file.read_text() == initial_content + append_content
        assert result.metadata["files_affected"] == 1

    @pytest.mark.asyncio
    async def test_append_file_new(self, tool, temp_dir):
        """Test appending to non-existent file (creates new)."""
        test_file = temp_dir / "new_append.txt"
        content = "New file content"

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(operation="append", path=str(test_file), content=content)

        assert result.status == ToolStatus.COMPLETED
        assert test_file.exists()
        assert test_file.read_text() == content

    @pytest.mark.asyncio
    async def test_delete_file_success(self, tool, temp_dir):
        """Test successful file deletion."""
        test_file = temp_dir / "delete_me.txt"
        test_file.write_text("Delete this")

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(operation="delete", path=str(test_file))

        assert result.status == ToolStatus.COMPLETED
        assert not test_file.exists()
        assert result.metadata["files_affected"] == 1

    @pytest.mark.asyncio
    async def test_delete_directory_recursive(self, tool, temp_dir):
        """Test recursive directory deletion."""
        test_dir = temp_dir / "delete_dir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("content3")

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(operation="delete", path=str(test_dir), recursive=True)

        assert result.status == ToolStatus.COMPLETED
        assert not test_dir.exists()
        assert result.metadata["files_affected"] == 3

    @pytest.mark.asyncio
    async def test_delete_empty_directory(self, tool, temp_dir):
        """Test deleting empty directory without recursive flag."""
        test_dir = temp_dir / "empty_dir"
        test_dir.mkdir()

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(operation="delete", path=str(test_dir), recursive=False)

        assert result.status == ToolStatus.COMPLETED
        assert not test_dir.exists()
        assert result.metadata["files_affected"] == 0

    @pytest.mark.asyncio
    async def test_copy_file_success(self, tool, temp_dir):
        """Test successful file copy."""
        src_file = temp_dir / "source.txt"
        dst_file = temp_dir / "destination.txt"
        content = "Copy this content"
        src_file.write_text(content)

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(
                operation="copy", path=str(src_file), destination=str(dst_file)
            )

        assert result.status == ToolStatus.COMPLETED
        assert dst_file.exists()
        assert dst_file.read_text() == content
        assert src_file.exists()  # Original should still exist
        assert result.metadata["files_affected"] == 1

    @pytest.mark.asyncio
    async def test_copy_directory_recursive(self, tool, temp_dir):
        """Test recursive directory copy."""
        src_dir = temp_dir / "src_dir"
        dst_dir = temp_dir / "dst_dir"
        src_dir.mkdir()
        (src_dir / "file1.txt").write_text("content1")
        (src_dir / "file2.txt").write_text("content2")

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(
                operation="copy", path=str(src_dir), destination=str(dst_dir), recursive=True
            )

        assert result.status == ToolStatus.COMPLETED
        assert dst_dir.exists()
        assert (dst_dir / "file1.txt").read_text() == "content1"
        assert (dst_dir / "file2.txt").read_text() == "content2"
        assert result.metadata["files_affected"] == 2

    @pytest.mark.asyncio
    async def test_move_file_success(self, tool, temp_dir):
        """Test successful file move."""
        src_file = temp_dir / "source.txt"
        dst_file = temp_dir / "moved.txt"
        content = "Move this content"
        src_file.write_text(content)

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(
                operation="move", path=str(src_file), destination=str(dst_file)
            )

        assert result.status == ToolStatus.COMPLETED
        assert dst_file.exists()
        assert dst_file.read_text() == content
        assert not src_file.exists()  # Original should be gone
        assert result.metadata["files_affected"] == 1

    @pytest.mark.asyncio
    async def test_create_directory(self, tool, temp_dir):
        """Test directory creation."""
        new_dir = temp_dir / "new_directory"

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(operation="create_dir", path=str(new_dir))

        assert result.status == ToolStatus.COMPLETED
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result.metadata["files_affected"] == 0

    @pytest.mark.asyncio
    async def test_list_directory(self, tool, temp_dir):
        """Test directory listing."""
        # Create test structure
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.py").write_text("content2")
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            with patch.object(FilesConfig, "should_ignore", return_value=False):
                result = await tool.execute(operation="list_dir", path=str(temp_dir))

        assert result.status == ToolStatus.COMPLETED
        entries = result.result
        assert len(entries) == 3

        # Check entry structure
        file_entry = next(e for e in entries if e["name"] == "file1.txt")
        assert file_entry["is_file"] is True
        assert file_entry["is_dir"] is False
        assert file_entry["size"] > 0

        dir_entry = next(e for e in entries if e["name"] == "subdir")
        assert dir_entry["is_file"] is False
        assert dir_entry["is_dir"] is True

    @pytest.mark.asyncio
    async def test_list_directory_recursive(self, tool, temp_dir):
        """Test recursive directory listing."""
        # Create nested structure
        (temp_dir / "file1.txt").write_text("content1")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("content2")

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            with patch.object(FilesConfig, "should_ignore", return_value=False):
                result = await tool.execute(
                    operation="list_dir", path=str(temp_dir), recursive=True
                )

        assert result.status == ToolStatus.COMPLETED
        entries = result.result
        assert len(entries) == 3  # file1, subdir, subdir/file2

    @pytest.mark.asyncio
    async def test_search_files(self, tool, temp_dir):
        """Test searching for text in files."""
        # Create test files
        (temp_dir / "file1.txt").write_text("Hello world\nPython is great")
        (temp_dir / "file2.txt").write_text("Hello Python\nWorld of code")
        (temp_dir / "file3.txt").write_text("No match here")

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            with patch.object(FilesConfig, "is_text_file", return_value=True):
                result = await tool.execute(
                    operation="search", path=str(temp_dir), search_text="Python", recursive=False
                )

        assert result.status == ToolStatus.COMPLETED
        matches = result.result
        assert len(matches) == 2

        # Check match structure
        match1 = next(m for m in matches if "file1.txt" in m["file"])
        assert match1["matches"] == 1
        assert len(match1["lines"]) == 1
        assert match1["lines"][0]["text"] == "Python"

    @pytest.mark.asyncio
    async def test_search_files_no_search_text(self, tool, temp_dir):
        """Test search operation without search_text."""
        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(operation="search", path=str(temp_dir))

        assert result.status == ToolStatus.FAILED
        assert "search_text required" in result.error

    @pytest.mark.asyncio
    async def test_find_files_by_pattern(self, tool, temp_dir):
        """Test finding files by glob pattern."""
        # Create test files
        (temp_dir / "test1.py").touch()
        (temp_dir / "test2.py").touch()
        (temp_dir / "data.txt").touch()
        subdir = temp_dir / "sub"
        subdir.mkdir()
        (subdir / "test3.py").touch()

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            with patch.object(FilesConfig, "should_ignore", return_value=False):
                result = await tool.execute(
                    operation="find", path=str(temp_dir), pattern="*.py", recursive=True
                )

        assert result.status == ToolStatus.COMPLETED
        found_files = result.result
        assert len(found_files) == 3
        assert all(f.endswith(".py") for f in found_files)

    @pytest.mark.asyncio
    async def test_find_files_no_pattern(self, tool, temp_dir):
        """Test find operation without pattern."""
        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(operation="find", path=str(temp_dir))

        assert result.status == ToolStatus.FAILED
        assert "pattern required" in result.error

    @pytest.mark.asyncio
    async def test_get_file_info(self, tool, temp_dir):
        """Test getting file information."""
        test_file = temp_dir / "info_test.txt"
        test_file.write_text("Test content")

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            with patch.object(FilesConfig, "is_text_file", return_value=True):
                result = await tool.execute(operation="get_info", path=str(test_file))

        assert result.status == ToolStatus.COMPLETED
        info = result.result
        assert info["name"] == "info_test.txt"
        assert info["is_file"] is True
        assert info["is_dir"] is False
        assert info["size"] > 0
        assert info["mime_type"] == "text/plain"
        assert "permissions" in info

    @pytest.mark.asyncio
    async def test_edit_file(self, tool, temp_dir):
        """Test editing file with search and replace."""
        test_file = temp_dir / "edit.txt"
        original = "Hello world\nThis is a test\nHello again"
        test_file.write_text(original)

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(
                operation="edit", path=str(test_file), search_text="Hello", replace_text="Hi"
            )

        assert result.status == ToolStatus.COMPLETED
        assert result.result["replacements"] == 2
        assert test_file.read_text() == "Hi world\nThis is a test\nHi again"
        assert result.metadata["files_affected"] == 1

    @pytest.mark.asyncio
    async def test_edit_file_no_search_text(self, tool, temp_dir):
        """Test edit operation without search_text."""
        test_file = temp_dir / "edit.txt"
        test_file.write_text("Content")

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(operation="edit", path=str(test_file))

        assert result.status == ToolStatus.FAILED
        assert "search_text required" in result.error

    @pytest.mark.asyncio
    async def test_batch_operations(self, tool, temp_dir):
        """Test batch operations."""
        # Prepare batch operations
        file1 = temp_dir / "batch1.txt"
        file2 = temp_dir / "batch2.txt"

        batch_ops = [
            {"operation": "write", "path": str(file1), "content": "Batch content 1"},
            {"operation": "write", "path": str(file2), "content": "Batch content 2"},
            {
                "operation": "copy",
                "path": str(file1),
                "destination": str(temp_dir / "batch1_copy.txt"),
            },
        ]

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(
                operation="batch",
                path="batch",  # Required field, even for batch operations
                batch_operations=batch_ops,
            )

        assert result.status == ToolStatus.COMPLETED
        batch_results = result.result
        assert len(batch_results) == 3
        assert all(r["success"] for r in batch_results)
        assert result.metadata["files_affected"] == 3

        # Verify files were created
        assert file1.exists()
        assert file2.exists()
        assert (temp_dir / "batch1_copy.txt").exists()

    @pytest.mark.asyncio
    async def test_batch_operations_with_failure(self, tool, temp_dir):
        """Test batch operations with some failures."""
        batch_ops = [
            {
                "operation": "write",
                "path": str(temp_dir / "batch_success.txt"),
                "content": "Success",
            },
            {"operation": "read", "path": "/non/existent/file.txt"},
        ]

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(
                operation="batch",
                path="batch",  # Required field, even for batch operations
                batch_operations=batch_ops,
            )

        assert result.status == ToolStatus.COMPLETED
        batch_results = result.result
        assert batch_results[0]["success"] is True
        assert batch_results[1]["success"] is False
        assert "error" in batch_results[1]

    @pytest.mark.asyncio
    async def test_batch_operations_no_operations(self, tool):
        """Test batch operation without operations list."""
        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await tool.execute(
                operation="batch",
                path="batch",  # Required field
            )

        assert result.status == ToolStatus.FAILED
        assert "batch_operations required" in result.error

    @pytest.mark.asyncio
    async def test_batch_operations_size_limit(self, tool):
        """Test batch operations exceeding size limit."""
        # Create too many operations
        batch_ops = [{"operation": "read", "path": f"/tmp/file{i}.txt"} for i in range(200)]

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            with patch.object(FilesConfig, "MAX_BATCH_SIZE", 100):
                result = await tool.execute(
                    operation="batch",
                    path="batch",  # Required field
                    batch_operations=batch_ops,
                )

        assert result.status == ToolStatus.FAILED
        assert "Batch size exceeds limit" in result.error

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
            "list_files",
        ]

        for tool_name in expected_tools:
            assert tool_name in pydantic_tools
            assert callable(pydantic_tools[tool_name])

    @pytest.mark.asyncio
    async def test_pydantic_read_file(self, tool, temp_dir):
        """Test PydanticAI read_file function."""
        test_file = temp_dir / "pydantic_test.txt"
        content = "PydanticAI test content"
        test_file.write_text(content)

        pydantic_tools = tool._create_pydantic_tools()
        read_file = pydantic_tools["read_file"]

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await read_file(str(test_file))

        assert result["content"] == content
        assert result["encoding"] == "utf-8"

    @pytest.mark.asyncio
    async def test_pydantic_write_file(self, tool, temp_dir):
        """Test PydanticAI write_file function."""
        test_file = temp_dir / "pydantic_write.txt"
        content = "Write via PydanticAI"

        pydantic_tools = tool._create_pydantic_tools()
        write_file = pydantic_tools["write_file"]

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await write_file(str(test_file), content)

        assert "bytes_written" in result
        assert test_file.read_text() == content

    @pytest.mark.asyncio
    async def test_pydantic_list_directory(self, tool, temp_dir):
        """Test PydanticAI list_directory function."""
        (temp_dir / "file1.txt").touch()
        (temp_dir / "file2.txt").touch()

        pydantic_tools = tool._create_pydantic_tools()
        list_directory = pydantic_tools["list_directory"]

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            with patch.object(FilesConfig, "should_ignore", return_value=False):
                result = await list_directory(str(temp_dir))

        assert len(result) == 2
        assert all(isinstance(entry, dict) for entry in result)

    @pytest.mark.asyncio
    async def test_pydantic_search_files(self, tool, temp_dir):
        """Test PydanticAI search_files function."""
        (temp_dir / "search1.txt").write_text("Find this text")
        (temp_dir / "search2.txt").write_text("Also find this")

        pydantic_tools = tool._create_pydantic_tools()
        search_files = pydantic_tools["search_files"]

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            with patch.object(FilesConfig, "is_text_file", return_value=True):
                result = await search_files(str(temp_dir), "find")

        assert len(result) == 2
        assert all(isinstance(match, dict) for match in result)

    @pytest.mark.asyncio
    async def test_pydantic_find_files(self, tool, temp_dir):
        """Test PydanticAI find_files function."""
        (temp_dir / "test1.py").touch()
        (temp_dir / "test2.py").touch()
        (temp_dir / "other.txt").touch()

        pydantic_tools = tool._create_pydantic_tools()
        find_files = pydantic_tools["find_files"]

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            with patch.object(FilesConfig, "should_ignore", return_value=False):
                result = await find_files(str(temp_dir), "*.py")

        assert len(result) == 2
        assert all(f.endswith(".py") for f in result)

    @pytest.mark.asyncio
    async def test_pydantic_create_dir(self, tool, temp_dir):
        """Test PydanticAI create_dir function."""
        new_dir = temp_dir / "pydantic_dir"

        pydantic_tools = tool._create_pydantic_tools()
        create_dir = pydantic_tools["create_dir"]

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            result = await create_dir(str(new_dir))

        assert result["created"] is True
        assert result["path"] == str(new_dir)
        assert new_dir.exists()

    @pytest.mark.asyncio
    async def test_pydantic_list_files(self, tool, temp_dir):
        """Test PydanticAI list_files function (alias for list_directory)."""
        (temp_dir / "file1.txt").touch()

        pydantic_tools = tool._create_pydantic_tools()
        list_files = pydantic_tools["list_files"]

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            with patch.object(FilesConfig, "should_ignore", return_value=False):
                result = await list_files(str(temp_dir))

        assert "files" in result
        assert len(result["files"]) == 1

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
    async def test_encoding_fallback(self, tool, temp_dir):
        """Test encoding fallback for non-UTF8 files."""
        test_file = temp_dir / "encoded.txt"
        # Write with latin-1 encoding
        test_file.write_bytes("Café".encode("latin-1"))

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            with patch.object(FilesConfig, "FALLBACK_ENCODINGS", ["latin-1"]):
                result = await tool.execute(
                    operation="read",
                    path=str(test_file),
                    encoding="utf-8",  # Will fail, should fallback
                )

        assert result.status == ToolStatus.COMPLETED
        assert "Café" in result.result["content"]
        assert result.result["encoding"] == "latin-1"

    @pytest.mark.asyncio
    async def test_binary_file_fallback(self, tool, temp_dir):
        """Test reading binary files."""
        test_file = temp_dir / "binary.bin"
        binary_data = b"\x00\x01\x02\x03\xff"
        test_file.write_bytes(binary_data)

        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            with patch.object(FilesConfig, "FALLBACK_ENCODINGS", []):
                result = await tool.execute(operation="read", path=str(test_file))

        assert result.status == ToolStatus.COMPLETED
        assert result.result["encoding"] == "binary"
        assert str(binary_data) in result.result["content"]

    @pytest.mark.asyncio
    async def test_route_unknown_operation(self, tool):
        """Test routing to unknown operation."""
        # Can't create FileOperationRequest with invalid operation
        # So we'll test the ValueError is raised in execute
        with patch.object(FilesConfig, "is_safe_path", return_value=True):
            # Patch FileOperation to allow invalid value temporarily
            with patch("aida.tools.files.models.FileOperation") as mock_enum:
                # This will cause the operation to not match any handler
                mock_enum.READ = "invalid_op"

                result = await tool.execute(operation="invalid_op", path="/tmp/test.txt")

                assert result.status == ToolStatus.FAILED


if __name__ == "__main__":
    pytest.main([__file__])
