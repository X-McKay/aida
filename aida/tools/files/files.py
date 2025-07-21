"""Main file operations tool implementation."""

from collections.abc import Callable
from datetime import datetime
import logging
from pathlib import Path
import re
import shutil
from typing import Any
import uuid

from aida.tools.base import ToolCapability, ToolParameter, ToolResult, ToolStatus
from aida.tools.base_tool import BaseModularTool

from .config import FilesConfig
from .models import FileInfo, FileOperation, FileOperationRequest, FileOperationResponse

logger = logging.getLogger(__name__)


class FileOperationsTool(BaseModularTool[FileOperationRequest, FileOperationResponse, FilesConfig]):
    """Tool for comprehensive file and directory operations."""

    def __init__(self):
        """Initialize the FileOperationsTool."""
        super().__init__()

    def _get_tool_name(self) -> str:
        return "file_operations"

    def _get_tool_version(self) -> str:
        return "2.0.0"

    def _get_tool_description(self) -> str:
        return "Comprehensive file and directory operations with search and editing capabilities"

    def _get_default_config(self):
        return FilesConfig

    def get_capability(self) -> ToolCapability:
        """Get tool capability descriptor."""
        return ToolCapability(
            name=self.name,
            version=self.version,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="operation",
                    type="str",
                    description="File operation to perform",
                    required=True,
                    choices=[op.value for op in FileOperation],
                ),
                ToolParameter(
                    name="path", type="str", description="File or directory path", required=True
                ),
                ToolParameter(
                    name="content",
                    type="str",
                    description="Content for write/append operations",
                    required=False,
                ),
                ToolParameter(
                    name="destination",
                    type="str",
                    description="Destination path for copy/move",
                    required=False,
                ),
                ToolParameter(
                    name="pattern",
                    type="str",
                    description="Search pattern (regex or glob)",
                    required=False,
                ),
                ToolParameter(
                    name="search_text", type="str", description="Text to search for", required=False
                ),
                ToolParameter(
                    name="replace_text",
                    type="str",
                    description="Text to replace with",
                    required=False,
                ),
                ToolParameter(
                    name="recursive",
                    type="bool",
                    description="Recursive operation",
                    required=False,
                    default=False,
                ),
                ToolParameter(
                    name="encoding",
                    type="str",
                    description="File encoding",
                    required=False,
                    default="utf-8",
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        """Execute file operation."""
        start_time = datetime.utcnow()

        try:
            # Ensure operation is provided
            if "operation" not in kwargs:
                return ToolResult(
                    tool_name=self.name,
                    execution_id=str(uuid.uuid4()),
                    status=ToolStatus.FAILED,
                    error="Missing required parameter: operation",
                    started_at=start_time,
                    completed_at=datetime.utcnow(),
                    metadata={"error_type": "validation_error"},
                )

            # Create request model
            request = FileOperationRequest(**kwargs)  # ty: ignore[missing-argument]

            # Check path safety
            if not FilesConfig.is_safe_path(request.path):
                raise ValueError(f"Access denied: {request.path} is not in allowed paths")

            # Route to appropriate operation
            response = await self._route_operation(request)

            # Create successful result
            return ToolResult(
                tool_name=self.name,
                execution_id=response.request_id,
                status=ToolStatus.COMPLETED,
                result=response.result,
                started_at=start_time,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                metadata={
                    "operation": request.operation.value,
                    "path": request.path,
                    "files_affected": response.files_affected,
                    "success": response.success,
                },
            )

        except Exception as e:
            logger.error(f"File operation failed: {e}")
            return ToolResult(
                tool_name=self.name,
                execution_id="",
                status=ToolStatus.FAILED,
                error=str(e),
                started_at=start_time,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )

    async def _route_operation(self, request: FileOperationRequest) -> FileOperationResponse:
        """Route to specific operation handler."""
        handlers = {
            FileOperation.READ: self._read_file,
            FileOperation.WRITE: self._write_file,
            FileOperation.APPEND: self._append_file,
            FileOperation.DELETE: self._delete_file,
            FileOperation.COPY: self._copy_file,
            FileOperation.MOVE: self._move_file,
            FileOperation.CREATE_DIR: self._create_directory,
            FileOperation.LIST_DIR: self._list_directory,
            FileOperation.SEARCH: self._search_files,
            FileOperation.FIND: self._find_files,
            FileOperation.GET_INFO: self._get_file_info,
            FileOperation.EDIT: self._edit_file,
            FileOperation.BATCH: self._batch_operations,
        }

        handler = handlers.get(request.operation)
        if not handler:
            raise ValueError(f"Unknown operation: {request.operation}")

        return await handler(request)

    async def _read_file(self, request: FileOperationRequest) -> FileOperationResponse:
        """Read file contents."""
        path = Path(request.path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        # Check file size
        if path.stat().st_size > FilesConfig.MAX_FILE_SIZE:
            raise ValueError(f"File too large: {path.stat().st_size} bytes")

        # Try to read with different encodings
        content = None
        encoding_used = None

        for encoding in [request.encoding] + FilesConfig.FALLBACK_ENCODINGS:
            try:
                content = path.read_text(encoding=encoding)
                encoding_used = encoding
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            # Read as binary if text decoding fails
            content = str(path.read_bytes())
            encoding_used = "binary"

        return FileOperationResponse(
            operation=request.operation,
            success=True,
            path=str(path),
            result={
                "content": content,
                "size_bytes": path.stat().st_size,
                "line_count": content.count("\n") + 1 if content else 0,
                "encoding": encoding_used,
            },
            files_affected=1,
            details={"encoding": encoding_used, "size": path.stat().st_size},
        )

    async def _write_file(self, request: FileOperationRequest) -> FileOperationResponse:
        """Write content to file."""
        path = Path(request.path)

        # Create parent directories if needed
        if request.create_parents and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        content = request.content or ""
        path.write_text(content, encoding=request.encoding)

        return FileOperationResponse(
            operation=request.operation,
            success=True,
            path=str(path),
            result={"bytes_written": len(content.encode(request.encoding)), "path": str(path)},
            files_affected=1,
            details={"size": len(content)},
        )

    async def _append_file(self, request: FileOperationRequest) -> FileOperationResponse:
        """Append content to file."""
        path = Path(request.path)

        # Create file if it doesn't exist
        if not path.exists():
            return await self._write_file(request)

        # Append content
        with open(path, "a", encoding=request.encoding) as f:
            f.write(request.content or "")

        return FileOperationResponse(
            operation=request.operation,
            success=True,
            path=str(path),
            files_affected=1,
            details={"appended_size": len(request.content or "")},
        )

    async def _delete_file(self, request: FileOperationRequest) -> FileOperationResponse:
        """Delete file or directory."""
        path = Path(request.path)

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        files_deleted = 0

        if path.is_file():
            path.unlink()
            files_deleted = 1
        elif path.is_dir():
            if request.recursive:
                # Count files before deletion
                files_deleted = sum(1 for _ in path.rglob("*") if _.is_file())
                shutil.rmtree(path)
            else:
                path.rmdir()  # Only works on empty directories
                files_deleted = 0

        return FileOperationResponse(
            operation=request.operation, success=True, path=str(path), files_affected=files_deleted
        )

    async def _copy_file(self, request: FileOperationRequest) -> FileOperationResponse:
        """Copy file or directory."""
        src = Path(request.path)
        dst = Path(request.destination)

        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")

        files_copied = 0

        if src.is_file():
            # Create parent directories if needed
            if request.create_parents and not dst.parent.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src, dst)
            files_copied = 1
        elif src.is_dir():
            if request.recursive:
                shutil.copytree(src, dst)
                files_copied = sum(1 for _ in dst.rglob("*") if _.is_file())
            else:
                raise ValueError("Use recursive=True to copy directories")

        return FileOperationResponse(
            operation=request.operation,
            success=True,
            path=str(src),
            result=str(dst),
            files_affected=files_copied,
        )

    async def _move_file(self, request: FileOperationRequest) -> FileOperationResponse:
        """Move file or directory."""
        src = Path(request.path)
        dst = Path(request.destination)

        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")

        # Create parent directories if needed
        if request.create_parents and not dst.parent.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)

        files_moved = 1
        if src.is_dir():
            files_moved = sum(1 for _ in src.rglob("*") if _.is_file())

        shutil.move(str(src), str(dst))

        return FileOperationResponse(
            operation=request.operation,
            success=True,
            path=str(src),
            result=str(dst),
            files_affected=files_moved,
        )

    async def _create_directory(self, request: FileOperationRequest) -> FileOperationResponse:
        """Create directory."""
        path = Path(request.path)

        path.mkdir(parents=request.create_parents, exist_ok=True)

        return FileOperationResponse(
            operation=request.operation, success=True, path=str(path), files_affected=0
        )

    async def _list_directory(self, request: FileOperationRequest) -> FileOperationResponse:
        """List directory contents."""
        path = Path(request.path)

        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")

        entries = []

        if request.recursive:
            for item in path.rglob("*"):
                if not FilesConfig.should_ignore(str(item)):
                    entries.append(
                        {
                            "path": str(item),
                            "name": item.name,
                            "is_file": item.is_file(),
                            "is_dir": item.is_dir(),
                            "size": item.stat().st_size if item.is_file() else 0,
                        }
                    )
        else:
            for item in path.iterdir():
                if not FilesConfig.should_ignore(str(item)):
                    entries.append(
                        {
                            "path": str(item),
                            "name": item.name,
                            "is_file": item.is_file(),
                            "is_dir": item.is_dir(),
                            "size": item.stat().st_size if item.is_file() else 0,
                        }
                    )

        return FileOperationResponse(
            operation=request.operation,
            success=True,
            path=str(path),
            result=entries,
            files_affected=len(entries),
        )

    async def _search_files(self, request: FileOperationRequest) -> FileOperationResponse:
        """Search for text in files."""
        path = Path(request.path)
        results = []

        if not request.search_text:
            raise ValueError("search_text required for search operation")

        pattern = re.compile(request.search_text, re.IGNORECASE)

        # Determine files to search
        if path.is_file():
            files_to_search = [path]
        elif path.is_dir() and request.recursive:
            files_to_search = [
                f for f in path.rglob("*") if f.is_file() and FilesConfig.is_text_file(str(f))
            ]
        elif path.is_dir():
            files_to_search = [
                f for f in path.iterdir() if f.is_file() and FilesConfig.is_text_file(str(f))
            ]
        else:
            files_to_search = []

        # Search files
        for file_path in files_to_search[: FilesConfig.MAX_SEARCH_RESULTS]:
            try:
                content = file_path.read_text(encoding=request.encoding)
                matches = list(pattern.finditer(content))

                if matches:
                    lines_with_matches = []
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        lines_with_matches.append(
                            {
                                "line": line_num,
                                "text": match.group(),
                                "start": match.start(),
                                "end": match.end(),
                            }
                        )

                    results.append(
                        {
                            "file": str(file_path),
                            "matches": len(matches),
                            "lines": lines_with_matches[:10],  # Limit matches per file
                        }
                    )
            except Exception as e:
                logger.debug(f"Error searching {file_path}: {e}")

        return FileOperationResponse(
            operation=request.operation,
            success=True,
            path=str(path),
            result=results,
            files_affected=len(results),
        )

    async def _find_files(self, request: FileOperationRequest) -> FileOperationResponse:
        """Find files by pattern."""
        path = Path(request.path)

        if not request.pattern:
            raise ValueError("pattern required for find operation")

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        results = []

        if request.recursive and path.is_dir():
            for item in path.rglob(request.pattern):
                if not FilesConfig.should_ignore(str(item)):
                    results.append(str(item))
        elif path.is_dir():
            for item in path.glob(request.pattern):
                if not FilesConfig.should_ignore(str(item)):
                    results.append(str(item))

        return FileOperationResponse(
            operation=request.operation,
            success=True,
            path=str(path),
            result=results[: FilesConfig.MAX_SEARCH_RESULTS],
            files_affected=len(results),
        )

    async def _get_file_info(self, request: FileOperationRequest) -> FileOperationResponse:
        """Get detailed file information."""
        path = Path(request.path)

        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        stat = path.stat()

        info = FileInfo(
            path=str(path),
            name=path.name,
            size=stat.st_size,
            is_file=path.is_file(),
            is_dir=path.is_dir(),
            created=datetime.fromtimestamp(stat.st_ctime),
            modified=datetime.fromtimestamp(stat.st_mtime),
            permissions=oct(stat.st_mode)[-3:],
        )

        # Add mime type for files
        if path.is_file():
            if FilesConfig.is_text_file(str(path)):
                info.mime_type = "text/plain"
            elif FilesConfig.is_binary_file(str(path)):
                info.mime_type = "application/octet-stream"

        return FileOperationResponse(
            operation=request.operation,
            success=True,
            path=str(path),
            result=info.dict(),
            files_affected=1,
        )

    async def _edit_file(self, request: FileOperationRequest) -> FileOperationResponse:
        """Edit file with search and replace."""
        if not request.search_text:
            raise ValueError("search_text required for edit operation")

        if request.replace_text is None:
            request.replace_text = ""

        # Read file
        read_response = await self._read_file(request)
        content = read_response.result

        # Perform replacement
        new_content = content.replace(request.search_text, request.replace_text)
        replacements = content.count(request.search_text)

        # Write back if changes were made
        if replacements > 0:
            request.content = new_content
            await self._write_file(request)

        return FileOperationResponse(
            operation=request.operation,
            success=True,
            path=request.path,
            result={"replacements": replacements},
            files_affected=1 if replacements > 0 else 0,
        )

    async def _batch_operations(self, request: FileOperationRequest) -> FileOperationResponse:
        """Execute multiple operations in batch."""
        if not request.batch_operations:
            raise ValueError("batch_operations required for batch operation")

        if len(request.batch_operations) > FilesConfig.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds limit of {FilesConfig.MAX_BATCH_SIZE}")

        results = []
        total_files_affected = 0

        for op_data in request.batch_operations:
            try:
                # Ensure operation is provided for batch item
                if "operation" not in op_data:
                    results.append(
                        {
                            "operation": "unknown",
                            "success": False,
                            "error": "Missing operation in batch item",
                        }
                    )
                    continue

                # Create request for individual operation
                op_request = FileOperationRequest(**op_data)  # ty: ignore[missing-argument]
                response = await self._route_operation(op_request)

                results.append(
                    {
                        "operation": op_request.operation.value,
                        "path": op_request.path,
                        "success": response.success,
                        "files_affected": response.files_affected,
                    }
                )

                total_files_affected += response.files_affected

            except Exception as e:
                results.append(
                    {
                        "operation": op_data.get("operation"),
                        "path": op_data.get("path"),
                        "success": False,
                        "error": str(e),
                    }
                )

        return FileOperationResponse(
            operation=request.operation,
            success=all(r["success"] for r in results),
            path="batch",
            result=results,
            files_affected=total_files_affected,
        )

    def _create_pydantic_tools(self) -> dict[str, Callable]:
        """Create PydanticAI-compatible tool functions."""

        async def read_file(path: str, encoding: str = "utf-8") -> dict[str, Any]:
            """Read file contents."""
            result = await self.execute(operation="read", path=path, encoding=encoding)
            # Result is already a dict with content and metadata
            return result.result

        async def write_file(path: str, content: str, encoding: str = "utf-8") -> dict[str, Any]:
            """Write content to file."""
            result = await self.execute(
                operation="write", path=path, content=content, encoding=encoding
            )
            return result.result

        async def list_directory(path: str, recursive: bool = False) -> list[dict[str, Any]]:
            """List directory contents."""
            result = await self.execute(operation="list_dir", path=path, recursive=recursive)
            return result.result

        async def search_files(
            path: str, search_text: str, recursive: bool = True
        ) -> list[dict[str, Any]]:
            """Search for text in files."""
            result = await self.execute(
                operation="search", path=path, search_text=search_text, recursive=recursive
            )
            return result.result

        async def find_files(path: str, pattern: str, recursive: bool = True) -> list[str]:
            """Find files by pattern."""
            result = await self.execute(
                operation="find", path=path, pattern=pattern, recursive=recursive
            )
            return result.result

        async def create_dir(path: str) -> dict[str, Any]:
            """Create a directory."""
            result = await self.execute(operation="create_dir", path=path)
            return {"created": result.status == ToolStatus.COMPLETED, "path": path}

        async def list_files(path: str, recursive: bool = False) -> dict[str, Any]:
            """List files in directory (alias for list_directory)."""
            result = await self.execute(operation="list_dir", path=path, recursive=recursive)
            return {"files": result.result}

        return {
            "read_file": read_file,
            "write_file": write_file,
            "list_directory": list_directory,
            "search_files": search_files,
            "find_files": find_files,
            "create_dir": create_dir,
            "list_files": list_files,
        }

    def _create_mcp_server(self):
        """Create MCP server instance."""
        from .mcp_server import FilesMCPServer

        return FilesMCPServer(self)

    def _create_observability(self, config: dict[str, Any]):
        """Create observability instance."""
        from .observability import FilesObservability

        return FilesObservability(self, config)
