"""File operations tool using MCP filesystem server."""

from datetime import datetime
import logging
from typing import Any
import uuid

from aida.providers.mcp.filesystem_client import MCPFilesystemClient
from aida.tools.base import ToolCapability, ToolParameter, ToolResult, ToolStatus
from aida.tools.base_tool import BaseModularTool

from .config import FilesConfig
from .models import FileOperation, FileOperationRequest, FileOperationResponse

logger = logging.getLogger(__name__)


class FileOperationsTool(BaseModularTool[FileOperationRequest, FileOperationResponse, FilesConfig]):
    """File operations tool using official MCP filesystem server."""

    def __init__(self, allowed_directories: list[str] | None = None):
        """Initialize file operations tool.

        Args:
            allowed_directories: List of directories the MCP server can access.
                               If None, uses configured safe paths.
        """
        super().__init__()

        # Use configured safe paths if no directories specified
        if allowed_directories is None:
            allowed_directories = list(FilesConfig.ALLOWED_BASE_PATHS)

        self.allowed_directories = allowed_directories
        self._client = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure MCP client is initialized and connected."""
        if not self._initialized:
            self._client = MCPFilesystemClient(self.allowed_directories)
            if await self._client.connect():
                self._initialized = True
            else:
                raise RuntimeError("Failed to connect to MCP filesystem server")

    def _get_tool_name(self) -> str:
        return "file_operations"

    def _get_tool_version(self) -> str:
        return "3.0.0"  # Major version bump for MCP integration

    def _get_tool_description(self) -> str:
        return "File operations using official MCP filesystem server"

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
                    description="Content for write operations",
                    required=False,
                ),
                ToolParameter(
                    name="destination",
                    type="str",
                    description="Destination path for move",
                    required=False,
                ),
                ToolParameter(
                    name="pattern",
                    type="str",
                    description="Search pattern (glob)",
                    required=False,
                ),
                ToolParameter(
                    name="search_text", type="str", description="Text to search for", required=False
                ),
                ToolParameter(
                    name="recursive",
                    type="bool",
                    description="Recursive operation",
                    required=False,
                    default=False,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        """Execute file operation using MCP filesystem server."""
        start_time = datetime.utcnow()

        try:
            # Ensure MCP client is connected
            await self._ensure_initialized()

            # Validate operation
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

            # Map operations to MCP tool names
            operation_map = {
                FileOperation.READ: "read_file",
                FileOperation.WRITE: "write_file",
                FileOperation.DELETE: "delete_file",
                FileOperation.MOVE: "move_file",
                FileOperation.CREATE_DIR: "create_directory",
                FileOperation.LIST_DIR: "list_directory",
                FileOperation.GET_INFO: "get_file_info",
                FileOperation.EDIT: "edit_file",
            }

            mcp_tool = operation_map.get(request.operation)

            if not mcp_tool:
                # Handle operations not directly supported by MCP
                return await self._handle_complex_operation(request, start_time)

            # Prepare arguments for MCP tool
            mcp_args = {"path": request.path}

            if request.operation == FileOperation.WRITE and request.content is not None:
                mcp_args["content"] = request.content
            elif request.operation == FileOperation.MOVE and request.destination:
                mcp_args["destination"] = request.destination
            elif request.operation == FileOperation.DELETE and request.recursive:
                mcp_args["recursive"] = request.recursive
            elif (
                request.operation == FileOperation.EDIT
                and request.search_text
                and request.replace_text is not None
            ):
                mcp_args["find"] = request.search_text
                mcp_args["replace"] = request.replace_text

            # Call MCP tool
            result = await self._client.call_tool(mcp_tool, mcp_args)

            # Create response
            response = FileOperationResponse(
                operation=request.operation,
                success=True,
                path=str(request.path),
                result=result,
                files_affected=1,
                details=result,
            )

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
            logger.error(f"MCP file operation failed: {e}")
            return ToolResult(
                tool_name=self.name,
                execution_id=str(uuid.uuid4()),
                status=ToolStatus.FAILED,
                error=str(e),
                started_at=start_time,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )

    async def _handle_complex_operation(
        self, request: FileOperationRequest, start_time: datetime
    ) -> ToolResult:
        """Handle operations that require multiple MCP calls."""
        try:
            if request.operation == FileOperation.APPEND:
                # Read existing content
                try:
                    read_result = await self._client.call_tool("read_file", {"path": request.path})
                    existing_content = read_result.get("content", "")
                except Exception:
                    existing_content = ""

                # Write combined content
                new_content = existing_content + (request.content or "")
                await self._client.call_tool(
                    "write_file", {"path": request.path, "content": new_content}
                )

                result = {"bytes_appended": len((request.content or "").encode("utf-8"))}

            elif request.operation == FileOperation.COPY:
                # Read source
                read_result = await self._client.call_tool("read_file", {"path": request.path})
                content = read_result.get("content", "")

                # Write to destination
                await self._client.call_tool(
                    "write_file", {"path": request.destination, "content": content}
                )

                result = {
                    "source": request.path,
                    "destination": request.destination,
                    "bytes_copied": len(content.encode("utf-8")),
                }

            elif request.operation == FileOperation.SEARCH:
                # Get list of files
                list_result = await self._client.call_tool("list_directory", {"path": request.path})

                results = []
                files = list_result.get("entries", [])

                # Search through files
                for entry in files:
                    if entry.get("type") == "file":
                        file_path = entry.get("path")
                        try:
                            read_result = await self._client.call_tool(
                                "read_file", {"path": file_path}
                            )
                            content = read_result.get("content", "")

                            if (
                                request.search_text
                                and request.search_text.lower() in content.lower()
                            ):
                                # Count occurrences
                                matches = content.lower().count(request.search_text.lower())
                                results.append(
                                    {
                                        "file": file_path,
                                        "matches": matches,
                                    }
                                )
                        except Exception:
                            logger.debug(f"Error reading {file_path} during search")

                result = {"results": results, "files_searched": len(files)}

            elif request.operation == FileOperation.FIND:
                # Use list_directory and filter by pattern
                import fnmatch

                list_result = await self._client.call_tool("list_directory", {"path": request.path})

                entries = list_result.get("entries", [])
                results = []

                for entry in entries:
                    name = entry.get("name", "")
                    if request.pattern and fnmatch.fnmatch(name, request.pattern):
                        results.append(entry.get("path"))

                result = {"results": results, "count": len(results)}

            elif request.operation == FileOperation.BATCH:
                # Execute batch operations
                results = []
                total_files = 0

                for op_data in request.batch_operations or []:
                    try:
                        op_request = FileOperationRequest(**op_data)  # ty: ignore[missing-argument]
                        op_result = await self.execute(**op_data)

                        result_entry = {
                            "operation": op_request.operation.value,
                            "path": op_request.path,
                            "success": op_result.status == ToolStatus.COMPLETED,
                        }

                        # Add error message if operation failed
                        if op_result.status == ToolStatus.FAILED and op_result.error:
                            result_entry["error"] = op_result.error

                        results.append(result_entry)

                        if op_result.status == ToolStatus.COMPLETED:
                            total_files += 1

                    except Exception as e:
                        results.append(
                            {
                                "operation": op_data.get("operation"),
                                "path": op_data.get("path"),
                                "success": False,
                                "error": str(e),
                            }
                        )

                result = {"results": results, "total_operations": len(results)}

            else:
                raise ValueError(f"Unsupported operation: {request.operation}")

            # Create response
            response = FileOperationResponse(
                operation=request.operation,
                success=True,
                path=str(request.path),
                result=result,
                files_affected=result.get("files_affected", 1),
            )

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
            logger.error(f"Complex operation failed: {e}")
            return ToolResult(
                tool_name=self.name,
                execution_id=str(uuid.uuid4()),
                status=ToolStatus.FAILED,
                error=str(e),
                started_at=start_time,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )

    async def cleanup(self):
        """Cleanup MCP client connection."""
        if self._client and self._initialized:
            await self._client.disconnect()
            self._initialized = False

    def _create_pydantic_tools(self) -> dict[str, Any]:
        """Create PydanticAI-compatible tool functions."""

        async def read_file(path: str, encoding: str = "utf-8") -> dict[str, Any]:
            """Read file contents."""
            result = await self.execute(operation="read", path=path, encoding=encoding)
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
            return result.result.get("results", [])

        async def find_files(path: str, pattern: str, recursive: bool = True) -> list[str]:
            """Find files by pattern."""
            result = await self.execute(
                operation="find", path=path, pattern=pattern, recursive=recursive
            )
            return result.result.get("results", [])

        async def create_dir(path: str) -> dict[str, Any]:
            """Create a directory."""
            result = await self.execute(operation="create_dir", path=path)
            return {"created": result.status == ToolStatus.COMPLETED, "path": path}

        return {
            "read_file": read_file,
            "write_file": write_file,
            "list_directory": list_directory,
            "search_files": search_files,
            "find_files": find_files,
            "create_dir": create_dir,
        }

    def _create_mcp_server(self):
        """Create MCP server instance."""
        from .mcp_server import FilesMCPServer

        return FilesMCPServer(self)

    def _create_observability(self, config: dict[str, Any]):
        """Create observability instance."""
        from .observability import FilesObservability

        return FilesObservability(self, config)
