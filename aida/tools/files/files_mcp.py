"""File operations tool using MCP filesystem server as backend."""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any
import uuid

from aida.providers.mcp.filesystem_client import MCPFilesystemAdapter, MCPFilesystemClient
from aida.tools.base import ToolCapability, ToolParameter, ToolResult, ToolStatus
from aida.tools.base_tool import BaseModularTool

from .config import FilesConfig
from .models import FileOperation, FileOperationRequest, FileOperationResponse

logger = logging.getLogger(__name__)


class MCPFileOperationsTool(
    BaseModularTool[FileOperationRequest, FileOperationResponse, FilesConfig]
):
    """File operations tool that uses MCP filesystem server as backend."""

    def __init__(self, allowed_directories: list[str] | None = None):
        """Initialize MCP-based file operations tool.

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
        self._adapter = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure MCP client is initialized and connected."""
        if not self._initialized:
            self._client = MCPFilesystemClient(self.allowed_directories)
            if await self._client.connect():
                self._adapter = MCPFilesystemAdapter(self._client)
                self._initialized = True
            else:
                raise RuntimeError("Failed to connect to MCP filesystem server")

    def _get_tool_name(self) -> str:
        return "mcp_file_operations"

    def _get_tool_version(self) -> str:
        return "1.0.0"

    def _get_tool_description(self) -> str:
        return "File operations using MCP filesystem server for secure, standardized access"

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

            # Check if path is within allowed directories
            path = Path(request.path).resolve()
            if not any(path.is_relative_to(Path(d).resolve()) for d in self.allowed_directories):
                raise ValueError(f"Access denied: {request.path} is not in allowed directories")

            # Translate and execute operation via MCP
            result = await self._adapter.translate_operation(
                request.operation.value, **request.model_dump(exclude={"operation"})
            )

            # Create response
            response = FileOperationResponse(
                operation=request.operation,
                success=True,
                path=str(request.path),
                result=result,
                files_affected=result.get("files_affected", 1),
                details=result,
            )

            # Return successful result
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
        # We don't need a separate MCP server since we're using an external one
        # But we can return a proxy that forwards to the external server
        from .mcp_server import FilesMCPServer

        return FilesMCPServer(self)

    def _create_observability(self, config: dict[str, Any]):
        """Create observability instance."""
        from .observability import FilesObservability

        return FilesObservability(self, config)
