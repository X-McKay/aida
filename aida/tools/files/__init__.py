"""File operations tool for comprehensive file and directory management."""

from .config import FilesConfig
from .models import (
    FileInfo,
    FileOperation,
    FileOperationRequest,
    FileOperationResponse,
    SearchScope,
)

# Conditionally import the appropriate implementation
if FilesConfig.USE_MCP_BACKEND:
    from .files_mcp import MCPFileOperationsTool as FileOperationsTool
else:
    from .files import FileOperationsTool

__all__ = [
    "FileOperationsTool",
    "FileOperation",
    "FileOperationRequest",
    "FileOperationResponse",
    "SearchScope",
    "FileInfo",
    "FilesConfig",
]
