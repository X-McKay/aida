"""File operations tool using official MCP filesystem server."""

from .config import FilesConfig
from .files import FileOperationsTool
from .models import (
    FileInfo,
    FileOperation,
    FileOperationRequest,
    FileOperationResponse,
    SearchScope,
)

__all__ = [
    "FileOperationsTool",
    "FileOperation",
    "FileOperationRequest",
    "FileOperationResponse",
    "SearchScope",
    "FileInfo",
    "FilesConfig",
]
