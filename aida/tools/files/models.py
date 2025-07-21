"""Pydantic models for file operations tool."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import uuid

from pydantic import BaseModel, Field, validator


class FileOperation(str, Enum):
    """Supported file operations."""

    READ = "read"
    WRITE = "write"
    APPEND = "append"
    DELETE = "delete"
    COPY = "copy"
    MOVE = "move"
    CREATE_DIR = "create_dir"
    LIST_DIR = "list_dir"
    SEARCH = "search"
    FIND = "find"
    GET_INFO = "get_info"
    EDIT = "edit"
    BATCH = "batch"


class SearchScope(str, Enum):
    """Search scope options."""

    FILE = "file"
    DIRECTORY = "directory"
    RECURSIVE = "recursive"


class FileInfo(BaseModel):
    """File information model."""

    path: str
    name: str
    size: int
    is_file: bool
    is_dir: bool
    created: datetime
    modified: datetime
    permissions: str | None = None
    mime_type: str | None = None


class FileOperationRequest(BaseModel):
    """Request model for file operations."""

    operation: FileOperation
    path: str = Field(..., description="File or directory path")
    content: str | None = Field(None, description="Content for write/append operations")
    destination: str | None = Field(None, description="Destination path for copy/move")
    pattern: str | None = Field(None, description="Search pattern (regex or glob)")
    search_text: str | None = Field(None, description="Text to search for")
    replace_text: str | None = Field(None, description="Text to replace with")
    scope: SearchScope = Field(SearchScope.FILE, description="Search scope")
    recursive: bool = Field(False, description="Recursive operation")
    create_parents: bool = Field(True, description="Create parent directories if needed")
    encoding: str = Field("utf-8", description="File encoding")
    batch_operations: list[dict[str, Any]] | None = Field(None, description="Batch operations")

    @validator("path")
    def validate_path(cls, v):
        """Validate and normalize path."""
        if not v:
            raise ValueError("Path cannot be empty")
        # Normalize path
        return str(Path(v).expanduser())

    @validator("destination")
    def validate_destination(cls, v, values):
        """Validate destination for copy/move operations."""
        if values.get("operation") in [FileOperation.COPY, FileOperation.MOVE] and not v:
            raise ValueError(f"Destination required for {values['operation']} operation")
        if v:
            return str(Path(v).expanduser())
        return v


class FileOperationResponse(BaseModel):
    """Response model for file operations."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation: FileOperation
    success: bool
    path: str
    result: Any | None = None
    files_affected: int = Field(0, description="Number of files affected")
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration for FileOperationResponse."""

        arbitrary_types_allowed = True
