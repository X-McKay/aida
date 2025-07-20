"""Pydantic models for file operations tool."""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
from pathlib import Path
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
    permissions: Optional[str] = None
    mime_type: Optional[str] = None


class FileOperationRequest(BaseModel):
    """Request model for file operations."""
    operation: FileOperation
    path: str = Field(..., description="File or directory path")
    content: Optional[str] = Field(None, description="Content for write/append operations")
    destination: Optional[str] = Field(None, description="Destination path for copy/move")
    pattern: Optional[str] = Field(None, description="Search pattern (regex or glob)")
    search_text: Optional[str] = Field(None, description="Text to search for")
    replace_text: Optional[str] = Field(None, description="Text to replace with")
    scope: SearchScope = Field(SearchScope.FILE, description="Search scope")
    recursive: bool = Field(False, description="Recursive operation")
    create_parents: bool = Field(True, description="Create parent directories if needed")
    encoding: str = Field("utf-8", description="File encoding")
    batch_operations: Optional[List[Dict[str, Any]]] = Field(None, description="Batch operations")
    
    @validator('path')
    def validate_path(cls, v):
        """Validate and normalize path."""
        if not v:
            raise ValueError("Path cannot be empty")
        # Normalize path
        return str(Path(v).expanduser())
    
    @validator('destination')
    def validate_destination(cls, v, values):
        """Validate destination for copy/move operations."""
        if values.get('operation') in [FileOperation.COPY, FileOperation.MOVE] and not v:
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
    result: Optional[Any] = None
    files_affected: int = Field(0, description="Number of files affected")
    error: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True