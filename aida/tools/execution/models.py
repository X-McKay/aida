"""Pydantic models for execution tool."""

from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime
import uuid

from pydantic import BaseModel, Field, validator


class ExecutionLanguage(str, Enum):
    """Supported execution languages/runtimes."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    BASH = "bash"
    NODE = "node"
    GO = "go"
    RUST = "rust"
    JAVA = "java"


class ExecutionEnvironment(str, Enum):
    """Execution environment types."""
    CONTAINER = "container"
    LOCAL = "local"  # For testing only
    SANDBOX = "sandbox"


class ExecutionRequest(BaseModel):
    """Request model for code execution."""
    language: ExecutionLanguage = Field(..., description="Programming language or runtime")
    code: str = Field(..., description="Code to execute")
    files: Optional[Dict[str, str]] = Field(None, description="Additional files needed")
    packages: Optional[List[str]] = Field(None, description="Package dependencies")
    env_vars: Optional[Dict[str, str]] = Field(None, description="Environment variables")
    timeout: int = Field(30, ge=1, le=300, description="Execution timeout in seconds")
    memory_limit: str = Field("512m", description="Memory limit (e.g., 512m, 1g)")
    environment: ExecutionEnvironment = Field(ExecutionEnvironment.CONTAINER, description="Execution environment")
    
    @validator('memory_limit')
    def validate_memory_limit(cls, v):
        """Validate memory limit format."""
        import re
        if not re.match(r'^\d+[mMgG]$', v):
            raise ValueError("Memory limit must be in format like '512m' or '1g'")
        return v


class ExecutionResponse(BaseModel):
    """Response model for code execution."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    language: ExecutionLanguage
    status: str = Field(..., description="Execution status (success, error, timeout)")
    output: str = Field("", description="Standard output")
    error: str = Field("", description="Standard error")
    exit_code: int = Field(0, description="Process exit code")
    execution_time: float = Field(0.0, description="Execution time in seconds")
    memory_used: Optional[str] = Field(None, description="Memory usage")
    files_created: Optional[List[str]] = Field(None, description="Files created during execution")
    metadata: Dict[str, Any] = Field(default_factory=dict)