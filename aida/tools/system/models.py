"""Pydantic models for system tool."""

from datetime import datetime
from enum import Enum
import uuid

from pydantic import BaseModel, Field, validator


class SystemOperation(str, Enum):
    """Supported system operations."""

    EXECUTE = "execute"
    SHELL = "shell"
    PROCESS_LIST = "process_list"
    PROCESS_INFO = "process_info"
    PROCESS_KILL = "process_kill"
    SYSTEM_INFO = "system_info"
    ENV_GET = "env_get"
    ENV_SET = "env_set"
    WHICH = "which"
    SCRIPT = "script"


class CommandResult(BaseModel):
    """Result from command execution."""

    command: str
    args: list[str] = Field(default_factory=list)
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False


class ProcessInfo(BaseModel):
    """Process information."""

    pid: int
    name: str
    status: str
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    create_time: datetime
    username: str | None = None
    cmdline: list[str] | None = None
    parent_pid: int | None = None


class SystemInfo(BaseModel):
    """System information."""

    platform: str
    hostname: str
    cpu_count: int
    memory_total: int
    memory_available: int
    disk_usage: dict[str, dict[str, int | float]]
    python_version: str
    env_vars: dict[str, str]


class SystemRequest(BaseModel):
    """Request model for system operations."""

    operation: SystemOperation
    command: str | None = Field(None, description="Command to execute")
    args: list[str] | None = Field(None, description="Command arguments")
    cwd: str | None = Field(None, description="Working directory")
    env: dict[str, str] | None = Field(None, description="Environment variables")
    timeout: int = Field(30, ge=1, le=300, description="Execution timeout in seconds")
    shell: bool = Field(False, description="Execute in shell")
    capture_output: bool = Field(True, description="Capture stdout/stderr")

    # Process operations
    pid: int | None = Field(None, description="Process ID")
    signal: str | None = Field("TERM", description="Signal to send")

    # Environment operations
    var_name: str | None = Field(None, description="Environment variable name")
    var_value: str | None = Field(None, description="Environment variable value")

    # Script operations
    script_content: str | None = Field(None, description="Script content to execute")
    interpreter: str | None = Field(None, description="Script interpreter")

    @validator("command")
    def validate_command(cls, v, values):
        """Validate command for execute operations."""
        op = values.get("operation")
        if op in [SystemOperation.EXECUTE, SystemOperation.SHELL] and not v:
            raise ValueError(f"Command required for {op} operation")
        return v

    @validator("pid")
    def validate_pid(cls, v, values):
        """Validate PID for process operations."""
        op = values.get("operation")
        if op in [SystemOperation.PROCESS_INFO, SystemOperation.PROCESS_KILL] and not v:
            raise ValueError(f"PID required for {op} operation")
        return v


class SystemResponse(BaseModel):
    """Response model for system operations."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    operation: SystemOperation
    success: bool
    result: (
        CommandResult | list[ProcessInfo] | ProcessInfo | SystemInfo | str | dict[str, str] | None
    ) = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    execution_time: float = 0.0
