"""Main system tool implementation."""

import asyncio
from asyncio import subprocess
from collections.abc import Callable
from datetime import datetime
import logging
import os
import re
import shutil
import signal
import sys
import tempfile
from typing import Any
import uuid

import psutil

from aida.tools.base import ToolCapability, ToolParameter, ToolResult, ToolStatus
from aida.tools.base_tool import BaseModularTool

from .config import SystemConfig
from .models import (
    CommandResult,
    ProcessInfo,
    SystemInfo,
    SystemOperation,
    SystemRequest,
    SystemResponse,
)

logger = logging.getLogger(__name__)


class SystemTool(BaseModularTool[SystemRequest, SystemResponse, SystemConfig]):
    """Tool for secure system command execution and system operations."""

    def __init__(self):
        """Initialize system tool for secure command execution."""
        super().__init__()

    def _get_tool_name(self) -> str:
        return "system"

    def _get_tool_version(self) -> str:
        return "2.0.0"

    def _get_tool_description(self) -> str:
        return "Secure command execution with logging and sandboxing"

    def _get_default_config(self):
        return SystemConfig

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
                    description="System operation to perform",
                    required=True,
                    choices=[op.value for op in SystemOperation],
                ),
                ToolParameter(
                    name="command", type="str", description="Command to execute", required=False
                ),
                ToolParameter(
                    name="args", type="list", description="Command arguments", required=False
                ),
                ToolParameter(
                    name="cwd", type="str", description="Working directory", required=False
                ),
                ToolParameter(
                    name="env", type="dict", description="Environment variables", required=False
                ),
                ToolParameter(
                    name="timeout",
                    type="int",
                    description="Execution timeout in seconds",
                    required=False,
                    default=SystemConfig.DEFAULT_TIMEOUT,
                    min_value=1,
                    max_value=SystemConfig.MAX_TIMEOUT,
                ),
                ToolParameter(
                    name="shell",
                    type="bool",
                    description="Execute in shell",
                    required=False,
                    default=False,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        """Execute system operation."""
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
            request = SystemRequest(**kwargs)  # ty: ignore[missing-argument]

            # Route to appropriate operation
            response = await self._route_operation(request)

            # Check if operation was successful
            if not response.success:
                return ToolResult(
                    tool_name=self.name,
                    execution_id=response.request_id,
                    status=ToolStatus.FAILED,
                    error=response.error or "Operation failed",
                    started_at=start_time,
                    completed_at=datetime.utcnow(),
                    duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                    metadata={"operation": request.operation.value, "warnings": response.warnings},
                )

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
                    "success": response.success,
                    "warnings": response.warnings,
                },
            )

        except Exception as e:
            logger.error(f"System operation failed: {e}")
            return ToolResult(
                tool_name=self.name,
                execution_id="",
                status=ToolStatus.FAILED,
                error=str(e),
                started_at=start_time,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )

    async def _route_operation(self, request: SystemRequest) -> SystemResponse:
        """Route to specific operation handler."""
        handlers = {
            SystemOperation.EXECUTE: self._execute_command,
            SystemOperation.SHELL: self._execute_shell,
            SystemOperation.PROCESS_LIST: self._list_processes,
            SystemOperation.PROCESS_INFO: self._get_process_info,
            SystemOperation.PROCESS_KILL: self._kill_process,
            SystemOperation.SYSTEM_INFO: self._get_system_info,
            SystemOperation.ENV_GET: self._get_env,
            SystemOperation.ENV_SET: self._set_env,
            SystemOperation.WHICH: self._which_command,
            SystemOperation.SCRIPT: self._execute_script,
        }

        handler = handlers.get(request.operation)
        if not handler:
            raise ValueError(f"Unknown operation: {request.operation}")

        return await handler(request)

    async def _execute_command(self, request: SystemRequest) -> SystemResponse:
        """Execute a command."""
        # Security check
        if request.command and not SystemConfig.is_command_allowed(request.command):
            return SystemResponse(
                operation=request.operation,
                success=False,
                error=f"Command not allowed: {request.command}",
                warnings=["Command blocked by security policy"],
            )

        # Check for dangerous patterns
        full_command = f"{request.command} {' '.join(request.args or [])}"
        for pattern in SystemConfig.DANGEROUS_PATTERNS:
            if re.search(pattern, full_command):
                return SystemResponse(
                    operation=request.operation,
                    success=False,
                    error="Dangerous pattern detected in command",
                    warnings=["Command contains potentially dangerous pattern"],
                )

        # Prepare command
        if not request.command:
            return SystemResponse(
                operation=request.operation,
                success=False,
                error="No command specified",
            )
        cmd = [request.command] + (request.args or [])

        # Prepare environment
        env = os.environ.copy()
        if request.env:
            env.update(request.env)

        # Execute command
        try:
            if not cmd:
                raise ValueError("Empty command")
            process = await asyncio.create_subprocess_exec(  # ty: ignore[missing-argument]
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=request.cwd,
                env=env,
            )

            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=request.timeout
                )

                result = CommandResult(
                    command=request.command or "",
                    args=request.args or [],
                    exit_code=process.returncode,
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                    duration=0.0,  # Will be set by parent
                    timed_out=False,
                )

                # Truncate output if too large
                if len(result.stdout) > SystemConfig.MAX_OUTPUT_SIZE:
                    result.stdout = (
                        result.stdout[: SystemConfig.MAX_OUTPUT_SIZE]
                        + SystemConfig.OUTPUT_TRUNCATE_MSG
                    )
                if len(result.stderr) > SystemConfig.MAX_OUTPUT_SIZE:
                    result.stderr = (
                        result.stderr[: SystemConfig.MAX_OUTPUT_SIZE]
                        + SystemConfig.OUTPUT_TRUNCATE_MSG
                    )

                return SystemResponse(
                    operation=request.operation, success=process.returncode == 0, result=result
                )

            except TimeoutError:
                process.kill()
                await process.wait()

                return SystemResponse(
                    operation=request.operation,
                    success=False,
                    result=CommandResult(
                        command=request.command or "",
                        args=request.args or [],
                        exit_code=-1,
                        stdout="",
                        stderr=f"Command timed out after {request.timeout} seconds",
                        duration=request.timeout,
                        timed_out=True,
                    ),
                )

        except Exception as e:
            return SystemResponse(operation=request.operation, success=False, error=str(e))

    async def _execute_shell(self, request: SystemRequest) -> SystemResponse:
        """Execute a shell command."""
        # For shell commands, we need to be extra careful
        request.shell = True
        return await self._execute_command(request)

    async def _list_processes(self, request: SystemRequest) -> SystemResponse:
        """List running processes."""
        try:
            processes = []
            for proc in psutil.process_iter(
                [
                    "pid",
                    "name",
                    "status",
                    "cpu_percent",
                    "memory_percent",
                    "create_time",
                    "username",
                    "cmdline",
                    "ppid",
                ]
            ):
                try:
                    info = proc.info
                    processes.append(
                        ProcessInfo(
                            pid=info["pid"],
                            name=info["name"],
                            status=info["status"],
                            cpu_percent=info.get("cpu_percent", 0.0),
                            memory_percent=info.get("memory_percent", 0.0),
                            create_time=datetime.fromtimestamp(info["create_time"]),
                            username=info.get("username"),
                            cmdline=info.get("cmdline"),
                            parent_pid=info.get("ppid"),
                        )
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return SystemResponse(operation=request.operation, success=True, result=processes)

        except Exception as e:
            return SystemResponse(operation=request.operation, success=False, error=str(e))

    async def _get_process_info(self, request: SystemRequest) -> SystemResponse:
        """Get information about a specific process."""
        try:
            proc = psutil.Process(request.pid)

            with proc.oneshot():
                info = ProcessInfo(
                    pid=proc.pid,
                    name=proc.name(),
                    status=proc.status(),
                    cpu_percent=proc.cpu_percent(),
                    memory_percent=proc.memory_percent(),
                    create_time=datetime.fromtimestamp(proc.create_time()),
                    username=proc.username(),
                    cmdline=proc.cmdline(),
                    parent_pid=proc.ppid(),
                )

            return SystemResponse(operation=request.operation, success=True, result=info)

        except psutil.NoSuchProcess:
            return SystemResponse(
                operation=request.operation, success=False, error=f"Process {request.pid} not found"
            )
        except Exception as e:
            return SystemResponse(operation=request.operation, success=False, error=str(e))

    async def _kill_process(self, request: SystemRequest) -> SystemResponse:
        """Kill a process."""
        try:
            # Validate signal
            if request.signal not in SystemConfig.ALLOWED_SIGNALS:
                return SystemResponse(
                    operation=request.operation,
                    success=False,
                    error=f"Signal {request.signal} not allowed",
                )

            proc = psutil.Process(request.pid)

            # Don't allow killing critical processes
            if proc.name() in ["init", "systemd", "kernel"]:
                return SystemResponse(
                    operation=request.operation,
                    success=False,
                    error="Cannot kill critical system process",
                )

            # Send signal
            sig = getattr(signal, f"SIG{request.signal}", signal.SIGTERM)
            proc.send_signal(sig)

            return SystemResponse(
                operation=request.operation,
                success=True,
                result=f"Sent {request.signal} to process {request.pid}",
            )

        except psutil.NoSuchProcess:
            return SystemResponse(
                operation=request.operation, success=False, error=f"Process {request.pid} not found"
            )
        except Exception as e:
            return SystemResponse(operation=request.operation, success=False, error=str(e))

    async def _get_system_info(self, request: SystemRequest) -> SystemResponse:
        """Get system information."""
        try:
            # Get disk usage for common mount points
            disk_usage = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = {
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "percent": usage.percent,
                    }
                except Exception:
                    continue

            # Get filtered environment variables
            env_vars = SystemConfig.filter_env_vars(dict(os.environ))

            info = SystemInfo(
                platform=sys.platform,
                hostname=os.uname().nodename,
                cpu_count=psutil.cpu_count(),
                memory_total=psutil.virtual_memory().total,
                memory_available=psutil.virtual_memory().available,
                disk_usage=disk_usage,
                python_version=sys.version,
                env_vars={k: v for k, v in env_vars.items() if k in SystemConfig.SAFE_ENV_VARS},
            )

            return SystemResponse(operation=request.operation, success=True, result=info)

        except Exception as e:
            return SystemResponse(operation=request.operation, success=False, error=str(e))

    async def _get_env(self, request: SystemRequest) -> SystemResponse:
        """Get environment variables."""
        if request.var_name:
            # Get specific variable
            value = os.environ.get(request.var_name)
            if value is None:
                return SystemResponse(
                    operation=request.operation,
                    success=False,
                    error=f"Environment variable {request.var_name} not found",
                )

            # Filter sensitive values
            if request.var_name.upper() in SystemConfig.FILTERED_ENV_VARS:
                value = "***FILTERED***"

            return SystemResponse(operation=request.operation, success=True, result=value)
        else:
            # Get all variables (filtered)
            env_vars = SystemConfig.filter_env_vars(dict(os.environ))
            return SystemResponse(operation=request.operation, success=True, result=env_vars)

    async def _set_env(self, request: SystemRequest) -> SystemResponse:
        """Set environment variable (for current process only)."""
        if not request.var_name:
            return SystemResponse(
                operation=request.operation, success=False, error="Variable name required"
            )

        # Don't allow setting sensitive variables
        if request.var_name.upper() in SystemConfig.FILTERED_ENV_VARS:
            return SystemResponse(
                operation=request.operation,
                success=False,
                error="Cannot set sensitive environment variable",
            )

        os.environ[request.var_name] = request.var_value or ""

        return SystemResponse(
            operation=request.operation,
            success=True,
            result=f"Set {request.var_name}={request.var_value}",
        )

    async def _which_command(self, request: SystemRequest) -> SystemResponse:
        """Find command in PATH."""
        if not request.command:
            return SystemResponse(
                operation=request.operation, success=False, error="Command required"
            )

        path = shutil.which(request.command)

        if path:
            return SystemResponse(operation=request.operation, success=True, result=path)
        else:
            return SystemResponse(
                operation=request.operation,
                success=False,
                error=f"Command {request.command} not found in PATH",
            )

    async def _execute_script(self, request: SystemRequest) -> SystemResponse:
        """Execute a script."""
        if not request.script_content:
            return SystemResponse(
                operation=request.operation, success=False, error="Script content required"
            )

        # Determine interpreter
        interpreter = request.interpreter or "bash"

        # Validate interpreter
        allowed_interpreters = []
        for interps in SystemConfig.ALLOWED_INTERPRETERS.values():
            allowed_interpreters.extend(interps)

        if interpreter not in allowed_interpreters:
            return SystemResponse(
                operation=request.operation,
                success=False,
                error=f"Interpreter {interpreter} not allowed",
            )

        # Create temporary script file
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".script", delete=False) as f:
                f.write(request.script_content)
                script_path = f.name

            # Execute script
            request.command = interpreter
            request.args = [script_path]
            result = await self._execute_command(request)

            # Clean up
            os.unlink(script_path)

            return result

        except Exception as e:
            return SystemResponse(operation=request.operation, success=False, error=str(e))

    def _create_pydantic_tools(self) -> dict[str, Callable]:
        """Create PydanticAI-compatible tool functions."""

        async def run_command(
            command: str, args: list[str] | None = None, timeout: int = 30
        ) -> dict[str, Any]:
            """Run a system command."""
            result = await self.execute(
                operation="execute", command=command, args=args, timeout=timeout
            )
            if result.status == ToolStatus.COMPLETED:
                cmd_result = result.result
                return {
                    "exit_code": cmd_result.exit_code,
                    "stdout": cmd_result.stdout,
                    "stderr": cmd_result.stderr,
                }
            else:
                raise Exception(result.error)

        async def get_system_info() -> dict[str, Any]:
            """Get system information."""
            result = await self.execute(operation="system_info")
            if result.status == ToolStatus.COMPLETED and result.result:
                return result.result.dict()
            return {}

        async def list_processes() -> list[dict[str, Any]]:
            """List running processes."""
            result = await self.execute(operation="process_list")
            if result.status == ToolStatus.COMPLETED and result.result:
                return [proc.dict() for proc in result.result]
            return []

        async def which_command(command: str) -> str | None:
            """Find command in PATH."""
            result = await self.execute(operation="which", command=command)
            return (
                result.result if result.status == ToolStatus.COMPLETED and result.result else None
            )

        return {
            "run_command": run_command,
            "get_system_info": get_system_info,
            "list_processes": list_processes,
            "which_command": which_command,
        }

    def _create_mcp_server(self):
        """Create MCP server instance."""
        from .mcp_server import SystemMCPServer

        return SystemMCPServer(self)

    def _create_observability(self, config: dict[str, Any]):
        """Create observability instance."""
        from .observability import SystemObservability

        return SystemObservability(self, config)
