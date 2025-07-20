"""System execution tool with hybrid architecture."""

import asyncio
import subprocess
import os
import sys
import shutil
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
import logging
import tempfile
import json
import time
from datetime import datetime
from contextlib import contextmanager

from aida.tools.base import Tool, ToolResult, ToolCapability, ToolParameter, ToolStatus

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


logger = logging.getLogger(__name__)


class SystemTool(Tool):
    """Secure system execution tool with hybrid architecture.
    
    Supports:
    - PydanticAI tool compatibility
    - MCP server integration
    - OpenTelemetry observability
    """
    
    def __init__(self):
        super().__init__(
            name="system",
            description="Secure command execution with logging and sandboxing",
            version="2.0.0"
        )
        self._pydantic_tools_cache = {}
        self._mcp_server = None
        self._observability = None
    
    def get_capability(self) -> ToolCapability:
        return ToolCapability(
            name=self.name,
            version=self.version,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="command",
                    type="str",
                    description="Command to execute",
                    required=True
                ),
                ToolParameter(
                    name="args",
                    type="list",
                    description="Command arguments",
                    required=False,
                    default=[]
                ),
                ToolParameter(
                    name="working_directory",
                    type="str",
                    description="Working directory for command execution",
                    required=False
                ),
                ToolParameter(
                    name="environment",
                    type="dict",
                    description="Environment variables",
                    required=False,
                    default={}
                ),
                ToolParameter(
                    name="timeout",
                    type="int",
                    description="Command timeout in seconds",
                    required=False,
                    default=30,
                    min_value=1,
                    max_value=300
                ),
                ToolParameter(
                    name="capture_output",
                    type="bool",
                    description="Capture stdout and stderr",
                    required=False,
                    default=True
                ),
                ToolParameter(
                    name="shell",
                    type="bool",
                    description="Execute through shell",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="input_data",
                    type="str",
                    description="Data to send to stdin",
                    required=False
                ),
                ToolParameter(
                    name="allowed_commands",
                    type="list",
                    description="List of allowed commands (security)",
                    required=False
                )
            ],
            required_permissions=["system_execution"],
            supported_platforms=["linux", "darwin", "windows"],
            dependencies=["psutil (optional)"]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute system command with security controls."""
        command = kwargs["command"]
        args = kwargs.get("args", [])
        working_directory = kwargs.get("working_directory")
        environment = kwargs.get("environment", {})
        timeout = kwargs.get("timeout", 30)
        capture_output = kwargs.get("capture_output", True)
        shell = kwargs.get("shell", False)
        input_data = kwargs.get("input_data")
        allowed_commands = kwargs.get("allowed_commands")
        
        execution_id = str(id(kwargs))
        started_at = datetime.now()
        
        try:
            # Security validation
            self._validate_command_security(command, allowed_commands)
            
            # Execute command
            result = await self._execute_command(
                command=command,
                args=args,
                working_directory=working_directory,
                environment=environment,
                timeout=timeout,
                capture_output=capture_output,
                shell=shell,
                input_data=input_data
            )
            
            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()
            
            return ToolResult(
                tool_name=self.name,
                execution_id=execution_id,
                status=ToolStatus.COMPLETED,
                result=result,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                metadata={
                    "command": command,
                    "exit_code": result.get("exit_code"),
                    "execution_time": result.get("execution_time"),
                    "security_validated": True
                }
            )
            
        except Exception as e:
            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()
            
            return ToolResult(
                tool_name=self.name,
                execution_id=execution_id,
                status=ToolStatus.FAILED,
                error=str(e),
                error_code="SYSTEM_EXECUTION_ERROR",
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=duration,
                metadata={"command": command}
            )
    
    def _validate_command_security(self, command: str, allowed_commands: Optional[List[str]]):
        """Validate command for security."""
        # Dangerous commands that should be blocked
        dangerous_commands = [
            "rm", "del", "format", "fdisk", "mkfs",
            "dd", "sudo", "su", "chmod", "chown",
            "passwd", "useradd", "userdel", "usermod",
            "systemctl", "service", "reboot", "shutdown",
            "halt", "poweroff", "init", "kill", "killall"
        ]
        
        # Extract base command
        base_command = command.split()[0] if " " in command else command
        base_command = Path(base_command).name  # Remove path
        
        # Check against dangerous commands
        if base_command.lower() in dangerous_commands:
            raise PermissionError(f"Command '{base_command}' is not allowed for security reasons")
        
        # Check against allowed list if provided
        if allowed_commands is not None:
            if base_command not in allowed_commands:
                raise PermissionError(f"Command '{base_command}' is not in the allowed commands list")
        
        # Additional security checks
        if any(dangerous in command.lower() for dangerous in ["../", "sudo", "su -", "rm -rf"]):
            raise PermissionError("Command contains potentially dangerous patterns")
    
    async def _execute_command(
        self,
        command: str,
        args: List[str],
        working_directory: Optional[str],
        environment: Dict[str, str],
        timeout: int,
        capture_output: bool,
        shell: bool,
        input_data: Optional[str]
    ) -> Dict[str, Any]:
        """Execute the command with specified parameters."""
        start_time = time.time()
        
        # Prepare command
        if shell:
            # Shell execution
            if args:
                full_command = f"{command} {' '.join(args)}"
            else:
                full_command = command
            cmd = full_command
        else:
            # Direct execution
            cmd = [command] + args
        
        # Prepare environment
        env = os.environ.copy()
        env.update(environment)
        
        # Prepare working directory
        cwd = working_directory if working_directory else os.getcwd()
        if not os.path.exists(cwd):
            raise FileNotFoundError(f"Working directory does not exist: {cwd}")
        
        try:
            # Execute command
            if shell:
                # Different shell commands for different platforms
                if os.name != "nt":
                    shell_cmd = ["/bin/bash", "-c", cmd]
                else:
                    shell_cmd = ["cmd", "/c", cmd]
                
                process = await asyncio.create_subprocess_exec(
                    *shell_cmd,
                    stdin=asyncio.subprocess.PIPE if input_data else None,
                    stdout=asyncio.subprocess.PIPE if capture_output else None,
                    stderr=asyncio.subprocess.PIPE if capture_output else None,
                    cwd=cwd,
                    env=env
                )
            else:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE if input_data else None,
                    stdout=asyncio.subprocess.PIPE if capture_output else None,
                    stderr=asyncio.subprocess.PIPE if capture_output else None,
                    cwd=cwd,
                    env=env
                )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=input_data.encode() if input_data else None),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Command timed out after {timeout} seconds")
            
            execution_time = time.time() - start_time
            
            return {
                "exit_code": process.returncode,
                "stdout": stdout.decode("utf-8", errors="replace") if stdout else "",
                "stderr": stderr.decode("utf-8", errors="replace") if stderr else "",
                "execution_time": execution_time,
                "command": cmd if not shell else full_command,
                "working_directory": cwd,
                "success": process.returncode == 0
            }
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Command not found: {command}")
        except Exception as e:
            raise RuntimeError(f"Command execution failed: {str(e)}")
    
    async def execute_script(self, script_content: str, language: str = "bash", **kwargs) -> ToolResult:
        """Execute a script from content."""
        # Create temporary script file in .aida/tmp
        script_extensions = {
            "bash": ".sh",
            "python": ".py",
            "powershell": ".ps1",
            "batch": ".bat"
        }
        
        extension = script_extensions.get(language.lower(), ".sh")
        
        # Use .aida/tmp for temporary files
        temp_dir = Path(".aida/tmp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=extension, delete=False, dir=str(temp_dir)) as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            # Determine interpreter
            interpreters = {
                "bash": "bash",
                "python": sys.executable,
                "powershell": "powershell",
                "batch": "cmd"
            }
            
            interpreter = interpreters.get(language.lower(), "bash")
            
            # Execute script
            if language.lower() == "batch":
                result = await self.execute(
                    command="cmd",
                    args=["/c", script_path],
                    **kwargs
                )
            else:
                result = await self.execute(
                    command=interpreter,
                    args=[script_path],
                    **kwargs
                )
            
            # Add script info to metadata
            result.metadata.update({
                "script_language": language,
                "script_size": len(script_content),
                "script_lines": len(script_content.splitlines())
            })
            
            return result
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(script_path)
            except Exception:
                pass
    
    async def check_command_exists(self, command: str) -> bool:
        """Check if a command exists in the system."""
        try:
            result = await self._execute_command(
                command="which" if os.name != "nt" else "where",
                args=[command],
                working_directory=None,
                environment={},
                timeout=5,
                capture_output=True,
                shell=False,
                input_data=None
            )
            return result["exit_code"] == 0
        except Exception:
            return False
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform
        
        info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            }
        }
        
        if HAS_PSUTIL:
            info["resources"] = {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": {
                    "total": psutil.disk_usage("/").total if os.name != "nt" else psutil.disk_usage("C:").total,
                    "free": psutil.disk_usage("/").free if os.name != "nt" else psutil.disk_usage("C:").free
                }
            }
        
        info["environment"] = {
            "path": os.environ.get("PATH", ""),
            "home": os.environ.get("HOME" if os.name != "nt" else "USERPROFILE", ""),
            "user": os.environ.get("USER" if os.name != "nt" else "USERNAME", ""),
            "shell": os.environ.get("SHELL", "")
        }
        
        return info
    
    async def run_health_check(self) -> Dict[str, Any]:
        """Run system health check."""
        checks = {}
        
        # Check basic commands
        basic_commands = ["echo", "ls" if os.name != "nt" else "dir", "pwd" if os.name != "nt" else "cd"]
        
        for cmd in basic_commands:
            try:
                result = await self._execute_command(
                    command=cmd,
                    args=["test"] if cmd == "echo" else [],
                    working_directory=None,
                    environment={},
                    timeout=5,
                    capture_output=True,
                    shell=False,
                    input_data=None
                )
                checks[f"command_{cmd}"] = {
                    "available": True,
                    "exit_code": result["exit_code"]
                }
            except Exception as e:
                checks[f"command_{cmd}"] = {
                    "available": False,
                    "error": str(e)
                }
        
        # Check file system access
        try:
            temp_dir = Path(".aida/tmp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = temp_dir / "aida_health_check.tmp"
            
            with open(temp_file, "w") as f:
                f.write("health check")
            
            temp_file.unlink()
            
            checks["filesystem_access"] = {"available": True}
        except Exception as e:
            checks["filesystem_access"] = {"available": False, "error": str(e)}
        
        # Overall health
        all_healthy = all(
            check.get("available", False) for check in checks.values()
        )
        
        return {
            "overall_health": "healthy" if all_healthy else "issues_detected",
            "checks": checks,
            "timestamp": time.time()
        }
    
    # ========== HYBRID ARCHITECTURE METHODS ==========
    
    def to_pydantic_tools(self) -> Dict[str, Callable]:
        """Convert to PydanticAI-compatible tool functions.
        
        Returns clean, typed functions for use with PydanticAI agents.
        """
        if self._pydantic_tools_cache:
            return self._pydantic_tools_cache
        
        async def execute_command(
            command: str,
            args: List[str] = None,
            working_dir: str = None,
            timeout: int = 30,
            capture_output: bool = True
        ) -> Dict[str, Any]:
            """Execute a system command with security controls.
            
            Args:
                command: Command to execute
                args: Command arguments
                working_dir: Working directory
                timeout: Execution timeout in seconds
                capture_output: Whether to capture stdout/stderr
                
            Returns:
                Command execution results
            """
            result = await self.execute(
                command=command,
                args=args or [],
                working_directory=working_dir,
                timeout=timeout,
                capture_output=capture_output
            )
            
            if result.status == ToolStatus.COMPLETED:
                return result.result
            else:
                raise RuntimeError(f"Command failed: {result.error}")
        
        async def run_script(
            script_content: str,
            language: str = "bash",
            working_dir: str = None,
            timeout: int = 30
        ) -> Dict[str, Any]:
            """Execute a script from content.
            
            Args:
                script_content: Script content to execute
                language: Script language (bash, python, powershell, batch)
                working_dir: Working directory
                timeout: Execution timeout
                
            Returns:
                Script execution results
            """
            result = await self.execute_script(
                script_content=script_content,
                language=language,
                working_directory=working_dir,
                timeout=timeout
            )
            
            if result.status == ToolStatus.COMPLETED:
                return result.result
            else:
                raise RuntimeError(f"Script failed: {result.error}")
        
        async def check_command(command: str) -> bool:
            """Check if a command exists in the system.
            
            Args:
                command: Command name to check
                
            Returns:
                True if command exists
            """
            return await self.check_command_exists(command)
        
        async def get_info() -> Dict[str, Any]:
            """Get system information.
            
            Returns:
                System information including platform, resources, environment
            """
            return await self.get_system_info()
        
        self._pydantic_tools_cache = {
            "execute_command": execute_command,
            "run_script": run_script,
            "check_command": check_command,
            "get_system_info": get_info
        }
        
        return self._pydantic_tools_cache
    
    def register_with_pydantic_agent(self, agent: Any) -> None:
        """Register all system tools with a PydanticAI agent.
        
        Args:
            agent: PydanticAI agent instance
        """
        tools = self.to_pydantic_tools()
        
        for tool_name, tool_func in tools.items():
            if hasattr(agent, 'tool'):
                # Use decorator style registration
                agent.tool(tool_func)
            else:
                # Fallback to direct registration
                agent.register_tool(tool_name, tool_func)
    
    def get_mcp_server(self) -> "SystemMCPServer":
        """Get MCP server interface for this tool.
        
        Returns Model Context Protocol server for universal AI compatibility.
        """
        if not self._mcp_server:
            self._mcp_server = SystemMCPServer(self)
        return self._mcp_server
    
    def enable_observability(self, config: Dict[str, Any]) -> "SystemObservability":
        """Enable OpenTelemetry observability for system operations.
        
        Args:
            config: Observability configuration
            
        Returns:
            Observability wrapper for the tool
        """
        if not self._observability:
            self._observability = SystemObservability(self, config)
        return self._observability


class SystemMCPServer:
    """MCP server wrapper for SystemTool."""
    
    def __init__(self, tool: SystemTool):
        self.tool = tool
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up MCP tool handlers."""
        self.handlers = {
            "system_execute_command": self._handle_execute_command,
            "system_run_script": self._handle_run_script,
            "system_check_command": self._handle_check_command,
            "system_get_info": self._handle_get_info,
            "system_health_check": self._handle_health_check
        }
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            MCP-formatted response
        """
        if tool_name not in self.handlers:
            return {
                "isError": True,
                "content": [{
                    "type": "text",
                    "text": f"Unknown tool: {tool_name}"
                }]
            }
        
        try:
            result = await self.handlers[tool_name](arguments)
            return {
                "isError": False,
                "content": [{
                    "type": "text", 
                    "text": json.dumps(result, indent=2)
                }]
            }
        except Exception as e:
            return {
                "isError": True,
                "content": [{
                    "type": "text",
                    "text": f"Error: {str(e)}"
                }]
            }
    
    async def _handle_execute_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle command execution via MCP."""
        result = await self.tool.execute(
            command=args["command"],
            args=args.get("args", []),
            working_directory=args.get("working_directory"),
            timeout=args.get("timeout", 30),
            capture_output=args.get("capture_output", True)
        )
        
        if result.status == ToolStatus.COMPLETED:
            return result.result
        else:
            raise RuntimeError(result.error)
    
    async def _handle_run_script(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle script execution via MCP."""
        result = await self.tool.execute_script(
            script_content=args["script_content"],
            language=args.get("language", "bash"),
            working_directory=args.get("working_directory"),
            timeout=args.get("timeout", 30)
        )
        
        if result.status == ToolStatus.COMPLETED:
            return result.result
        else:
            raise RuntimeError(result.error)
    
    async def _handle_check_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle command existence check via MCP."""
        exists = await self.tool.check_command_exists(args["command"])
        return {"exists": exists, "command": args["command"]}
    
    async def _handle_get_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system info request via MCP."""
        return await self.tool.get_system_info()
    
    async def _handle_health_check(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check via MCP."""
        return await self.tool.run_health_check()


class SystemObservability:
    """OpenTelemetry observability for SystemTool."""
    
    def __init__(self, tool: SystemTool, config: Dict[str, Any]):
        self.tool = tool
        self.config = config
        self.enabled = config.get("enabled", False)
        
        if self.enabled:
            self._setup_telemetry()
    
    def _setup_telemetry(self):
        """Set up OpenTelemetry instrumentation."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            
            # Set up tracer
            provider = TracerProvider()
            processor = BatchSpanProcessor(OTLPSpanExporter())
            provider.add_span_processor(processor)
            trace.set_tracer_provider(provider)
            
            self.tracer = trace.get_tracer(
                self.config.get("service_name", "aida-system-tool"),
                self.tool.version
            )
        except ImportError:
            logger.warning("OpenTelemetry not available, observability disabled")
            self.enabled = False
    
    @contextmanager
    def trace_operation(self, operation: str, **attributes):
        """Trace a system operation.
        
        Args:
            operation: Operation name
            **attributes: Span attributes
        """
        if not self.enabled:
            yield
            return
        
        span = self.tracer.start_span(f"system.{operation}")
        
        # Add attributes
        span.set_attribute("tool.name", self.tool.name)
        span.set_attribute("tool.version", self.tool.version)
        span.set_attribute("operation.type", operation)
        
        for key, value in attributes.items():
            span.set_attribute(f"operation.{key}", str(value))
        
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
        finally:
            span.end()