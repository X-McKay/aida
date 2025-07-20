"""System execution tool for AIDA."""

import asyncio
import subprocess
import os
import sys
import shutil
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging
import tempfile
import json

from aida.tools.base import Tool, ToolResult, ToolCapability, ToolParameter


logger = logging.getLogger(__name__)


class SystemTool(Tool):
    """Secure system execution tool with logging and sandboxing."""
    
    def __init__(self):
        super().__init__(
            name="system",
            description="Secure command execution with logging and sandboxing",
            version="1.0.0"
        )
    
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
            dependencies=[]
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
            
            return ToolResult(
                tool_name=self.name,
                execution_id="",
                status="completed",
                result=result,
                started_at=None,
                metadata={
                    "command": command,
                    "exit_code": result.get("exit_code"),
                    "execution_time": result.get("execution_time"),
                    "security_validated": True
                }
            )
            
        except Exception as e:
            raise Exception(f"System command execution failed: {str(e)}")
    
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
        import time
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
            process = await asyncio.create_subprocess_exec(
                *cmd if not shell else ["/bin/bash", "-c", cmd] if os.name != "nt" else ["cmd", "/c", cmd],
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
                "command": cmd if shell else " ".join(cmd),
                "working_directory": cwd,
                "success": process.returncode == 0
            }
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Command not found: {command}")
        except Exception as e:
            raise RuntimeError(f"Command execution failed: {str(e)}")
    
    async def execute_script(self, script_content: str, language: str = "bash", **kwargs) -> ToolResult:
        """Execute a script from content."""
        # Create temporary script file
        script_extensions = {
            "bash": ".sh",
            "python": ".py",
            "powershell": ".ps1",
            "batch": ".bat"
        }
        
        extension = script_extensions.get(language.lower(), ".sh")
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=extension, delete=False) as f:
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
                result = await self.execute_async(
                    command="cmd",
                    args=["/c", script_path],
                    **kwargs
                )
            else:
                result = await self.execute_async(
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
        import psutil
        
        info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "resources": {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": {
                    "total": psutil.disk_usage("/").total if os.name != "nt" else psutil.disk_usage("C:").total,
                    "free": psutil.disk_usage("/").free if os.name != "nt" else psutil.disk_usage("C:").free
                }
            },
            "environment": {
                "path": os.environ.get("PATH", ""),
                "home": os.environ.get("HOME" if os.name != "nt" else "USERPROFILE", ""),
                "user": os.environ.get("USER" if os.name != "nt" else "USERNAME", ""),
                "shell": os.environ.get("SHELL", "")
            }
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
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, "aida_health_check.tmp")
            
            with open(temp_file, "w") as f:
                f.write("health check")
            
            os.unlink(temp_file)
            
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