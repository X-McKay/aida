"""Main execution tool implementation."""

import asyncio
from collections.abc import Callable
from datetime import datetime
import logging
from pathlib import Path
import tempfile
from typing import Any

import dagger

from aida.tools.base import ToolCapability, ToolParameter, ToolResult, ToolStatus
from aida.tools.base_tool import BaseModularTool

from .config import ExecutionConfig
from .models import ExecutionEnvironment, ExecutionLanguage, ExecutionRequest, ExecutionResponse

logger = logging.getLogger(__name__)


class ExecutionTool(BaseModularTool[ExecutionRequest, ExecutionResponse, ExecutionConfig]):
    """Tool for executing code in secure containerized environments."""

    def __init__(self):
        super().__init__()
        self._dagger_client = None

    def _get_tool_name(self) -> str:
        return "execution"

    def _get_tool_version(self) -> str:
        return "2.0.0"

    def _get_tool_description(self) -> str:
        return "Execute code and commands in secure containerized environments"

    def _get_default_config(self):
        return ExecutionConfig

    def get_capability(self) -> ToolCapability:
        """Get tool capability descriptor."""
        return ToolCapability(
            name=self.name,
            version=self.version,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="language",
                    type="str",
                    description="Programming language or runtime",
                    required=True,
                    choices=[lang.value for lang in ExecutionLanguage],
                ),
                ToolParameter(
                    name="code", type="str", description="Code to execute", required=True
                ),
                ToolParameter(
                    name="files",
                    type="dict",
                    description="Additional files needed (filename -> content)",
                    required=False,
                ),
                ToolParameter(
                    name="packages",
                    type="list",
                    description="Package dependencies to install",
                    required=False,
                ),
                ToolParameter(
                    name="env_vars",
                    type="dict",
                    description="Environment variables",
                    required=False,
                ),
                ToolParameter(
                    name="timeout",
                    type="int",
                    description="Execution timeout in seconds",
                    required=False,
                    default=ExecutionConfig.DEFAULT_TIMEOUT,
                    min_value=1,
                    max_value=ExecutionConfig.MAX_TIMEOUT,
                ),
                ToolParameter(
                    name="memory_limit",
                    type="str",
                    description="Memory limit (e.g., 512m, 1g)",
                    required=False,
                    default=ExecutionConfig.DEFAULT_MEMORY_LIMIT,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        """Execute code in a containerized environment."""
        start_time = datetime.utcnow()

        try:
            # Create request model
            request = ExecutionRequest(**kwargs)

            # Execute based on environment
            if request.environment == ExecutionEnvironment.CONTAINER:
                response = await self._execute_in_container(request)
            else:
                # For now, only container execution is supported
                raise ValueError(f"Unsupported environment: {request.environment}")

            # Determine status based on response
            if response.status == "timeout" or response.status == "error":
                status = ToolStatus.FAILED
                error = response.error
            else:
                status = ToolStatus.COMPLETED
                error = None

            # Create result
            return ToolResult(
                tool_name=self.name,
                execution_id=response.request_id,
                status=status,
                result={
                    "output": response.output,
                    "error": response.error,
                    "exit_code": response.exit_code,
                    "execution_time": response.execution_time,
                },
                error=error,
                started_at=start_time,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                metadata={
                    "language": request.language.value,
                    "timeout": request.timeout,
                    "memory_limit": request.memory_limit,
                    "files_created": response.files_created if response.files_created else [],
                },
            )

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ToolResult(
                tool_name=self.name,
                execution_id="",
                status=ToolStatus.FAILED,
                error=str(e),
                started_at=start_time,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )

    async def _execute_in_container(self, request: ExecutionRequest) -> ExecutionResponse:
        """Execute code in a Dagger container."""
        async with dagger.Connection(dagger.Config(log_output=None)) as client:
            # Get base container image
            container = client.container().from_(
                ExecutionConfig.LANGUAGE_IMAGES[request.language.value]
            )

            # Set up working directory
            container = container.with_workdir(ExecutionConfig.CONTAINER_WORK_DIR)

            # Install packages if needed
            if request.packages:
                container = await self._install_packages(container, request)

            # Add environment variables
            if request.env_vars:
                for key, value in request.env_vars.items():
                    if key in ExecutionConfig.ALLOWED_ENV_VARS:
                        container = container.with_env_variable(key, value)

            # Create temporary directory for code and files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write main code file
                code_file = self._write_code_file(temp_dir, request)

                # Write additional files
                if request.files:
                    for filename, content in request.files.items():
                        file_path = Path(temp_dir) / filename
                        file_path.write_text(content)

                # Mount the directory
                container = container.with_mounted_directory(
                    ExecutionConfig.CONTAINER_WORK_DIR, client.host().directory(temp_dir)
                )

                # Prepare execution command
                command = self._prepare_command(request, code_file)

                # Execute with timeout
                try:
                    exec_start = datetime.utcnow()
                    result = await asyncio.wait_for(
                        container.with_exec(command).stdout(), timeout=request.timeout
                    )
                    exec_time = (datetime.utcnow() - exec_start).total_seconds()

                    return ExecutionResponse(
                        language=request.language,
                        status="success",
                        output=result,
                        exit_code=0,
                        execution_time=exec_time,
                    )

                except TimeoutError:
                    return ExecutionResponse(
                        language=request.language,
                        status="timeout",
                        error=f"Execution timed out after {request.timeout} seconds",
                        exit_code=-1,
                        execution_time=request.timeout,
                    )
                except Exception as e:
                    return ExecutionResponse(
                        language=request.language,
                        status="error",
                        error=str(e),
                        exit_code=1,
                        execution_time=0,
                    )

    async def _install_packages(
        self, container: dagger.Container, request: ExecutionRequest
    ) -> dagger.Container:
        """Install packages in the container."""
        pkg_manager = ExecutionConfig.PACKAGE_MANAGERS.get(request.language.value)
        if not pkg_manager:
            return container

        # Install packages one by one
        for package in request.packages:
            command = pkg_manager["command"] + [package]
            container = container.with_exec(command)

        return container

    def _write_code_file(self, temp_dir: str, request: ExecutionRequest) -> str:
        """Write code to a file and return the filename."""
        extension = ExecutionConfig.LANGUAGE_FILE_EXTENSIONS[request.language.value]
        filename = f"main{extension}"
        file_path = Path(temp_dir) / filename
        file_path.write_text(request.code)
        return filename

    def _prepare_command(self, request: ExecutionRequest, code_file: str) -> list[str]:
        """Prepare the execution command."""
        base_command = ExecutionConfig.LANGUAGE_COMMANDS[request.language.value]

        if request.language == ExecutionLanguage.BASH:
            # For bash, pass the code directly
            return base_command + [request.code]
        else:
            # For other languages, execute the file
            return base_command + [code_file]

    def _create_pydantic_tools(self) -> dict[str, Callable]:
        """Create PydanticAI-compatible tool functions."""

        async def execute_code(
            language: str, code: str, packages: list[str] | None = None, timeout: int = 30
        ) -> dict[str, Any]:
            """Execute code in a secure container."""
            result = await self.execute(
                language=language, code=code, packages=packages, timeout=timeout
            )
            return result.result

        async def run_python(code: str, packages: list[str] | None = None) -> str:
            """Run Python code."""
            result = await self.execute(language="python", code=code, packages=packages)
            return result.result.get("output", "")

        async def run_javascript(code: str, packages: list[str] | None = None) -> str:
            """Run JavaScript code."""
            result = await self.execute(language="javascript", code=code, packages=packages)
            return result.result.get("output", "")

        async def run_bash(command: str) -> str:
            """Run a bash command."""
            result = await self.execute(language="bash", code=command)
            return result.result.get("output", "")

        return {
            "execute_code": execute_code,
            "run_python": run_python,
            "run_javascript": run_javascript,
            "run_bash": run_bash,
        }

    def _create_mcp_server(self):
        """Create MCP server instance."""
        from .mcp_server import ExecutionMCPServer

        return ExecutionMCPServer(self)

    def _create_observability(self, config: dict[str, Any]):
        """Create observability instance."""
        from .observability import ExecutionObservability

        return ExecutionObservability(self, config)
