"""MCP Filesystem Server integration for worker agents.

This module provides a wrapper around the @modelcontextprotocol/server-filesystem
MCP server, managing its lifecycle and configuration for use with worker agents.
"""

import asyncio
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from aida.core.mcp_executor import MCPExecutor

logger = logging.getLogger(__name__)


class MockMCPFilesystemClient:
    """Mock MCP client for testing without real MCP server."""

    def __init__(self, allowed_directories: list[str]):
        self.allowed_directories = allowed_directories

    async def execute_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Mock tool execution."""
        if tool_name == "read_file":
            path = Path(args["path"])
            try:
                content = path.read_text()
                return {"content": content, "success": True}
            except Exception as e:
                return {"error": str(e), "success": False}

        elif tool_name == "write_file":
            path = Path(args["path"])
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(args["content"])
                return {"success": True}
            except Exception as e:
                return {"error": str(e), "success": False}

        elif tool_name == "list_directory":
            path = Path(args["path"])
            try:
                files = [str(f) for f in path.iterdir()]
                return {"files": files, "success": True}
            except Exception as e:
                return {"error": str(e), "success": False}

        else:
            return {"error": f"Unknown tool: {tool_name}", "success": False}

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools."""
        return [
            {"name": "read_file", "description": "Read file content"},
            {"name": "write_file", "description": "Write file content"},
            {"name": "list_directory", "description": "List directory contents"},
        ]


class FilesystemMCPServer:
    """Manages the MCP filesystem server for worker agents.

    This class handles:
    - Starting/stopping the filesystem MCP server
    - Configuring allowed directories
    - Providing a client interface for workers
    """

    def __init__(
        self,
        allowed_directories: list[str] | None = None,
        server_name: str = "filesystem",
        port: int = 0,  # 0 means auto-assign
    ):
        """Initialize the filesystem MCP server manager.

        Args:
            allowed_directories: List of directories the server can access
            server_name: Name for the MCP server
            port: Port to run the server on (0 for auto)
        """
        self.server_name = server_name
        self.port = port
        self.allowed_directories = allowed_directories or ["/tmp", ".aida"]

        self._process: subprocess.Popen | None = None
        self._client: MCPExecutor | None = None
        self._server_config: dict[str, Any] = {}

    async def start(self, use_mock: bool = False) -> dict[str, Any]:
        """Start the MCP filesystem server.

        Args:
            use_mock: Force use of mock server (default False - use real server)

        Returns:
            Server configuration including connection details
        """
        if self._process and self._process.poll() is None:
            logger.warning("MCP filesystem server already running")
            return self._server_config

        logger.info(f"Starting MCP filesystem server with access to: {self.allowed_directories}")

        # Check if we should use mock
        if use_mock:
            logger.info("Using mock MCP filesystem server as requested")
            return self._start_mock_server()

        # Check if npm is available
        if not shutil.which("npm"):
            raise RuntimeError(
                "npm is required to run the MCP filesystem server. "
                "Please install Node.js and npm, or set use_mock=True"
            )

        # Start the real MCP server
        return await self._start_real_server()

    async def _start_mock_server(self) -> dict[str, Any]:
        """Start a mock MCP filesystem server."""
        self._server_config = {
            "name": self.server_name,
            "type": "filesystem",
            "pid": -1,  # Mock PID
            "allowed_directories": self.allowed_directories,
            "status": "mock",
        }

        # Create mock client
        self._client = MockMCPFilesystemClient(self.allowed_directories)

        logger.info("Mock MCP filesystem server initialized")
        return self._server_config

    async def _start_real_server(self) -> dict[str, Any]:
        """Start the real MCP filesystem server."""
        # Build command to start the server
        cmd = ["npx", "-y", "@modelcontextprotocol/server-filesystem"]

        # Add allowed directories
        cmd.extend(self.allowed_directories)

        logger.info(f"Starting MCP server with command: {' '.join(cmd)}")

        # Start the server process
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
            )

            # Wait a bit for server to start
            await asyncio.sleep(2)

            # Check if process is still running
            if self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                raise RuntimeError(f"MCP server failed to start: {stderr}")

            # Create MCP client
            from aida.core.mcp_executor import MCPExecutor

            # Initialize the executor
            self._client = MCPExecutor()

            # Add the filesystem provider
            # The MCPExecutor expects the provider name and optional config
            await self._client.add_provider("filesystem", {"command": cmd, "args": [], "env": {}})

            # Store configuration
            self._server_config = {
                "name": self.server_name,
                "type": "filesystem",
                "pid": self._process.pid,
                "allowed_directories": self.allowed_directories,
                "status": "running",
            }

            logger.info(f"MCP filesystem server started with PID {self._process.pid}")
            return self._server_config

        except Exception as e:
            logger.error(f"Failed to start MCP filesystem server: {e}")
            if self._process:
                self._process.terminate()
                self._process = None
            raise

    async def stop(self) -> None:
        """Stop the MCP filesystem server."""
        if not self._process:
            logger.warning("No MCP filesystem server to stop")
            return

        logger.info("Stopping MCP filesystem server")

        # Cleanup client
        if self._client:
            # TODO: Add cleanup method to MCPExecutor
            self._client = None

        # Terminate the process
        self._process.terminate()

        # Wait for it to exit
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("MCP server did not terminate gracefully, killing")
            self._process.kill()
            self._process.wait()

        self._process = None
        self._server_config = {}

        logger.info("MCP filesystem server stopped")

    async def get_client(self):
        """Get MCP client for the filesystem server.

        Returns:
            MCPExecutor or mock client configured for this server
        """
        if not self._client:
            raise RuntimeError("MCP filesystem server not started")

        return self._client

    async def list_tools(self) -> list[dict[str, Any]]:
        """List available tools from the filesystem server.

        Returns:
            List of tool descriptions
        """
        if not self._client:
            raise RuntimeError("MCP filesystem server not started")

        tools = await self._client.list_tools()
        return tools

    async def read_file(self, path: str) -> dict[str, Any]:
        """Read a file using the MCP filesystem server.

        Args:
            path: Path to the file

        Returns:
            File content and metadata
        """
        if not self._client:
            raise RuntimeError("MCP filesystem server not started")

        # Validate path is within allowed directories
        abs_path = Path(path).resolve()
        allowed = False

        for allowed_dir in self.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                abs_path.relative_to(allowed_path)
                allowed = True
                break
            except ValueError:
                continue

        if not allowed:
            raise ValueError(f"Path {path} is not within allowed directories")

        # Execute read_file tool
        result = await self._client.execute_tool("read_file", {"path": str(abs_path)})

        return result

    async def write_file(self, path: str, content: str) -> dict[str, Any]:
        """Write a file using the MCP filesystem server.

        Args:
            path: Path to the file
            content: File content

        Returns:
            Write result
        """
        if not self._client:
            raise RuntimeError("MCP filesystem server not started")

        # Validate path is within allowed directories
        abs_path = Path(path).resolve()
        allowed = False

        for allowed_dir in self.allowed_directories:
            allowed_path = Path(allowed_dir).resolve()
            try:
                abs_path.relative_to(allowed_path)
                allowed = True
                break
            except ValueError:
                continue

        if not allowed:
            raise ValueError(f"Path {path} is not within allowed directories")

        # Execute write_file tool
        result = await self._client.execute_tool(
            "write_file", {"path": str(abs_path), "content": content}
        )

        return result

    async def list_directory(self, path: str) -> dict[str, Any]:
        """List directory contents using the MCP filesystem server.

        Args:
            path: Directory path

        Returns:
            Directory listing
        """
        if not self._client:
            raise RuntimeError("MCP filesystem server not started")

        # Execute list_directory tool
        result = await self._client.execute_tool("list_directory", {"path": path})

        return result

    async def _check_server_installed(self) -> bool:
        """Check if the MCP filesystem server is installed.

        Returns:
            True if installed
        """
        try:
            # Check if package exists
            result = subprocess.run(
                ["npm", "list", "@modelcontextprotocol/server-filesystem", "--depth=0"],
                capture_output=True,
                text=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    async def _install_server(self) -> None:
        """Install the MCP filesystem server package."""
        logger.info("Installing @modelcontextprotocol/server-filesystem")

        try:
            # Install the package
            result = subprocess.run(
                ["npm", "install", "@modelcontextprotocol/server-filesystem"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info("MCP filesystem server installed successfully")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install MCP filesystem server: {e.stderr}")

    def is_running(self) -> bool:
        """Check if the server is running.

        Returns:
            True if running
        """
        return self._process is not None and self._process.poll() is None

    def get_status(self) -> dict[str, Any]:
        """Get server status.

        Returns:
            Status information
        """
        if not self._process:
            return {"status": "stopped"}

        if self._process.poll() is None:
            return {
                "status": "running",
                "pid": self._process.pid,
                "allowed_directories": self.allowed_directories,
            }
        else:
            return {"status": "crashed", "exit_code": self._process.returncode}


# Singleton instance for shared filesystem server
_filesystem_server: FilesystemMCPServer | None = None


async def get_filesystem_server(
    allowed_directories: list[str] | None = None, use_mock: bool = True
) -> FilesystemMCPServer:
    """Get or create the singleton filesystem MCP server.

    Args:
        allowed_directories: Directories to allow access to
        use_mock: Whether to use mock server (default True for backwards compatibility)

    Returns:
        FilesystemMCPServer instance
    """
    global _filesystem_server

    if _filesystem_server is None:
        _filesystem_server = FilesystemMCPServer(allowed_directories)
        await _filesystem_server.start(use_mock=use_mock)

    return _filesystem_server


async def stop_filesystem_server() -> None:
    """Stop the singleton filesystem MCP server."""
    global _filesystem_server

    if _filesystem_server:
        await _filesystem_server.stop()
        _filesystem_server = None


class FilesystemMCPTools:
    """High-level interface to filesystem operations via MCP.

    This provides a cleaner API for workers to use filesystem operations.
    """

    def __init__(self, server: FilesystemMCPServer):
        """Initialize with a filesystem server.

        Args:
            server: FilesystemMCPServer instance
        """
        self.server = server

    async def read(self, path: str) -> str:
        """Read file content.

        Args:
            path: File path

        Returns:
            File content as string
        """
        result = await self.server.read_file(path)
        return result.get("content", "")

    async def write(self, path: str, content: str) -> bool:
        """Write file content.

        Args:
            path: File path
            content: Content to write

        Returns:
            True if successful
        """
        result = await self.server.write_file(path, content)
        return result.get("success", False)

    async def exists(self, path: str) -> bool:
        """Check if file exists.

        Args:
            path: File path

        Returns:
            True if exists
        """
        try:
            await self.server.read_file(path)
            return True
        except Exception:
            return False

    async def list_files(self, directory: str, pattern: str | None = None) -> list[str]:
        """List files in directory.

        Args:
            directory: Directory path
            pattern: Optional glob pattern

        Returns:
            List of file paths
        """
        result = await self.server.list_directory(directory)
        files = result.get("files", [])

        if pattern:
            # Simple pattern matching
            import fnmatch

            files = [f for f in files if fnmatch.fnmatch(f, pattern)]

        return files

    async def create_directory(self, path: str) -> bool:
        """Create a directory.

        Args:
            path: Directory path

        Returns:
            True if successful
        """
        # Create by writing a placeholder file
        placeholder = os.path.join(path, ".aida_placeholder")
        try:
            await self.write(placeholder, "# Directory created by AIDA")
            return True
        except Exception:
            return False

    async def delete(self, path: str) -> bool:
        """Delete a file.

        Args:
            path: File path

        Returns:
            True if successful
        """
        # Note: Basic MCP filesystem may not support delete
        # This would need to be implemented in the MCP server
        logger.warning("File deletion not implemented in basic MCP filesystem")
        return False
