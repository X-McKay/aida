"""MCP Filesystem Client for connecting to the official MCP filesystem server."""

import asyncio
import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Any

from aida.providers.mcp.base import MCPMessage, MCPProvider

logger = logging.getLogger(__name__)


class MCPFilesystemClient(MCPProvider):
    """Client for connecting to the official MCP filesystem server."""

    def __init__(self, allowed_directories: list[str] | None = None):
        """Initialize MCP filesystem client.

        Args:
            allowed_directories: List of directories the server can access.
                               If None, will use current working directory.
        """
        self.allowed_directories = allowed_directories or [os.getcwd()]
        super().__init__("filesystem", {"directories": self.allowed_directories})

        self._process = None
        self._reader = None
        self._writer = None
        self._read_task = None
        self._pending_responses = {}

    async def connect(self) -> bool:
        """Connect to MCP filesystem server by spawning Node.js process."""
        try:
            # Prepare command to run the MCP filesystem server
            cmd = [
                "npx",
                "-y",
                "@modelcontextprotocol/server-filesystem",
                *self.allowed_directories,
            ]

            logger.info(f"Starting MCP filesystem server with command: {' '.join(cmd)}")

            # Start the server process
            self._process = await asyncio.create_subprocess_exec(  # ty: ignore[missing-argument]
                *cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            self._reader = self._process.stdout
            self._writer = self._process.stdin

            # Start reading messages from server
            self._read_task = asyncio.create_task(self._read_messages())

            # Initialize the session
            await self.initialize_session()

            self._connected = True
            logger.info("Successfully connected to MCP filesystem server")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP filesystem server: {e}")
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Disconnect from MCP server and cleanup process."""
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()

        if self._process:
            self._process.terminate()
            await self._process.wait()

        self._process = None
        self._reader = None
        self._writer = None
        self._connected = False

        await super().disconnect()

    async def _read_messages(self):
        """Read messages from the server process."""
        while self._connected and self._reader:
            try:
                # Read line from server
                line = await self._reader.readline()
                if not line:
                    break

                # Parse JSON-RPC message
                try:
                    message_data = json.loads(line.decode("utf-8"))
                    message = MCPMessage(**message_data)

                    # Handle response
                    if message.id is not None and message.id in self._pending_responses:
                        self._pending_responses[message.id].set_result(message)

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error reading message: {e}")
                break

    async def send_message(self, method: str, params: dict[str, Any] | None = None) -> MCPMessage:
        """Send MCP message and get response."""
        if not self._connected or not self._writer:
            raise RuntimeError("MCP filesystem client not connected")

        self._message_id += 1
        message = MCPMessage(id=self._message_id, method=method, params=params or {})

        # Create future for response
        response_future = asyncio.Future()
        self._pending_responses[message.id] = response_future

        # Send message
        message_json = message.model_dump_json(exclude_none=True) + "\n"
        self._writer.write(message_json.encode("utf-8"))
        await self._writer.drain()

        # Wait for response
        try:
            response = await asyncio.wait_for(response_future, timeout=30.0)
            return response
        except TimeoutError:
            del self._pending_responses[message.id]
            raise TimeoutError(f"Timeout waiting for response to {method}")
        finally:
            if message.id in self._pending_responses:
                del self._pending_responses[message.id]


class MCPFilesystemAdapter:
    """Adapter to translate between AIDA file operations and MCP filesystem server."""

    def __init__(self, client: MCPFilesystemClient):
        """Initialize adapter with MCP client."""
        self.client = client

    async def translate_operation(self, operation: str, **kwargs) -> dict[str, Any]:
        """Translate AIDA file operation to MCP tool call.

        Args:
            operation: AIDA operation name
            **kwargs: Operation parameters

        Returns:
            Result from MCP server
        """
        # Map AIDA operations to MCP tools
        operation_map = {
            "read": "read_file",
            "write": "write_file",
            "append": "write_file",  # Handle append specially
            "delete": "delete_file",
            "copy": None,  # Will handle with read + write
            "move": "move_file",
            "create_dir": "create_directory",
            "list_dir": "list_directory",
            "search": None,  # Will handle with custom logic
            "find": None,  # Will handle with list_directory
            "get_info": "get_file_info",
            "edit": "edit_file",
        }

        mcp_tool = operation_map.get(operation)

        if mcp_tool is None:
            # Handle special cases
            if operation == "copy":
                return await self._handle_copy(**kwargs)
            elif operation == "search":
                return await self._handle_search(**kwargs)
            elif operation == "find":
                return await self._handle_find(**kwargs)
            elif operation == "append":
                return await self._handle_append(**kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")

        # Prepare arguments for MCP tool
        mcp_args = self._prepare_mcp_args(mcp_tool, kwargs)

        # Call MCP tool
        response = await self.client.call_tool(mcp_tool, mcp_args)

        # Transform response back to AIDA format
        return self._transform_response(operation, response)

    def _prepare_mcp_args(self, tool: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Prepare arguments for MCP tool call."""
        # Map AIDA parameter names to MCP parameter names
        param_map = {
            "path": "path",
            "content": "content",
            "destination": "destination",
            "recursive": "recursive",
            "encoding": "encoding",
            "search_text": "find",
            "replace_text": "replace",
        }

        mcp_args = {}
        for aida_param, mcp_param in param_map.items():
            if aida_param in kwargs:
                mcp_args[mcp_param] = kwargs[aida_param]

        return mcp_args

    def _transform_response(self, operation: str, response: dict[str, Any]) -> dict[str, Any]:
        """Transform MCP response to AIDA format."""
        # Most responses can be returned as-is
        # Add any necessary transformations here
        return response

    async def _handle_copy(self, path: str, destination: str, **kwargs) -> dict[str, Any]:
        """Handle copy operation using read + write."""
        # Read source file
        read_response = await self.client.call_tool("read_file", {"path": path})
        content = read_response.get("content", "")

        # Write to destination
        await self.client.call_tool("write_file", {"path": destination, "content": content})

        return {
            "success": True,
            "source": path,
            "destination": destination,
            "bytes_copied": len(content.encode("utf-8")),
        }

    async def _handle_append(self, path: str, content: str, **kwargs) -> dict[str, Any]:
        """Handle append operation by reading existing content first."""
        # Try to read existing content
        try:
            read_response = await self.client.call_tool("read_file", {"path": path})
            existing_content = read_response.get("content", "")
        except Exception:
            existing_content = ""

        # Write combined content
        new_content = existing_content + content
        await self.client.call_tool("write_file", {"path": path, "content": new_content})

        return {"success": True, "path": path, "bytes_appended": len(content.encode("utf-8"))}

    async def _handle_search(self, path: str, search_text: str, **kwargs) -> dict[str, Any]:
        """Handle search operation with custom logic."""
        # List files first
        recursive = kwargs.get("recursive", True)
        list_response = await self.client.call_tool(
            "list_directory", {"path": path, "recursive": recursive}
        )

        results = []
        files = list_response.get("entries", [])

        # Search through files
        for file_info in files:
            if file_info.get("is_file"):
                file_path = file_info.get("path")
                try:
                    # Read file content
                    read_response = await self.client.call_tool("read_file", {"path": file_path})
                    content = read_response.get("content", "")

                    # Search for text
                    if search_text.lower() in content.lower():
                        # Find line numbers
                        lines = content.split("\n")
                        matches = []
                        for i, line in enumerate(lines):
                            if search_text.lower() in line.lower():
                                matches.append(
                                    {"line": i + 1, "text": search_text, "context": line.strip()}
                                )

                        if matches:
                            results.append(
                                {
                                    "file": file_path,
                                    "matches": len(matches),
                                    "lines": matches[:10],  # Limit to 10 matches
                                }
                            )

                except Exception as e:
                    logger.debug(f"Error searching {file_path}: {e}")

        return {"results": results, "files_searched": len(files)}

    async def _handle_find(self, path: str, pattern: str, **kwargs) -> dict[str, Any]:
        """Handle find operation using list_directory."""
        import fnmatch

        recursive = kwargs.get("recursive", True)
        list_response = await self.client.call_tool(
            "list_directory", {"path": path, "recursive": recursive}
        )

        results = []
        entries = list_response.get("entries", [])

        for entry in entries:
            entry_path = entry.get("path", "")
            entry_name = Path(entry_path).name

            if fnmatch.fnmatch(entry_name, pattern):
                results.append(entry_path)

        return {"results": results, "count": len(results)}
