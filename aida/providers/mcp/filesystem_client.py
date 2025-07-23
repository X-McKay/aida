"""MCP Filesystem Client for connecting to the official MCP filesystem server."""

import asyncio
import json
import logging
import os
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
