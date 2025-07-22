"""Tests for system MCP server."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock
import uuid

import pytest

from aida.tools.base import ToolResult, ToolStatus
from aida.tools.system.mcp_server import SystemMCPServer
from aida.tools.system.models import (
    CommandResult,
    ProcessInfo,
    SystemInfo,
)


class TestSystemMCPServer:
    """Test SystemMCPServer class."""

    def _create_tool_result(self, status=ToolStatus.COMPLETED, result=None, error=None):
        """Helper to create a valid ToolResult."""
        return ToolResult(
            tool_name="system",
            execution_id=str(uuid.uuid4()),
            status=status,
            result=result,
            error=error,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow()
            if status in [ToolStatus.COMPLETED, ToolStatus.FAILED]
            else None,
            duration_seconds=0.1 if status in [ToolStatus.COMPLETED, ToolStatus.FAILED] else None,
        )

    @pytest.fixture
    def mock_tool(self):
        """Create a mock system tool."""
        tool = Mock()
        tool.execute = AsyncMock()
        tool.name = "system"
        tool.version = "1.0.0"
        tool.description = "System tool"
        return tool

    @pytest.fixture
    def server(self, mock_tool):
        """Create a system MCP server."""
        return SystemMCPServer(mock_tool)

    def test_initialization(self, server):
        """Test server initialization."""
        assert server.tool is not None
        assert "execute" in server.operations
        assert "system_info" in server.operations
        assert "processes" in server.operations
        assert "which" in server.operations
        assert "env" in server.operations

    def test_operation_schemas(self, server):
        """Test operation schemas are properly defined."""
        # Check execute operation
        execute_op = server.operations["execute"]
        assert execute_op["description"] == "Execute a system command"
        assert "command" in execute_op["parameters"]
        assert "args" in execute_op["parameters"]
        assert "timeout" in execute_op["parameters"]
        assert "cwd" in execute_op["parameters"]
        assert execute_op["required"] == ["command"]

        # Check system_info operation
        info_op = server.operations["system_info"]
        assert info_op["description"] == "Get system information"
        assert info_op["parameters"] == {}
        assert info_op["required"] == []

        # Check processes operation
        proc_op = server.operations["processes"]
        assert proc_op["description"] == "List running processes"
        assert proc_op["parameters"] == {}
        assert proc_op["required"] == []

        # Check which operation
        which_op = server.operations["which"]
        assert which_op["description"] == "Find command in PATH"
        assert "command" in which_op["parameters"]
        assert which_op["required"] == ["command"]

        # Check env operation
        env_op = server.operations["env"]
        assert env_op["description"] == "Get environment variables"
        assert "name" in env_op["parameters"]
        assert env_op["required"] == []

    @pytest.mark.asyncio
    async def test_handle_execute_success(self, server, mock_tool):
        """Test successful command execution."""
        cmd_result = CommandResult(
            command="echo",
            args=["hello"],
            exit_code=0,
            stdout="hello\\n",
            stderr="",
            duration=0.1,
            timed_out=False,
        )

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=cmd_result
        )

        arguments = {"command": "echo", "args": ["hello"], "timeout": 10, "cwd": "/tmp"}

        result = await server._handle_execute(arguments)

        mock_tool.execute.assert_called_once_with(
            operation="execute", command="echo", args=["hello"], timeout=10, cwd="/tmp"
        )

        assert result["exit_code"] == 0
        assert result["stdout"] == "hello\\n"
        assert result["stderr"] == ""
        assert result["timed_out"] is False

    @pytest.mark.asyncio
    async def test_handle_execute_default_args(self, server, mock_tool):
        """Test command execution with default arguments."""
        cmd_result = CommandResult(
            command="ls",
            args=[],
            exit_code=0,
            stdout="file1\\nfile2\\n",
            stderr="",
            duration=0.05,
            timed_out=False,
        )

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=cmd_result
        )

        # Only provide required command
        arguments = {"command": "ls"}

        result = await server._handle_execute(arguments)

        mock_tool.execute.assert_called_once_with(
            operation="execute",
            command="ls",
            args=[],
            timeout=30,  # Default timeout
            cwd=None,
        )

        assert result["exit_code"] == 0
        assert result["stdout"] == "file1\\nfile2\\n"

    @pytest.mark.asyncio
    async def test_handle_execute_failure(self, server, mock_tool):
        """Test failed command execution."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.FAILED, error="Command not found"
        )

        arguments = {"command": "nonexistent"}

        with pytest.raises(Exception, match="Command not found"):
            await server._handle_execute(arguments)

    @pytest.mark.asyncio
    async def test_handle_system_info_success(self, server, mock_tool):
        """Test successful system info retrieval."""
        sys_info = SystemInfo(
            platform="Linux",
            hostname="test-host",
            cpu_count=4,
            memory_total=16384,
            memory_available=8192,
            disk_usage={"/": {"total": 100000, "used": 50000, "free": 50000}},
            python_version="3.11.0",
            env_vars={"PATH": "/usr/bin:/bin"},
        )

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=sys_info
        )

        result = await server._handle_system_info({})

        mock_tool.execute.assert_called_once_with(operation="system_info")

        assert result["platform"] == "Linux"
        assert result["hostname"] == "test-host"
        assert result["cpu_count"] == 4
        assert result["memory_total"] == 16384
        assert result["memory_available"] == 8192
        assert result["python_version"] == "3.11.0"

    @pytest.mark.asyncio
    async def test_handle_system_info_failure(self, server, mock_tool):
        """Test failed system info retrieval."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.FAILED, error="Permission denied"
        )

        with pytest.raises(Exception, match="Permission denied"):
            await server._handle_system_info({})

    @pytest.mark.asyncio
    async def test_handle_processes_success(self, server, mock_tool):
        """Test successful process list."""
        processes = [
            ProcessInfo(
                pid=1234,
                name="python",
                status="running",
                cpu_percent=10.5,
                memory_percent=2.3,
                create_time=datetime.utcnow(),
                username="user1",
            ),
            ProcessInfo(
                pid=5678,
                name="bash",
                status="sleeping",
                cpu_percent=0.0,
                memory_percent=0.5,
                create_time=datetime.utcnow(),
                username="user2",
            ),
        ]

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=processes
        )

        result = await server._handle_processes({})

        mock_tool.execute.assert_called_once_with(operation="process_list")

        assert len(result) == 2
        assert result[0]["pid"] == 1234
        assert result[0]["name"] == "python"
        assert result[0]["status"] == "running"
        assert result[0]["cpu_percent"] == 10.5
        assert result[0]["memory_percent"] == 2.3

        assert result[1]["pid"] == 5678
        assert result[1]["name"] == "bash"

    @pytest.mark.asyncio
    async def test_handle_processes_limit(self, server, mock_tool):
        """Test process list limits to 20 processes."""
        # Create 30 processes
        processes = [
            ProcessInfo(
                pid=i,
                name=f"proc{i}",
                status="running",
                cpu_percent=0.0,
                memory_percent=0.0,
                create_time=datetime.utcnow(),
            )
            for i in range(30)
        ]

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=processes
        )

        result = await server._handle_processes({})

        # Should be limited to 20
        assert len(result) == 20
        assert result[0]["pid"] == 0
        assert result[19]["pid"] == 19

    @pytest.mark.asyncio
    async def test_handle_processes_failure(self, server, mock_tool):
        """Test failed process list."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.FAILED, error="Access denied"
        )

        with pytest.raises(Exception, match="Access denied"):
            await server._handle_processes({})

    @pytest.mark.asyncio
    async def test_handle_which_success(self, server, mock_tool):
        """Test successful which command."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result="/usr/bin/python"
        )

        arguments = {"command": "python"}

        result = await server._handle_which(arguments)

        mock_tool.execute.assert_called_once_with(operation="which", command="python")

        assert result["path"] == "/usr/bin/python"

    @pytest.mark.asyncio
    async def test_handle_which_not_found(self, server, mock_tool):
        """Test which command when not found."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.FAILED, error="Command not found"
        )

        arguments = {"command": "nonexistent"}

        result = await server._handle_which(arguments)

        assert result["path"] is None
        assert result["error"] == "Command not found"

    @pytest.mark.asyncio
    async def test_handle_env_get_all(self, server, mock_tool):
        """Test getting all environment variables."""
        env_vars = {"PATH": "/usr/bin:/bin", "HOME": "/home/user", "USER": "testuser"}

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=env_vars
        )

        # No name provided - get all
        arguments = {}

        result = await server._handle_env(arguments)

        mock_tool.execute.assert_called_once_with(operation="env_get", var_name=None)

        assert result == env_vars

    @pytest.mark.asyncio
    async def test_handle_env_get_specific(self, server, mock_tool):
        """Test getting specific environment variable."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result="/usr/bin:/bin"
        )

        arguments = {"name": "PATH"}

        result = await server._handle_env(arguments)

        mock_tool.execute.assert_called_once_with(operation="env_get", var_name="PATH")

        assert result == "/usr/bin:/bin"

    @pytest.mark.asyncio
    async def test_handle_env_failure(self, server, mock_tool):
        """Test failed environment variable retrieval."""
        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.FAILED, error="Variable not found"
        )

        arguments = {"name": "NONEXISTENT"}

        with pytest.raises(Exception, match="Variable not found"):
            await server._handle_env(arguments)

    @pytest.mark.asyncio
    async def test_inherited_from_simple_mcp_server(self, server):
        """Test that server inherits from SimpleMCPServer correctly."""
        # Should have methods from SimpleMCPServer
        assert hasattr(server, "call_tool")
        assert hasattr(server, "list_tools")
        assert hasattr(server, "operations")
        assert hasattr(server, "tool")
        assert hasattr(server, "server_info")

    @pytest.mark.asyncio
    async def test_call_tool_integration(self, server, mock_tool):
        """Test integration with SimpleMCPServer's call_tool method."""
        # Set up mock for execute operation
        cmd_result = CommandResult(
            command="test",
            args=[],
            exit_code=0,
            stdout="test output",
            stderr="",
            duration=0.1,
            timed_out=False,
        )

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=cmd_result
        )

        # Call through the inherited call_tool method
        result = await server.call_tool(
            name="system_execute",  # Should be prefixed with tool name
            arguments={"command": "test"},
        )

        # Should return MCP formatted response
        assert "content" in result
        assert isinstance(result["content"], list)
        assert result["content"][0]["type"] == "text"
        # The text should contain the JSON result
        assert "exit_code" in result["content"][0]["text"]
        assert "test output" in result["content"][0]["text"]

    def test_list_tools(self, server):
        """Test listing available MCP tools."""
        tools = server.list_tools()

        # Should have tools for each operation
        tool_names = [tool["name"] for tool in tools]
        assert "system_execute" in tool_names
        assert "system_system_info" in tool_names
        assert "system_processes" in tool_names
        assert "system_which" in tool_names
        assert "system_env" in tool_names

        # Check tool definitions
        execute_tool = next(t for t in tools if t["name"] == "system_execute")
        assert execute_tool["description"] == "Execute a system command"
        assert "inputSchema" in execute_tool
        assert execute_tool["inputSchema"]["type"] == "object"
        assert "command" in execute_tool["inputSchema"]["properties"]
        assert execute_tool["inputSchema"]["required"] == ["command"]


if __name__ == "__main__":
    pytest.main([__file__])
