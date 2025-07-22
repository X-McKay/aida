"""Tests for Execution MCP server implementation."""

from datetime import datetime
from unittest.mock import patch
import uuid

import pytest

from aida.tools.base import ToolResult, ToolStatus
from aida.tools.execution.execution import ExecutionTool
from aida.tools.execution.mcp_server import ExecutionMCPServer


@pytest.fixture
def execution_tool():
    """Create an execution tool instance."""
    return ExecutionTool()


@pytest.fixture
def mcp_server(execution_tool):
    """Create an execution MCP server instance."""
    return ExecutionMCPServer(execution_tool)


def _create_tool_result(status=ToolStatus.COMPLETED, result=None, error=None):
    """Helper to create a valid ToolResult."""
    return ToolResult(
        tool_name="execution",
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


class TestExecutionMCPServer:
    """Test ExecutionMCPServer class."""

    def test_initialization(self, mcp_server, execution_tool):
        """Test MCP server initialization."""
        assert mcp_server.tool is execution_tool
        assert mcp_server.server_info["name"] == "aida-execution"

    @pytest.mark.asyncio
    async def test_execute_operation(self, mcp_server):
        """Test execute operation."""
        mock_result = _create_tool_result(result={"output": "Hello, World!", "exit_code": 0})

        with patch.object(mcp_server.tool, "execute", return_value=mock_result):
            result = await mcp_server.call_tool(
                "execution_execute", {"language": "python", "code": 'print("Hello, World!")'}
            )

        import json

        response_text = result["content"][0]["text"]
        response_data = json.loads(response_text)
        assert response_data["output"] == "Hello, World!"
        assert response_data["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_execute_with_packages(self, mcp_server):
        """Test execute operation with packages."""
        mock_result = _create_tool_result(result={"output": "Package installed", "exit_code": 0})

        with patch.object(mcp_server.tool, "execute", return_value=mock_result):
            result = await mcp_server.call_tool(
                "execution_execute",
                {"language": "python", "code": "import requests", "packages": ["requests"]},
            )

        import json

        response_text = result["content"][0]["text"]
        response_data = json.loads(response_text)
        assert response_data["output"] == "Package installed"
        assert response_data["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_execute_with_files(self, mcp_server):
        """Test execute operation with files."""
        mock_result = _create_tool_result(
            result={"output": "File content: test data", "exit_code": 0}
        )

        with patch.object(mcp_server.tool, "execute", return_value=mock_result):
            result = await mcp_server.call_tool(
                "execution_execute",
                {
                    "language": "python",
                    "code": 'with open("data.txt") as f: print(f"File content: {f.read()}")',
                    "files": {"data.txt": "test data"},
                },
            )

        import json

        response_text = result["content"][0]["text"]
        response_data = json.loads(response_text)
        assert "File content: test data" in response_data["output"]

    @pytest.mark.asyncio
    async def test_execution_error(self, mcp_server):
        """Test execution error handling."""
        mock_result = _create_tool_result(
            result={"output": "", "error": "SyntaxError: invalid syntax", "exit_code": 1}
        )

        with patch.object(mcp_server.tool, "execute", return_value=mock_result):
            result = await mcp_server.call_tool(
                "execution_execute", {"language": "python", "code": "invalid syntax"}
            )

        import json

        response_text = result["content"][0]["text"]
        response_data = json.loads(response_text)
        assert response_data["exit_code"] == 1
        assert "SyntaxError" in response_data["error"]

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mcp_server):
        """Test timeout handling."""
        mock_result = _create_tool_result(
            status=ToolStatus.FAILED, error="Execution timed out after 5 seconds"
        )

        with patch.object(mcp_server.tool, "execute", return_value=mock_result):
            result = await mcp_server.call_tool(
                "execution_execute",
                {"language": "python", "code": "import time; time.sleep(10)", "timeout": 5},
            )

        # Should return error format
        assert result["isError"] is True
        assert "timed out" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_unknown_operation(self, mcp_server):
        """Test handling of unknown operations."""
        result = await mcp_server.call_tool("unknown_operation", {})

        # Should return error response, not raise exception
        assert result["isError"] is True
        assert "Unknown tool" in result["content"][0]["text"]
