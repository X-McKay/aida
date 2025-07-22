"""Tests for LLM Response MCP server implementation."""

from datetime import datetime
from unittest.mock import patch
import uuid

import pytest

from aida.tools.base import ToolResult, ToolStatus
from aida.tools.llm_response.llm_response import LLMResponseTool
from aida.tools.llm_response.mcp_server import LLMResponseMCPServer


@pytest.fixture
def llm_response_tool():
    """Create an LLM response tool instance."""
    return LLMResponseTool()


@pytest.fixture
def mcp_server(llm_response_tool):
    """Create an LLM response MCP server instance."""
    return LLMResponseMCPServer(llm_response_tool)


def _create_tool_result(status=ToolStatus.COMPLETED, result=None, error=None):
    """Helper to create a valid ToolResult."""
    return ToolResult(
        tool_name="llm_response",
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


class TestLLMResponseMCPServer:
    """Test LLMResponseMCPServer class."""

    def test_initialization(self, mcp_server, llm_response_tool):
        """Test MCP server initialization."""
        assert mcp_server.tool is llm_response_tool
        assert mcp_server.server_info["name"] == "aida-llm_response"

    @pytest.mark.asyncio
    async def test_answer_operation(self, mcp_server):
        """Test answer operation."""
        mock_result = _create_tool_result(
            result={"response": "Generated response", "confidence": 0.9}
        )

        with patch.object(mcp_server.tool, "execute", return_value=mock_result):
            result = await mcp_server.call_tool(
                "llm_response_answer",
                {"question": "What is AI?", "context": "General context", "max_length": 1000},
            )

        import json

        response_text = result["content"][0]["text"]
        response_data = json.loads(response_text)
        assert response_data["response"] == "Generated response"
        assert response_data["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_explain_operation(self, mcp_server):
        """Test explain operation."""
        mock_result = _create_tool_result(result={"response": "Explanation response"})

        with patch.object(mcp_server.tool, "execute", return_value=mock_result):
            result = await mcp_server.call_tool(
                "llm_response_explain",
                {"concept": "machine learning", "context": "basic introduction"},
            )

        import json

        response_text = result["content"][0]["text"]
        response_data = json.loads(response_text)
        assert response_data["response"] == "Explanation response"

    @pytest.mark.asyncio
    async def test_error_handling(self, mcp_server):
        """Test error handling in operations."""
        mock_result = _create_tool_result(status=ToolStatus.FAILED, error="LLM service unavailable")

        with patch.object(mcp_server.tool, "execute", return_value=mock_result):
            result = await mcp_server.call_tool(
                "llm_response_answer", {"question": "Test question"}
            )

        # Should return error format
        assert result["isError"] is True
        assert "LLM service unavailable" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_unknown_operation(self, mcp_server):
        """Test handling of unknown operations."""
        result = await mcp_server.call_tool("unknown_operation", {})

        # Should return error response, not raise exception
        assert result["isError"] is True
        assert "Unknown tool" in result["content"][0]["text"]
