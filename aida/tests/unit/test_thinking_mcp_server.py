"""Tests for thinking MCP server."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock
import uuid

import pytest

from aida.tools.base import ToolResult, ToolStatus
from aida.tools.thinking.mcp_server import ThinkingMCPServer
from aida.tools.thinking.models import Perspective, ReasoningType


class TestThinkingMCPServer:
    """Test ThinkingMCPServer class."""

    def _create_tool_result(self, status=ToolStatus.COMPLETED, result=None, error=None):
        """Helper to create a valid ToolResult."""
        return ToolResult(
            tool_name="thinking",
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
        """Create a mock thinking tool."""
        tool = Mock()
        tool.execute = AsyncMock()
        tool.name = "thinking"
        tool.version = "2.0.0"
        tool.description = "Advanced reasoning and analysis tool"
        return tool

    @pytest.fixture
    def server(self, mock_tool):
        """Create a thinking MCP server."""
        return ThinkingMCPServer(mock_tool)

    def test_initialization(self, server):
        """Test server initialization."""
        assert server.tool is not None
        assert "analyze" in server.operations
        assert "brainstorm" in server.operations
        assert "decide" in server.operations

    def test_operation_schemas(self, server):
        """Test operation schemas are properly defined."""
        # Check analyze operation
        analyze_op = server.operations["analyze"]
        assert analyze_op["description"] == "Analyze problems using various reasoning methods"
        assert "problem" in analyze_op["parameters"]
        assert "context" in analyze_op["parameters"]
        assert "reasoning_type" in analyze_op["parameters"]
        assert "depth" in analyze_op["parameters"]
        assert "perspective" in analyze_op["parameters"]
        assert analyze_op["required"] == ["problem"]

        # Check reasoning type enum values
        reasoning_enum = analyze_op["parameters"]["reasoning_type"]["enum"]
        assert set(reasoning_enum) == {t.value for t in ReasoningType}

        # Check perspective enum values
        perspective_enum = analyze_op["parameters"]["perspective"]["enum"]
        assert set(perspective_enum) == {p.value for p in Perspective}

        # Check depth constraints
        depth_param = analyze_op["parameters"]["depth"]
        assert depth_param["minimum"] == 1
        assert depth_param["maximum"] == 5

        # Check brainstorm operation
        brainstorm_op = server.operations["brainstorm"]
        assert brainstorm_op["description"] == "Brainstorm creative solutions"
        assert brainstorm_op["required"] == ["problem"]
        assert "handler" in brainstorm_op

        # Check decide operation
        decide_op = server.operations["decide"]
        assert decide_op["description"] == "Analyze a decision with options"
        assert "options" in decide_op["parameters"]
        assert decide_op["parameters"]["options"]["type"] == "array"
        assert decide_op["required"] == ["decision", "options"]
        assert "handler" in decide_op

    @pytest.mark.asyncio
    async def test_analyze_default_handler(self, server, mock_tool):
        """Test analyze operation uses default handler."""
        mock_result = {
            "analysis": "Systematic analysis result",
            "key_points": ["Point 1", "Point 2"],
            "recommendations": ["Rec 1", "Rec 2"],
        }

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=mock_result
        )

        # The analyze operation doesn't have a custom handler, so it uses default
        handler = server.operations["analyze"].get("handler")
        assert handler is None  # Should use default handler

    @pytest.mark.asyncio
    async def test_brainstorm_handler(self, server, mock_tool):
        """Test brainstorm handler."""
        mock_result = {
            "ideas": ["Idea 1", "Idea 2", "Idea 3"],
            "categories": {"technical": ["Idea 1"], "process": ["Idea 2", "Idea 3"]},
        }

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=mock_result
        )

        arguments = {
            "problem": "How to improve team productivity?",
            "context": "Remote team of 10 developers",
        }

        result = await server._handle_brainstorm(arguments)

        mock_tool.execute.assert_called_once_with(
            problem="How to improve team productivity?",
            context="Remote team of 10 developers",
            reasoning_type="brainstorming",
            depth=4,
        )

        assert result == mock_result

    @pytest.mark.asyncio
    async def test_brainstorm_handler_no_context(self, server, mock_tool):
        """Test brainstorm handler without context."""
        mock_result = {"ideas": ["Idea 1", "Idea 2"]}

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=mock_result
        )

        arguments = {"problem": "Generate new product ideas"}

        await server._handle_brainstorm(arguments)

        mock_tool.execute.assert_called_once_with(
            problem="Generate new product ideas",
            context="",  # Default empty context
            reasoning_type="brainstorming",
            depth=4,
        )

    @pytest.mark.asyncio
    async def test_decide_handler(self, server, mock_tool):
        """Test decide handler."""
        mock_result = {
            "decision_analysis": {
                "best_option": "Option A",
                "analysis": {
                    "Option A": {"pros": ["Pro 1"], "cons": ["Con 1"], "score": 8},
                    "Option B": {"pros": ["Pro 2"], "cons": ["Con 2"], "score": 6},
                },
            }
        }

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=mock_result
        )

        arguments = {
            "decision": "Which framework to use?",
            "options": ["React", "Vue", "Angular"],
            "context": "Building a large enterprise application",
        }

        result = await server._handle_decide(arguments)

        # Check the call - context should include options
        call_args = mock_tool.execute.call_args[1]
        assert call_args["problem"] == "Which framework to use?"
        assert "Options:" in call_args["context"]
        assert "- React" in call_args["context"]
        assert "- Vue" in call_args["context"]
        assert "- Angular" in call_args["context"]
        assert "Building a large enterprise application" in call_args["context"]
        assert call_args["reasoning_type"] == "decision_analysis"
        assert call_args["depth"] == 4

        assert result == mock_result

    @pytest.mark.asyncio
    async def test_decide_handler_no_context(self, server, mock_tool):
        """Test decide handler without additional context."""
        mock_result = {"decision_analysis": "Analysis result"}

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=mock_result
        )

        arguments = {"decision": "Choose a database", "options": ["PostgreSQL", "MongoDB"]}

        await server._handle_decide(arguments)

        # Check context only contains options
        call_args = mock_tool.execute.call_args[1]
        context = call_args["context"]
        assert context.startswith("Options:\n")
        assert "- PostgreSQL" in context
        assert "- MongoDB" in context

    @pytest.mark.asyncio
    async def test_inherited_from_simple_mcp_server(self, server):
        """Test that server inherits from SimpleMCPServer correctly."""
        # Should have methods from SimpleMCPServer
        assert hasattr(server, "call_tool")
        assert hasattr(server, "list_tools")
        assert hasattr(server, "operations")
        assert hasattr(server, "tool")
        assert hasattr(server, "server_info")

    def test_list_tools(self, server):
        """Test listing available MCP tools."""
        tools = server.list_tools()

        # Should have tools for each operation
        tool_names = [tool["name"] for tool in tools]
        assert "thinking_analyze" in tool_names
        assert "thinking_brainstorm" in tool_names
        assert "thinking_decide" in tool_names

        # Check analyze tool definition
        analyze_tool = next(t for t in tools if t["name"] == "thinking_analyze")
        assert analyze_tool["description"] == "Analyze problems using various reasoning methods"
        assert "inputSchema" in analyze_tool
        assert analyze_tool["inputSchema"]["type"] == "object"
        assert "problem" in analyze_tool["inputSchema"]["properties"]
        assert analyze_tool["inputSchema"]["required"] == ["problem"]

        # Check reasoning type enum in schema
        reasoning_prop = analyze_tool["inputSchema"]["properties"]["reasoning_type"]
        assert set(reasoning_prop["enum"]) == {t.value for t in ReasoningType}

    @pytest.mark.asyncio
    async def test_call_tool_brainstorm(self, server, mock_tool):
        """Test calling brainstorm through call_tool method."""
        mock_result = {"ideas": ["Idea 1", "Idea 2"]}

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=mock_result
        )

        # Call through the inherited call_tool method
        result = await server.call_tool(
            name="thinking_brainstorm", arguments={"problem": "Test problem"}
        )

        # Should return MCP formatted response
        assert "content" in result
        assert isinstance(result["content"], list)
        assert result["content"][0]["type"] == "text"
        # The result should be JSON formatted
        assert "ideas" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_call_tool_error_handling(self, server, mock_tool):
        """Test error handling in call_tool."""
        mock_tool.execute.side_effect = Exception("Tool execution failed")

        result = await server.call_tool(name="thinking_brainstorm", arguments={"problem": "Test"})

        assert result.get("isError") is True
        assert "Tool execution failed" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_call_tool_analyze_with_defaults(self, server, mock_tool):
        """Test calling analyze with default handler."""
        mock_result = {"analysis": "Complete analysis"}

        mock_tool.execute.return_value = self._create_tool_result(
            status=ToolStatus.COMPLETED, result=mock_result
        )

        # Call analyze which uses default handler
        result = await server.call_tool(
            name="thinking_analyze",
            arguments={
                "problem": "Analyze this problem",
                "reasoning_type": "systematic_analysis",
                "depth": 3,
            },
        )

        # Should call tool.execute with the arguments
        mock_tool.execute.assert_called_once()
        call_kwargs = mock_tool.execute.call_args[1]
        assert call_kwargs["problem"] == "Analyze this problem"
        assert call_kwargs["reasoning_type"] == "systematic_analysis"
        assert call_kwargs["depth"] == 3

        # Should return formatted result
        assert "content" in result
        assert "analysis" in result["content"][0]["text"]


if __name__ == "__main__":
    pytest.main([__file__])
