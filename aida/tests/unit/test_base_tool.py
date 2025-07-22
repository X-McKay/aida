"""Tests for base_tool module."""

from collections.abc import Callable
from datetime import datetime
from typing import Any
from unittest.mock import Mock, patch
import uuid

from pydantic import BaseModel
import pytest

from aida.tools.base import ToolResult, ToolStatus
from aida.tools.base_tool import (
    BaseMCPServer,
    BaseModularTool,
    BaseObservability,
    BaseToolConfig,
    SimpleToolBase,
)


# Test models
class TestRequest(BaseModel):
    """Test request model."""

    data: str


class TestResponse(BaseModel):
    """Test response model."""

    result: str


class TestConfig(BaseToolConfig):
    """Test configuration."""

    def __init__(self, setting: str = "default"):
        super().__init__()
        self.setting = setting


# Concrete implementations for testing
class TestModularTool(BaseModularTool[TestRequest, TestResponse, TestConfig]):
    """Concrete implementation for testing."""

    def _get_tool_name(self) -> str:
        return "test_tool"

    def _get_tool_version(self) -> str:
        return "1.0.0"

    def _get_tool_description(self) -> str:
        return "Test tool for unit tests"

    def _get_default_config(self) -> TestConfig:
        return TestConfig()

    async def execute(self, **kwargs) -> ToolResult:
        """Execute test operation."""
        execution_id = str(uuid.uuid4())
        started_at = datetime.utcnow()

        try:
            # Simulate some work
            result = {"output": f"Processed: {kwargs.get('input', 'none')}"}

            return ToolResult(
                tool_name=self.name,
                execution_id=execution_id,
                status=ToolStatus.COMPLETED,
                result=result,
                started_at=started_at,
                completed_at=datetime.utcnow(),
                duration_seconds=0.1,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                execution_id=execution_id,
                status=ToolStatus.FAILED,
                error=str(e),
                started_at=started_at,
                completed_at=datetime.utcnow(),
                duration_seconds=0.1,
            )

    def _create_mcp_server(self):
        """Create mock MCP server."""
        return TestMCPServer(self)

    def _create_observability(self, config: dict[str, Any]) -> Any:
        """Create mock observability."""
        return TestObservability(self, config)

    def _create_pydantic_tools(self) -> dict[str, Callable]:
        """Create mock pydantic tools."""
        return {"test_tool": Mock(name="test_tool_func")}


class TestMCPServer(BaseMCPServer):
    """Test MCP server implementation."""

    def list_tools(self) -> list[dict[str, Any]]:
        """List available MCP tools."""
        return [
            {
                "name": self.tool.name,
                "description": self.tool.description,
                "inputSchema": {"type": "object"},
            }
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle MCP tool call."""
        if name == self.tool.name:
            result = await self.tool.execute(**arguments)
            return self._format_mcp_response(
                result.result or result.error, is_error=result.status == ToolStatus.FAILED
            )
        return self._format_mcp_response(f"Unknown tool: {name}", is_error=True)


class TestObservability(BaseObservability):
    """Test observability implementation."""

    def _setup_metrics(self):
        """Setup custom metrics."""
        self.operation_counter = Mock()
        self.duration_histogram = Mock()

    def record_operation(self, operation: str, duration: float, success: bool):
        """Record operation metrics."""
        self.operation_counter.add(1, {"operation": operation, "success": str(success)})
        self.duration_histogram.record(duration, {"operation": operation})


class TestSimpleTool(SimpleToolBase):
    """Test simple tool implementation."""

    def _get_tool_name(self) -> str:
        return "test_simple_tool"

    def _get_tool_version(self) -> str:
        return "1.0.0"

    def _get_tool_description(self) -> str:
        return "Test simple tool"

    def _get_default_config(self) -> BaseToolConfig:
        return BaseToolConfig()

    def _create_processors(self) -> dict[str, Callable]:
        """Create operation processors."""

        async def process_data(**kwargs):
            return {"processed": kwargs.get("data", "none")}

        async def transform_data(**kwargs):
            data = kwargs.get("data", "")
            return {"transformed": data.upper()}

        return {
            "process": process_data,
            "transform": transform_data,
        }

    def _get_default_operation(self) -> str | None:
        """Get default operation."""
        return "process"

    def _create_mcp_server(self):
        return TestMCPServer(self)

    def _create_observability(self, config: dict[str, Any]):
        return TestObservability(self, config)

    def _create_pydantic_tools(self) -> dict[str, Callable]:
        return {"simple_tool": Mock()}


class TestBaseToolConfig:
    """Test BaseToolConfig class."""

    def test_default_config_values(self):
        """Test default configuration values."""
        assert BaseToolConfig.TRACE_ENABLED is True
        assert BaseToolConfig.METRICS_ENABLED is True
        assert BaseToolConfig.SERVICE_NAME_PREFIX == "aida"


class TestBaseModularTool:
    """Test BaseModularTool class."""

    def test_initialization_with_default_config(self):
        """Test initialization with default config."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()

        assert tool.name == "test_tool"
        assert tool.version == "1.0.0"
        assert tool.description == "Test tool for unit tests"
        assert isinstance(tool.config, TestConfig)
        assert tool.config.setting == "default"
        assert tool._mcp_server is None
        assert tool._observability is None
        assert tool._pydantic_tools is None

    def test_initialization_with_custom_config(self):
        """Test initialization with custom config."""
        custom_config = TestConfig(setting="custom")
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool(config=custom_config)

        assert tool.config is custom_config
        assert tool.config.setting == "custom"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execution."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()
        result = await tool.execute(input="test_data")

        assert result.tool_name == "test_tool"
        assert result.status == ToolStatus.COMPLETED
        assert result.result == {"output": "Processed: test_data"}
        assert result.error is None
        assert result.duration_seconds == 0.1

    @pytest.mark.asyncio
    async def test_execute_no_input(self):
        """Test execution without input."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()
        result = await tool.execute()

        assert result.result == {"output": "Processed: none"}

    def test_get_mcp_server_lazy_initialization(self):
        """Test lazy initialization of MCP server."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()

        # Initially None
        assert tool._mcp_server is None

        # Get MCP server
        mcp_server = tool.get_mcp_server()
        assert mcp_server is not None
        assert isinstance(mcp_server, TestMCPServer)
        assert tool._mcp_server is mcp_server

        # Second call returns same instance
        mcp_server2 = tool.get_mcp_server()
        assert mcp_server2 is mcp_server

    def test_to_mcp_tool(self):
        """Test converting to MCP tool."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()
        mcp_tool = tool.to_mcp_tool()

        assert mcp_tool is not None
        assert mcp_tool is tool.get_mcp_server()

    def test_enable_observability_default_config(self):
        """Test enabling observability with default config."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()

        # Initially None
        assert tool._observability is None

        # Enable observability
        obs = tool.enable_observability()
        assert obs is not None
        assert isinstance(obs, TestObservability)
        assert tool._observability is obs

        # Check config was set correctly
        expected_config = {
            "service_name": "aida-test_tool",
            "trace_enabled": True,
            "metrics_enabled": True,
        }
        assert obs.config == expected_config

        # Second call returns same instance
        obs2 = tool.enable_observability()
        assert obs2 is obs

    def test_enable_observability_custom_config(self):
        """Test enabling observability with custom config."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()

        custom_config = {
            "service_name": "custom-service",
            "trace_enabled": False,
            "custom_setting": "value",
        }

        obs = tool.enable_observability(custom_config)

        # Check custom config was preserved and defaults added
        expected_config = {
            "service_name": "custom-service",
            "trace_enabled": False,
            "metrics_enabled": True,  # Default added
            "custom_setting": "value",
        }
        assert obs.config == expected_config

    def test_to_pydantic_tools(self):
        """Test getting PydanticAI-compatible tools."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()

        # Initially None
        assert tool._pydantic_tools is None

        # Get pydantic tools
        pydantic_tools = tool.to_pydantic_tools()
        assert pydantic_tools is not None
        assert "test_tool" in pydantic_tools
        assert tool._pydantic_tools is pydantic_tools

        # Second call returns same instance
        pydantic_tools2 = tool.to_pydantic_tools()
        assert pydantic_tools2 is pydantic_tools


class TestBaseMCPServer:
    """Test BaseMCPServer class."""

    def test_initialization(self):
        """Test MCP server initialization."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()
        server = TestMCPServer(tool)

        assert server.tool is tool
        assert server.server_info["name"] == "aida-test_tool"
        assert server.server_info["version"] == "1.0.0"
        assert server.server_info["description"] == "Test tool for unit tests"

    def test_format_mcp_response_string(self):
        """Test formatting string response."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()
        server = TestMCPServer(tool)

        response = server._format_mcp_response("Test message")
        assert response == {"content": [{"type": "text", "text": "Test message"}]}

    def test_format_mcp_response_list(self):
        """Test formatting list response."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()
        server = TestMCPServer(tool)

        response = server._format_mcp_response(["item1", "item2", "item3"])
        expected_text = "• item1\n• item2\n• item3"
        assert response == {"content": [{"type": "text", "text": expected_text}]}

    def test_format_mcp_response_dict(self):
        """Test formatting dict response."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()
        server = TestMCPServer(tool)

        test_dict = {"key1": "value1", "key2": "value2"}
        response = server._format_mcp_response(test_dict)

        # Check it's formatted as JSON
        assert response["content"][0]["type"] == "text"
        assert "key1" in response["content"][0]["text"]
        assert "value1" in response["content"][0]["text"]

    def test_format_mcp_response_error(self):
        """Test formatting error response."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()
        server = TestMCPServer(tool)

        response = server._format_mcp_response("Error occurred", is_error=True)
        assert response == {
            "content": [{"type": "text", "text": "Error occurred"}],
            "isError": True,
        }

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test successful tool call."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()
        server = TestMCPServer(tool)

        response = await server.call_tool("test_tool", {"input": "test_data"})
        # Parse the JSON to compare content, not formatting
        import json

        response_data = json.loads(response["content"][0]["text"])
        assert response_data == {"output": "Processed: test_data"}

    @pytest.mark.asyncio
    async def test_call_tool_unknown(self):
        """Test calling unknown tool."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()
        server = TestMCPServer(tool)

        response = await server.call_tool("unknown_tool", {})
        assert response["isError"] is True
        assert "Unknown tool: unknown_tool" in response["content"][0]["text"]


class TestBaseObservability:
    """Test BaseObservability class."""

    def test_initialization_enabled(self):
        """Test observability initialization when enabled."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()

        config = {"trace_enabled": True, "metrics_enabled": True}

        # Create observability instance - OpenTelemetry imports happen inside _setup
        obs = TestObservability(tool, config)

        assert obs.tool is tool
        assert obs.config is config
        assert obs.enabled is True

    def test_initialization_disabled(self):
        """Test observability initialization when disabled."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()

        config = {"trace_enabled": False}
        obs = TestObservability(tool, config)

        assert obs.enabled is False
        assert obs.tracer is None
        assert obs.meter is None

    def test_initialization_import_error(self):
        """Test handling import error for OpenTelemetry."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestModularTool()

        config = {"trace_enabled": True}

        # Simulate ImportError during OpenTelemetry import
        with patch("builtins.__import__", side_effect=ImportError("OpenTelemetry not available")):
            obs = TestObservability(tool, config)
            # Should handle gracefully - enabled becomes False
            assert obs.enabled is False
            assert obs.tracer is None
            assert obs.meter is None


class TestSimpleToolBase:
    """Test SimpleToolBase class."""

    def test_initialization(self):
        """Test simple tool initialization."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestSimpleTool()

        assert tool.name == "test_simple_tool"
        assert tool.version == "1.0.0"
        assert tool.description == "Test simple tool"
        assert tool._processors is not None
        assert "process" in tool._processors
        assert "transform" in tool._processors

    @pytest.mark.asyncio
    async def test_execute_with_operation(self):
        """Test executing with specific operation."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestSimpleTool()

        result = await tool.execute(operation="transform", data="hello")

        assert result.status == ToolStatus.COMPLETED
        assert result.result == {"transformed": "HELLO"}

    @pytest.mark.asyncio
    async def test_execute_default_operation(self):
        """Test executing with default operation."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestSimpleTool()

        # No operation specified, should use default
        result = await tool.execute(data="test_data")

        assert result.status == ToolStatus.COMPLETED
        assert result.result == {"processed": "test_data"}

    @pytest.mark.asyncio
    async def test_execute_unknown_operation(self):
        """Test executing with unknown operation."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestSimpleTool()

        result = await tool.execute(operation="unknown", data="test")

        assert result.status == ToolStatus.FAILED
        assert "Unknown operation: unknown" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_error(self):
        """Test execution that raises an error."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestSimpleTool()

        # Mock processor to raise error
        async def failing_processor(**kwargs):
            raise ValueError("Test error")

        tool._processors["fail"] = failing_processor

        result = await tool.execute(operation="fail")

        assert result.status == ToolStatus.FAILED
        assert result.error == "Test error"

    @pytest.mark.asyncio
    async def test_execute_with_observability(self):
        """Test execution with observability enabled."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestSimpleTool()

        # Enable observability
        tool.enable_observability()

        # Mock the record_operation method
        tool._observability.record_operation = Mock()

        # Execute
        await tool.execute(operation="process", data="test")

        # Check metrics were recorded
        tool._observability.record_operation.assert_called_once()
        call_args = tool._observability.record_operation.call_args[1]
        assert call_args["operation"] == "process"
        assert call_args["success"] is True
        assert call_args["duration"] > 0

    def test_get_default_operation(self):
        """Test getting default operation."""
        with patch("aida.tools.base_tool.logger"):
            tool = TestSimpleTool()

        assert tool._get_default_operation() == "process"


if __name__ == "__main__":
    pytest.main([__file__])
