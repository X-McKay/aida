"""Tests for base tool classes."""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aida.tools.base import (
    ToolCapability,
    ToolParameter,
    ToolRegistry,
    ToolResult,
    ToolStatus,
    get_tool_registry,
    initialize_default_tools,
)


class TestToolParameter:
    """Test ToolParameter class."""

    def test_tool_parameter_creation(self):
        """Test creating a tool parameter."""
        param = ToolParameter(
            name="test_param",
            type="string",
            description="Test parameter",
            required=True,
            default="default_value",
            choices=["option1", "option2"],
            min_value=0,
            max_value=100,
        )

        assert param.name == "test_param"
        assert param.type == "string"
        assert param.description == "Test parameter"
        assert param.required is True
        assert param.default == "default_value"
        assert param.choices == ["option1", "option2"]
        assert param.min_value == 0
        assert param.max_value == 100

    def test_tool_parameter_defaults(self):
        """Test tool parameter default values."""
        param = ToolParameter(
            name="test",
            type="string",
            description="Test",
        )

        assert param.required is True  # Default is True
        assert param.default is None
        assert param.choices is None
        assert param.min_value is None
        assert param.max_value is None

    def test_tool_parameter_to_dict(self):
        """Test converting parameter to dict."""
        param = ToolParameter(
            name="test",
            type="int",
            description="Test param",
            required=True,
            min_value=0,
            max_value=10,
        )

        # Use model_dump() instead of to_dict()
        param_dict = param.model_dump()
        assert param_dict["name"] == "test"
        assert param_dict["type"] == "int"
        assert param_dict["description"] == "Test param"
        assert param_dict["required"] is True
        assert param_dict["min_value"] == 0
        assert param_dict["max_value"] == 10


class TestToolCapability:
    """Test ToolCapability class."""

    def test_tool_capability_creation(self):
        """Test creating a tool capability."""
        params = [
            ToolParameter(name="param1", type="string", description="Param 1"),
            ToolParameter(name="param2", type="int", description="Param 2", required=True),
        ]

        capability = ToolCapability(
            name="test_tool",
            version="1.0.0",
            description="Test tool capability",
            parameters=params,
            required_permissions=["read", "write"],
            supported_platforms=["linux", "darwin"],
            dependencies=["numpy", "pandas"],
        )

        assert capability.name == "test_tool"
        assert capability.version == "1.0.0"
        assert capability.description == "Test tool capability"
        assert len(capability.parameters) == 2
        assert capability.required_permissions == ["read", "write"]
        assert capability.supported_platforms == ["linux", "darwin"]
        assert capability.dependencies == ["numpy", "pandas"]

    def test_tool_capability_defaults(self):
        """Test tool capability default values."""
        capability = ToolCapability(
            name="test",
            description="Test capability",
        )

        assert capability.version == "1.0.0"
        assert capability.parameters == []
        assert capability.required_permissions == []
        assert capability.supported_platforms == []
        assert capability.dependencies == []

    def test_tool_capability_to_dict(self):
        """Test converting capability to dict."""
        params = [
            ToolParameter(name="test", type="string", description="Test"),
        ]

        capability = ToolCapability(
            name="test_tool",
            description="Test",
            parameters=params,
        )

        # Use model_dump() instead of to_dict()
        cap_dict = capability.model_dump()
        assert cap_dict["name"] == "test_tool"
        assert cap_dict["description"] == "Test"
        assert len(cap_dict["parameters"]) == 1
        assert cap_dict["parameters"][0]["name"] == "test"


class TestToolResult:
    """Test ToolResult class."""

    def test_tool_result_success(self):
        """Test creating a successful tool result."""
        started = datetime.utcnow()
        completed = datetime.utcnow()

        result = ToolResult(
            tool_name="test_tool",
            execution_id="exec_123",
            status=ToolStatus.COMPLETED,
            result={"output": "Success"},
            error=None,
            started_at=started,
            completed_at=completed,
            duration_seconds=1.5,
            metadata={"extra": "data"},
        )

        assert result.tool_name == "test_tool"
        assert result.execution_id == "exec_123"
        assert result.status == ToolStatus.COMPLETED
        assert result.result == {"output": "Success"}
        assert result.error is None
        assert result.started_at == started
        assert result.completed_at == completed
        assert result.duration_seconds == 1.5
        assert result.metadata == {"extra": "data"}

    def test_tool_result_failure(self):
        """Test creating a failed tool result."""
        result = ToolResult(
            tool_name="test_tool",
            execution_id="exec_456",
            status=ToolStatus.FAILED,
            error="Something went wrong",
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_seconds=0.5,
        )

        assert result.status == ToolStatus.FAILED
        assert result.error == "Something went wrong"
        assert result.result is None

    def test_tool_result_defaults(self):
        """Test tool result default values."""
        result = ToolResult(
            tool_name="test",
            execution_id="123",
            status=ToolStatus.COMPLETED,
            started_at=datetime.utcnow(),
        )

        assert result.result is None
        assert result.error is None
        assert result.metadata == {}  # Default is empty dict

    def test_tool_result_to_dict(self):
        """Test converting result to dict."""
        result = ToolResult(
            tool_name="test",
            execution_id="123",
            status=ToolStatus.COMPLETED,
            result={"key": "value"},
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_seconds=1.0,
        )

        result_dict = result.model_dump()
        assert result_dict["tool_name"] == "test"
        assert result_dict["execution_id"] == "123"
        assert result_dict["status"] == ToolStatus.COMPLETED
        assert result_dict["result"] == {"key": "value"}


class TestToolStatus:
    """Test ToolStatus enum."""

    def test_tool_status_values(self):
        """Test tool status enum values."""
        assert ToolStatus.PENDING == "pending"
        assert ToolStatus.RUNNING == "running"
        assert ToolStatus.COMPLETED == "completed"
        assert ToolStatus.FAILED == "failed"
        assert ToolStatus.CANCELLED == "cancelled"


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str, capability: ToolCapability):
        self.name = name
        self.version = "1.0.0"
        self.capability = capability
        self.execute_async = AsyncMock()

    def get_capability(self) -> ToolCapability:
        return self.capability


class TestToolRegistry:
    """Test ToolRegistry class."""

    @pytest.fixture
    def registry(self):
        """Create a fresh tool registry."""
        return ToolRegistry()

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        capability = ToolCapability(
            name="mock_tool",
            description="Mock tool for testing",
            parameters=[
                ToolParameter(name="input", type="string", description="Input", required=True)
            ],
        )
        return MockTool("mock_tool", capability)

    @pytest.mark.asyncio
    async def test_register_tool(self, registry, mock_tool):
        """Test registering a tool."""
        await registry.register_tool(mock_tool)

        # Tool should be in registry
        assert "mock_tool" in registry._tools
        assert registry._tools["mock_tool"] == mock_tool

    @pytest.mark.asyncio
    async def test_register_duplicate_tool(self, registry, mock_tool):
        """Test registering duplicate tool name."""
        await registry.register_tool(mock_tool)

        # The current implementation doesn't raise error on duplicate, it overwrites
        # Let's test that behavior
        await registry.register_tool(mock_tool)  # Should not raise
        assert registry._tools["mock_tool"] == mock_tool

    @pytest.mark.asyncio
    async def test_get_tool(self, registry, mock_tool):
        """Test getting a registered tool."""
        await registry.register_tool(mock_tool)

        # Get the tool
        tool = await registry.get_tool("mock_tool")
        assert tool == mock_tool

    @pytest.mark.asyncio
    async def test_get_nonexistent_tool(self, registry):
        """Test getting a tool that doesn't exist."""
        tool = await registry.get_tool("nonexistent")
        assert tool is None

    @pytest.mark.asyncio
    async def test_list_tools(self, registry, mock_tool):
        """Test listing registered tools."""
        # Initially empty
        tools = await registry.list_tools()
        assert tools == []

        # Register a tool
        await registry.register_tool(mock_tool)

        # Should list the tool
        tools = await registry.list_tools()
        assert tools == ["mock_tool"]

        # Register another tool
        capability2 = ToolCapability(name="tool2", description="Tool 2")
        tool2 = MockTool("tool2", capability2)
        await registry.register_tool(tool2)

        # Should list both tools
        tools = await registry.list_tools()
        assert len(tools) == 2
        assert "mock_tool" in tools
        assert "tool2" in tools

    @pytest.mark.asyncio
    async def test_get_capabilities(self, registry, mock_tool):
        """Test getting all tool capabilities."""
        await registry.register_tool(mock_tool)

        # Get all capabilities (returns list)
        capabilities = await registry.get_capabilities()
        assert len(capabilities) == 1
        assert capabilities[0] == mock_tool.capability

        # Get specific capability
        capability = await registry.get_capabilities("mock_tool")
        assert capability == mock_tool.capability

    @pytest.mark.asyncio
    async def test_unregister_tool(self, registry, mock_tool):
        """Test unregistering a tool."""
        await registry.register_tool(mock_tool)

        # Unregister the tool
        success = await registry.unregister_tool("mock_tool")
        assert success is True

        # Tool should be gone
        tool = await registry.get_tool("mock_tool")
        assert tool is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_tool(self, registry):
        """Test unregistering a tool that doesn't exist."""
        success = await registry.unregister_tool("nonexistent")
        assert success is False

    @pytest.mark.asyncio
    async def test_clear_tools(self, registry, mock_tool):
        """Test clearing all tools - but the API doesn't have a clear method."""
        # Register multiple tools
        await registry.register_tool(mock_tool)

        capability2 = ToolCapability(name="tool2", description="Tool 2")
        tool2 = MockTool("tool2", capability2)
        await registry.register_tool(tool2)

        # Check tools are registered
        tools = await registry.list_tools()
        assert len(tools) == 2

        # The API doesn't have a clear method, so let's test unregister instead
        await registry.unregister_tool("mock_tool")
        await registry.unregister_tool("tool2")

        # Registry should be empty
        tools = await registry.list_tools()
        assert tools == []


class TestGlobalRegistry:
    """Test global registry functions."""

    def test_get_tool_registry_singleton(self):
        """Test that get_tool_registry returns singleton."""
        registry1 = get_tool_registry()
        registry2 = get_tool_registry()
        assert registry1 is registry2

    @pytest.mark.asyncio
    async def test_initialize_default_tools(self):
        """Test initializing default tools."""
        # Mock the tool imports
        mock_tools = {
            "thinking": Mock(name="thinking"),
            "execution": Mock(name="execution"),
            "file_operations": Mock(name="file_operations"),
            "llm_response": Mock(name="llm_response"),
            "context": Mock(name="context"),
            "system": Mock(name="system"),
        }

        # Create mock classes
        mock_thinking_tool = Mock(return_value=mock_tools["thinking"])
        mock_execution_tool = Mock(return_value=mock_tools["execution"])
        mock_file_operations_tool = Mock(return_value=mock_tools["file_operations"])
        mock_llm_response_tool = Mock(return_value=mock_tools["llm_response"])
        mock_context_tool = Mock(return_value=mock_tools["context"])
        mock_system_tool = Mock(return_value=mock_tools["system"])

        with patch("aida.tools.base.get_tool_registry") as mock_get_registry:
            mock_registry = AsyncMock()
            mock_get_registry.return_value = mock_registry

            # Mock the imports directly in the function
            with (
                patch("aida.tools.thinking.ThinkingTool", mock_thinking_tool),
                patch("aida.tools.execution.ExecutionTool", mock_execution_tool),
                patch("aida.tools.files.FileOperationsTool", mock_file_operations_tool),
                patch("aida.tools.llm_response.LLMResponseTool", mock_llm_response_tool),
                patch("aida.tools.context.ContextTool", mock_context_tool),
                patch("aida.tools.system.SystemTool", mock_system_tool),
            ):
                # Need to patch the imports inside initialize_default_tools
                import_patches = {
                    "aida.tools.thinking.ThinkingTool": mock_thinking_tool,
                    "aida.tools.execution.ExecutionTool": mock_execution_tool,
                    "aida.tools.files.FileOperationsTool": mock_file_operations_tool,
                    "aida.tools.llm_response.LLMResponseTool": mock_llm_response_tool,
                    "aida.tools.context.ContextTool": mock_context_tool,
                    "aida.tools.system.SystemTool": mock_system_tool,
                }

                with patch.dict("sys.modules", import_patches):
                    await initialize_default_tools()

                    # Should have registered all default tools
                    assert mock_registry.register_tool.call_count == 6

                    # Check that each tool was registered
                    registered_tools = [
                        call[0][0] for call in mock_registry.register_tool.call_args_list
                    ]
                    for tool in mock_tools.values():
                        assert tool in registered_tools


if __name__ == "__main__":
    pytest.main([__file__])
