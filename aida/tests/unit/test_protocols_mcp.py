"""Tests for MCP protocol implementation."""

from unittest.mock import AsyncMock

import pytest

from aida.core.protocols.mcp import (
    MCPCapability,
    MCPMessage,
    MCPMessageType,
    MCPProtocol,
    MCPResource,
    MCPResourceType,
    MCPTool,
)


class TestMCPMessageType:
    """Test MCPMessageType enum."""

    def test_message_type_values(self):
        """Test MCPMessageType enum values."""
        assert MCPMessageType.INITIALIZE == "initialize"
        assert MCPMessageType.INITIALIZED == "initialized"
        assert MCPMessageType.NOTIFICATION == "notification"
        assert MCPMessageType.REQUEST == "request"
        assert MCPMessageType.RESPONSE == "response"
        assert MCPMessageType.ERROR == "error"
        assert MCPMessageType.PROGRESS == "progress"


class TestMCPResourceType:
    """Test MCPResourceType enum."""

    def test_resource_type_values(self):
        """Test MCPResourceType enum values."""
        assert MCPResourceType.TEXT == "text"
        assert MCPResourceType.JSON == "json"
        assert MCPResourceType.BINARY == "binary"
        assert MCPResourceType.STREAM == "stream"


class TestMCPCapability:
    """Test MCPCapability model."""

    def test_capability_creation(self):
        """Test MCPCapability creation with all fields."""
        capability = MCPCapability(
            name="context_management",
            version="2.0.0",
            description="Context management capability",
            parameters={"max_size": 1000},
        )

        assert capability.name == "context_management"
        assert capability.version == "2.0.0"
        assert capability.description == "Context management capability"
        assert capability.parameters == {"max_size": 1000}

    def test_capability_defaults(self):
        """Test MCPCapability default values."""
        capability = MCPCapability(name="basic_capability")

        assert capability.name == "basic_capability"
        assert capability.version == "1.0.0"
        assert capability.description is None
        assert capability.parameters == {}

    def test_capability_serialization(self):
        """Test MCPCapability serialization."""
        capability = MCPCapability(
            name="test_capability",
            version="1.5.0",
            description="Test capability",
            parameters={"param1": "value1"},
        )

        data = capability.dict()
        assert data["name"] == "test_capability"
        assert data["version"] == "1.5.0"
        assert data["description"] == "Test capability"
        assert data["parameters"] == {"param1": "value1"}


class TestMCPResource:
    """Test MCPResource model."""

    def test_resource_creation(self):
        """Test MCPResource creation with all fields."""
        resource = MCPResource(
            uri="file:///path/to/resource.txt",
            type=MCPResourceType.TEXT,
            metadata={"size": 1024, "encoding": "utf-8"},
            content="Hello, world!",
        )

        assert resource.uri == "file:///path/to/resource.txt"
        assert resource.type == MCPResourceType.TEXT
        assert resource.metadata == {"size": 1024, "encoding": "utf-8"}
        assert resource.content == "Hello, world!"

    def test_resource_defaults(self):
        """Test MCPResource default values."""
        resource = MCPResource(uri="http://example.com/resource", type=MCPResourceType.JSON)

        assert resource.uri == "http://example.com/resource"
        assert resource.type == MCPResourceType.JSON
        assert resource.metadata == {}
        assert resource.content is None

    def test_resource_with_binary_content(self):
        """Test MCPResource with binary content."""
        binary_data = b"\x89PNG\r\n\x1a\n"
        resource = MCPResource(
            uri="data:image/png", type=MCPResourceType.BINARY, content=binary_data
        )

        assert resource.content == binary_data
        assert resource.type == MCPResourceType.BINARY

    def test_resource_with_json_content(self):
        """Test MCPResource with JSON content."""
        json_data = {"key": "value", "number": 42}
        resource = MCPResource(
            uri="data:application/json", type=MCPResourceType.JSON, content=json_data
        )

        assert resource.content == json_data
        assert resource.type == MCPResourceType.JSON


class TestMCPTool:
    """Test MCPTool model."""

    def test_tool_creation(self):
        """Test MCPTool creation with all fields."""
        tool = MCPTool(
            name="file_processor",
            description="Process files with various operations",
            parameters={
                "type": "object",
                "properties": {"operation": {"type": "string"}, "file_path": {"type": "string"}},
            },
            required_capabilities=["file_system", "text_processing"],
        )

        assert tool.name == "file_processor"
        assert tool.description == "Process files with various operations"
        assert tool.parameters["type"] == "object"
        assert tool.required_capabilities == ["file_system", "text_processing"]

    def test_tool_defaults(self):
        """Test MCPTool default values."""
        tool = MCPTool(name="simple_tool", description="A simple tool")

        assert tool.name == "simple_tool"
        assert tool.description == "A simple tool"
        assert tool.parameters == {}
        assert tool.required_capabilities == []


class TestMCPMessage:
    """Test MCPMessage class."""

    def test_mcp_message_creation(self):
        """Test MCPMessage creation with all fields."""
        message = MCPMessage(
            sender_id="agent1",
            message_type=MCPMessageType.REQUEST,
            method="context/get",
            params={"key": "test_key"},
            result=None,
            error=None,
            jsonrpc="2.0",
        )

        assert message.sender_id == "agent1"
        assert message.message_type == MCPMessageType.REQUEST
        assert message.method == "context/get"
        assert message.params == {"key": "test_key"}
        assert message.result is None
        assert message.error is None
        assert message.jsonrpc == "2.0"

    def test_mcp_message_defaults(self):
        """Test MCPMessage default values."""
        message = MCPMessage(sender_id="agent1", message_type=MCPMessageType.REQUEST)

        assert message.method is None
        assert message.params is None
        assert message.result is None
        assert message.error is None
        assert message.jsonrpc == "2.0"

    def test_mcp_message_methods_constants(self):
        """Test MCPMessage.Methods constants."""
        # Context management
        assert MCPMessage.Methods.CONTEXT_GET == "context/get"
        assert MCPMessage.Methods.CONTEXT_SET == "context/set"
        assert MCPMessage.Methods.CONTEXT_UPDATE == "context/update"
        assert MCPMessage.Methods.CONTEXT_DELETE == "context/delete"
        assert MCPMessage.Methods.CONTEXT_LIST == "context/list"

        # Resource management
        assert MCPMessage.Methods.RESOURCE_GET == "resource/get"
        assert MCPMessage.Methods.RESOURCE_SET == "resource/set"
        assert MCPMessage.Methods.RESOURCE_LIST == "resource/list"
        assert MCPMessage.Methods.RESOURCE_WATCH == "resource/watch"

        # Tool management
        assert MCPMessage.Methods.TOOL_CALL == "tool/call"
        assert MCPMessage.Methods.TOOL_LIST == "tool/list"
        assert MCPMessage.Methods.TOOL_GET == "tool/get"

        # Capability management
        assert MCPMessage.Methods.CAPABILITY_GET == "capability/get"
        assert MCPMessage.Methods.CAPABILITY_LIST == "capability/list"

        # Session management
        assert MCPMessage.Methods.SESSION_START == "session/start"
        assert MCPMessage.Methods.SESSION_END == "session/end"
        assert MCPMessage.Methods.SESSION_STATUS == "session/status"


class TestMCPProtocol:
    """Test MCPProtocol class."""

    @pytest.fixture
    def capabilities(self):
        """Create sample capabilities."""
        return [
            MCPCapability(name="context", version="1.0.0"),
            MCPCapability(name="resources", version="1.0.0"),
        ]

    @pytest.fixture
    def protocol(self, capabilities):
        """Create MCP protocol instance."""
        return MCPProtocol(agent_id="test_agent", transport="stdio", capabilities=capabilities)

    def test_protocol_initialization(self, protocol, capabilities):
        """Test MCPProtocol initialization."""
        assert protocol.agent_id == "test_agent"
        assert protocol.transport == "stdio"
        assert protocol.capabilities == capabilities

        # Check internal state
        assert protocol._initialized is False
        assert protocol._session_id is None
        assert protocol._context_store == {}
        assert protocol._resources == {}
        assert protocol._tools == {}
        assert protocol._message_id_counter == 0
        assert protocol._pending_requests == {}
        assert protocol._reader is None
        assert protocol._writer is None
        assert protocol._tasks == set()

    def test_protocol_initialization_defaults(self):
        """Test MCPProtocol initialization with defaults."""
        protocol = MCPProtocol("test_agent")

        assert protocol.agent_id == "test_agent"
        assert protocol.transport == "stdio"
        assert protocol.capabilities == []

    def test_register_capability(self, protocol):
        """Test registering a capability."""
        new_capability = MCPCapability(
            name="tool_execution", version="2.0.0", description="Execute tools"
        )

        initial_count = len(protocol.capabilities)
        protocol.register_capability(new_capability)

        assert len(protocol.capabilities) == initial_count + 1
        assert new_capability in protocol.capabilities

    def test_register_resource(self, protocol):
        """Test registering a resource."""
        resource = MCPResource(
            uri="file:///test.txt", type=MCPResourceType.TEXT, content="Test content"
        )

        protocol.register_resource(resource)

        assert protocol._resources[resource.uri] == resource

    def test_register_tool(self, protocol):
        """Test registering a tool."""
        tool = MCPTool(name="test_tool", description="A test tool", parameters={"param1": "string"})

        protocol.register_tool(tool)

        assert protocol._tools[tool.name] == tool

    def test_register_multiple_resources(self, protocol):
        """Test registering multiple resources."""
        resource1 = MCPResource(uri="file:///test1.txt", type=MCPResourceType.TEXT)
        resource2 = MCPResource(uri="file:///test2.json", type=MCPResourceType.JSON)

        protocol.register_resource(resource1)
        protocol.register_resource(resource2)

        assert len(protocol._resources) == 2
        assert protocol._resources["file:///test1.txt"] == resource1
        assert protocol._resources["file:///test2.json"] == resource2

    def test_register_multiple_tools(self, protocol):
        """Test registering multiple tools."""
        tool1 = MCPTool(name="tool1", description="First tool")
        tool2 = MCPTool(name="tool2", description="Second tool")

        protocol.register_tool(tool1)
        protocol.register_tool(tool2)

        assert len(protocol._tools) == 2
        assert protocol._tools["tool1"] == tool1
        assert protocol._tools["tool2"] == tool2

    @pytest.mark.asyncio
    async def test_send_mcp_message(self, protocol):
        """Test sending MCP message."""
        # Mock the writer to avoid actual I/O
        mock_writer = AsyncMock()
        protocol._writer = mock_writer

        message = MCPMessage(
            sender_id="test_agent",
            message_type=MCPMessageType.REQUEST,
            method="context/get",
            params={"key": "test"},
        )

        result = await protocol.send(message)

        assert result is True
        mock_writer.write.assert_called_once()
        mock_writer.drain.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_no_writer(self, protocol):
        """Test sending message without writer."""
        message = MCPMessage(sender_id="test_agent", message_type=MCPMessageType.REQUEST)

        result = await protocol.send(message)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_protocol_message_conversion(self, protocol):
        """Test sending generic ProtocolMessage gets converted to MCPMessage."""
        from aida.core.protocols.base import ProtocolMessage

        mock_writer = AsyncMock()
        protocol._writer = mock_writer

        generic_message = ProtocolMessage(sender_id="test_agent", message_type="test")

        result = await protocol.send(generic_message)

        assert result is True
        # Should have called write (message was converted and sent)
        mock_writer.write.assert_called_once()

        # Check the serialized message contains MCP-specific fields
        call_args = mock_writer.write.call_args[0][0].decode()
        assert '"jsonrpc":"2.0"' in call_args

    @pytest.mark.asyncio
    async def test_receive_no_reader(self, protocol):
        """Test receiving message without reader."""
        result = await protocol.receive()
        assert result is None

    @pytest.mark.asyncio
    async def test_receive_empty_line(self, protocol):
        """Test receiving empty line."""
        mock_reader = AsyncMock()
        mock_reader.readline.return_value = b""
        protocol._reader = mock_reader

        result = await protocol.receive()
        assert result is None
