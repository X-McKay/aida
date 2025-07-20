# Hybrid Architecture Overview

## Introduction

AIDA's hybrid architecture enables tools to support multiple AI frameworks simultaneously. This approach provides maximum flexibility and compatibility across different AI ecosystems.

## Architecture Goals

1. **Framework Flexibility** - Support for PydanticAI, MCP, and future frameworks
2. **Zero Lock-in** - Use any interface or combine multiple interfaces
3. **Minimal Overhead** - Less than 5% performance impact
4. **Production Ready** - Built-in observability and monitoring

## Supported Interfaces

### 1. PydanticAI Interface

Clean, typed functions designed for modern AI agents:

```python
tools = tool.to_pydantic_tools()
content = await tools["read_file"]("/data/file.txt")
# Returns: Simple dict with results
```

**Benefits:**
- Type-safe operations
- Clean function signatures
- Easy integration with AI agents
- Simplified responses

### 2. MCP Server Interface

Universal AI compatibility via Model Context Protocol:

```python
mcp = tool.get_mcp_server()
result = await mcp.call_tool("file_read_file", {
    "path": "/data/file.txt"
})
# Returns: MCP-formatted response
```

**Benefits:**
- Works with Claude, GPT, and other AI systems
- Standardized tool protocol
- Language agnostic
- Remote execution capable

### 3. OpenTelemetry Observability

Production monitoring and tracing:

```python
obs = tool.enable_observability({
    "enabled": True,
    "service_name": "my-service"
})

with obs.trace_operation("bulk_process"):
    # Operations are traced
    pass
```

**Benefits:**
- Distributed tracing
- Performance metrics
- Error tracking
- Production debugging

## Implementation Pattern

### Tool Structure

```python
class HybridTool(Tool):
    """Example hybrid tool implementation."""

    def __init__(self):
        super().__init__(name="hybrid_tool", version="1.0.0")
        self._pydantic_tools_cache = {}
        self._mcp_server = None
        self._observability = None

    async def execute(self, **kwargs) -> ToolResult:
        """Tool execution method."""
        # Implementation
        pass

    def to_pydantic_tools(self) -> Dict[str, Callable]:
        """Convert to PydanticAI tools."""
        # Return dict of clean functions
        pass

    def get_mcp_server(self) -> MCPServer:
        """Get MCP server interface."""
        # Return MCP wrapper
        pass

    def enable_observability(self, config: Dict) -> Observability:
        """Enable OpenTelemetry."""
        # Return observability wrapper
        pass
```

### PydanticAI Adapter Pattern

```python
def to_pydantic_tools(self) -> Dict[str, Callable]:
    if self._pydantic_tools_cache:
        return self._pydantic_tools_cache

    async def clean_function(param1: str, param2: int = None) -> Dict:
        """Clean function for PydanticAI."""
        result = await self.execute(
            operation="something",
            param1=param1,
            param2=param2
        )

        if result.status == ToolStatus.COMPLETED:
            return result.result
        else:
            raise RuntimeError(result.error)

    self._pydantic_tools_cache = {
        "clean_function": clean_function
    }
    return self._pydantic_tools_cache
```

### MCP Server Pattern

```python
class ToolMCPServer:
    def __init__(self, tool):
        self.tool = tool
        self.handlers = {
            "tool_operation": self._handle_operation
        }

    async def call_tool(self, name: str, args: Dict) -> Dict:
        if name not in self.handlers:
            return {"isError": True, "content": [{"type": "text", "text": "Unknown tool"}]}

        try:
            result = await self.handlers[name](args)
            return {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(result)}]
            }
        except Exception as e:
            return {"isError": True, "content": [{"type": "text", "text": str(e)}]}
```

## Usage Patterns

### Single Interface Usage

Use one interface exclusively:

```python
# Pure AIDA
tool = MyTool()
result = await tool.execute(operation="something")

# Pure PydanticAI
tools = tool.to_pydantic_tools()
result = await tools["something"]()

# Pure MCP
mcp = tool.get_mcp_server()
result = await mcp.call_tool("tool_something", {})
```

### Mixed Interface Usage

Combine interfaces for different needs:

```python
tool = MyTool()

# Use AIDA for rich metadata
result = await tool.execute(operation="analyze", path="/data")
execution_time = result.duration_seconds

# Use PydanticAI for AI agent
agent = Agent("gpt-4")
tool.register_with_pydantic_agent(agent)
response = await agent.run("Process the analysis")

# Use MCP for external integration
mcp = tool.get_mcp_server()
external_result = await mcp.call_tool("tool_export", {"format": "json"})

# Monitor with observability
with tool.enable_observability(config).trace_operation("pipeline"):
    # All operations are traced
    pass
```

## Benefits by Use Case

### For AI Agent Development

- **PydanticAI**: Clean functions for agent tools
- **Type Safety**: Full typing support
- **Simple Returns**: No complex metadata
- **Easy Integration**: Direct agent registration

### For Cross-Platform Integration

- **MCP Protocol**: Universal AI compatibility
- **Remote Execution**: Network-capable tools
- **Language Agnostic**: Works with any language
- **Standardized**: Industry protocol

### For Production Systems

- **Observability**: Full tracing support
- **Monitoring**: Performance metrics
- **Debugging**: Detailed operation tracking
- **Reliability**: Error tracking and alerts


## Implementation Checklist

When implementing hybrid architecture:

- [ ] Implement `execute()` method for core functionality
- [ ] Implement `to_pydantic_tools()` with clean functions
- [ ] Create MCP server wrapper class
- [ ] Add observability support
- [ ] Cache instances to avoid recreation
- [ ] Handle errors consistently across interfaces
- [ ] Document all interfaces
- [ ] Create comprehensive tests

## Performance Considerations

### Overhead Analysis

- **Interface Adaptation**: < 1ms per call
- **Object Creation**: Cached after first use
- **Memory Impact**: ~10KB per tool instance
- **Overall Impact**: < 5% in typical usage

### Optimization Tips

1. **Cache Tool Instances**
   ```python
   _tools_cache = {}

   def get_tools():
       if "my_tool" not in _tools_cache:
           _tools_cache["my_tool"] = MyTool()
       return _tools_cache["my_tool"]
   ```

2. **Reuse Interface Objects**
   ```python
   # Good: Create once
   pydantic_tools = tool.to_pydantic_tools()
   for item in items:
       await pydantic_tools["process"](item)

   # Bad: Create repeatedly
   for item in items:
       tools = tool.to_pydantic_tools()
       await tools["process"](item)
   ```

3. **Batch Operations**
   ```python
   # Process multiple items in one call
   result = await tool.execute(
       operation="batch_process",
       items=items
   )
   ```

## Testing Strategy

### Interface Testing

Test each interface independently:

```python
async def test_aida_interface():
    result = await tool.execute(operation="test")
    assert result.status == ToolStatus.COMPLETED

async def test_pydantic_interface():
    tools = tool.to_pydantic_tools()
    result = await tools["test"]()
    assert "success" in result

async def test_mcp_interface():
    mcp = tool.get_mcp_server()
    result = await mcp.call_tool("tool_test", {})
    assert not result["isError"]
```

### Cross-Interface Testing

Verify consistency across interfaces:

```python
async def test_consistency():
    # Same operation, different interfaces
    aida_result = await tool.execute(operation="get_data")

    pydantic_tools = tool.to_pydantic_tools()
    pydantic_result = await pydantic_tools["get_data"]()

    mcp = tool.get_mcp_server()
    mcp_result = await mcp.call_tool("tool_get_data", {})
    mcp_data = json.loads(mcp_result["content"][0]["text"])

    # Verify same data returned
    assert aida_result.result["value"] == pydantic_result["value"]
    assert pydantic_result["value"] == mcp_data["value"]
```

## Future Extensibility

The hybrid architecture is designed for future growth:

### Adding New Interfaces

```python
def to_langchain_tools(self) -> List[LangChainTool]:
    """Future: LangChain compatibility."""
    pass

def get_grpc_server(self) -> GRPCServer:
    """Future: gRPC interface."""
    pass

def enable_graphql(self) -> GraphQLSchema:
    """Future: GraphQL API."""
    pass
```

### Protocol Evolution

- Support new MCP versions
- Add streaming capabilities
- Implement async generators
- Support WebSocket connections

## Conclusion

The hybrid architecture provides a future-proof foundation for AIDA tools, enabling:

- Seamless integration with any AI framework
- Production-ready observability
- Minimal performance overhead
- Maximum flexibility

Tools implementing this architecture can be used in any context, from simple scripts to complex production systems.
