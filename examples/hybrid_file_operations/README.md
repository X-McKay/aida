# Hybrid FileOperationsTool Examples

This directory demonstrates the hybrid architecture approach for AIDA tools, using FileOperationsTool as the example implementation.

## Architecture Overview

The hybrid FileOperationsTool provides **three different interfaces** for maximum compatibility and interoperability:

### 1. 🔧 Original AIDA Interface
```python
from aida.tools.files import FileOperationsTool

file_tool = FileOperationsTool()
result = await file_tool.execute(
    operation="read_file",
    path="/path/to/file.txt"
)
```

### 2. 🤖 PydanticAI Compatibility
```python
file_tool = FileOperationsTool()
pydantic_tools = file_tool.to_pydantic_tools()

# Use individual tool functions
content = await pydantic_tools["read_file"]("/path/to/file.txt")
```

### 3. 🌐 MCP Server Interface
```python
file_tool = FileOperationsTool()
mcp_server = file_tool.get_mcp_server()

# Compatible with Claude, other MCP clients
result = await mcp_server.call_tool("file_read_file", {"path": "/path/to/file.txt"})
```

## Key Benefits

- ✅ **Backward Compatible**: Existing AIDA code continues to work
- ✅ **Industry Standard**: PydanticAI integration for modern AI frameworks
- ✅ **Interoperable**: MCP protocol works with Claude, other AI systems
- ✅ **Observable**: OpenTelemetry integration for production monitoring
- ✅ **Future-Proof**: Supports emerging AI agent ecosystem

## Files

- `basic_usage.py` - Demonstrates all three interfaces
- `pydantic_agent_example.py` - Full PydanticAI agent integration
- `mcp_client_example.py` - MCP client usage patterns
- `observability_example.py` - OpenTelemetry monitoring setup

## Running Examples

```bash
# Basic usage demo
python examples/hybrid_file_operations/basic_usage.py

# PydanticAI agent example  
python examples/hybrid_file_operations/pydantic_agent_example.py

# MCP client example
python examples/hybrid_file_operations/mcp_client_example.py
```

## Installation Requirements

```bash
# Basic functionality (always available)
pip install aida

# PydanticAI support
pip install pydantic-ai

# MCP support
pip install model-context-protocol

# Observability support
pip install opentelemetry-api opentelemetry-sdk
```

## Migration Guide

### From Original AIDA Tools
No changes needed! Existing code continues to work:

```python
# This still works exactly as before
from aida.tools.files import FileOperationsTool
```

### To PydanticAI
```python
from aida.tools.files import FileOperationsTool
from pydantic_ai import Agent

agent = Agent(model='openai:gpt-4')
file_tool = FileOperationsTool()

# Register all file operations with agent
file_tool.register_with_pydantic_agent(agent)

# Or use individual tools
tools = file_tool.to_pydantic_tools(agent)
```

### To MCP Integration
```python
from aida.tools.files import FileOperationsTool

file_tool = FileOperationsTool()
mcp_server = file_tool.get_mcp_server()

# Now compatible with any MCP client
```

## Architecture Pattern

This hybrid approach can be applied to any AIDA tool:

1. **Keep Original Interface**: Maintains backward compatibility
2. **Add PydanticAI Methods**: `to_pydantic_tools()`, `register_with_pydantic_agent()`
3. **Add MCP Wrapper**: `get_mcp_server()` returns MCP-compatible interface
4. **Add Observability**: `enable_observability()` for production monitoring

This pattern allows gradual migration while supporting multiple AI agent frameworks simultaneously.