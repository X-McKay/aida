# AIDA Tools Documentation

## Overview

AIDA implements a **hybrid tool architecture** where each tool supports three interfaces:
1. **Native AIDA** - Direct async execution via `execute()` method
2. **PydanticAI** - Integration with PydanticAI agents
3. **MCP Server** - Model Context Protocol for standardized tool access

This design allows tools to work seamlessly across different AI frameworks while maintaining a consistent implementation.

## Tool Architecture

### Base Classes

All tools inherit from `ToolBase` in `aida/tools/base.py`:

```python
class ToolBase(ABC):
    """Base class for hybrid tools."""

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Native AIDA execution interface."""
        pass

    @abstractmethod
    def to_pydantic_tool(self) -> list[Callable]:
        """Convert to PydanticAI-compatible tools."""
        pass

    @abstractmethod
    async def to_mcp_tool(self) -> list[dict]:
        """Convert to MCP tool definitions."""
        pass
```

### Tool Result Format

Tools return standardized `ToolResult` objects:

```python
@dataclass
class ToolResult:
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Optional[dict] = None
```

## Available Tools

### 1. FileOperationsTool

File I/O operations via MCP filesystem server.

**Location**: `aida/tools/file_operations/`

**Operations**:
- `read` - Read file contents
- `write` - Write content to file
- `append` - Append content to file
- `delete` - Delete file
- `list` - List directory contents
- `exists` - Check if file/directory exists
- `mkdir` - Create directory
- `move` - Move/rename file
- `copy` - Copy file
- `get_info` - Get file metadata
- `batch` - Execute multiple operations

**Example**:
```python
from aida.tools.file_operations import FileOperationsTool

tool = FileOperationsTool()
result = await tool.execute(
    operation="read",
    path="/path/to/file.py"
)
```

### 2. SystemExecutionTool

Secure command execution with comprehensive controls.

**Location**: `aida/tools/system/`

**Features**:
- Timeout control
- Working directory specification
- Environment variable management
- Output capture (stdout/stderr)
- Async execution support
- Security validation

**Example**:
```python
from aida.tools.system import SystemExecutionTool

tool = SystemExecutionTool()
result = await tool.execute(
    command="python script.py",
    working_dir="/project",
    timeout=30
)
```

### 3. WebSearchTool

Web search and content extraction via MCP SearXNG.

**Location**: `aida/tools/websearch/`

**Operations**:
- `search` - Search web with various categories
- `get_website` - Extract content from URL
- `get_datetime` - Get current datetime

**Categories**:
- General, Images, Videos, Files, Maps, Social Media

**Example**:
```python
from aida.tools.websearch import WebSearchTool

tool = WebSearchTool()
result = await tool.execute(
    operation="search",
    query="Python tutorials",
    max_results=10,
    scrape_content=True
)
```

### 4. ThinkingTool

Structured reasoning and analysis capabilities.

**Location**: `aida/tools/thinking/`

**Operations**:
- `analyze` - Deep analysis of content/problems
- `reason` - Step-by-step reasoning
- `critique` - Critical evaluation
- `brainstorm` - Creative idea generation
- `summarize` - Content summarization
- `explain` - Detailed explanations

**Example**:
```python
from aida.tools.thinking import ThinkingTool

tool = ThinkingTool()
result = await tool.execute(
    operation="analyze",
    content="Complex problem description...",
    analysis_type="problem_solving"
)
```

### 5. LLMResponseTool

Direct LLM interaction for specialized responses.

**Location**: `aida/tools/llm_response/`

**Features**:
- Purpose-based model selection
- Streaming support
- Context management
- Response formatting

**Example**:
```python
from aida.tools.llm_response import LLMResponseTool

tool = LLMResponseTool()
result = await tool.execute(
    prompt="Explain quantum computing",
    purpose="reasoning",
    stream=False
)
```

## Creating Custom Tools

### 1. Tool Implementation

```python
from aida.tools.base import ToolBase, ToolResult, ToolParameter

class MyCustomTool(ToolBase):
    """Custom tool implementation."""

    def __init__(self):
        super().__init__(
            name="my_tool",
            description="Tool description",
            version="1.0.0"
        )

    async def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute tool operation."""
        try:
            if operation == "my_operation":
                result = await self._do_operation(**kwargs)
                return ToolResult(success=True, result=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def get_parameters(self) -> dict[str, list[ToolParameter]]:
        """Define operation parameters."""
        return {
            "my_operation": [
                ToolParameter(
                    name="param1",
                    type="string",
                    description="Parameter description",
                    required=True
                )
            ]
        }
```

### 2. PydanticAI Integration

```python
def to_pydantic_tool(self) -> list[Callable]:
    """Convert to PydanticAI tools."""

    async def my_operation(ctx, param1: str) -> str:
        """PydanticAI-compatible function."""
        result = await self.execute(
            operation="my_operation",
            param1=param1
        )
        return result.result if result.success else f"Error: {result.error}"

    return [my_operation]
```

### 3. MCP Integration

```python
async def to_mcp_tool(self) -> list[dict]:
    """Convert to MCP tool definitions."""
    return [{
        "name": "my_operation",
        "description": "Operation description",
        "inputSchema": {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            "required": ["param1"]
        }
    }]
```

## Tool Registration

Tools are registered in `aida/tools/__init__.py`:

```python
from aida.tools.registry import ToolRegistry

# Initialize registry
registry = ToolRegistry()

# Register tools
registry.register(FileOperationsTool())
registry.register(SystemExecutionTool())
registry.register(WebSearchTool())
registry.register(ThinkingTool())

# Get all tools
all_tools = registry.get_all_tools()
```

## Using Tools in Workers

Workers can access tools through the registry:

```python
from aida.tools import get_tool

class MyWorker(WorkerAgent):
    async def execute_task(self, task_data):
        # Get tool
        file_tool = get_tool("file_operations")

        # Use tool
        result = await file_tool.execute(
            operation="read",
            path=task_data["file_path"]
        )

        if result.success:
            content = result.result
            # Process content...
```

## Observability

All tools include OpenTelemetry instrumentation:

- **Traces**: Track operation execution, duration, errors
- **Metrics**: Count operations, measure performance
- **Attributes**: Operation type, parameters, results

Example metrics:
- `aida.tool.{tool_name}.operations` - Operation count
- `aida.tool.{tool_name}.errors` - Error count
- `aida.tool.{tool_name}.duration` - Execution time

## Best Practices

1. **Error Handling**
   - Always return `ToolResult` with proper error messages
   - Include context in error messages
   - Log errors for debugging

2. **Parameter Validation**
   - Validate required parameters
   - Provide clear parameter descriptions
   - Use type hints consistently

3. **Security**
   - Validate file paths (no path traversal)
   - Sanitize command inputs
   - Respect configured restrictions

4. **Performance**
   - Implement timeouts for long operations
   - Use async operations when possible
   - Cache results when appropriate

5. **Documentation**
   - Document all operations clearly
   - Provide usage examples
   - List all parameters and types

## Testing Tools

```python
# Unit test example
async def test_tool():
    tool = MyCustomTool()

    # Test successful operation
    result = await tool.execute(
        operation="my_operation",
        param1="test"
    )
    assert result.success
    assert result.result == expected_value

    # Test error handling
    result = await tool.execute(
        operation="invalid_op"
    )
    assert not result.success
    assert "Invalid operation" in result.error
```

## Common Issues

1. **MCP Server Connection**
   - Ensure MCP servers are running
   - Check network connectivity
   - Verify server configuration

2. **Tool Not Found**
   - Check tool registration
   - Verify tool name matches
   - Ensure imports are correct

3. **Parameter Errors**
   - Review parameter definitions
   - Check required vs optional
   - Validate parameter types

4. **Timeout Issues**
   - Increase timeout for long operations
   - Implement progress callbacks
   - Consider async execution
