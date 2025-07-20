# AIDA Tools - Modular Architecture Guide

## Overview

AIDA tools follow a consistent modular architecture pattern that promotes code organization, maintainability, and extensibility. Each tool is structured as a Python package with clearly defined components.

## Base Classes

AIDA provides base classes to reduce code duplication and ensure consistency across all tools:

### BaseModularTool
The main base class for all tools, providing:
- Consistent initialization pattern
- Built-in support for MCP, PydanticAI, and observability
- Abstract methods that enforce implementation of required functionality

### BaseMCPServer
Base implementation for MCP servers with:
- Consistent response formatting
- Error handling
- Tool definition helpers

### BaseObservability
Base OpenTelemetry integration providing:
- Automatic tracer and meter setup
- Common metrics (operation count, duration, errors)
- Extensible custom metrics support

### SimpleToolBase
Simplified base for tools with basic operations:
- Automatic operation routing
- Built-in error handling and metrics
- Reduced boilerplate for simple tools

## Directory Structure

Each tool should be organized as follows:

```
tools/
├── tool_name/
│   ├── __init__.py          # Package initialization and exports
│   ├── README.md            # Tool-specific documentation
│   ├── config.py            # Configuration constants and settings
│   ├── models.py            # Pydantic models for requests/responses
│   ├── tool_name.py         # Main tool implementation
│   ├── prompt_builder.py    # Prompt construction logic (for LLM tools)
│   ├── response_parser.py   # Response parsing logic (for LLM tools)
│   ├── mcp_server.py        # MCP server implementation
│   └── observability.py     # OpenTelemetry instrumentation
```

## Component Descriptions

### 1. `__init__.py`
Defines the public API of the tool package:
```python
from .tool_name import ToolNameTool
from .models import (
    ToolRequest,
    ToolResponse,
    # Other models...
)
from .config import ToolConfig

__all__ = ["ToolNameTool", "ToolRequest", "ToolResponse", "ToolConfig"]
```

### 2. `config.py`
Contains all configuration constants and settings:
```python
class ToolConfig:
    # LLM Configuration
    LLM_PURPOSE = Purpose.DEFAULT

    # Processing Configuration
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3

    # Feature flags
    ENABLE_CACHING = True

    @classmethod
    def get_mcp_config(cls) -> Dict[str, Any]:
        """Get MCP server configuration."""
        return {...}
```

### 3. `models.py`
Defines Pydantic models for type safety and validation:
```python
from pydantic import BaseModel, Field, validator
from enum import Enum

class OperationType(str, Enum):
    """Enumeration of operation types."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"

class ToolRequest(BaseModel):
    """Request model for tool operations."""
    operation: OperationType
    parameters: Dict[str, Any]

    @validator('parameters')
    def validate_parameters(cls, v, values):
        # Custom validation logic
        return v

class ToolResponse(BaseModel):
    """Response model for tool operations."""
    request_id: str
    status: str
    result: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

### 4. `tool_name.py`
Main tool implementation following the hybrid architecture:
```python
from aida.tools.base import Tool, ToolResult, ToolCapability, ToolParameter

class ToolNameTool(Tool):
    """Main tool implementation."""

    def __init__(self):
        super().__init__(
            name="tool_name",
            description="Tool description",
            version="1.0.0"
        )
        self._pydantic_tools_cache = {}
        self._mcp_server = None
        self._observability = None

    def get_capability(self) -> ToolCapability:
        """Define tool capabilities."""
        return ToolCapability(...)

    async def execute(self, **kwargs) -> ToolResult:
        """Execute tool operation."""
        # Implementation

    # Hybrid architecture methods
    def to_pydantic_tools(self, agent=None) -> Dict[str, Callable]:
        """Convert to PydanticAI-compatible tools."""
        # Implementation

    def get_mcp_server(self):
        """Get MCP server instance."""
        if self._mcp_server is None:
            from .mcp_server import ToolMCPServer
            self._mcp_server = ToolMCPServer(self)
        return self._mcp_server
```

### 5. `prompt_builder.py` (For LLM-based tools)
Handles prompt construction:
```python
class PromptBuilder:
    """Builds prompts for LLM operations."""

    def build(self, request: ToolRequest) -> str:
        """Build prompt from request."""
        # Implementation
```

### 6. `response_parser.py` (For LLM-based tools)
Handles response parsing:
```python
class ResponseParser:
    """Parses LLM responses into structured data."""

    def parse(self, response: str, request: ToolRequest) -> ToolResponse:
        """Parse response into structured format."""
        # Implementation
```

### 7. `mcp_server.py`
MCP (Model Context Protocol) server implementation:
```python
class ToolMCPServer:
    """MCP server wrapper for tool."""

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        # Implementation

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool calls."""
        # Implementation
```

### 8. `observability.py`
OpenTelemetry instrumentation:
```python
class ToolObservability:
    """OpenTelemetry observability for tool."""

    def __init__(self, tool, config: Dict[str, Any]):
        # Setup tracing and metrics

    @contextmanager
    def trace_operation(self, operation: str, **attributes):
        """Create trace span for operation."""
        # Implementation
```

### 9. `README.md` (Tool-Specific Documentation)
Each tool MUST include its own README.md with the following sections:

```markdown
# Tool Name

## Overview
Brief description of what the tool does and its primary use cases.

## Features
- List of key features
- Supported operations
- Integration capabilities

## Configuration
Required and optional configuration parameters, environment variables, etc.

## Usage Examples

### Basic Usage
```python
from aida.tools.tool_name import ToolNameTool

tool = ToolNameTool()
result = await tool.execute(operation="example", param="value")
```

### PydanticAI Integration
```python
from pydantic_ai import Agent
from aida.tools.tool_name import ToolNameTool

tool = ToolNameTool()
agent = Agent(tools=tool.to_pydantic_tools())
```

### MCP Server Usage
```python
mcp_server = tool.get_mcp_server()
tools = await mcp_server.list_tools()
```

## Operations

### operation_name
**Description**: What this operation does
**Parameters**:
- `param1` (type): Description
- `param2` (type, optional): Description

**Returns**: Description of return value

**Example**:
```python
result = await tool.execute(
    operation="operation_name",
    param1="value",
    param2="optional_value"
)
```

## Error Handling
Common errors and how to handle them.

## Performance Considerations
- Rate limits
- Timeout settings
- Best practices for optimal performance

## Dependencies
External dependencies or services required.

## Changelog
Notable changes in recent versions.
```

## Best Practices

### 1. Documentation
- Every tool MUST have its own README.md
- Follow the tool-specific README template
- Keep documentation updated with code changes
- Include real, working examples

### 2. Type Safety
- Use Pydantic models for all inputs/outputs
- Define enums for fixed choices
- Add validators for complex validation logic

### 3. Error Handling
- Use specific exceptions for different error types
- Provide meaningful error messages
- Log errors appropriately

### 4. Configuration
- Keep all constants in config.py
- Use class methods for computed configurations
- Support environment variable overrides

### 5. Testing
- Write unit tests for each component
- Mock external dependencies
- Test error cases thoroughly

### 6. Code Quality
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Keep functions focused and single-purpose
- Add type hints to all functions

## Using Base Classes

### Example: Simple Tool with Base Classes

```python
# my_tool.py
from aida.tools.base_tool import SimpleToolBase
from aida.tools.base_mcp import SimpleMCPServer
from aida.tools.base_observability import SimpleObservability

class MyTool(SimpleToolBase):
    """Example tool using base classes."""

    def _get_tool_name(self) -> str:
        return "my_tool"

    def _get_tool_version(self) -> str:
        return "1.0.0"

    def _get_tool_description(self) -> str:
        return "Example tool demonstrating base classes"

    def _get_default_config(self):
        return MyToolConfig()

    def _create_processors(self) -> Dict[str, Callable]:
        return {
            "process": self._process_data,
            "analyze": self._analyze_data
        }

    async def _process_data(self, data: str, **kwargs) -> Dict[str, Any]:
        # Implementation
        return {"processed": data.upper()}

    async def _analyze_data(self, data: str, **kwargs) -> Dict[str, Any]:
        # Implementation
        return {"length": len(data), "words": len(data.split())}

    def _create_mcp_server(self):
        return SimpleMCPServer(self, {
            "process": {
                "description": "Process data",
                "parameters": {
                    "data": {"type": "string", "description": "Data to process"}
                },
                "required": ["data"]
            },
            "analyze": {
                "description": "Analyze data",
                "parameters": {
                    "data": {"type": "string", "description": "Data to analyze"}
                },
                "required": ["data"]
            }
        })

    def _create_observability(self, config: Dict[str, Any]):
        return SimpleObservability(self, config, {
            "data_processed": {
                "type": "counter",
                "description": "Number of data items processed"
            },
            "data_size": {
                "type": "histogram",
                "description": "Size of processed data",
                "unit": "bytes"
            }
        })

    def _create_pydantic_tools(self) -> Dict[str, Callable]:
        async def process_data(data: str) -> str:
            result = await self.execute(operation="process", data=data)
            return result.result["processed"]

        async def analyze_data(data: str) -> Dict[str, int]:
            result = await self.execute(operation="analyze", data=data)
            return result.result

        return {
            "process_data": process_data,
            "analyze_data": analyze_data
        }
```

## Creating a New Tool

1. **Create the directory structure**:
   ```bash
   mkdir -p aida/tools/new_tool
   ```

2. **Start with models.py**:
   - Define request/response models
   - Add validation logic
   - Define enums for choices

3. **Create config.py**:
   - Add configuration constants
   - Define default values
   - Add configuration methods

4. **Implement the main tool**:
   - Inherit from `Tool` base class
   - Implement `get_capability()` method
   - Implement `execute()` method
   - Add hybrid architecture methods

5. **Add supporting components**:
   - Prompt builder (if LLM-based)
   - Response parser (if LLM-based)
   - MCP server wrapper
   - Observability setup

6. **Create __init__.py**:
   - Export public API
   - Define __all__

7. **Create README.md**:
   - Follow the tool-specific README template above
   - Include all operations with examples
   - Document configuration requirements
   - Add usage examples for all integration methods

8. **Register the tool**:
   - Add to `tools/__init__.py`
   - Add to `initialize_default_tools()` in `base.py`

## Example: Web Search Tool Structure

```
tools/
├── web_search/
│   ├── __init__.py
│   ├── config.py           # API keys, endpoints, defaults
│   ├── models.py           # SearchRequest, SearchResult, etc.
│   ├── web_search.py       # Main WebSearchTool class
│   ├── search_engines.py   # Different search engine adapters
│   ├── result_parser.py    # Parse and rank search results
│   ├── mcp_server.py       # MCP protocol support
│   └── observability.py    # Metrics and tracing
```

## Migration Guide

To migrate an existing tool to the modular pattern:

1. Create the directory structure
2. Extract models to `models.py`
3. Extract configuration to `config.py`
4. Move prompt/parsing logic to separate files
5. Update imports in `__init__.py`
6. Test thoroughly to ensure compatibility

## Benefits of Base Classes

Using the base classes provides:

- **Reduced Boilerplate**: Common functionality is implemented once
- **Consistent Interface**: All tools behave the same way
- **Automatic Features**: MCP, PydanticAI, and observability come for free
- **Type Safety**: Generic types ensure proper typing throughout
- **Easy Testing**: Base functionality is pre-tested
- **Future-Proof**: New features can be added to base classes

## Benefits of Modular Architecture

- **Consistency**: All tools follow the same pattern
- **Maintainability**: Clear separation of concerns
- **Testability**: Each component can be tested independently
- **Extensibility**: Easy to add new features or capabilities
- **Discoverability**: Predictable structure makes navigation easier

## Tool Documentation Standards

Each tool's README should:

1. **Be self-contained**: Users should understand the tool without reading other docs
2. **Include working examples**: All code examples should be copy-paste ready
3. **Document all operations**: Every operation with parameters and return values
4. **Explain error scenarios**: Common errors and how to resolve them
5. **Show integration methods**: Examples for AIDA, PydanticAI, and MCP usage
6. **List dependencies**: External services, APIs, or configurations needed
7. **Track changes**: Maintain a changelog for version history

### README Quality Checklist
- [ ] Overview clearly explains the tool's purpose
- [ ] All features are listed and explained
- [ ] Configuration section covers all settings
- [ ] Usage examples work when copy-pasted
- [ ] All operations are documented with examples
- [ ] Error handling section covers common issues
- [ ] Performance tips help users optimize usage
- [ ] Dependencies are clearly stated
- [ ] Changelog tracks significant changes
