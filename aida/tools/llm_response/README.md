# LLM Response Tool

## Overview
The LLM Response Tool provides direct access to language model capabilities for answering questions, generating content, and providing information without requiring specialized processing or external data sources.

## Features
- Direct LLM access for general knowledge questions
- Context-aware responses using conversation history
- Configurable response length limits
- Support for multiple LLM backends through AIDA's LLM system
- Full hybrid architecture support (AIDA, PydanticAI, MCP)

## Configuration
The tool uses the following configuration parameters:

- `LLM_PURPOSE`: Set to `Purpose.DEFAULT` for general responses
- `DEFAULT_MAX_LENGTH`: Default 2000 characters
- `MIN_LENGTH`: Minimum 100 characters
- `MAX_LENGTH`: Maximum 10000 characters

## Usage Examples

### Basic Usage
```python
from aida.tools.llm_response import LLMResponseTool

tool = LLMResponseTool()
result = await tool.execute(
    question="What are the benefits of test-driven development?",
    max_length=500
)
print(result.result)
```

### With Context
```python
# Provide additional context for the response
result = await tool.execute(
    question="Which of those would be most important for a startup?",
    context="Previous discussion about TDD benefits: faster debugging, better design, documentation",
    max_length=300
)
```

### PydanticAI Integration
```python
from pydantic_ai import Agent
from aida.tools.llm_response import LLMResponseTool

tool = LLMResponseTool()
agent = Agent(
    "You are a helpful assistant",
    tools=tool.to_pydantic_tools()
)

response = await agent.run("Explain quantum computing")
```

### MCP Server Usage
```python
mcp_server = tool.get_mcp_server()

# List available tools
tools = await mcp_server.list_tools()

# Call the tool via MCP
result = await mcp_server.call_tool(
    "llm_response_answer",
    {
        "question": "What is machine learning?",
        "max_length": 500
    }
)
```

## Operations

### answer (default)
**Description**: Get an LLM response to a question or request

**Parameters**:
- `question` (str, required): The question or request to answer
- `context` (str, optional): Additional context for the question
- `max_length` (int, optional): Maximum response length in characters (default: 2000)

**Returns**: String containing the LLM's response

**Example**:
```python
result = await tool.execute(
    question="Explain the concept of recursion with an example",
    context="For a beginner programmer",
    max_length=1000
)
```

## Error Handling
Common errors and solutions:

- **LLM not available**: Ensure Ollama is running and the configured model is pulled
- **Response too long**: Adjust max_length parameter or ask for a more concise response
- **Context format error**: Ensure context is a string (JSON objects will be serialized)

## Performance Considerations
- Response time depends on the LLM backend and model size
- Longer max_length values may increase response time
- Context should be kept concise to avoid token limits
- The tool automatically manages token limits based on the LLM configuration

## Dependencies
- AIDA LLM system (configured via `aida.llm`)
- Ollama (default) or other configured LLM backend
- No external API keys required with default Ollama setup

## Changelog
- **1.0.0**: Initial implementation with full hybrid architecture support
- **1.0.1**: Added context parameter for conversation continuity
- **1.0.2**: Improved error handling and response formatting
