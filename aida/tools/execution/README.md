# Execution Tool

## Overview
The Execution Tool provides secure code execution capabilities in containerized environments using Dagger.io. It supports multiple programming languages and runtimes, with built-in security controls, resource limits, and package management.

## Features
- Secure containerized execution using Dagger
- Support for Python, JavaScript/Node, Bash, Go, Rust, and Java
- Automatic package/dependency installation
- Configurable timeouts and memory limits
- File I/O support for multi-file projects
- Environment variable management
- Full hybrid architecture support (AIDA, PydanticAI, MCP)

## Configuration
The tool uses the following configuration parameters:

- `DEFAULT_TIMEOUT`: 30 seconds
- `MAX_TIMEOUT`: 300 seconds (5 minutes)
- `DEFAULT_MEMORY_LIMIT`: 512MB
- `MAX_MEMORY_LIMIT`: 2GB
- `CONTAINER_WORK_DIR`: /workspace

## Usage Examples

### Basic Usage
```python
from aida.tools.execution import ExecutionTool

tool = ExecutionTool()
result = await tool.execute(
    language="python",
    code="print('Hello, World!')",
    timeout=10
)
print(result.result["output"])  # Hello, World!
```

### With Package Dependencies
```python
result = await tool.execute(
    language="python",
    code="""
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Mean: {np.mean(arr)}")
""",
    packages=["numpy"]
)
```

### Multi-file Project
```python
result = await tool.execute(
    language="python",
    code="from helper import greet\ngreet('AIDA')",
    files={
        "helper.py": "def greet(name):\n    print(f'Hello, {name}!')"
    }
)
```

### With Environment Variables
```python
result = await tool.execute(
    language="python",
    code="import os\nprint(os.environ.get('MY_VAR', 'not set'))",
    env_vars={"MY_VAR": "test_value"}
)
```

### PydanticAI Integration
```python
from pydantic_ai import Agent
from aida.tools.execution import ExecutionTool

tool = ExecutionTool()
agent = Agent(
    "You are a code execution assistant",
    tools=tool.to_pydantic_tools()
)

# Agent can now execute code
response = await agent.run("Calculate fibonacci of 10 in Python")
```

### MCP Server Usage
```python
mcp_server = tool.get_mcp_server()

# Execute via MCP
result = await mcp_server.call_tool(
    "execution_run_python",
    {
        "code": "print([x**2 for x in range(10)])",
        "packages": []
    }
)
```

## Operations

### execute (default)
**Description**: Execute code in a containerized environment

**Parameters**:
- `language` (str, required): Programming language - "python", "javascript", "bash", etc.
- `code` (str, required): Code to execute
- `files` (dict, optional): Additional files as {filename: content}
- `packages` (list, optional): Package dependencies to install
- `env_vars` (dict, optional): Environment variables
- `timeout` (int, optional): Timeout in seconds (default: 30)
- `memory_limit` (str, optional): Memory limit like "512m" or "1g" (default: "512m")

**Returns**: Dictionary with execution results
```json
{
    "output": "Standard output",
    "error": "Standard error",
    "exit_code": 0,
    "execution_time": 1.23
}
```

**Example**:
```python
result = await tool.execute(
    language="javascript",
    code="console.log(Math.PI * 2)",
    timeout=5
)
```

## Supported Languages

### Python
- Image: `python:3.11-slim`
- Package Manager: pip
- File Extension: .py

### JavaScript/Node
- Image: `node:18-slim`
- Package Manager: npm
- File Extension: .js

### Bash
- Image: `alpine:latest`
- Command: sh -c
- File Extension: .sh

### Go
- Image: `golang:1.21-alpine`
- Package Manager: go get
- File Extension: .go

### Rust
- Image: `rust:1.73-slim`
- Package Manager: cargo
- File Extension: .rs

### Java
- Image: `openjdk:17-slim`
- File Extension: .java

## Security Considerations

### Blocked Patterns
The tool blocks potentially dangerous code patterns:
- `subprocess.call`, `subprocess.run`, `os.system`
- `eval()`, `exec()`, `__import__`
- Direct file operations: `open()`, `file()`

### Environment Variables
Only whitelisted environment variables are allowed:
- `HOME`, `USER`, `PATH`
- `LANG`, `LC_ALL`
- Language-specific: `PYTHONPATH`, `NODE_PATH`, `GOPATH`

### Resource Limits
- Execution timeout: Max 5 minutes
- Memory limit: Max 2GB
- CPU: Shared with host (Dagger default)
- Network: Isolated by default

## Error Handling
Common errors and solutions:

- **Timeout Error**: Code exceeded timeout limit
  - Solution: Optimize code or increase timeout (max 300s)
- **Package Installation Failed**: Unable to install dependencies
  - Solution: Check package names and availability
- **Memory Limit Exceeded**: Process killed due to memory usage
  - Solution: Optimize memory usage or increase limit
- **Syntax Error**: Code has syntax errors
  - Solution: Validate code syntax before execution

## Performance Considerations
- Container startup time: ~1-3 seconds per execution
- Package installation adds overhead on first use
- Reuse containers when possible for multiple executions
- Consider batching related executions
- File I/O operations add minimal overhead

## Dependencies
- Dagger.io: Container orchestration engine
- Docker: Required by Dagger (must be running)
- No external API keys required

## Best Practices
1. **Validate Input**: Check code for dangerous patterns before execution
2. **Set Appropriate Timeouts**: Use shortest reasonable timeout
3. **Limit Package Installation**: Install only necessary packages
4. **Handle Output Streams**: Check both stdout and stderr
5. **Use Memory Limits**: Prevent runaway processes
6. **Clean Up Resources**: Containers are automatically cleaned up

## Troubleshooting

### Docker Not Running
```
Error: Cannot connect to Docker daemon
Solution: Start Docker Desktop or Docker service
```

### Dagger Connection Failed
```
Error: Failed to connect to Dagger
Solution: Check Docker is running and accessible
```

### Package Not Found
```
Error: Package 'xyz' not found
Solution: Verify package name and availability for the language
```

## Changelog
- **2.0.0**: Complete rewrite with Dagger.io backend
- **2.0.1**: Added support for Go, Rust, and Java
- **2.0.2**: Improved error handling and resource limits
- **2.0.3**: Added file I/O and multi-file project support
