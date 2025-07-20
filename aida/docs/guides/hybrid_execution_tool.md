# Hybrid ExecutionTool Guide

## Overview

The ExecutionTool provides secure code execution in containerized environments using Dagger.io. It supports multiple programming languages and includes features like package installation, file management, and resource limits.

**Version:** 2.0.0

## Key Features

- **Multi-language Support**: Python, JavaScript, Bash, Node.js, Go, Rust, Java
- **Containerized Execution**: Secure isolation using Dagger.io
- **Package Management**: Install dependencies at runtime
- **File Operations**: Include additional files in execution
- **Resource Limits**: Control memory and execution time
- **Environment Variables**: Set custom environment variables
- **Hybrid Architecture**: Compatible with AIDA, PydanticAI, MCP, and OpenTelemetry

## Architecture

The ExecutionTool implements a hybrid architecture that supports:

1. **Core Tool Interface** - Primary execution method
2. **PydanticAI Tools** - Language-specific convenience functions
3. **MCP Server** - Universal AI compatibility via Model Context Protocol
4. **OpenTelemetry** - Production-ready observability

## Usage Examples

### 1. Core Interface

```python
from aida.tools.execution import ExecutionTool

tool = ExecutionTool()

# Execute Python code
result = await tool.execute(
    language="python",
    code="""
import math
print(f"Pi is approximately {math.pi:.5f}")
print(f"Square root of 2 is {math.sqrt(2):.5f}")
""",
    timeout=30
)

print(f"Output: {result.result['stdout']}")
print(f"Exit code: {result.result['exit_code']}")

# Execute with packages
result = await tool.execute(
    language="python",
    code="""
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Mean: {arr.mean()}")
print(f"Std: {arr.std()}")
""",
    packages=["numpy"],
    timeout=60
)

# Execute bash script
result = await tool.execute(
    language="bash",
    code="""
#!/bin/bash
echo "System information:"
uname -a
echo ""
echo "Current directory:"
pwd
echo ""
echo "Directory contents:"
ls -la
""",
    timeout=30
)
```

### 2. PydanticAI Interface

```python
from aida.tools.execution import ExecutionTool
from pydantic_ai import Agent

# Get PydanticAI-compatible tools
tool = ExecutionTool()
tools = tool.to_pydantic_tools()

# Use with PydanticAI agent
agent = Agent("gpt-4")
tool.register_with_pydantic_agent(agent)

# Or use functions directly
# Execute Python with packages
result = await tools["execute_python"](
    ctx,  # RunContext from PydanticAI
    code="""
import pandas as pd
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}
df = pd.DataFrame(data)
print(df.to_string())
""",
    packages=["pandas"]
)

# Execute JavaScript
result = await tools["execute_javascript"](
    ctx,
    code="""
const axios = require('axios');
console.log('Making HTTP request...');
// Simulated request
const data = {status: 'success', value: 42};
console.log(JSON.stringify(data, null, 2));
""",
    packages=["axios"]
)

# Execute bash scripts
result = await tools["execute_bash"](
    ctx,
    script="""
curl --version
echo "Installing jq..."
which jq || echo "jq not found"
""",
    packages=["curl", "jq"]
)
```

### 3. MCP Server Interface

```python
from aida.tools.execution import ExecutionTool

tool = ExecutionTool()
mcp_server = tool.get_mcp_server()

# Execute code via MCP
result = await mcp_server.call_tool("execution_execute_code", {
    "language": "python",
    "code": "print('Hello from MCP!')",
    "timeout": 30
})

# Execute Python via MCP shorthand
result = await mcp_server.call_tool("execution_run_python", {
    "code": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")
""",
    "timeout": 30
})

# Execute bash via MCP
result = await mcp_server.call_tool("execution_run_bash", {
    "script": "echo $HOME && ls -la /",
    "timeout": 20
})
```

### 4. Advanced Usage

```python
# Execute with files
result = await tool.execute(
    language="python",
    code="""
import json

# Read config file
with open('config.json', 'r') as f:
    config = json.load(f)

print(f"App name: {config['name']}")
print(f"Version: {config['version']}")

# Process data file
with open('data.csv', 'r') as f:
    lines = f.readlines()
    print(f"Data rows: {len(lines) - 1}")  # Minus header
""",
    files={
        "config.json": json.dumps({
            "name": "MyApp",
            "version": "1.0.0",
            "debug": True
        }),
        "data.csv": "id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,300"
    },
    timeout=30
)

# Execute with environment variables
result = await tool.execute(
    language="python",
    code="""
import os
print(f"API_KEY: {os.environ.get('API_KEY', 'not set')}")
print(f"DEBUG: {os.environ.get('DEBUG', 'false')}")
print(f"HOME: {os.environ.get('HOME', 'not set')}")
""",
    env_vars={
        "API_KEY": "secret-key-123",
        "DEBUG": "true"
    },
    timeout=30
)

# Execute with custom working directory
result = await tool.execute(
    language="bash",
    code="""
pwd
mkdir -p output
echo "Test file" > output/test.txt
ls -la output/
""",
    working_dir="/app",
    timeout=30
)

# Execute with memory limit
result = await tool.execute(
    language="python",
    code="""
# Memory-intensive operation
data = []
for i in range(1000000):
    data.append([0] * 100)
print(f"Created {len(data)} arrays")
""",
    memory_limit="256m",  # Limit to 256MB
    timeout=30
)
```

### 5. Script and Notebook Execution

```python
# Execute a script file
result = await tool.execute_script(
    "/path/to/script.py",
    packages=["requests", "beautifulsoup4"],
    timeout=120
)

# Execute a Jupyter notebook
result = await tool.execute_notebook(
    "/path/to/analysis.ipynb",
    packages=["pandas", "matplotlib"],
    timeout=300
)
```

### 6. With Observability

```python
# Enable tracing
observability = tool.enable_observability({
    "enabled": True,
    "service_name": "code-execution-service",
    "endpoint": "http://localhost:4317"
})

# Traced execution
with observability.trace_execution("python", len(code)):
    result = await tool.execute(
        language="python",
        code=code,
        packages=["numpy", "scipy"]
    )
    
    # Record metrics
    observability.record_execution(
        "python",
        result.duration_seconds,
        result.status == ToolStatus.COMPLETED
    )
```

## Supported Languages

### Python
- **Base Image**: `python:3.11-slim`
- **Package Manager**: pip
- **Example Packages**: numpy, pandas, requests, matplotlib

### JavaScript/Node.js
- **Base Image**: `node:18-slim`
- **Package Manager**: npm
- **Example Packages**: axios, lodash, express, cheerio

### Bash
- **Base Image**: `ubuntu:22.04`
- **Package Manager**: apt-get
- **Example Packages**: curl, jq, git, wget

### Go
- **Base Image**: `golang:1.21-alpine`
- **Package Manager**: go modules
- **Notes**: Automatically initializes go module

### Rust
- **Base Image**: `rust:1.75-slim`
- **Package Manager**: cargo (requires Cargo.toml)
- **Notes**: Compiles and runs the code

### Java
- **Base Image**: `openjdk:17-slim`
- **Package Manager**: maven/gradle (requires build files)
- **Notes**: Compiles and runs the code

## Parameters

### Core Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `language` | str | Programming language | Required |
| `code` | str | Code to execute | Required |
| `files` | dict | Additional files (name -> content) | {} |
| `timeout` | int | Execution timeout in seconds | 300 |
| `memory_limit` | str | Memory limit (e.g., '512m', '1g') | '512m' |
| `env_vars` | dict | Environment variables | {} |
| `packages` | list | Packages to install | [] |
| `working_dir` | str | Working directory in container | '/workspace' |

### Language Choices
- `python` - Python 3.11
- `javascript` or `node` - Node.js 18
- `bash` - Bash on Ubuntu 22.04
- `go` - Go 1.21
- `rust` - Rust 1.75
- `java` - Java 17

## Return Values

### Execution Result

```python
{
    "stdout": "Program output here",
    "stderr": "Error output if any",
    "exit_code": 0,  # 0 for success, non-zero for failure
    "language": "python",
    "execution_time": None  # Could be measured in future
}
```

### Exit Codes
- `0` - Success
- `1` - General error
- `2` - Container setup error
- `124` - Timeout

## Security Features

### Container Isolation
- Each execution runs in an isolated container
- No access to host filesystem
- Network access controlled by container runtime

### Resource Limits
- Memory limits enforced
- CPU limits (via container runtime)
- Execution timeout prevents infinite loops

### Package Security
- Packages installed fresh for each execution
- No persistent package cache between executions
- Standard package managers with official repositories

## Best Practices

1. **Set Appropriate Timeouts**
   - Short scripts: 30 seconds
   - Data processing: 120-300 seconds
   - Complex analysis: 300-600 seconds

2. **Manage Memory Usage**
   - Default 512MB is suitable for most tasks
   - Increase for data-intensive operations
   - Monitor for out-of-memory errors

3. **Package Installation**
   - Only install required packages
   - Consider execution time with many packages
   - Use specific package versions when needed

4. **Error Handling**
   ```python
   try:
       result = await tool.execute(
           language="python",
           code=code,
           timeout=60
       )
       
       if result.status == ToolStatus.COMPLETED:
           if result.result["exit_code"] == 0:
               print("Success:", result.result["stdout"])
           else:
               print("Error:", result.result["stderr"])
       else:
           print("Execution failed:", result.error)
           
   except Exception as e:
       print(f"Tool error: {e}")
   ```

5. **File Management**
   - Keep files small (under 1MB each)
   - Use appropriate file formats
   - Clean up temporary files in code

## Fallback Mode

When Dagger.io is not available (e.g., in some environments), the tool operates in fallback mode:

```python
if result.metadata.get("fallback", False):
    print("Running in fallback mode")
    # Output will show the code instead of execution results
```

In fallback mode:
- Code is displayed instead of executed
- Useful for code review and debugging
- No actual execution occurs

## Performance Considerations

### Container Startup
- First execution may be slower (pulling images)
- Subsequent executions use cached images
- Consider keeping containers warm for production

### Package Installation
- Package installation adds overhead
- Cache common packages in custom images
- Use requirements files for complex dependencies

### Memory and CPU
- Set appropriate limits for workload
- Monitor resource usage
- Scale horizontally for parallel executions

## MCP Tools

The MCP server provides these tools:

- `execution_execute_code` - Execute code with full options
- `execution_run_python` - Simplified Python execution
- `execution_run_bash` - Simplified Bash execution

## PydanticAI Functions

Available typed functions:

- `execute_code(ctx, language, code, ...)` - Full execution control
- `execute_python(ctx, code, packages, timeout)` - Python shortcuts
- `execute_bash(ctx, script, packages, timeout)` - Bash shortcuts
- `execute_javascript(ctx, code, packages, timeout)` - JavaScript shortcuts

## Integration Examples

### With FileOperationsTool

```python
from aida.tools.execution import ExecutionTool
from aida.tools.files import FileOperationsTool

exec_tool = ExecutionTool()
file_tool = FileOperationsTool()

# Read data file
data_result = await file_tool.execute(
    operation="read_file",
    path="/data/input.csv"
)

# Process with Python
exec_result = await exec_tool.execute(
    language="python",
    code="""
import pandas as pd
import io

# Data passed as file
df = pd.read_csv('input.csv')
print(f"Processing {len(df)} rows")

# Analysis
summary = df.describe()
print(summary)

# Save results
summary.to_csv('output.csv')
""",
    files={
        "input.csv": data_result.result["content"]
    },
    packages=["pandas"],
    timeout=120
)

# Save results
if exec_result.status == ToolStatus.COMPLETED:
    # Extract output.csv from stdout or handle differently
    pass
```

### Data Pipeline

```python
async def run_analysis_pipeline(data_path: str):
    tool = ExecutionTool()
    
    # Step 1: Data validation
    validation_result = await tool.execute(
        language="python",
        code="""
import json
with open('data.json', 'r') as f:
    data = json.load(f)
    
# Validate structure
required_fields = ['id', 'timestamp', 'value']
for item in data:
    for field in required_fields:
        assert field in item, f"Missing field: {field}"
        
print(f"Validated {len(data)} records")
""",
        files={
            "data.json": open(data_path).read()
        },
        timeout=30
    )
    
    if validation_result.result["exit_code"] != 0:
        raise ValueError("Data validation failed")
    
    # Step 2: Process data
    process_result = await tool.execute(
        language="python",
        code="""
import json
import statistics

with open('data.json', 'r') as f:
    data = json.load(f)

# Calculate statistics
values = [item['value'] for item in data]
stats = {
    'count': len(values),
    'mean': statistics.mean(values),
    'median': statistics.median(values),
    'stdev': statistics.stdev(values) if len(values) > 1 else 0
}

print(json.dumps(stats, indent=2))
""",
        files={
            "data.json": open(data_path).read()
        },
        timeout=60
    )
    
    return json.loads(process_result.result["stdout"])
```

## Troubleshooting

### Dagger Connection Issues
- Ensure Docker is running
- Check Dagger daemon status
- Fallback mode activates automatically

### Package Installation Failures
- Verify package names
- Check for system dependencies
- Use specific versions if needed

### Memory Limit Errors
- Increase memory_limit parameter
- Optimize code for memory usage
- Process data in chunks

### Timeout Errors
- Increase timeout for long operations
- Add progress indicators in code
- Consider breaking into smaller tasks

### Container Startup Failures
- Check Docker permissions
- Verify image availability
- Check disk space

## Related Documentation

- [Hybrid Architecture Overview](./hybrid_architecture.md)
- [SystemTool Guide](./hybrid_system_tool.md)
- [FileOperationsTool Guide](./hybrid_file_operations_tool.md)
- [Dagger.io Documentation](https://docs.dagger.io/)