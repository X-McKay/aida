# Hybrid SystemTool Guide

## Overview

The SystemTool provides secure command execution capabilities with a hybrid architecture supporting multiple AI frameworks. It includes built-in security controls, script execution, and system information gathering.

**Version:** 2.0.0

## Key Features

- **Secure Command Execution**: Built-in security validation and dangerous command blocking
- **Script Execution**: Support for Bash, Python, PowerShell, and Batch scripts
- **Cross-Platform**: Works on Linux, macOS, and Windows
- **Hybrid Architecture**: Compatible with AIDA, PydanticAI, MCP, and OpenTelemetry
- **Resource Management**: Timeout controls and output capture
- **System Information**: Platform details, resource usage, and health checks

## Architecture

The SystemTool implements a hybrid architecture that supports:

1. **Core Tool Interface** - Primary execution method
2. **PydanticAI Tools** - Clean, typed functions for modern AI agents
3. **MCP Server** - Universal AI compatibility via Model Context Protocol
4. **OpenTelemetry** - Production-ready observability

## Usage Examples

### 1. Core Interface

```python
from aida.tools.system import SystemTool

tool = SystemTool()

# Execute a simple command
result = await tool.execute(
    command="echo",
    args=["Hello, World!"]
)

print(f"Status: {result.status}")
print(f"Output: {result.result['stdout']}")
print(f"Exit code: {result.result['exit_code']}")

# Execute with working directory
result = await tool.execute(
    command="ls",
    working_directory="/home/user/projects"
)

# Execute with timeout and environment
result = await tool.execute(
    command="python",
    args=["script.py"],
    environment={"PYTHONPATH": "/custom/path"},
    timeout=60,
    capture_output=True
)
```

### 2. PydanticAI Interface

```python
from aida.tools.system import SystemTool
from pydantic_ai import Agent

# Get PydanticAI-compatible tools
tool = SystemTool()
tools = tool.to_pydantic_tools()

# Use with PydanticAI agent
agent = Agent("gpt-4")
tool.register_with_pydantic_agent(agent)

# Or use functions directly
result = await tools["execute_command"](
    command="git",
    args=["status"],
    working_dir="/path/to/repo"
)

# Check if command exists
exists = await tools["check_command"]("docker")
print(f"Docker installed: {exists}")

# Get system information
info = await tools["get_system_info"]()
print(f"Platform: {info['platform']['system']}")
print(f"Python: {info['platform']['python_version']}")
```

### 3. MCP Server Interface

```python
from aida.tools.system import SystemTool

tool = SystemTool()
mcp_server = tool.get_mcp_server()

# Execute command via MCP
result = await mcp_server.call_tool("system_execute_command", {
    "command": "npm",
    "args": ["install"],
    "working_directory": "/project",
    "timeout": 120
})

# Run health check
health = await mcp_server.call_tool("system_health_check", {})
```

### 4. Script Execution

```python
# Execute a Bash script
bash_script = """#!/bin/bash
echo "Starting deployment..."
git pull origin main
npm install
npm run build
echo "Deployment complete!"
"""

result = await tool.execute_script(
    script_content=bash_script,
    language="bash",
    working_directory="/app"
)

# Execute a Python script
python_script = """
import os
import json

data = {"status": "ready", "version": "1.0.0"}
print(json.dumps(data))
"""

result = await tool.execute_script(
    script_content=python_script,
    language="python"
)
```

### 5. With Observability

```python
# Enable tracing
observability = tool.enable_observability({
    "enabled": True,
    "service_name": "my-automation",
    "endpoint": "http://localhost:4317"
})

# Traced operations
with observability.trace_operation("deploy", environment="production"):
    result = await tool.execute(
        command="./deploy.sh",
        args=["--prod"]
    )
```

## Security Features

### Dangerous Command Blocking

The following commands are automatically blocked:
- `rm`, `del`, `format`, `fdisk`, `mkfs`
- `dd`, `sudo`, `su`, `chmod`, `chown`
- `passwd`, `useradd`, `userdel`, `usermod`
- `systemctl`, `service`, `reboot`, `shutdown`
- `halt`, `poweroff`, `init`, `kill`, `killall`

### Allowed Commands List

Restrict execution to specific commands:

```python
result = await tool.execute(
    command="npm",
    args=["test"],
    allowed_commands=["npm", "yarn", "pnpm"]
)
```

### Pattern Detection

The tool detects dangerous patterns like:
- Path traversal (`../`)
- Privilege escalation (`sudo`, `su -`)
- Destructive operations (`rm -rf`)

## Parameters

### execute() Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `command` | str | Yes | - | Command to execute |
| `args` | list | No | [] | Command arguments |
| `working_directory` | str | No | Current | Working directory |
| `environment` | dict | No | {} | Environment variables |
| `timeout` | int | No | 30 | Timeout in seconds (1-300) |
| `capture_output` | bool | No | True | Capture stdout/stderr |
| `shell` | bool | No | False | Execute through shell |
| `input_data` | str | No | None | Data for stdin |
| `allowed_commands` | list | No | None | Whitelist of commands |

### execute_script() Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `script_content` | str | Yes | - | Script content |
| `language` | str | No | "bash" | Script language |
| `**kwargs` | - | No | - | Same as execute() |

Supported languages: `bash`, `python`, `powershell`, `batch`

## Return Values

### ToolResult Structure

```python
{
    "tool_name": "system",
    "execution_id": "unique-id",
    "status": ToolStatus.COMPLETED,  # or FAILED
    "result": {
        "exit_code": 0,
        "stdout": "command output",
        "stderr": "error output",
        "execution_time": 1.23,
        "command": "echo test",
        "working_directory": "/current/dir",
        "success": true
    },
    "error": None,  # Error message if failed
    "started_at": datetime,
    "completed_at": datetime,
    "duration_seconds": 1.23,
    "metadata": {
        "command": "echo",
        "exit_code": 0,
        "execution_time": 1.23,
        "security_validated": true
    }
}
```

## MCP Tools

The MCP server provides these tools:

- `system_execute_command` - Execute system commands
- `system_run_script` - Run scripts
- `system_check_command` - Check command existence
- `system_get_info` - Get system information
- `system_health_check` - Run health checks

## Best Practices

1. **Always validate user input** before passing to commands
2. **Use timeout** for long-running commands
3. **Capture output** for debugging and logging
4. **Specify working directory** explicitly when needed
5. **Use allowed_commands** for additional security
6. **Handle errors** gracefully with try/except
7. **Clean up** temporary files in scripts

## Error Handling

```python
try:
    result = await tool.execute(
        command="some-command",
        args=["--flag"]
    )

    if result.status == ToolStatus.COMPLETED:
        print(f"Success: {result.result['stdout']}")
    else:
        print(f"Failed: {result.error}")

except Exception as e:
    print(f"Execution error: {e}")
```

## Platform Differences

### Linux/macOS
- Shell: `/bin/bash`
- Path separator: `/`
- Commands: `ls`, `pwd`, `which`

### Windows
- Shell: `cmd.exe`
- Path separator: `\`
- Commands: `dir`, `cd`, `where`

The tool automatically handles these differences.

## Performance Considerations

- **Overhead**: Hybrid architecture adds <5% overhead
- **Timeout**: Default 30s, max 300s
- **Output Size**: Large outputs may impact performance
- **Concurrent Execution**: Supports parallel commands

## Troubleshooting

### Command Not Found
- Check if command exists: `await tool.check_command_exists("cmd")`
- Verify PATH environment variable
- Use full path to executable

### Permission Denied
- Check file permissions
- Verify user has execute rights
- Some commands may be blocked by security

### Timeout Errors
- Increase timeout parameter
- Consider breaking into smaller operations
- Use background execution for long tasks

## Integration Examples

### With Todo Orchestrator

```python
from aida.core.orchestrator import TodoOrchestrator
from aida.tools.system import SystemTool

orchestrator = TodoOrchestrator()
system_tool = SystemTool()

# Register tool
orchestrator.register_tool("system", system_tool)

# Use in plan
plan = await orchestrator.create_plan(
    "Run tests and deploy if successful"
)
```

### With GitHub Actions

```yaml
- name: Run AIDA System Commands
  run: |
    python -c "
    import asyncio
    from aida.tools.system import SystemTool

    async def main():
        tool = SystemTool()
        result = await tool.execute(
            command='npm',
            args=['test']
        )
        exit(result.result['exit_code'])

    asyncio.run(main())
    "
```


## Related Documentation

- [Hybrid Architecture Overview](./hybrid_architecture.md)
- [FileOperationsTool Guide](./hybrid_file_operations_tool.md)
- [Security Best Practices](./security.md)
- [MCP Integration](./mcp_integration.md)
