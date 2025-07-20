# System Tool

## Overview
The System Tool provides secure command execution and system management capabilities with built-in security controls, command whitelisting, and comprehensive process management. It's designed for safe system interactions with configurable restrictions.

## Features
- Secure command execution with whitelisting
- Process listing and management
- System information gathering
- Environment variable management
- Command location resolution (which)
- Script execution with interpreter control
- Timeout and output size limits
- Dangerous pattern detection
- Full hybrid architecture support (AIDA, PydanticAI, MCP)

## Configuration
The tool uses the following configuration parameters:

- `DEFAULT_TIMEOUT`: 30 seconds
- `MAX_TIMEOUT`: 300 seconds (5 minutes)
- `MAX_OUTPUT_SIZE`: 1MB per stream
- Whitelisted commands only (configurable)
- Forbidden commands blocked
- Sensitive environment variables filtered

## Usage Examples

### Basic Command Execution
```python
from aida.tools.system import SystemTool

tool = SystemTool()

# Run a simple command
result = await tool.execute(
    operation="execute",
    command="ls",
    args=["-la", "/tmp"]
)
print(result.result.stdout)

# Run with timeout
result = await tool.execute(
    operation="execute",
    command="ping",
    args=["-c", "5", "google.com"],
    timeout=10
)
```

### Process Management
```python
# List all processes
result = await tool.execute(operation="process_list")
for proc in result.result[:10]:  # First 10 processes
    print(f"{proc.pid}: {proc.name} ({proc.status})")

# Get specific process info
result = await tool.execute(
    operation="process_info",
    pid=1234
)
if result.status == "completed":
    proc = result.result
    print(f"Process {proc.name} using {proc.memory_percent:.1f}% memory")

# Kill a process (safely)
result = await tool.execute(
    operation="process_kill",
    pid=1234,
    signal="TERM"
)
```

### System Information
```python
# Get comprehensive system info
result = await tool.execute(operation="system_info")
info = result.result
print(f"Platform: {info.platform}")
print(f"CPU Count: {info.cpu_count}")
print(f"Memory: {info.memory_available / 1024**3:.1f}GB available")
print(f"Disk Usage:")
for mount, usage in info.disk_usage.items():
    print(f"  {mount}: {usage['percent']}% used")
```

### Environment Variables
```python
# Get specific environment variable
result = await tool.execute(
    operation="env_get",
    var_name="PATH"
)
print(f"PATH: {result.result}")

# Get all environment variables (filtered)
result = await tool.execute(operation="env_get")
for key, value in result.result.items():
    print(f"{key}={value}")

# Set environment variable (current process only)
result = await tool.execute(
    operation="env_set",
    var_name="MY_VAR",
    var_value="test_value"
)
```

### Command Resolution
```python
# Find command location
result = await tool.execute(
    operation="which",
    command="python3"
)
if result.status == "completed":
    print(f"Python3 found at: {result.result}")
```

### Script Execution
```python
# Execute a Python script
result = await tool.execute(
    operation="script",
    script_content="""
import sys
print(f"Python version: {sys.version}")
print("Hello from script!")
""",
    interpreter="python3"
)
print(result.result.stdout)

# Execute a bash script
result = await tool.execute(
    operation="script",
    script_content="""
#!/bin/bash
echo "Current directory: $(pwd)"
echo "User: $(whoami)"
ls -la
""",
    interpreter="bash"
)
```

### PydanticAI Integration
```python
from pydantic_ai import Agent
from aida.tools.system import SystemTool

tool = SystemTool()
agent = Agent(
    "You are a system administrator assistant",
    tools=tool.to_pydantic_tools()
)

# Agent can now execute commands, check processes, etc.
response = await agent.run("Check if nginx is running")
```

### MCP Server Usage
```python
mcp_server = tool.get_mcp_server()

# Execute command via MCP
result = await mcp_server.call_tool(
    "system_execute",
    {
        "command": "df",
        "args": ["-h"],
        "timeout": 5
    }
)
```

## Operations

### execute
**Description**: Execute a system command
**Parameters**:
- `command` (str, required): Command to execute
- `args` (list, optional): Command arguments
- `cwd` (str, optional): Working directory
- `env` (dict, optional): Environment variables
- `timeout` (int, optional): Timeout in seconds (default: 30)
- `shell` (bool, optional): Execute in shell (default: false)

**Returns**: CommandResult with exit_code, stdout, stderr

### shell
**Description**: Execute a shell command
**Parameters**: Same as execute, but shell=true by default

### process_list
**Description**: List all running processes
**Parameters**: None

**Returns**: List of ProcessInfo objects

### process_info
**Description**: Get information about a specific process
**Parameters**:
- `pid` (int, required): Process ID

**Returns**: ProcessInfo object

### process_kill
**Description**: Send signal to a process
**Parameters**:
- `pid` (int, required): Process ID
- `signal` (str, optional): Signal name (default: "TERM")

**Returns**: Success message

### system_info
**Description**: Get comprehensive system information
**Parameters**: None

**Returns**: SystemInfo object with platform, CPU, memory, disk usage

### env_get
**Description**: Get environment variables
**Parameters**:
- `var_name` (str, optional): Specific variable name

**Returns**: Variable value or all variables (filtered)

### env_set
**Description**: Set environment variable (current process only)
**Parameters**:
- `var_name` (str, required): Variable name
- `var_value` (str, required): Variable value

**Returns**: Success message

### which
**Description**: Find command in PATH
**Parameters**:
- `command` (str, required): Command to find

**Returns**: Full path to command

### script
**Description**: Execute a script with specified interpreter
**Parameters**:
- `script_content` (str, required): Script content
- `interpreter` (str, optional): Interpreter (default: "bash")
- Other parameters from execute operation

**Returns**: CommandResult with script output

## Security Features

### Command Whitelisting
Only approved commands can be executed:
- System info: `uname`, `hostname`, `date`, `uptime`
- File operations: `ls`, `cat`, `grep`, `find`
- Network: `ping`, `curl`, `wget` (limited)
- Development: `git`, `python`, `node`, etc.

### Forbidden Commands
These commands are always blocked:
- Destructive: `rm`, `rmdir`, `dd`, `mkfs`
- System control: `shutdown`, `reboot`, `systemctl`
- User management: `useradd`, `passwd`, `sudo`
- Process control: `kill` (use process_kill instead)

### Dangerous Pattern Detection
Commands are scanned for dangerous patterns:
- Output redirection to device files
- Command injection attempts
- Piping to sudo
- Command substitution
- Curl/wget piped to shell

### Environment Variable Filtering
Sensitive variables are automatically filtered:
- AWS credentials
- API keys and tokens
- Database URLs
- Passwords and secrets

## Error Handling
Common errors and solutions:

- **Command not allowed**: Command is not in whitelist
  - Solution: Use only approved commands
- **Timeout**: Command exceeded time limit
  - Solution: Increase timeout or optimize command
- **Process not found**: PID doesn't exist
  - Solution: Verify process exists with process_list
- **Permission denied**: Insufficient privileges
  - Solution: Check user permissions

## Performance Considerations
- Commands run in subprocess with overhead
- Large outputs are truncated to 1MB
- Process listing can be slow with many processes
- System info gathering may take time
- Timeout prevents hanging commands

## Best Practices
1. **Use specific commands**: Avoid shell=true when possible
2. **Set appropriate timeouts**: Don't use maximum timeout unnecessarily
3. **Handle exit codes**: Check command success/failure
4. **Validate inputs**: Sanitize user-provided arguments
5. **Monitor resources**: Watch for high CPU/memory processes
6. **Use process management**: Prefer process_kill over kill command

## Allowed Signals
For process_kill operation:
- `TERM`: Graceful termination (default)
- `INT`: Interrupt (Ctrl+C)
- `HUP`: Hangup
- `USR1`, `USR2`: User-defined signals

## Dependencies
- psutil: Process and system information
- Standard library: subprocess, os, sys
- No external command dependencies

## Changelog
- **2.0.0**: Complete rewrite with modular architecture
- **2.0.1**: Added script execution support
- **2.0.2**: Enhanced security with pattern detection
- **2.0.3**: Improved process management features