# MCP Filesystem Integration

AIDA uses the official Model Context Protocol (MCP) filesystem server for all file operations. This provides standardized, secure file access with excellent interoperability.

## Overview

The file operations tool leverages the official `@modelcontextprotocol/server-filesystem` package from Anthropic. This significantly reduces code complexity while providing a robust, well-tested solution for file system access.

## Benefits

- **Standards Compliance**: Uses the official MCP protocol
- **Reduced Complexity**: ~400 lines of code vs ~700 lines for custom implementation
- **Better Security**: Built-in access controls and sandboxing
- **Active Maintenance**: Community-maintained by Anthropic
- **Interoperability**: Works with any MCP-compatible system

## Requirements

- Node.js and npm/npx installed
- Internet connection (for initial package download)

The MCP server package is automatically downloaded via npx when first used.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│                 │     │                  │     │                 │
│ FileOperations  │────▶│ MCPFilesystem    │────▶│ @modelcontext   │
│     Tool        │     │    Client        │     │ protocol/server │
│                 │     │                  │     │  -filesystem    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                                                  │
        │                                                  │
        ▼                                                  ▼
┌─────────────────┐                              ┌─────────────────┐
│   AIDA Agent    │                              │   File System   │
│   Interface     │                              │   (sandboxed)   │
└─────────────────┘                              └─────────────────┘
```

## Usage

### Basic Usage

```python
from aida.tools.files import FileOperationsTool

# Create tool with allowed directories
tool = FileOperationsTool(allowed_directories=[
    "/home/user/projects",
    "/tmp/workspace"
])

# Read a file
result = await tool.execute(
    operation="read",
    path="/home/user/projects/file.txt"
)

# Write a file
result = await tool.execute(
    operation="write",
    path="/tmp/workspace/output.txt",
    content="Hello from MCP!"
)
```

### Specifying Allowed Directories

By default, the tool uses safe paths from configuration. You can override:

```python
# Use specific directories
tool = FileOperationsTool(allowed_directories=[
    "/path/to/project",
    "/tmp/safe-area"
])

# Use default safe paths (home, /tmp, /var/tmp)
tool = FileOperationsTool()
```

## Supported Operations

All operations are handled through the MCP filesystem server:

### Direct MCP Operations
- **read**: Read file contents
- **write**: Write content to file
- **delete**: Delete files or directories
- **move**: Move or rename files
- **create_dir**: Create directories
- **list_dir**: List directory contents
- **get_info**: Get file metadata
- **edit**: Edit files with find/replace

### Composite Operations
These operations use multiple MCP calls:
- **append**: Read + write to append content
- **copy**: Read + write to copy files
- **search**: List + read to search in files
- **find**: List + filter to find files by pattern
- **batch**: Execute multiple operations

## Testing

Run the test script to verify functionality:

```bash
python tests/scripts/test_mcp_filesystem.py
```

## Security

The MCP filesystem server provides:
- **Directory Access Control**: Only specified directories are accessible
- **Path Traversal Protection**: Prevents access outside allowed paths
- **Process Isolation**: Runs in separate Node.js process
- **No Direct File Access**: All operations go through MCP protocol

## Performance

- Initial startup: ~1-2 seconds (Node.js process spawn)
- Subsequent operations: Minimal overhead
- File operations are I/O bound, protocol overhead is negligible

## Troubleshooting

### Node.js Not Found

```bash
# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version
npx --version
```

### Permission Denied

Ensure the process has permission to:
1. Execute npx
2. Access specified directories
3. Create subprocess

### MCP Server Fails to Start

Check logs for errors. Common issues:
- Network connectivity (for package download)
- Firewall blocking npm registry
- Insufficient disk space

## Implementation Details

The FileOperationsTool:
1. Spawns MCP filesystem server via npx
2. Communicates using JSON-RPC over stdin/stdout
3. Translates AIDA operations to MCP tools
4. Handles composite operations with multiple calls
5. Manages connection lifecycle

## Future Enhancements

- WebSocket transport for better performance
- Caching for frequently accessed files
- Batch operation optimizations
- Additional MCP server integrations
