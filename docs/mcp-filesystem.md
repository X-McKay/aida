# MCP Filesystem Integration

AIDA now supports using the official Model Context Protocol (MCP) filesystem server as a backend for file operations. This provides standardized, secure file access with better interoperability.

## Overview

The MCP filesystem integration allows AIDA to use the official `@modelcontextprotocol/server-filesystem` package instead of the built-in file operations implementation. This reduces code complexity and leverages a well-tested, community-maintained solution.

## Benefits

- **Standards Compliance**: Uses the official MCP protocol for file operations
- **Reduced Maintenance**: Leverages community-maintained code
- **Better Security**: Built-in access controls and sandboxing
- **Interoperability**: Compatible with other MCP-enabled tools

## Configuration

### Enabling MCP Backend

Set the environment variable to enable MCP filesystem backend:

```bash
export AIDA_FILES_USE_MCP=true
```

### Specifying Allowed Directories

When creating a `FileOperationsTool` instance, specify allowed directories:

```python
from aida.tools.files import FileOperationsTool

# Use MCP backend with specific directories
tool = FileOperationsTool(allowed_directories=[
    "/home/user/projects",
    "/tmp/workspace"
])
```

If no directories are specified, the tool uses the configured safe paths from `FilesConfig.SAFE_PATHS`.

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

## Supported Operations

All standard file operations are supported through the MCP backend:

- **read**: Read file contents
- **write**: Write content to file
- **append**: Append content to existing file
- **delete**: Delete files or directories
- **copy**: Copy files (implemented via read + write)
- **move**: Move or rename files
- **create_dir**: Create directories
- **list_dir**: List directory contents
- **search**: Search for text in files
- **find**: Find files by pattern
- **get_info**: Get file metadata
- **edit**: Edit files with search/replace

## Testing

Run the MCP filesystem test script:

```bash
python tests/scripts/test_mcp_filesystem.py
```

This will test all file operations using both MCP and native backends for comparison.

## Migration Guide

### For Existing Code

No changes needed! The `FileOperationsTool` interface remains the same. Just set the environment variable to enable MCP backend.

### For New Implementations

```python
import os
from aida.tools.files import FileOperationsTool

# Enable MCP backend
os.environ["AIDA_FILES_USE_MCP"] = "true"

# Create tool with allowed directories
tool = FileOperationsTool(allowed_directories=[
    "/path/to/project",
    "/tmp/workspace"
])

# Use as normal
result = await tool.execute(
    operation="read",
    path="/path/to/project/file.txt"
)
```

## Performance Considerations

- Initial startup is slower due to Node.js process spawn
- Subsequent operations have minimal overhead
- File operations are generally I/O bound, so protocol overhead is negligible

## Security

The MCP filesystem server provides:

- Directory access control (only specified directories are accessible)
- Path traversal protection
- Read-only mount options (for Docker)
- Process isolation

## Troubleshooting

### Node.js Not Found

Ensure Node.js is installed:
```bash
# Check Node.js
node --version

# Check npx
npx --version
```

### Permission Denied

Ensure the AIDA process has permission to:
1. Execute npx
2. Access the specified directories
3. Create subprocess

### MCP Server Fails to Start

Check logs for detailed error messages. Common issues:
- Network connectivity (for package download)
- Insufficient permissions
- Port conflicts (if using network transport)

## Future Enhancements

- Support for MCP roots protocol for dynamic directory management
- Integration with MCP resource exposure
- Performance optimizations for batch operations
- Support for additional MCP filesystem servers (Go, Rust implementations)
