# Hybrid FileOperationsTool Guide

## Overview

The FileOperationsTool provides comprehensive file and directory operations with a hybrid architecture supporting multiple AI frameworks. It includes advanced features like pattern matching, safe file operations, and content search capabilities.

**Version:** 2.0.0

## Key Features

- **File Operations**: Read, write, create, delete, move, copy files
- **Directory Management**: Create, list, remove directories
- **Pattern Matching**: Glob patterns and recursive search
- **Content Search**: Search within files with regex support
- **Safe Operations**: Automatic backups and atomic writes
- **Batch Operations**: Process multiple files efficiently
- **Hybrid Architecture**: Compatible with AIDA, PydanticAI, MCP, and OpenTelemetry

## Architecture

The FileOperationsTool implements a hybrid architecture that supports:

1. **Core Tool Interface** - Primary execution method
2. **PydanticAI Tools** - Clean, typed functions for modern AI agents
3. **MCP Server** - Universal AI compatibility via Model Context Protocol
4. **OpenTelemetry** - Production-ready observability

## Usage Examples

### 1. Core Interface

```python
from aida.tools.files import FileOperationsTool

tool = FileOperationsTool()

# Read a file
result = await tool.execute(
    operation="read_file",
    path="/path/to/file.txt"
)
print(f"Content: {result.result['content']}")
print(f"Lines: {result.result['line_count']}")

# Write a file
result = await tool.execute(
    operation="write_file",
    path="/path/to/output.txt",
    content="Hello, World!",
    create_dirs=True
)

# List files with pattern
result = await tool.execute(
    operation="list_files",
    path="/project",
    pattern="*.py",
    recursive=True
)
for file in result.result['files']:
    print(f"{file['name']} - {file['size']} bytes")
```

### 2. PydanticAI Interface

```python
from aida.tools.files import FileOperationsTool
from pydantic_ai import Agent

# Get PydanticAI-compatible tools
tool = FileOperationsTool()
tools = tool.to_pydantic_tools()

# Use with PydanticAI agent
agent = Agent("gpt-4")
tool.register_with_pydantic_agent(agent)

# Or use functions directly
content = await tools["read_file"]("/path/to/config.json")
data = json.loads(content["content"])

# Write with automatic directory creation
await tools["write_file"](
    "/new/path/file.txt",
    "Content here"
)

# Create directory structure
await tools["create_directory"]("/project/src/components")

# List files
files = await tools["list_files"]("/project", pattern="*.js")
```

### 3. MCP Server Interface

```python
from aida.tools.files import FileOperationsTool

tool = FileOperationsTool()
mcp_server = tool.get_mcp_server()

# Read file via MCP
result = await mcp_server.call_tool("file_read_file", {
    "path": "/path/to/document.md"
})

# Write file via MCP
result = await mcp_server.call_tool("file_write_file", {
    "path": "/output/report.txt",
    "content": "Analysis complete"
})

# List directory via MCP
result = await mcp_server.call_tool("file_list_files", {
    "path": "/data",
    "pattern": "*.csv"
})
```

### 4. Advanced Operations

```python
# Search in files
result = await tool.execute(
    operation="search_in_files",
    path="/src",
    pattern="TODO|FIXME",
    file_pattern="*.py",
    case_sensitive=False
)

# Copy with backup
result = await tool.execute(
    operation="copy_file",
    source="/important.conf",
    destination="/backup/important.conf",
    create_backup=True
)

# Batch operations
files_to_process = ["file1.txt", "file2.txt", "file3.txt"]
for filepath in files_to_process:
    result = await tool.execute(
        operation="read_file",
        path=filepath
    )
    # Process content...

# Safe write with atomic operation
result = await tool.execute(
    operation="write_file",
    path="/critical/config.json",
    content=json.dumps(config_data),
    atomic=True,
    create_backup=True
)
```

### 5. With Observability

```python
# Enable tracing
observability = tool.enable_observability({
    "enabled": True,
    "service_name": "file-processor",
    "endpoint": "http://localhost:4317"
})

# Traced operations
with observability.trace_operation("bulk_process", file_count=100):
    for file in files:
        result = await tool.execute(
            operation="read_file",
            path=file
        )
        # Process...
```

## Operations

### File Operations

| Operation | Description | Key Parameters |
|-----------|-------------|----------------|
| `read_file` | Read file content | `path`, `encoding`, `offset`, `limit` |
| `write_file` | Write content to file | `path`, `content`, `encoding`, `create_dirs`, `atomic` |
| `delete_file` | Delete a file | `path`, `create_backup` |
| `copy_file` | Copy file | `source`, `destination`, `overwrite` |
| `move_file` | Move/rename file | `source`, `destination`, `overwrite` |
| `get_file_info` | Get file metadata | `path` |

### Directory Operations

| Operation | Description | Key Parameters |
|-----------|-------------|----------------|
| `create_directory` | Create directory | `path`, `parents` |
| `list_files` | List directory contents | `path`, `pattern`, `recursive` |
| `delete_directory` | Remove directory | `path`, `recursive` |
| `copy_directory` | Copy directory tree | `source`, `destination` |

### Search Operations

| Operation | Description | Key Parameters |
|-----------|-------------|----------------|
| `search_in_files` | Search content | `path`, `pattern`, `file_pattern` |
| `find_files` | Find by name | `path`, `name_pattern`, `recursive` |

## Parameters

### Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str | File or directory path |
| `encoding` | str | Text encoding (default: utf-8) |
| `create_dirs` | bool | Create parent directories |
| `create_backup` | bool | Backup before modifying |
| `atomic` | bool | Use atomic write operations |

### Pattern Matching

- **Glob patterns**: `*.txt`, `**/*.py`, `[a-z]*.log`
- **Regex patterns**: For content search
- **Case sensitivity**: Configurable

## Return Values

### File Read Result

```python
{
    "path": "/absolute/path/to/file.txt",
    "content": "file contents",
    "lines": ["line1", "line2"],
    "line_count": 2,
    "character_count": 20,
    "encoding": "utf-8",
    "size_bytes": 20
}
```

### Directory Listing Result

```python
{
    "path": "/directory",
    "files": [
        {
            "name": "file.txt",
            "path": "/directory/file.txt",
            "size": 1024,
            "is_file": true,
            "is_directory": false,
            "modified_time": "2024-01-01T12:00:00",
            "permissions": "rw-r--r--"
        }
    ],
    "total_files": 10,
    "total_directories": 3,
    "total_size": 52428
}
```

## Safety Features

### Automatic Backups

Files are backed up before modification:
```python
# Original: config.json
# Backup: config.json.backup.1234567890
```

### Atomic Writes

Critical files are written atomically:
```python
result = await tool.execute(
    operation="write_file",
    path="/important.conf",
    content=data,
    atomic=True  # Write to temp, then rename
)
```

### Path Validation

- Prevents path traversal attacks
- Validates file permissions
- Checks disk space before write

## Best Practices

1. **Always use absolute paths** for clarity
2. **Enable backups** for important files
3. **Use atomic writes** for critical data
4. **Check file existence** before operations
5. **Handle encoding** explicitly for text files
6. **Use patterns** for batch operations
7. **Monitor disk space** for large operations

## Error Handling

```python
try:
    result = await tool.execute(
        operation="read_file",
        path="/path/to/file.txt"
    )

    if result.status == ToolStatus.COMPLETED:
        content = result.result["content"]
    else:
        print(f"Error: {result.error}")

except FileNotFoundError:
    print("File does not exist")
except PermissionError:
    print("No permission to access file")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

### Large Files

```python
# Read in chunks
result = await tool.execute(
    operation="read_file",
    path="/large/file.txt",
    offset=0,
    limit=1000  # Read first 1000 lines
)
```

### Batch Operations

```python
# Process multiple files efficiently
files = await tool.execute(
    operation="list_files",
    path="/data",
    pattern="*.csv"
)

for file_info in files.result["files"]:
    # Process each file
    pass
```

### Pattern Optimization

- Use specific patterns: `*.py` vs `*`
- Limit recursion depth when possible
- Pre-filter with file patterns

## MCP Tools

The MCP server provides these tools:

- `file_read_file` - Read file content
- `file_write_file` - Write content to file
- `file_create_directory` - Create directories
- `file_list_files` - List directory contents
- `file_delete_file` - Delete files
- `file_copy_file` - Copy files
- `file_move_file` - Move/rename files
- `file_search_in_files` - Search content

## PydanticAI Functions

Available typed functions:

- `read_file(path: str) -> Dict[str, Any]`
- `write_file(path: str, content: str) -> Dict[str, Any]`
- `create_directory(path: str) -> Dict[str, Any]`
- `list_files(path: str, pattern: str = None) -> Dict[str, Any]`

## Integration Examples

### With SystemTool

```python
from aida.tools.files import FileOperationsTool
from aida.tools.system import SystemTool

file_tool = FileOperationsTool()
system_tool = SystemTool()

# Read config, process, execute
config = await file_tool.execute(
    operation="read_file",
    path="/config/app.json"
)

data = json.loads(config.result["content"])

result = await system_tool.execute(
    command=data["command"],
    args=data["args"]
)

# Save results
await file_tool.execute(
    operation="write_file",
    path="/output/results.json",
    content=json.dumps(result.result)
)
```

### Data Processing Pipeline

```python
async def process_csv_files(directory: str):
    tool = FileOperationsTool()

    # Find all CSV files
    files = await tool.execute(
        operation="list_files",
        path=directory,
        pattern="*.csv",
        recursive=True
    )

    results = []
    for file_info in files.result["files"]:
        # Read each file
        content = await tool.execute(
            operation="read_file",
            path=file_info["path"]
        )

        # Process content
        processed = process_csv_content(content.result["content"])
        results.append(processed)

        # Write processed file
        output_path = file_info["path"].replace(".csv", "_processed.csv")
        await tool.execute(
            operation="write_file",
            path=output_path,
            content=processed
        )

    return results
```

## Troubleshooting

### Permission Errors
- Check file ownership and permissions
- Ensure parent directory is writable
- Run with appropriate user privileges

### Encoding Issues
- Specify encoding explicitly: `encoding="utf-8"`
- Use `errors="replace"` for problematic files
- Try different encodings: `latin-1`, `cp1252`

### Path Problems
- Use absolute paths when possible
- Normalize paths with `Path` objects
- Check path exists before operations

### Performance Issues
- Use patterns to filter files early
- Process large files in chunks
- Consider parallel processing for many files


## Related Documentation

- [Hybrid Architecture Overview](./hybrid_architecture.md)
- [SystemTool Guide](./hybrid_system_tool.md)
- [MCP Integration](./mcp_integration.md)
- [Performance Optimization](./performance.md)
