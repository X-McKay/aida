# File Operations Tool

## Overview
The File Operations Tool provides comprehensive file and directory management capabilities with built-in safety checks, encoding detection, and batch operations support. It's designed for secure file system interactions with configurable access controls.

## Features
- Read, write, append, and delete files
- Copy, move, and rename files/directories
- Directory listing and creation
- Text search within files
- File pattern matching (glob/regex)
- Batch operations support
- Automatic encoding detection
- Safe path validation
- Ignore patterns for common artifacts
- Full hybrid architecture support (AIDA, PydanticAI, MCP)

## Configuration
The tool uses the following configuration parameters:

- `MAX_FILE_SIZE`: 10MB limit for file operations
- `MAX_BATCH_SIZE`: 100 operations per batch
- `MAX_SEARCH_RESULTS`: 1000 results limit
- `DEFAULT_ENCODING`: UTF-8 with fallback options
- Safe paths: User home, /tmp, /var/tmp
- Forbidden paths: System directories (/etc, /sys, etc.)

## Usage Examples

### Basic File Operations
```python
from aida.tools.files import FileOperationsTool

tool = FileOperationsTool()

# Read a file
result = await tool.execute(
    operation="read",
    path="~/documents/readme.txt"
)
print(result.result)

# Write a file
result = await tool.execute(
    operation="write",
    path="~/documents/new_file.txt",
    content="Hello, World!",
    create_parents=True
)

# Append to file
result = await tool.execute(
    operation="append",
    path="~/documents/log.txt",
    content="\nNew log entry"
)
```

### Directory Operations
```python
# List directory
result = await tool.execute(
    operation="list_dir",
    path="~/projects",
    recursive=True
)
for entry in result.result:
    print(f"{entry['name']} - {'File' if entry['is_file'] else 'Dir'}")

# Create directory structure
result = await tool.execute(
    operation="create_dir",
    path="~/projects/new_project/src",
    create_parents=True
)
```

### Search and Find
```python
# Search for text in files
result = await tool.execute(
    operation="search",
    path="~/projects",
    search_text="TODO|FIXME",
    recursive=True
)
for file_result in result.result:
    print(f"{file_result['file']}: {file_result['matches']} matches")

# Find files by pattern
result = await tool.execute(
    operation="find",
    path="~/projects",
    pattern="*.py",
    recursive=True
)
print(f"Found {len(result.result)} Python files")
```

### File Manipulation
```python
# Copy file
result = await tool.execute(
    operation="copy",
    path="~/documents/original.txt",
    destination="~/backup/original_backup.txt",
    create_parents=True
)

# Move/rename file
result = await tool.execute(
    operation="move",
    path="~/documents/old_name.txt",
    destination="~/documents/new_name.txt"
)

# Edit file (search and replace)
result = await tool.execute(
    operation="edit",
    path="~/documents/config.json",
    search_text="localhost",
    replace_text="production.server.com"
)
print(f"Made {result.result['replacements']} replacements")
```

### Batch Operations
```python
# Execute multiple operations
result = await tool.execute(
    operation="batch",
    path="batch",
    batch_operations=[
        {
            "operation": "create_dir",
            "path": "~/temp/batch_test"
        },
        {
            "operation": "write",
            "path": "~/temp/batch_test/file1.txt",
            "content": "First file"
        },
        {
            "operation": "write",
            "path": "~/temp/batch_test/file2.txt",
            "content": "Second file"
        }
    ]
)
```

### PydanticAI Integration
```python
from pydantic_ai import Agent
from aida.tools.files import FileOperationsTool

tool = FileOperationsTool()
agent = Agent(
    "You are a file management assistant",
    tools=tool.to_pydantic_tools()
)

# Agent can now read, write, search files
response = await agent.run("Find all Python files in the project")
```

### MCP Server Usage
```python
mcp_server = tool.get_mcp_server()

# Read file via MCP
result = await mcp_server.call_tool(
    "files_read",
    {
        "path": "~/documents/data.json",
        "encoding": "utf-8"
    }
)
```

## Operations

### read
**Description**: Read file contents
**Parameters**:
- `path` (str, required): File path to read
- `encoding` (str, optional): File encoding (default: "utf-8")

**Returns**: File contents as string

### write
**Description**: Write content to file
**Parameters**:
- `path` (str, required): File path to write
- `content` (str, required): Content to write
- `encoding` (str, optional): File encoding (default: "utf-8")
- `create_parents` (bool, optional): Create parent directories (default: true)

**Returns**: Success status

### append
**Description**: Append content to file
**Parameters**:
- `path` (str, required): File path
- `content` (str, required): Content to append
- `encoding` (str, optional): File encoding

**Returns**: Success status

### delete
**Description**: Delete file or directory
**Parameters**:
- `path` (str, required): Path to delete
- `recursive` (bool, optional): Delete directories recursively

**Returns**: Number of files deleted

### copy
**Description**: Copy file or directory
**Parameters**:
- `path` (str, required): Source path
- `destination` (str, required): Destination path
- `recursive` (bool, optional): Copy directories recursively
- `create_parents` (bool, optional): Create parent directories

**Returns**: Destination path and files copied

### move
**Description**: Move or rename file/directory
**Parameters**:
- `path` (str, required): Source path
- `destination` (str, required): Destination path
- `create_parents` (bool, optional): Create parent directories

**Returns**: Destination path and files moved

### create_dir
**Description**: Create directory
**Parameters**:
- `path` (str, required): Directory path
- `create_parents` (bool, optional): Create parent directories

**Returns**: Success status

### list_dir
**Description**: List directory contents
**Parameters**:
- `path` (str, required): Directory path
- `recursive` (bool, optional): List recursively

**Returns**: Array of file/directory entries

### search
**Description**: Search for text in files
**Parameters**:
- `path` (str, required): Path to search in
- `search_text` (str, required): Text or regex pattern
- `recursive` (bool, optional): Search recursively

**Returns**: Array of files with matches

### find
**Description**: Find files by pattern
**Parameters**:
- `path` (str, required): Path to search in
- `pattern` (str, required): Glob pattern (e.g., "*.py")
- `recursive` (bool, optional): Search recursively

**Returns**: Array of matching file paths

### get_info
**Description**: Get detailed file information
**Parameters**:
- `path` (str, required): File or directory path

**Returns**: File metadata (size, dates, permissions, etc.)

### edit
**Description**: Edit file with search and replace
**Parameters**:
- `path` (str, required): File path
- `search_text` (str, required): Text to find
- `replace_text` (str, required): Replacement text

**Returns**: Number of replacements made

### batch
**Description**: Execute multiple operations
**Parameters**:
- `batch_operations` (list, required): Array of operation definitions

**Returns**: Results for each operation

## Security Considerations

### Path Validation
- Only allowed paths: User home, /tmp, /var/tmp
- Forbidden paths: System directories
- Path traversal prevention

### File Size Limits
- Maximum file size: 10MB
- Prevents memory exhaustion
- Large file handling requires chunking

### Encoding Safety
- Automatic encoding detection
- Fallback encodings for compatibility
- Binary file detection

## Error Handling
Common errors and solutions:

- **FileNotFoundError**: Path doesn't exist
  - Solution: Check path or use create_parents=true
- **PermissionError**: Insufficient permissions
  - Solution: Check file permissions and ownership
- **ValueError**: Invalid operation or parameters
  - Solution: Verify operation name and required parameters
- **UnicodeDecodeError**: Encoding mismatch
  - Solution: Specify correct encoding or let auto-detection work

## Performance Considerations
- Large directory listings are paginated
- Search operations limited to 1000 results
- Batch operations process sequentially
- Text file detection based on extensions
- Recursive operations may be slow on large trees

## Best Practices
1. **Always validate paths**: Use safe base paths
2. **Handle encodings**: Let auto-detection work or specify explicitly
3. **Use batch operations**: For multiple related operations
4. **Set recursive carefully**: Can affect many files
5. **Check file existence**: Before operations that require it
6. **Use patterns efficiently**: Glob patterns are faster than regex

## Ignored Patterns
The following are automatically ignored:
- Python: `*.pyc`, `__pycache__`, `*.pyo`
- Version control: `.git`, `.svn`, `.hg`
- Editors: `*.swp`, `*.swo`, `*~`
- OS: `.DS_Store`, `Thumbs.db`
- Dependencies: `node_modules`, `venv`, `.env`

## Dependencies
- No external dependencies
- Uses Python stdlib: pathlib, shutil, os
- Optional: chardet for better encoding detection

## Changelog
- **2.0.0**: Complete rewrite with modular architecture
- **2.0.1**: Added batch operations support
- **2.0.2**: Improved encoding detection
- **2.0.3**: Enhanced security with path validation
