"""File operations tool suite for AIDA with hybrid architecture support."""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import json
import re
import logging
from datetime import datetime

from aida.tools.base import Tool, ToolResult, ToolCapability, ToolParameter


logger = logging.getLogger(__name__)


class FileOperationsTool(Tool):
    """Comprehensive file operations tool with hybrid architecture support.
    
    Supports:
    - Original AIDA tool interface
    - PydanticAI tool compatibility
    - MCP server integration
    - OpenTelemetry observability
    """
    
    def __init__(self):
        super().__init__(
            name="file_operations",
            description="Comprehensive file and directory operations with search and editing capabilities",
            version="2.0.0"
        )
        self._pydantic_tools_cache = {}
        self._mcp_server = None
        self._observability = None
    
    def get_capability(self) -> ToolCapability:
        return ToolCapability(
            name=self.name,
            version=self.version,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="operation",
                    type="str",
                    description="File operation to perform",
                    required=True,
                    choices=[
                        "list_files", "read_file", "write_file", "edit_file",
                        "search_files", "create_directory", "delete_file",
                        "copy_file", "move_file", "get_file_info", "find_files"
                    ]
                ),
                ToolParameter(
                    name="path",
                    type="str",
                    description="File or directory path",
                    required=True
                ),
                ToolParameter(
                    name="content",
                    type="str",
                    description="Content for write operations",
                    required=False
                ),
                ToolParameter(
                    name="search_query",
                    type="str",
                    description="Search query for content search",
                    required=False
                ),
                ToolParameter(
                    name="pattern",
                    type="str",
                    description="File name pattern for filtering",
                    required=False,
                    default="*"
                ),
                ToolParameter(
                    name="recursive",
                    type="bool",
                    description="Recursive operation",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="encoding",
                    type="str",
                    description="File encoding",
                    required=False,
                    default="utf-8"
                ),
                ToolParameter(
                    name="backup",
                    type="bool",
                    description="Create backup before modifying",
                    required=False,
                    default=True
                ),
                ToolParameter(
                    name="line_number",
                    type="int",
                    description="Line number for line-based operations",
                    required=False
                ),
                ToolParameter(
                    name="destination",
                    type="str",
                    description="Destination path for copy/move operations",
                    required=False
                )
            ],
            required_permissions=["file_system"],
            supported_platforms=["linux", "darwin", "windows"],
            dependencies=[]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        operation = kwargs["operation"]
        path = kwargs["path"]
        
        # Expand environment variables and user paths
        path = os.path.expandvars(os.path.expanduser(path))
        
        try:
            if operation == "list_files":
                result = await self._list_files(
                    path, 
                    kwargs.get("pattern", "*"),
                    kwargs.get("recursive", False)
                )
            elif operation == "read_file":
                result = await self._read_file(path, kwargs.get("encoding", "utf-8"))
            elif operation == "write_file":
                result = await self._write_file(
                    path, 
                    kwargs["content"],
                    kwargs.get("encoding", "utf-8"),
                    kwargs.get("backup", True)
                )
            elif operation == "edit_file":
                result = await self._edit_file(
                    path,
                    kwargs.get("content"),
                    kwargs.get("line_number"),
                    kwargs.get("backup", True),
                    kwargs.get("encoding", "utf-8")
                )
            elif operation == "search_files":
                result = await self._search_files(
                    path,
                    kwargs["search_query"],
                    kwargs.get("pattern", "*"),
                    kwargs.get("recursive", False)
                )
            elif operation == "create_directory":
                result = await self._create_directory(path)
            elif operation == "delete_file":
                result = await self._delete_file(path)
            elif operation == "copy_file":
                result = await self._copy_file(path, kwargs["destination"])
            elif operation == "move_file":
                result = await self._move_file(path, kwargs["destination"])
            elif operation == "get_file_info":
                result = await self._get_file_info(path)
            elif operation == "find_files":
                result = await self._find_files(
                    path,
                    kwargs.get("pattern", "*"),
                    kwargs.get("recursive", False)
                )
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return ToolResult(
                tool_name=self.name,
                execution_id="",
                status="completed",
                result=result,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                duration_seconds=0.1,
                metadata={
                    "operation": operation,
                    "path": path,
                    "files_processed": result.get("files_processed", 1)
                }
            )
            
        except Exception as e:
            raise Exception(f"File operation failed: {str(e)}")
    
    async def _list_files(self, path: str, pattern: str, recursive: bool) -> Dict[str, Any]:
        """List files in directory."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        if not path_obj.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        
        files = []
        
        if recursive:
            for item in path_obj.rglob(pattern):
                files.append(self._get_file_entry(item))
        else:
            for item in path_obj.glob(pattern):
                files.append(self._get_file_entry(item))
        
        return {
            "path": str(path_obj.absolute()),
            "pattern": pattern,
            "recursive": recursive,
            "total_files": len(files),
            "files": files
        }
    
    async def _read_file(self, path: str, encoding: str) -> Dict[str, Any]:
        """Read file content."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        
        if not path_obj.is_file():
            raise IsADirectoryError(f"Path is not a file: {path}")
        
        try:
            content = path_obj.read_text(encoding=encoding)
            lines = content.splitlines()
            
            return {
                "path": str(path_obj.absolute()),
                "content": content,
                "lines": lines,
                "line_count": len(lines),
                "character_count": len(content),
                "encoding": encoding,
                "size_bytes": path_obj.stat().st_size
            }
        except UnicodeDecodeError as e:
            raise ValueError(f"Cannot decode file with encoding {encoding}: {e}")
    
    async def _write_file(self, path: str, content: str, encoding: str, backup: bool) -> Dict[str, Any]:
        """Write content to file."""
        path_obj = Path(path)
        
        logger.info(f"📝 Writing file: {path}")
        
        # Create backup if file exists
        backup_path = None
        if backup and path_obj.exists():
            backup_path = path_obj.with_suffix(path_obj.suffix + f".backup.{int(datetime.now().timestamp())}")
            shutil.copy2(path_obj, backup_path)
            logger.debug(f"📋 Created backup: {backup_path}")
        
        # Create parent directories if needed
        if not path_obj.parent.exists():
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"📁 Created directory: {path_obj.parent}")
        
        # Write content
        path_obj.write_text(content, encoding=encoding)
        logger.info(f"✅ File written successfully: {path} ({len(content)} chars)")
        
        return {
            "path": str(path_obj.absolute()),
            "bytes_written": len(content.encode(encoding)),
            "lines_written": len(content.splitlines()),
            "encoding": encoding,
            "backup_created": backup_path is not None,
            "backup_path": str(backup_path) if backup_path else None
        }
    
    async def _edit_file(self, path: str, content: str, line_number: Optional[int], backup: bool, encoding: str) -> Dict[str, Any]:
        """Edit file content."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        
        # Read existing content
        existing_content = path_obj.read_text(encoding=encoding)
        lines = existing_content.splitlines()
        
        # Create backup
        backup_path = None
        if backup:
            backup_path = path_obj.with_suffix(path_obj.suffix + f".backup.{int(datetime.now().timestamp())}")
            shutil.copy2(path_obj, backup_path)
        
        # Apply edit
        if line_number is not None:
            # Line-based edit
            if 1 <= line_number <= len(lines):
                lines[line_number - 1] = content
            else:
                raise ValueError(f"Line number {line_number} out of range (1-{len(lines)})")
            new_content = "\n".join(lines)
        else:
            # Replace entire content
            new_content = content
        
        # Write modified content
        path_obj.write_text(new_content, encoding=encoding)
        
        return {
            "path": str(path_obj.absolute()),
            "edit_type": "line_edit" if line_number else "full_replace",
            "line_number": line_number,
            "original_lines": len(lines),
            "new_lines": len(new_content.splitlines()),
            "backup_created": backup_path is not None,
            "backup_path": str(backup_path) if backup_path else None
        }
    
    async def _search_files(self, path: str, query: str, pattern: str, recursive: bool) -> Dict[str, Any]:
        """Search for content in files."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        matches = []
        files_searched = 0
        
        # Get files to search
        if path_obj.is_file():
            files_to_search = [path_obj]
        else:
            if recursive:
                files_to_search = list(path_obj.rglob(pattern))
            else:
                files_to_search = list(path_obj.glob(pattern))
        
        # Search each file
        for file_path in files_to_search:
            if file_path.is_file():
                try:
                    file_matches = await self._search_file_content(file_path, query)
                    if file_matches:
                        matches.append({
                            "file": str(file_path),
                            "matches": file_matches
                        })
                    files_searched += 1
                except Exception as e:
                    logger.warning(f"Failed to search file {file_path}: {e}")
        
        return {
            "search_path": str(path_obj.absolute()),
            "query": query,
            "pattern": pattern,
            "recursive": recursive,
            "files_searched": files_searched,
            "files_with_matches": len(matches),
            "total_matches": sum(len(m["matches"]) for m in matches),
            "matches": matches
        }
    
    async def _search_file_content(self, file_path: Path, query: str) -> List[Dict[str, Any]]:
        """Search content within a single file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines()
            matches = []
            
            # Simple text search (could be enhanced with regex)
            for line_num, line in enumerate(lines, 1):
                if query.lower() in line.lower():
                    matches.append({
                        "line_number": line_num,
                        "line_content": line.strip(),
                        "match_positions": [m.start() for m in re.finditer(re.escape(query.lower()), line.lower())]
                    })
            
            return matches
        except Exception:
            return []
    
    async def _create_directory(self, path: str) -> Dict[str, Any]:
        """Create directory."""
        path_obj = Path(path)
        
        if path_obj.exists():
            if path_obj.is_dir():
                return {
                    "path": str(path_obj.absolute()),
                    "created": False,
                    "reason": "Directory already exists"
                }
            else:
                raise FileExistsError(f"Path exists but is not a directory: {path}")
        
        path_obj.mkdir(parents=True, exist_ok=True)
        
        return {
            "path": str(path_obj.absolute()),
            "created": True,
            "parents_created": True
        }
    
    async def _delete_file(self, path: str) -> Dict[str, Any]:
        """Delete file or directory."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        if path_obj.is_file():
            size = path_obj.stat().st_size
            path_obj.unlink()
            return {
                "path": str(path_obj.absolute()),
                "type": "file",
                "size_deleted": size
            }
        elif path_obj.is_dir():
            # Count items before deletion
            item_count = len(list(path_obj.rglob("*")))
            shutil.rmtree(path_obj)
            return {
                "path": str(path_obj.absolute()),
                "type": "directory",
                "items_deleted": item_count
            }
        else:
            raise ValueError(f"Unknown path type: {path}")
    
    async def _copy_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Copy file or directory."""
        source_obj = Path(source)
        dest_obj = Path(destination)
        
        if not source_obj.exists():
            raise FileNotFoundError(f"Source does not exist: {source}")
        
        # Create destination parent directories
        dest_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if source_obj.is_file():
            shutil.copy2(source_obj, dest_obj)
            return {
                "source": str(source_obj.absolute()),
                "destination": str(dest_obj.absolute()),
                "type": "file",
                "size_copied": source_obj.stat().st_size
            }
        elif source_obj.is_dir():
            shutil.copytree(source_obj, dest_obj, dirs_exist_ok=True)
            item_count = len(list(dest_obj.rglob("*")))
            return {
                "source": str(source_obj.absolute()),
                "destination": str(dest_obj.absolute()),
                "type": "directory",
                "items_copied": item_count
            }
        else:
            raise ValueError(f"Unknown source type: {source}")
    
    async def _move_file(self, source: str, destination: str) -> Dict[str, Any]:
        """Move file or directory."""
        source_obj = Path(source)
        dest_obj = Path(destination)
        
        if not source_obj.exists():
            raise FileNotFoundError(f"Source does not exist: {source}")
        
        # Create destination parent directories
        dest_obj.parent.mkdir(parents=True, exist_ok=True)
        
        original_size = source_obj.stat().st_size if source_obj.is_file() else 0
        item_count = len(list(source_obj.rglob("*"))) if source_obj.is_dir() else 1
        
        shutil.move(source_obj, dest_obj)
        
        return {
            "source": str(source_obj.absolute()),
            "destination": str(dest_obj.absolute()),
            "type": "file" if original_size > 0 else "directory",
            "size_moved": original_size,
            "items_moved": item_count
        }
    
    async def _get_file_info(self, path: str) -> Dict[str, Any]:
        """Get detailed file information."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        stat = path_obj.stat()
        
        info = {
            "path": str(path_obj.absolute()),
            "name": path_obj.name,
            "type": "file" if path_obj.is_file() else "directory",
            "size_bytes": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
            "is_readable": os.access(path_obj, os.R_OK),
            "is_writable": os.access(path_obj, os.W_OK),
            "is_executable": os.access(path_obj, os.X_OK)
        }
        
        if path_obj.is_file():
            info["extension"] = path_obj.suffix
            info["stem"] = path_obj.stem
        elif path_obj.is_dir():
            info["item_count"] = len(list(path_obj.iterdir()))
        
        return info
    
    async def _find_files(self, path: str, pattern: str, recursive: bool) -> Dict[str, Any]:
        """Find files matching pattern."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        
        if not path_obj.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        
        found_files = []
        
        if recursive:
            for item in path_obj.rglob(pattern):
                if item.is_file():
                    found_files.append(str(item.absolute()))
        else:
            for item in path_obj.glob(pattern):
                if item.is_file():
                    found_files.append(str(item.absolute()))
        
        return {
            "search_path": str(path_obj.absolute()),
            "pattern": pattern,
            "recursive": recursive,
            "files_found": len(found_files),
            "files": found_files
        }
    
    def _get_file_entry(self, path: Path) -> Dict[str, Any]:
        """Get file entry information."""
        try:
            stat = path.stat()
            return {
                "name": path.name,
                "path": str(path.absolute()),
                "type": "file" if path.is_file() else "directory",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": path.suffix if path.is_file() else None
            }
        except Exception as e:
            return {
                "name": path.name,
                "path": str(path.absolute()),
                "type": "unknown",
                "error": str(e)
            }
    
    # ============================================================================
    # HYBRID ARCHITECTURE METHODS
    # ============================================================================
    
    def to_pydantic_tools(self, agent=None) -> Dict[str, Callable]:
        """Convert to PydanticAI-compatible tool functions.
        
        Args:
            agent: PydanticAI agent instance (optional, for caching)
            
        Returns:
            Dictionary of tool functions that can be registered with PydanticAI
        """
        if agent and id(agent) in self._pydantic_tools_cache:
            return self._pydantic_tools_cache[id(agent)]
        
        # Create individual tool functions for each operation
        tools = {}
        
        async def read_file(path: str, encoding: str = "utf-8") -> Dict[str, Any]:
            """Read content from a file."""
            result = await self.execute(operation="read_file", path=path, encoding=encoding)
            return result.result
        
        async def write_file(path: str, content: str, encoding: str = "utf-8", backup: bool = True) -> Dict[str, Any]:
            """Write content to a file."""
            result = await self.execute(operation="write_file", path=path, content=content, encoding=encoding, backup=backup)
            return result.result
        
        async def list_files(path: str, pattern: str = "*", recursive: bool = False) -> Dict[str, Any]:
            """List files in a directory."""
            result = await self.execute(operation="list_files", path=path, pattern=pattern, recursive=recursive)
            return result.result
        
        async def search_files(path: str, search_query: str, pattern: str = "*", recursive: bool = False) -> Dict[str, Any]:
            """Search for content in files."""
            result = await self.execute(operation="search_files", path=path, search_query=search_query, pattern=pattern, recursive=recursive)
            return result.result
        
        async def create_directory(path: str) -> Dict[str, Any]:
            """Create a directory."""
            result = await self.execute(operation="create_directory", path=path)
            return result.result
        
        async def delete_file(path: str) -> Dict[str, Any]:
            """Delete a file or directory."""
            result = await self.execute(operation="delete_file", path=path)
            return result.result
        
        async def copy_file(source_path: str, destination_path: str) -> Dict[str, Any]:
            """Copy a file or directory."""
            result = await self.execute(operation="copy_file", path=source_path, destination=destination_path)
            return result.result
        
        async def move_file(source_path: str, destination_path: str) -> Dict[str, Any]:
            """Move a file or directory."""
            result = await self.execute(operation="move_file", path=source_path, destination=destination_path)
            return result.result
        
        async def get_file_info(path: str) -> Dict[str, Any]:
            """Get detailed information about a file or directory."""
            result = await self.execute(operation="get_file_info", path=path)
            return result.result
        
        async def find_files(path: str, pattern: str = "*", recursive: bool = False) -> Dict[str, Any]:
            """Find files matching a pattern."""
            result = await self.execute(operation="find_files", path=path, pattern=pattern, recursive=recursive)
            return result.result
        
        tools = {
            "read_file": read_file,
            "write_file": write_file,
            "list_files": list_files,
            "search_files": search_files,
            "create_directory": create_directory,
            "delete_file": delete_file,
            "copy_file": copy_file,
            "move_file": move_file,
            "get_file_info": get_file_info,
            "find_files": find_files,
        }
        
        # Cache for this agent
        if agent:
            self._pydantic_tools_cache[id(agent)] = tools
        
        return tools
    
    def register_with_pydantic_agent(self, agent) -> None:
        """Register all file operations as individual tools with a PydanticAI agent.
        
        Usage:
            file_tool = FileOperationsTool()
            file_tool.register_with_pydantic_agent(agent)
        """
        tools = self.to_pydantic_tools(agent)
        
        # Register each tool function with the agent
        for tool_name, tool_func in tools.items():
            # Use agent.tool decorator if available
            if hasattr(agent, 'tool'):
                decorated_func = agent.tool(tool_func)
                setattr(agent, f"_file_{tool_name}", decorated_func)
            else:
                logger.warning(f"Agent does not have 'tool' decorator method. Cannot register {tool_name}")
    
    def get_mcp_server(self):
        """Get or create MCP server wrapper for this tool.
        
        Returns:
            MCP server instance that exposes file operations
        """
        if self._mcp_server is None:
            self._mcp_server = FileOperationsMCPServer(self)
        return self._mcp_server
    
    def enable_observability(self, config: Optional[Dict[str, Any]] = None):
        """Enable OpenTelemetry observability for file operations.
        
        Args:
            config: Optional configuration for observability setup
        """
        if self._observability is None:
            self._observability = FileOperationsObservability(self, config or {})
        return self._observability


class FileOperationsMCPServer:
    """MCP server wrapper for FileOperationsTool.
    
    Provides Model Context Protocol compatible interface for file operations.
    """
    
    def __init__(self, file_tool: FileOperationsTool):
        self.file_tool = file_tool
        self.server_info = {
            "name": "aida-file-operations",
            "version": file_tool.version,
            "description": file_tool.description
        }
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        capability = self.file_tool.get_capability()
        
        # Convert each operation to an MCP tool
        operations = [
            "read_file", "write_file", "list_files", "search_files",
            "create_directory", "delete_file", "copy_file", "move_file",
            "get_file_info", "find_files"
        ]
        
        tools = []
        for op in operations:
            tools.append({
                "name": f"file_{op}",
                "description": f"File operation: {op.replace('_', ' ').title()}",
                "inputSchema": {
                    "type": "object",
                    "properties": self._get_operation_schema(op),
                    "required": self._get_required_params(op)
                }
            })
        
        return tools
    
    def _get_operation_schema(self, operation: str) -> Dict[str, Any]:
        """Get JSON schema for operation parameters."""
        base_props = {
            "path": {
                "type": "string",
                "description": "File or directory path"
            }
        }
        
        if operation in ["write_file", "edit_file"]:
            base_props["content"] = {
                "type": "string", 
                "description": "Content to write"
            }
        
        if operation == "search_files":
            base_props["search_query"] = {
                "type": "string",
                "description": "Search query"
            }
        
        if operation in ["copy_file", "move_file"]:
            base_props["destination"] = {
                "type": "string",
                "description": "Destination path"
            }
        
        return base_props
    
    def _get_required_params(self, operation: str) -> List[str]:
        """Get required parameters for operation."""
        required = ["path"]
        
        if operation in ["write_file", "search_files"]:
            required.append("content" if operation == "write_file" else "search_query")
        
        if operation in ["copy_file", "move_file"]:
            required.append("destination")
        
        return required
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool call."""
        # Extract operation from tool name (remove 'file_' prefix)
        operation = name.replace("file_", "")
        
        try:
            result = await self.file_tool.execute(operation=operation, **arguments)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result.result, indent=2)
                    }
                ]
            }
        except Exception as e:
            return {
                "content": [
                    {
                        "type": "text", 
                        "text": f"Error: {str(e)}"
                    }
                ],
                "isError": True
            }


class FileOperationsObservability:
    """OpenTelemetry observability for file operations."""
    
    def __init__(self, file_tool: FileOperationsTool, config: Dict[str, Any]):
        self.file_tool = file_tool
        self.config = config
        self.enabled = config.get("enabled", True)
        
        if self.enabled:
            self._setup_tracing()
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        try:
            # Import OpenTelemetry components
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            
            # Setup tracer
            self.tracer = trace.get_tracer(__name__)
            logger.debug("OpenTelemetry tracing enabled for file operations")
        except ImportError:
            logger.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")
            self.enabled = False
    
    def trace_operation(self, operation: str, **kwargs):
        """Create a trace span for file operation."""
        if not self.enabled:
            return None
        
        return self.tracer.start_span(
            f"file_operation.{operation}",
            attributes={
                "file.operation": operation,
                "file.path": kwargs.get("path", "unknown"),
                "tool.name": self.file_tool.name,
                "tool.version": self.file_tool.version
            }
        )