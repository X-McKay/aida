"""Base tool interface for AIDA."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime
from enum import Enum
import uuid
import logging

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ToolStatus(str, Enum):
    """Tool execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolError(Exception):
    """Base exception for tool errors."""
    
    def __init__(self, message: str, error_code: str = "TOOL_ERROR", details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ToolResult(BaseModel):
    """Tool execution result."""
    
    tool_name: str
    execution_id: str
    status: ToolStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class ToolParameter(BaseModel):
    """Tool parameter definition."""
    
    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None


class ToolCapability(BaseModel):
    """Tool capability descriptor."""
    
    name: str
    version: str = "1.0.0"
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)
    required_permissions: List[str] = Field(default_factory=list)
    supported_platforms: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)


class Tool(ABC):
    """Abstract base class for all AIDA tools."""
    
    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        self.name = name
        self.description = description
        self.version = version
        self._capability: Optional[ToolCapability] = None
        self._execution_history: List[ToolResult] = []
        self._active_executions: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_duration": 0.0
        }
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    @abstractmethod
    def get_capability(self) -> ToolCapability:
        """Get tool capability descriptor."""
        pass
    
    async def execute_async(self, execution_id: Optional[str] = None, **kwargs) -> ToolResult:
        """Execute tool asynchronously with tracking."""
        execution_id = execution_id or str(uuid.uuid4())
        
        async with self._lock:
            if execution_id in self._active_executions:
                raise ToolError(
                    f"Execution {execution_id} already active",
                    "EXECUTION_ALREADY_ACTIVE"
                )
        
        # Create task for execution
        task = asyncio.create_task(self._execute_with_tracking(execution_id, **kwargs))
        
        async with self._lock:
            self._active_executions[execution_id] = task
        
        try:
            result = await task
            return result
        finally:
            async with self._lock:
                self._active_executions.pop(execution_id, None)
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active execution."""
        async with self._lock:
            task = self._active_executions.get(execution_id)
            if task and not task.done():
                task.cancel()
                return True
            return False
    
    async def get_execution_status(self, execution_id: str) -> Optional[ToolStatus]:
        """Get status of an execution."""
        async with self._lock:
            task = self._active_executions.get(execution_id)
            if task:
                if task.done():
                    if task.cancelled():
                        return ToolStatus.CANCELLED
                    elif task.exception():
                        return ToolStatus.FAILED
                    else:
                        return ToolStatus.COMPLETED
                else:
                    return ToolStatus.RUNNING
        
        # Check history
        for result in self._execution_history:
            if result.execution_id == execution_id:
                return result.status
        
        return None
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[ToolResult]:
        """Get execution history."""
        if limit is None:
            return self._execution_history.copy()
        else:
            return self._execution_history[-limit:]
    
    def get_active_executions(self) -> List[str]:
        """Get list of active execution IDs."""
        return list(self._active_executions.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool statistics."""
        return {
            **self._stats,
            "active_executions": len(self._active_executions),
            "history_size": len(self._execution_history)
        }
    
    def validate_parameters(self, **kwargs) -> None:
        """Validate tool parameters."""
        capability = self.get_capability()
        
        # Check required parameters
        required_params = {p.name for p in capability.parameters if p.required}
        provided_params = set(kwargs.keys())
        missing_params = required_params - provided_params
        
        if missing_params:
            raise ToolError(
                f"Missing required parameters: {missing_params}",
                "MISSING_PARAMETERS"
            )
        
        # Validate parameter types and constraints
        for param in capability.parameters:
            if param.name not in kwargs:
                continue
            
            value = kwargs[param.name]
            
            # Type validation (basic)
            if param.type == "str" and not isinstance(value, str):
                raise ToolError(
                    f"Parameter {param.name} must be a string",
                    "INVALID_PARAMETER_TYPE"
                )
            elif param.type == "int" and not isinstance(value, int):
                raise ToolError(
                    f"Parameter {param.name} must be an integer",
                    "INVALID_PARAMETER_TYPE"
                )
            elif param.type == "float" and not isinstance(value, (int, float)):
                raise ToolError(
                    f"Parameter {param.name} must be a number",
                    "INVALID_PARAMETER_TYPE"
                )
            elif param.type == "bool" and not isinstance(value, bool):
                raise ToolError(
                    f"Parameter {param.name} must be a boolean",
                    "INVALID_PARAMETER_TYPE"
                )
            
            # Choices validation
            if param.choices and value not in param.choices:
                raise ToolError(
                    f"Parameter {param.name} must be one of: {param.choices}",
                    "INVALID_PARAMETER_VALUE"
                )
            
            # Range validation
            if param.min_value is not None and value < param.min_value:
                raise ToolError(
                    f"Parameter {param.name} must be >= {param.min_value}",
                    "PARAMETER_OUT_OF_RANGE"
                )
            
            if param.max_value is not None and value > param.max_value:
                raise ToolError(
                    f"Parameter {param.name} must be <= {param.max_value}",
                    "PARAMETER_OUT_OF_RANGE"
                )
            
            # Pattern validation
            if param.pattern and isinstance(value, str):
                import re
                if not re.match(param.pattern, value):
                    raise ToolError(
                        f"Parameter {param.name} does not match pattern: {param.pattern}",
                        "INVALID_PARAMETER_FORMAT"
                    )
    
    async def _execute_with_tracking(self, execution_id: str, **kwargs) -> ToolResult:
        """Execute tool with tracking and error handling."""
        started_at = datetime.utcnow()
        
        result = ToolResult(
            tool_name=self.name,
            execution_id=execution_id,
            status=ToolStatus.RUNNING,
            started_at=started_at
        )
        
        try:
            # Validate parameters
            self.validate_parameters(**kwargs)
            
            # Execute tool
            result = await self.execute(**kwargs)
            result.execution_id = execution_id
            result.started_at = started_at
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (result.completed_at - started_at).total_seconds()
            result.status = ToolStatus.COMPLETED
            
            # Update statistics
            self._stats["successful_executions"] += 1
            
        except asyncio.CancelledError:
            result.status = ToolStatus.CANCELLED
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (result.completed_at - started_at).total_seconds()
            raise
            
        except Exception as e:
            result.status = ToolStatus.FAILED
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (result.completed_at - started_at).total_seconds()
            
            if isinstance(e, ToolError):
                result.error = e.message
                result.error_code = e.error_code
                result.metadata.update(e.details)
            else:
                result.error = str(e)
                result.error_code = "EXECUTION_ERROR"
            
            self._stats["failed_executions"] += 1
            
            logger.debug(f"Tool {self.name} execution failed: {result.error}")
        
        finally:
            # Update statistics
            self._stats["total_executions"] += 1
            
            # Update average duration
            if self._stats["total_executions"] > 0:
                total_duration = (
                    self._stats["average_duration"] * (self._stats["total_executions"] - 1) +
                    (result.duration_seconds or 0)
                )
                self._stats["average_duration"] = total_duration / self._stats["total_executions"]
            
            # Store in history
            self._execution_history.append(result)
            
            # Limit history size
            if len(self._execution_history) > 1000:
                self._execution_history = self._execution_history[-1000:]
        
        return result


class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._capabilities: Dict[str, ToolCapability] = {}
        self._lock = asyncio.Lock()
    
    async def register_tool(self, tool: Tool) -> None:
        """Register a tool."""
        async with self._lock:
            self._tools[tool.name] = tool
            self._capabilities[tool.name] = tool.get_capability()
            
        logger.debug(f"Tool registered: {tool.name} v{tool.version}")
    
    async def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool."""
        async with self._lock:
            tool = self._tools.pop(tool_name, None)
            self._capabilities.pop(tool_name, None)
            
        if tool:
            logger.info(f"Tool unregistered: {tool_name}")
            return True
        return False
    
    async def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        async with self._lock:
            return self._tools.get(tool_name)
    
    async def list_tools(self) -> List[str]:
        """List all registered tool names."""
        async with self._lock:
            return list(self._tools.keys())
    
    async def get_capabilities(self, tool_name: Optional[str] = None) -> Union[ToolCapability, List[ToolCapability]]:
        """Get tool capabilities."""
        async with self._lock:
            if tool_name:
                return self._capabilities.get(tool_name)
            else:
                return list(self._capabilities.values())
    
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = await self.get_tool(tool_name)
        if not tool:
            raise ToolError(
                f"Tool not found: {tool_name}",
                "TOOL_NOT_FOUND"
            )
        
        return await tool.execute_async(**kwargs)
    
    async def get_tool_stats(self, tool_name: Optional[str] = None) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Get tool statistics."""
        if tool_name:
            tool = await self.get_tool(tool_name)
            if tool:
                return tool.get_stats()
            return {}
        else:
            async with self._lock:
                return {
                    name: tool.get_stats()
                    for name, tool in self._tools.items()
                }


# Global tool registry
_global_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _global_tool_registry
    if _global_tool_registry is None:
        _global_tool_registry = ToolRegistry()
    return _global_tool_registry


async def initialize_default_tools() -> ToolRegistry:
    """Initialize and register all default tools."""
    registry = get_tool_registry()
    
    # Import tools here to avoid circular imports
    from aida.tools.execution import ExecutionTool
    from aida.tools.files import FileOperationsTool
    from aida.tools.system import SystemTool
    from aida.tools.context import ContextTool
    from aida.tools.llm_response import LLMResponseTool
    
    # Create and register all tools
    # NOTE: Removed non-refactored tools (thinking, maintenance, project, architecture)
    # to meet deadline. These can be added back later if needed.
    tools = [
        ExecutionTool(),
        FileOperationsTool(),
        SystemTool(),
        ContextTool(),
        LLMResponseTool(),
    ]
    
    for tool in tools:
        await registry.register_tool(tool)
    
    logger.debug(f"Initialized {len(tools)} default tools")
    return registry