"""Base classes for AIDA tools following modular pattern."""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Optional, TypeVar, Generic
from datetime import datetime
from contextlib import contextmanager

from pydantic import BaseModel, Field
from aida.tools.base import ToolResult, ToolStatus

logger = logging.getLogger(__name__)

# Type variables for generic models
TRequest = TypeVar('TRequest', bound=BaseModel)
TResponse = TypeVar('TResponse', bound=BaseModel)
TConfig = TypeVar('TConfig')


class BaseToolConfig:
    """Base configuration for all tools."""
    TRACE_ENABLED = True
    METRICS_ENABLED = True
    SERVICE_NAME_PREFIX = "aida"
    

class BaseModularTool(ABC, Generic[TRequest, TResponse, TConfig]):
    """Base class for modular tools with consistent interface."""
    
    def __init__(self, config: Optional[TConfig] = None):
        """Initialize base tool with optional config."""
        self.config = config or self._get_default_config()
        self.name = self._get_tool_name()
        self.version = self._get_tool_version()
        self.description = self._get_tool_description()
        self._mcp_server = None
        self._observability = None
        self._pydantic_tools = None
        logger.info(f"Initialized {self.name} v{self.version}")
    
    @abstractmethod
    def _get_tool_name(self) -> str:
        """Get the tool name."""
        pass
    
    @abstractmethod
    def _get_tool_version(self) -> str:
        """Get the tool version."""
        pass
    
    @abstractmethod
    def _get_tool_description(self) -> str:
        """Get the tool description."""
        pass
    
    @abstractmethod
    def _get_default_config(self) -> TConfig:
        """Get default configuration."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool operation."""
        pass
    
    def get_mcp_server(self):
        """Get MCP server instance."""
        if self._mcp_server is None:
            self._mcp_server = self._create_mcp_server()
        return self._mcp_server
    
    def to_mcp_tool(self):
        """Convert to MCP tool format."""
        return self.get_mcp_server()
    
    def enable_observability(self, config: Optional[Dict[str, Any]] = None) -> Any:
        """Enable OpenTelemetry observability."""
        if self._observability is None:
            obs_config = config or {}
            obs_config.setdefault("service_name", f"{BaseToolConfig.SERVICE_NAME_PREFIX}-{self.name}")
            obs_config.setdefault("trace_enabled", BaseToolConfig.TRACE_ENABLED)
            obs_config.setdefault("metrics_enabled", BaseToolConfig.METRICS_ENABLED)
            self._observability = self._create_observability(obs_config)
        return self._observability
    
    def to_pydantic_tools(self) -> Dict[str, Callable]:
        """Get PydanticAI-compatible tool functions."""
        if self._pydantic_tools is None:
            self._pydantic_tools = self._create_pydantic_tools()
        return self._pydantic_tools
    
    @abstractmethod
    def _create_mcp_server(self):
        """Create MCP server instance."""
        pass
    
    @abstractmethod
    def _create_observability(self, config: Dict[str, Any]):
        """Create observability instance."""
        pass
    
    @abstractmethod
    def _create_pydantic_tools(self) -> Dict[str, Callable]:
        """Create PydanticAI tool functions."""
        pass


class BaseMCPServer:
    """Base MCP server implementation."""
    
    def __init__(self, tool: BaseModularTool):
        """Initialize MCP server with tool reference."""
        self.tool = tool
        self.server_info = {
            "name": f"aida-{tool.name}",
            "version": tool.version,
            "description": tool.description
        }
    
    @abstractmethod
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        pass
    
    @abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP tool call."""
        pass
    
    def _format_mcp_response(self, result: Any, is_error: bool = False) -> Dict[str, Any]:
        """Format response for MCP protocol."""
        if is_error:
            return {
                "content": [{
                    "type": "text",
                    "text": str(result)
                }],
                "isError": True
            }
        
        # Handle different result types
        if isinstance(result, str):
            response_text = result
        elif isinstance(result, list):
            response_text = "\n".join(f"• {item}" for item in result)
        elif isinstance(result, dict):
            import json
            response_text = json.dumps(result, indent=2)
        else:
            response_text = str(result)
        
        return {
            "content": [{
                "type": "text",
                "text": response_text
            }]
        }


class BaseObservability:
    """Base OpenTelemetry observability implementation."""
    
    def __init__(self, tool: BaseModularTool, config: Dict[str, Any]):
        """Initialize observability with tool reference and config."""
        self.tool = tool
        self.config = config
        self.enabled = config.get("trace_enabled", True)
        self.tracer = None
        self.meter = None
        
        if self.enabled:
            self._setup_observability()
    
    def _setup_observability(self):
        """Setup OpenTelemetry components."""
        try:
            from opentelemetry import trace, metrics
            
            # Setup tracer
            self.tracer = trace.get_tracer(
                self.config.get("service_name", f"aida-{self.tool.name}")
            )
            
            # Setup metrics if enabled
            if self.config.get("metrics_enabled", True):
                self.meter = metrics.get_meter(
                    self.config.get("service_name", f"aida-{self.tool.name}")
                )
                self._setup_metrics()
            
            logger.debug(f"OpenTelemetry observability enabled for {self.tool.name}")
            
        except ImportError:
            logger.warning(
                "OpenTelemetry not available. Install with: "
                "pip install opentelemetry-api opentelemetry-sdk"
            )
            self.enabled = False
    
    @abstractmethod
    def _setup_metrics(self):
        """Setup custom metrics for the specific tool."""
        pass
    
    @contextmanager
    def trace_operation(self, operation: str, **attributes):
        """Create a trace span for tool operation."""
        if not self.enabled or not self.tracer:
            yield None
            return
            
        with self.tracer.start_as_current_span(
            f"{self.tool.name}_{operation}",
            attributes={
                "tool.name": self.tool.name,
                "tool.version": self.tool.version,
                "tool.operation": operation,
                **attributes
            }
        ) as span:
            yield span
    
    def record_operation(self, operation: str, duration: float, success: bool, **metrics):
        """Record common operation metrics."""
        if not self.enabled:
            return
        
        # Override in subclasses to record specific metrics
        pass


class SimpleToolBase(BaseModularTool[BaseModel, BaseModel, BaseToolConfig]):
    """Simplified base for tools that don't need complex request/response models."""
    
    def __init__(self):
        """Initialize simple tool."""
        super().__init__()
        self._processors = self._create_processors()
    
    @abstractmethod
    def _create_processors(self) -> Dict[str, Callable]:
        """Create operation processors."""
        pass
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute tool operation with simple routing."""
        start_time = datetime.utcnow()
        
        try:
            # Get operation from kwargs
            operation = kwargs.get("operation")
            if not operation:
                # Single-operation tools can have a default
                operation = self._get_default_operation()
            
            # Route to processor
            processor = self._processors.get(operation)
            if not processor:
                return ToolResult(
                    tool_name=self.name,
                    execution_id=str(uuid.uuid4()),
                    status=ToolStatus.FAILED,
                    error=f"Unknown operation: {operation}",
                    started_at=start_time,
                    completed_at=datetime.utcnow(),
                    duration_seconds=(datetime.utcnow() - start_time).total_seconds()
                )
            
            # Execute processor
            result = await processor(**kwargs)
            
            # Record metrics if observability enabled
            if self._observability:
                duration = (datetime.utcnow() - start_time).total_seconds()
                self._observability.record_operation(
                    operation=operation,
                    duration=duration,
                    success=True
                )
            
            return ToolResult(
                tool_name=self.name,
                execution_id=str(uuid.uuid4()),
                status=ToolStatus.COMPLETED,
                result=result,
                started_at=start_time,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds()
            )
            
        except Exception as e:
            logger.error(f"{self.name} operation failed: {e}")
            
            # Record failure metrics
            if self._observability:
                duration = (datetime.utcnow() - start_time).total_seconds()
                self._observability.record_operation(
                    operation=kwargs.get("operation", "unknown"),
                    duration=duration,
                    success=False
                )
            
            return ToolResult(
                tool_name=self.name,
                execution_id=str(uuid.uuid4()),
                status=ToolStatus.FAILED,
                error=str(e),
                started_at=start_time,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds()
            )
    
    def _get_default_operation(self) -> Optional[str]:
        """Get default operation for single-operation tools."""
        return None