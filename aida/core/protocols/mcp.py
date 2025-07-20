"""Model Context Protocol (MCP) implementation for AIDA."""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import logging

from pydantic import BaseModel, Field

from aida.core.protocols.base import Protocol, ProtocolMessage


logger = logging.getLogger(__name__)


class MCPMessageType(str, Enum):
    """MCP message types."""
    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    NOTIFICATION = "notification"
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    PROGRESS = "progress"


class MCPResourceType(str, Enum):
    """MCP resource types."""
    TEXT = "text"
    JSON = "json"
    BINARY = "binary"
    STREAM = "stream"


class MCPCapability(BaseModel):
    """MCP capability descriptor."""
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class MCPResource(BaseModel):
    """MCP resource descriptor."""
    uri: str
    type: MCPResourceType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content: Optional[Union[str, bytes, Dict[str, Any]]] = None


class MCPTool(BaseModel):
    """MCP tool descriptor."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_capabilities: List[str] = Field(default_factory=list)


class MCPMessage(ProtocolMessage):
    """MCP-specific message format."""
    
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    jsonrpc: str = "2.0"
    
    class Methods:
        """Standard MCP methods."""
        # Context management
        CONTEXT_GET = "context/get"
        CONTEXT_SET = "context/set"
        CONTEXT_UPDATE = "context/update"
        CONTEXT_DELETE = "context/delete"
        CONTEXT_LIST = "context/list"
        
        # Resource management
        RESOURCE_GET = "resource/get"
        RESOURCE_SET = "resource/set"
        RESOURCE_LIST = "resource/list"
        RESOURCE_WATCH = "resource/watch"
        
        # Tool management
        TOOL_CALL = "tool/call"
        TOOL_LIST = "tool/list"
        TOOL_GET = "tool/get"
        
        # Capability management
        CAPABILITY_GET = "capability/get"
        CAPABILITY_LIST = "capability/list"
        
        # Session management
        SESSION_START = "session/start"
        SESSION_END = "session/end"
        SESSION_STATUS = "session/status"


class MCPProtocol(Protocol):
    """Model Context Protocol implementation."""
    
    def __init__(
        self, 
        agent_id: str,
        transport: str = "stdio",
        capabilities: Optional[List[MCPCapability]] = None
    ):
        super().__init__(agent_id)
        self.transport = transport
        self.capabilities = capabilities or []
        
        # MCP state
        self._initialized = False
        self._session_id: Optional[str] = None
        self._context_store: Dict[str, Any] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._tools: Dict[str, MCPTool] = {}
        
        # Message handling
        self._message_id_counter = 0
        self._pending_requests: Dict[str, asyncio.Future] = {}
        
        # Transport
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        
        # Tasks
        self._tasks: Set[asyncio.Task] = set()
    
    async def connect(self) -> bool:
        """Initialize MCP connection."""
        try:
            if self.transport == "stdio":
                await self._connect_stdio()
            else:
                raise ValueError(f"Unsupported transport: {self.transport}")
            
            # Start message processing
            self._tasks.add(asyncio.create_task(self._process_messages()))
            
            # Send initialization
            await self._initialize()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect MCP: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Close MCP connection."""
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        # Close transport
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        
        self._initialized = False
        logger.info("MCP protocol disconnected")
    
    async def send(self, message: ProtocolMessage) -> bool:
        """Send MCP message."""
        if not isinstance(message, MCPMessage):
            message = MCPMessage(**message.dict())
        
        try:
            if not self._writer:
                return False
            
            # Serialize message
            message_data = message.json() + "\n"
            self._writer.write(message_data.encode())
            await self._writer.drain()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send MCP message: {e}")
            return False
    
    async def receive(self) -> Optional[MCPMessage]:
        """Receive MCP message."""
        try:
            if not self._reader:
                return None
            
            line = await self._reader.readline()
            if not line:
                return None
            
            message_data = line.decode().strip()
            message = MCPMessage.parse_raw(message_data)
            
            return message
            
        except Exception as e:
            logger.error(f"Failed to receive MCP message: {e}")
            return None
    
    async def call_method(
        self, 
        method: str, 
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0
    ) -> Any:
        """Call an MCP method and wait for response."""
        if not self._initialized:
            raise RuntimeError("MCP not initialized")
        
        # Create request
        message_id = str(self._message_id_counter)
        self._message_id_counter += 1
        
        request = MCPMessage(
            id=message_id,
            sender_id=self.agent_id,
            message_type=MCPMessageType.REQUEST,
            method=method,
            params=params or {}
        )
        
        # Setup future for response
        future = asyncio.get_event_loop().create_future()
        self._pending_requests[message_id] = future
        
        try:
            # Send request
            await self.send(request)
            
            # Wait for response
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"MCP method {method} timed out")
        
        finally:
            self._pending_requests.pop(message_id, None)
    
    async def get_context(self, key: str) -> Any:
        """Get context value."""
        return await self.call_method(
            MCPMessage.Methods.CONTEXT_GET,
            {"key": key}
        )
    
    async def set_context(self, key: str, value: Any) -> bool:
        """Set context value."""
        result = await self.call_method(
            MCPMessage.Methods.CONTEXT_SET,
            {"key": key, "value": value}
        )
        return result.get("success", False)
    
    async def update_context(self, updates: Dict[str, Any]) -> bool:
        """Update multiple context values."""
        result = await self.call_method(
            MCPMessage.Methods.CONTEXT_UPDATE,
            {"updates": updates}
        )
        return result.get("success", False)
    
    async def list_context(self) -> List[str]:
        """List available context keys."""
        result = await self.call_method(MCPMessage.Methods.CONTEXT_LIST)
        return result.get("keys", [])
    
    async def get_resource(self, uri: str) -> Optional[MCPResource]:
        """Get a resource by URI."""
        result = await self.call_method(
            MCPMessage.Methods.RESOURCE_GET,
            {"uri": uri}
        )
        
        if result:
            return MCPResource(**result)
        return None
    
    async def list_resources(self) -> List[MCPResource]:
        """List available resources."""
        result = await self.call_method(MCPMessage.Methods.RESOURCE_LIST)
        resources = result.get("resources", [])
        return [MCPResource(**res) for res in resources]
    
    async def call_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Any:
        """Call a tool."""
        return await self.call_method(
            MCPMessage.Methods.TOOL_CALL,
            {"name": tool_name, "arguments": arguments}
        )
    
    async def list_tools(self) -> List[MCPTool]:
        """List available tools."""
        result = await self.call_method(MCPMessage.Methods.TOOL_LIST)
        tools = result.get("tools", [])
        return [MCPTool(**tool) for tool in tools]
    
    def register_capability(self, capability: MCPCapability) -> None:
        """Register a new capability."""
        self.capabilities.append(capability)
    
    def register_resource(self, resource: MCPResource) -> None:
        """Register a new resource."""
        self._resources[resource.uri] = resource
    
    def register_tool(self, tool: MCPTool) -> None:
        """Register a new tool."""
        self._tools[tool.name] = tool
    
    async def _connect_stdio(self) -> None:
        """Connect using stdio transport."""
        self._reader = asyncio.StreamReader()
        self._writer = None
        
        # For stdio, we use stdin/stdout
        # This is a simplified implementation
        logger.info("MCP stdio transport connected")
    
    async def _initialize(self) -> None:
        """Send MCP initialization."""
        init_message = MCPMessage(
            sender_id=self.agent_id,
            message_type=MCPMessageType.INITIALIZE,
            method="initialize",
            params={
                "protocolVersion": "1.0.0",
                "clientInfo": {
                    "name": "aida",
                    "version": "1.0.0"
                },
                "capabilities": [cap.dict() for cap in self.capabilities]
            }
        )
        
        await self.send(init_message)
        self._initialized = True
        logger.info("MCP initialized")
    
    async def _process_messages(self) -> None:
        """Process incoming MCP messages."""
        while True:
            try:
                message = await self.receive()
                if not message:
                    continue
                
                await self._handle_message(message)
                
            except Exception as e:
                logger.error(f"Error processing MCP message: {e}")
    
    async def _handle_message(self, message: MCPMessage) -> None:
        """Handle incoming MCP message."""
        # Handle responses
        if message.message_type == MCPMessageType.RESPONSE and message.id:
            future = self._pending_requests.get(message.id)
            if future and not future.done():
                if message.error:
                    future.set_exception(Exception(message.error.get("message", "MCP error")))
                else:
                    future.set_result(message.result)
            return
        
        # Handle requests
        if message.message_type == MCPMessageType.REQUEST:
            await self._handle_request(message)
            return
        
        # Handle notifications
        if message.message_type == MCPMessageType.NOTIFICATION:
            await self._handle_notification(message)
            return
    
    async def _handle_request(self, message: MCPMessage) -> None:
        """Handle MCP request."""
        try:
            result = None
            error = None
            
            # Route to appropriate handler
            if message.method == MCPMessage.Methods.CONTEXT_GET:
                key = message.params.get("key")
                result = self._context_store.get(key)
            
            elif message.method == MCPMessage.Methods.CONTEXT_SET:
                key = message.params.get("key")
                value = message.params.get("value")
                self._context_store[key] = value
                result = {"success": True}
            
            elif message.method == MCPMessage.Methods.CONTEXT_LIST:
                result = {"keys": list(self._context_store.keys())}
            
            elif message.method == MCPMessage.Methods.RESOURCE_LIST:
                result = {
                    "resources": [res.dict() for res in self._resources.values()]
                }
            
            elif message.method == MCPMessage.Methods.TOOL_LIST:
                result = {
                    "tools": [tool.dict() for tool in self._tools.values()]
                }
            
            else:
                error = {
                    "code": -32601,
                    "message": f"Method not found: {message.method}"
                }
            
            # Send response
            response = MCPMessage(
                id=message.id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MCPMessageType.RESPONSE,
                result=result,
                error=error
            )
            
            await self.send(response)
            
        except Exception as e:
            # Send error response
            error_response = MCPMessage(
                id=message.id,
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MCPMessageType.ERROR,
                error={
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            )
            await self.send(error_response)
    
    async def _handle_notification(self, message: MCPMessage) -> None:
        """Handle MCP notification."""
        logger.info(f"Received MCP notification: {message.method}")
        # Custom notification handling can be added here