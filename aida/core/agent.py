"""Base agent implementation for AIDA."""

import asyncio
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Type
from datetime import datetime
import logging

from pydantic import BaseModel, Field

from aida.core.protocols.base import Protocol
from aida.core.protocols.a2a import A2AProtocol
from aida.core.protocols.mcp import MCPProtocol
from aida.core.events import Event, EventBus
from aida.core.state import AgentState
from aida.core.memory import MemoryManager


logger = logging.getLogger(__name__)


class AgentCapability(BaseModel):
    """Agent capability descriptor."""
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    required_tools: List[str] = Field(default_factory=list)


class AgentConfig(BaseModel):
    """Agent configuration."""
    agent_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    capabilities: List[AgentCapability] = Field(default_factory=list)
    protocols: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    tools: List[str] = Field(default_factory=list)
    memory_config: Dict[str, Any] = Field(default_factory=dict)
    security_config: Dict[str, Any] = Field(default_factory=dict)
    max_concurrent_tasks: int = 10
    heartbeat_interval: float = 30.0
    

class BaseAgent(ABC):
    """Abstract base class for all AIDA agents."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id or str(uuid.uuid4())
        self.name = config.name
        
        # Core components
        self.state = AgentState(agent_id=self.agent_id)
        self.event_bus = EventBus()
        self.memory = MemoryManager(config.memory_config)
        
        # Communication protocols
        self.protocols: Dict[str, Protocol] = {}
        self._setup_protocols()
        
        # Tools and capabilities
        self.tools: Dict[str, Any] = {}
        self.capabilities: Dict[str, AgentCapability] = {
            cap.name: cap for cap in config.capabilities
        }
        
        # Task management
        self._tasks: Set[asyncio.Task] = set()
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._task_semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        
        # Lifecycle
        self._started = False
        self._shutdown_event = asyncio.Event()
        
        # Setup event handlers
        self._setup_event_handlers()
    
    @abstractmethod
    async def process_message(self, message: Any) -> Optional[Any]:
        """Process an incoming message."""
        pass
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a task."""
        pass
    
    async def start(self) -> None:
        """Start the agent."""
        if self._started:
            return
        
        try:
            # Start core components
            await self.memory.start()
            
            # Connect protocols
            for protocol in self.protocols.values():
                await protocol.connect()
            
            # Start background tasks
            self._tasks.add(asyncio.create_task(self._message_loop()))
            self._tasks.add(asyncio.create_task(self._heartbeat_loop()))
            self._tasks.add(asyncio.create_task(self._maintenance_loop()))
            
            # Update state
            self.state.status = "running"
            self.state.started_at = datetime.utcnow()
            self._started = True
            
            # Emit started event
            await self.event_bus.emit(Event(
                type="agent.started",
                source=self.agent_id,
                data={"agent_id": self.agent_id, "name": self.name}
            ))
            
            logger.debug(f"Agent {self.name} ({self.agent_id}) started")
            
        except Exception as e:
            logger.error(f"Failed to start agent: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the agent."""
        if not self._started:
            return
        
        try:
            # Signal shutdown
            self._shutdown_event.set()
            
            # Cancel all tasks
            for task in self._tasks:
                task.cancel()
            
            for task in self._running_tasks.values():
                task.cancel()
            
            # Wait for tasks to complete
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Disconnect protocols
            for protocol in self.protocols.values():
                await protocol.disconnect()
            
            # Stop memory manager
            await self.memory.stop()
            
            # Update state
            self.state.status = "stopped"
            self._started = False
            
            # Emit stopped event
            await self.event_bus.emit(Event(
                type="agent.stopped",
                source=self.agent_id,
                data={"agent_id": self.agent_id, "name": self.name}
            ))
            
            logger.debug(f"Agent {self.name} ({self.agent_id}) stopped")
            
        except Exception as e:
            logger.error(f"Error stopping agent: {e}")
    
    async def send_message(
        self, 
        protocol_name: str, 
        message: Any, 
        recipient: Optional[str] = None
    ) -> bool:
        """Send a message using specified protocol."""
        protocol = self.protocols.get(protocol_name)
        if not protocol:
            logger.error(f"Protocol {protocol_name} not found")
            return False
        
        try:
            if hasattr(message, 'recipient_id') and recipient:
                message.recipient_id = recipient
            
            return await protocol.send(message)
            
        except Exception as e:
            logger.error(f"Failed to send message via {protocol_name}: {e}")
            return False
    
    async def add_capability(self, capability: AgentCapability) -> None:
        """Add a new capability to the agent."""
        self.capabilities[capability.name] = capability
        
        # Update A2A protocol capabilities
        if "a2a" in self.protocols:
            self.protocols["a2a"].add_capability(capability.name)
        
        # Emit capability added event
        await self.event_bus.emit(Event(
            type="agent.capability_added",
            source=self.agent_id,
            data={"capability": capability.dict()}
        ))
    
    async def remove_capability(self, capability_name: str) -> None:
        """Remove a capability from the agent."""
        if capability_name in self.capabilities:
            del self.capabilities[capability_name]
            
            # Emit capability removed event
            await self.event_bus.emit(Event(
                type="agent.capability_removed",
                source=self.agent_id,
                data={"capability_name": capability_name}
            ))
    
    def add_tool(self, name: str, tool: Any) -> None:
        """Add a tool to the agent."""
        self.tools[name] = tool
        logger.info(f"Added tool {name} to agent {self.agent_id}")
    
    def get_tool(self, name: str) -> Optional[Any]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    async def submit_task(self, task: Dict[str, Any]) -> str:
        """Submit a task for execution."""
        task_id = str(uuid.uuid4())
        task["id"] = task_id
        
        # Create task coroutine
        coro = self._execute_task_with_tracking(task)
        
        # Submit task
        async with self._task_semaphore:
            task_obj = asyncio.create_task(coro)
            self._running_tasks[task_id] = task_obj
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[str]:
        """Get the status of a task."""
        task = self._running_tasks.get(task_id)
        if not task:
            return None
        
        if task.done():
            if task.exception():
                return "failed"
            else:
                return "completed"
        else:
            return "running"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.state.status,
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
            "capabilities": list(self.capabilities.keys()),
            "tools": list(self.tools.keys()),
            "active_tasks": len(self._running_tasks),
            "total_tasks": self.state.tasks_completed + len(self._running_tasks),
            "memory_usage": self.memory.get_basic_stats() if self.memory else {},
            "protocols": {
                name: protocol.get_stats() if hasattr(protocol, 'get_stats') else {}
                for name, protocol in self.protocols.items()
            }
        }
    
    def _setup_protocols(self) -> None:
        """Setup communication protocols."""
        # Setup A2A protocol
        if "a2a" in self.config.protocols:
            a2a_config = self.config.protocols["a2a"]
            self.protocols["a2a"] = A2AProtocol(
                agent_id=self.agent_id,
                **a2a_config
            )
        
        # Setup MCP protocol
        if "mcp" in self.config.protocols:
            mcp_config = self.config.protocols["mcp"]
            self.protocols["mcp"] = MCPProtocol(
                agent_id=self.agent_id,
                **mcp_config
            )
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers."""
        # Register core event handlers
        self.event_bus.subscribe("agent.task_completed", self._on_task_completed)
        self.event_bus.subscribe("agent.task_failed", self._on_task_failed)
        self.event_bus.subscribe("agent.error", self._on_error)
    
    async def _message_loop(self) -> None:
        """Main message processing loop."""
        while not self._shutdown_event.is_set():
            try:
                # Check all protocols for messages
                for protocol_name, protocol in self.protocols.items():
                    message = await protocol.receive()
                    if message:
                        # Process message
                        try:
                            result = await self.process_message(message)
                            
                            # Send response if needed
                            if result and hasattr(message, 'requires_response'):
                                await protocol.send(result)
                                
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in message loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat."""
        while not self._shutdown_event.is_set():
            try:
                # Update state
                self.state.last_heartbeat = datetime.utcnow()
                
                # Emit heartbeat event
                await self.event_bus.emit(Event(
                    type="agent.heartbeat",
                    source=self.agent_id,
                    data={"timestamp": datetime.utcnow().isoformat()}
                ))
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)
    
    async def _maintenance_loop(self) -> None:
        """Periodic maintenance tasks."""
        while not self._shutdown_event.is_set():
            try:
                # Clean up completed tasks
                completed_tasks = [
                    task_id for task_id, task in self._running_tasks.items()
                    if task.done()
                ]
                
                for task_id in completed_tasks:
                    del self._running_tasks[task_id]
                
                # Memory cleanup
                await self.memory.cleanup()
                
                # Wait before next maintenance cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(300)
    
    async def _execute_task_with_tracking(self, task: Dict[str, Any]) -> Any:
        """Execute task with tracking and error handling."""
        task_id = task["id"]
        
        try:
            # Emit task started event
            await self.event_bus.emit(Event(
                type="agent.task_started",
                source=self.agent_id,
                data={"task_id": task_id, "task": task}
            ))
            
            # Execute task
            result = await self.execute_task(task)
            
            # Update state
            self.state.tasks_completed += 1
            
            # Emit task completed event
            await self.event_bus.emit(Event(
                type="agent.task_completed",
                source=self.agent_id,
                data={"task_id": task_id, "result": result}
            ))
            
            return result
            
        except Exception as e:
            # Update state
            self.state.tasks_failed += 1
            
            # Emit task failed event
            await self.event_bus.emit(Event(
                type="agent.task_failed",
                source=self.agent_id,
                data={"task_id": task_id, "error": str(e)}
            ))
            
            raise
        
        finally:
            # Cleanup
            self._running_tasks.pop(task_id, None)
    
    async def _on_task_completed(self, event: Event) -> None:
        """Handle task completed event."""
        logger.info(f"Task {event.data['task_id']} completed")
    
    async def _on_task_failed(self, event: Event) -> None:
        """Handle task failed event."""
        logger.error(f"Task {event.data['task_id']} failed: {event.data['error']}")
    
    async def _on_error(self, event: Event) -> None:
        """Handle error event."""
        logger.error(f"Agent error: {event.data}")


class Agent(BaseAgent):
    """Concrete agent implementation."""
    
    async def process_message(self, message: Any) -> Optional[Any]:
        """Process an incoming message."""
        # Default message processing
        logger.info(f"Processing message: {message}")
        
        # Handle different message types
        if hasattr(message, 'message_type'):
            if message.message_type == "task_request":
                # Submit task for execution
                task_id = await self.submit_task(message.payload)
                
                # Return task ID
                if hasattr(message, 'requires_response'):
                    return message.__class__(
                        sender_id=self.agent_id,
                        recipient_id=message.sender_id,
                        message_type="task_response",
                        payload={"task_id": task_id}
                    )
            
            elif message.message_type == "capability_discovery":
                # Return capabilities
                return message.__class__(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="capability_response",
                    payload={
                        "capabilities": [cap.dict() for cap in self.capabilities.values()]
                    }
                )
        
        return None
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a task."""
        task_type = task.get("type", "unknown")
        
        logger.info(f"Executing task {task['id']} of type {task_type}")
        
        # Basic task execution
        if task_type == "echo":
            return {"result": task.get("data", "Hello from AIDA!")}
        
        elif task_type == "compute":
            # Simulate computation
            await asyncio.sleep(1)
            return {"result": "computation completed"}
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")