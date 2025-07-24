# AIDA API Reference

## Core APIs

### TodoOrchestrator

The main entry point for using AIDA programmatically.

```python
from aida.core.orchestrator import get_orchestrator

# Get singleton instance
orchestrator = get_orchestrator()
```

#### Methods

##### `execute_request(request: str | dict, context: dict = None) -> dict`

Execute a user request through the coordinator-worker system.

**Parameters:**
- `request` (str | dict): Natural language request or structured task
- `context` (dict, optional): Additional context for the request

**Returns:**
- `dict`: Result with status, plan_id, result data, and metadata

**Example:**
```python
result = await orchestrator.execute_request(
    "Create a fibonacci function with memoization"
)

# Result structure:
{
    "status": "completed",
    "plan_id": "plan_0001_1234567890",
    "result": {
        "generated_code": "def fibonacci(n)...",
        "validation": {...}
    },
    "message": "Successfully generated code"
}
```

### CoordinatorAgent

The coordinator that manages task planning and delegation.

```python
from aida.agents.base import AgentConfig
from aida.agents.coordination.coordinator_agent import CoordinatorAgent

config = AgentConfig(
    agent_id="coordinator",
    agent_type="coordinator",
    capabilities=["planning", "task_delegation"]
)
coordinator = CoordinatorAgent(config)
```

#### Methods

##### `start() -> None`
Start the coordinator agent and A2A server.

##### `stop() -> None`
Stop the coordinator and cleanup resources.

##### `execute_request(request: str, context: dict = None) -> dict`
Process a user request by creating and executing a plan.

##### `execute_plan(plan: TodoPlan) -> dict`
Execute an existing plan through workers.

##### `create_plan(request: str, context: dict = None) -> TodoPlan`
Create a plan without executing it.

### Worker Agents

Base class for all worker implementations.

```python
from aida.agents.base import WorkerAgent, WorkerConfig

class MyWorker(WorkerAgent):
    def __init__(self, worker_id: str):
        config = WorkerConfig(
            agent_id=worker_id,
            agent_type="my_worker",
            capabilities=["my_capability"]
        )
        super().__init__(config)

    async def execute_task(self, task_data: dict) -> dict:
        # Implement task execution
        pass
```

#### Required Methods

##### `execute_task(task_data: dict) -> dict`
Execute a task assigned by the coordinator.

**Task Data Structure:**
```python
{
    "capability": "code_generation",
    "parameters": {
        "specification": "...",
        "language": "python"
    },
    "timeout": 300.0
}
```

#### Inherited Methods

##### `start() -> None`
Start the worker and register with coordinator.

##### `stop() -> None`
Stop the worker and cleanup.

##### `report_progress(progress: float, message: str) -> None`
Report task progress (0.0 to 1.0).

## Tool APIs

### ToolBase

Base class for all hybrid tools.

```python
from aida.tools.base import ToolBase, ToolResult

class MyTool(ToolBase):
    async def execute(self, **kwargs) -> ToolResult:
        # Native execution
        pass

    def to_pydantic_tool(self) -> list[Callable]:
        # PydanticAI compatibility
        pass

    async def to_mcp_tool(self) -> list[dict]:
        # MCP definitions
        pass
```

### ToolResult

Standard result format for all tools.

```python
@dataclass
class ToolResult:
    success: bool           # Operation success status
    result: Any            # Operation result data
    error: str | None      # Error message if failed
    metadata: dict | None  # Additional metadata
```

### Tool Registry

Access and manage available tools.

```python
from aida.tools import get_tool, list_tools

# Get specific tool
file_tool = get_tool("file_operations")

# List all available tools
tools = list_tools()
# ["file_operations", "system_execution", "websearch", "thinking", "llm_response"]
```

## LLM APIs

### LLM Manager

Manage LLM providers and model selection.

```python
from aida.llm import get_llm

# Get LLM manager instance
manager = get_llm()

# Create LLM client for purpose
llm = manager.get_llm_for_purpose("coding")

# Generate response
response = await llm.generate(
    prompt="Create a hello world function",
    temperature=0.7
)
```

#### Available Purposes
- `DEFAULT` - General purpose tasks
- `CODING` - Code generation and analysis
- `REASONING` - Complex reasoning tasks
- `MULTIMODAL` - Image and text tasks
- `QUICK` - Fast, simple responses

## Plan Models

### TodoPlan

Representation of an execution plan.

```python
from aida.agents.coordination.plan_models import TodoPlan, TodoStep

@dataclass
class TodoPlan:
    id: str                          # Unique plan ID
    user_request: str                # Original request
    context: dict                    # Execution context
    steps: list[TodoStep]           # Execution steps
    status: str                     # pending|active|completed|failed
    created_at: datetime            # Creation timestamp
    updated_at: datetime            # Last update
    metadata: dict                  # Additional data
```

### TodoStep

Individual step in a plan.

```python
@dataclass
class TodoStep:
    id: str                         # Step ID
    description: str                # What to do
    capability: str                 # Required capability
    parameters: dict               # Step parameters
    dependencies: list[str]        # Step IDs this depends on
    status: str                    # pending|active|completed|failed
    assigned_to: str | None        # Worker ID
    result: dict | None           # Execution result
    error: str | None             # Error if failed
    started_at: datetime | None   # Start time
    completed_at: datetime | None # Completion time
```

## Storage APIs

### CoordinatorPlanStorage

Manage plan persistence.

```python
from aida.agents.coordination.storage import CoordinatorPlanStorage

storage = CoordinatorPlanStorage(".aida/orchestrator")

# Save plan
storage.save_plan(plan)

# Load plan
plan = storage.load_plan(plan_id)

# List plans
active_plans = storage.list_plans("active")
archived_plans = storage.list_plans("archived")

# Move plan between directories
storage.move_plan(plan_id, "archived")
```

## Event System

### EventBus

Central event distribution system.

```python
from aida.core.events import EventBus, Event

bus = EventBus()

# Subscribe to events
def handler(event: Event):
    print(f"Received: {event.type}")

bus.subscribe("task.completed", handler)

# Publish events
bus.publish(Event(
    type="task.completed",
    data={"task_id": "123", "result": "success"}
))
```

## A2A Protocol

### Message Format

```python
from aida.core.protocols.a2a import A2AMessage

message = A2AMessage(
    id="msg_123",
    sender_id="worker_001",
    recipient_id="coordinator",
    message_type="task_response",
    payload={
        "task_id": "task_456",
        "status": "completed",
        "result": {...}
    },
    timestamp=datetime.now(),
    requires_ack=True
)
```

### Message Types
- `agent_discovery` - Find available agents
- `agent_registration` - Register with coordinator
- `task_assignment` - Assign task to worker
- `task_response` - Worker task result
- `progress_update` - Task progress report
- `health_check` - Agent health status

## MCP Integration

### MCPExecutor

Execute tools via MCP servers.

```python
from aida.core.mcp_executor import MCPExecutor

executor = MCPExecutor()

# Initialize MCP server
await executor.initialize_server("filesystem", {
    "allowed_directories": ["/workspace"]
})

# Execute tool
result = await executor.execute_tool(
    "filesystem",
    "read_file",
    {"path": "example.py"}
)

# List available tools
tools = await executor.list_tools("filesystem")
```

## CLI Command API

Commands available via `uv run aida`:

### Chat Commands
- `chat` - Start interactive chat session
- `chat --continue` - Continue previous session
- `chat --model MODEL` - Use specific model

### TUI Commands
- `tui` - Start Text User Interface

### Test Commands
- `test list` - List available test suites
- `test run --suite SUITE` - Run specific test suite

### Agent Commands
- `status` - Check system status
- `execute "request"` - Execute single request

## Error Handling

All APIs follow consistent error patterns:

```python
try:
    result = await orchestrator.execute_request("task")
    if result["status"] == "failed":
        print(f"Task failed: {result['error']}")
    else:
        print(f"Success: {result['result']}")
except Exception as e:
    print(f"System error: {e}")
```

## Best Practices

1. **Use Type Hints**
   ```python
   async def process_code(code: str) -> dict[str, Any]:
       # Implementation
   ```

2. **Handle Timeouts**
   ```python
   result = await asyncio.wait_for(
       orchestrator.execute_request("task"),
       timeout=300.0
   )
   ```

3. **Check Status**
   ```python
   if result["status"] == "completed":
       # Handle success
   else:
       # Handle failure
   ```

4. **Use Context**
   ```python
   result = await orchestrator.execute_request(
       "analyze code",
       context={
           "language": "python",
           "style_guide": "pep8"
       }
   )
   ```
