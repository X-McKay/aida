# AIDA Agent Architecture

## Overview

AIDA implements a **coordinator-worker architecture** where specialized agents collaborate through the Agent-to-Agent (A2A) protocol to complete complex tasks.

## Core Components

### 1. CoordinatorAgent (`aida/agents/coordination/`)

The central orchestrator that manages task execution:

**Key Responsibilities:**
- **Planning**: Creates `TodoPlan` objects with steps and dependencies
- **Worker Management**: Tracks workers via `WorkerProxy` objects
- **Task Delegation**: Routes tasks using `TaskDispatcher` with capability matching
- **Progress Monitoring**: Real-time tracking via A2A messages
- **Storage**: Persists plans to `.aida/orchestrator/`

**Key Classes:**
- `CoordinatorAgent`: Main coordinator implementation
- `TodoPlan`: Plan representation with steps
- `TaskDispatcher`: Intelligent task routing
- `WorkerProxy`: Remote worker representation
- `CoordinatorPlanStorage`: Plan persistence

### 2. Worker Agents (`aida/agents/worker/`)

Specialized agents that execute specific tasks:

**Available Workers:**
- **CodingWorker**: Code generation, analysis, refactoring, testing
  - Capabilities: `code_generation`, `code_analysis`, `code_review`, `test_generation`
  - MCP Integration: Uses filesystem server for file operations

**Worker Features:**
- Inherits from `WorkerAgent` base class
- Auto-registration with coordinator via A2A
- Progress reporting during task execution
- MCP tool integration for enhanced capabilities

### 3. Agent Communication (A2A Protocol)

Built on WebSockets for reliable agent communication:

- **Message Types**: Registration, task assignment, progress updates, completion
- **Automatic Reconnection**: Resilient to network issues
- **Message Routing**: Type-based handler dispatch
- **Health Monitoring**: Heartbeat and status tracking

## Architecture Flow

```
User Request → TodoOrchestrator (compatibility wrapper)
                    ↓
             CoordinatorAgent
                    ↓
            Creates TodoPlan
                    ↓
         TaskDispatcher finds workers
                    ↓
    WorkerProxy delegates to remote workers
                    ↓
         Workers execute tasks
                    ↓
      Results returned via A2A
```

## Creating New Workers

### 1. Implement Worker Class

```python
from aida.agents.base import WorkerAgent, WorkerConfig

class MyWorker(WorkerAgent):
    """Custom worker implementation."""

    def __init__(self, worker_id: str):
        config = WorkerConfig(
            agent_id=worker_id,
            agent_type="my_worker",
            capabilities=["my_capability"],
            allowed_mcp_servers=["filesystem"]  # Optional MCP servers
        )
        super().__init__(config)

    async def execute_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Execute assigned task."""
        capability = task_data["capability"]
        parameters = task_data.get("parameters", {})

        if capability == "my_capability":
            # Implement task logic
            result = await self._do_work(parameters)
            return {"status": "completed", "result": result}
```

### 2. Task Execution

Workers receive tasks with this structure:
```python
{
    "capability": "code_generation",
    "parameters": {
        "specification": "Create a fibonacci function",
        "language": "python"
    },
    "timeout": 300.0
}
```

### 3. Progress Reporting

```python
async def execute_task(self, task_data):
    # Report progress during execution
    await self.report_progress(0.25, "Starting analysis")
    # ... do work ...
    await self.report_progress(0.75, "Generating output")
    # ... finish ...
    return result
```

## MCP Integration

Workers can use MCP servers for enhanced capabilities:

```python
# In worker initialization
self.mcp_executor = MCPExecutor()
await self.mcp_executor.initialize_server("filesystem", {
    "allowed_directories": ["/path/to/workspace"]
})

# Using MCP tools
result = await self.mcp_executor.execute_tool(
    "filesystem",
    "read_file",
    {"path": "example.py"}
)
```

## Plan Storage

Plans are stored in `.aida/orchestrator/`:
- `active/`: Currently executing plans
- `archived/`: Completed plans (auto-archived)
- `failed/`: Failed plans for debugging

Plan files contain:
- User request and context
- Generated steps with dependencies
- Execution status and results
- Timing and performance data

## Testing Workers

```python
# Integration test example
async def test_worker():
    # 1. Create coordinator
    coordinator = CoordinatorAgent(config)
    await coordinator.start()

    # 2. Create and start worker
    worker = MyWorker("test_worker")
    await worker.start()

    # 3. Execute request through coordinator
    result = await coordinator.execute_request(
        "Do something with my capability"
    )

    assert result["status"] == "completed"
```

## Configuration

### Coordinator Configuration
```python
config = AgentConfig(
    agent_id="coordinator",
    agent_type="coordinator",
    capabilities=["planning", "task_delegation"],
    storage_dir=".aida/orchestrator"  # Plan storage location
)
```

### Worker Configuration
```python
config = WorkerConfig(
    agent_id="worker_001",
    agent_type="coding_worker",
    capabilities=["code_generation", "code_analysis"],
    max_concurrent_tasks=3,
    allowed_mcp_servers=["filesystem", "searxng"]
)
```

## TodoOrchestrator (Compatibility Layer)

The `TodoOrchestrator` in `aida/core/orchestrator/` provides backward compatibility:

```python
from aida.core.orchestrator import get_orchestrator

# Get singleton instance
orchestrator = get_orchestrator()

# Execute request (creates plan and manages execution)
result = await orchestrator.execute_request("Create a hello world function")
```

This wrapper:
- Creates and manages a CoordinatorAgent instance
- Provides a simplified API for single requests
- Maintains storage in `.aida/orchestrator/`
- Auto-creates a default CodingWorker

## Best Practices

1. **Capability Design**
   - Use specific, descriptive capability names
   - One capability = one well-defined task type
   - Document required parameters

2. **Error Handling**
   - Return structured error responses
   - Include context for debugging
   - Let coordinator handle retries

3. **Resource Management**
   - Clean up temporary files
   - Handle cancellation gracefully
   - Respect timeout limits

4. **Testing**
   - Test workers in isolation
   - Use integration tests for full flow
   - Verify plan storage and retrieval

## Debugging

### Common Issues

1. **Worker Not Found**
   - Check worker started successfully
   - Verify capability names match
   - Ensure A2A connection established

2. **Task Timeout**
   - Increase timeout in task data
   - Check for blocking operations
   - Add progress reporting

3. **Plan Storage Issues**
   - Verify storage directory exists
   - Check file permissions
   - Monitor disk space

### Useful Commands

```bash
# View active plans
ls -la .aida/orchestrator/active/

# Monitor plan execution
tail -f .aida/logs/coordinator.log

# Check worker registration
grep "Worker registered" .aida/logs/coordinator.log
```
