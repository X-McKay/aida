# AIDA Agent Architecture

## Overview

AIDA uses a distributed agent architecture with specialized workers coordinated by a central orchestrator. This design enables parallel task execution, specialized capabilities, and scalable workloads.

## Core Components

### 1. Coordinator Agent (`aida/agents/coordination/`)

The coordinator serves as the central brain of the system:

- **Task Planning**: Creates execution plans from user requests
- **Worker Management**: Tracks available workers and their capabilities
- **Task Delegation**: Routes tasks to appropriate workers based on capabilities
- **Progress Monitoring**: Tracks task execution and handles failures
- **Plan Storage**: Persists plans for debugging and recovery

Key features:
- Deterministic planning for simple tasks
- LLM-based planning for complex requests
- Automatic retry with exponential backoff
- Plan versioning and archival

### 2. Worker Agents (`aida/agents/worker/`)

Specialized agents that execute specific types of tasks:

- **CodingWorker**: Code analysis, generation, refactoring, test generation
- **ResearchWorker**: (Planned) Web search, documentation lookup
- **DataWorker**: (Planned) Data analysis, visualization

Workers features:
- Auto-registration with coordinator
- Capability-based task routing
- Progress reporting
- Sandboxed execution (Dagger containers)
- MCP tool integration

### 3. Communication Protocol (A2A)

Agent-to-Agent (A2A) protocol built on WebSockets:

- Bidirectional message passing
- Automatic reconnection
- Message type routing
- Health monitoring

## Architecture Flow

```
User Request
     │
     ▼
Coordinator Agent
     │
     ├─► Creates Plan (TodoPlan)
     │   └─► Steps with dependencies
     │
     ├─► Discovers Workers
     │   └─► Capability matching
     │
     └─► Delegates Tasks
         │
         ▼
Worker Agents (parallel)
     │
     ├─► Execute in Sandbox
     ├─► Use MCP Tools
     └─► Report Progress
         │
         ▼
Results Aggregation
```

## MCP (Model Context Protocol) Integration

Workers can use MCP servers for enhanced capabilities:

### Filesystem Access
```python
# MCP filesystem server provides safe file operations
result = await self._fs_client.execute_tool("read_file", {"path": file_path})

# Response format:
{
  "content": [
    {
      "type": "text",
      "text": "file contents here"
    }
  ]
}
```

### Adding New MCP Servers
1. Configure in `WorkerConfig.allowed_mcp_servers`
2. Base agent automatically initializes connections
3. Access via `self.mcp_clients[server_name]`

## Creating a New Worker

### 1. Define the Worker Class

```python
from aida.agents.base import WorkerAgent, WorkerConfig

class MyWorker(WorkerAgent):
    """Worker for specific tasks."""

    def __init__(self, worker_id: str, config: Optional[WorkerConfig] = None):
        if config is None:
            config = WorkerConfig(
                agent_id=worker_id,
                agent_type="my_worker",
                capabilities=["capability1", "capability2"],
                allowed_mcp_servers=["filesystem"],  # Optional
            )
        super().__init__(config)

    async def execute_task_logic(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Implement task execution."""
        capability = task_data["capability"]
        parameters = task_data["parameters"]

        # Task implementation
        if capability == "capability1":
            result = await self._do_capability1(parameters)

        return result
```

### 2. Register Worker Capabilities

Workers automatically register their capabilities with the coordinator on startup.

### 3. Handle Task Assignment

The coordinator will route tasks based on capabilities:

```python
# Coordinator finds workers
workers = self._get_workers_for_capability("capability1")

# Selects best worker (load balancing, performance metrics)
worker = self._select_best_worker(workers)

# Delegates task
await self._delegate_to_worker(worker_id, task_data)
```

## Testing

### Integration Tests

```python
# See aida/agents/tests/test_coordinator_worker_improved.py

async def test_worker_integration():
    # 1. Start coordinator
    coordinator = CoordinatorAgent(config)
    await coordinator.start()

    # 2. Start worker
    worker = CodingWorker("worker1", config)
    await worker.start()

    # 3. Wait for registration
    await asyncio.sleep(2)

    # 4. Submit task
    response = await coordinator.execute_request({
        "task_type": "code_analysis",
        "file_path": "example.py"
    })

    # 5. Check results
    assert response["status"] == "completed"
```

### Test Utilities

- `TestResult`: Assertion tracking with detailed error messages
- Plan storage verification
- Worker registration checks
- Error handling validation

## Configuration

### Coordinator Configuration

```python
from aida.agents.coordination.models import CoordinatorConfig

config = CoordinatorConfig(
    agent_id="coordinator_001",
    planning_timeout=30.0,
    execution_timeout=300.0,
    max_retries=3,
    storage_path=".aida/coordinator/plans"
)
```

### Worker Configuration

```python
from aida.agents.base import WorkerConfig, SandboxConfig

config = WorkerConfig(
    agent_id="worker_001",
    agent_type="coding_worker",
    capabilities=["code_analysis", "code_generation"],
    max_concurrent_tasks=3,
    allowed_mcp_servers=["filesystem"],
    sandbox_config=SandboxConfig(
        isolation_level=SandboxIsolationLevel.CONTAINER,
        resource_limits=ResourceLimits(cpu_cores=2, memory_mb=1024)
    )
)
```

## Debugging

### Plan Storage

Plans are stored in `.aida/coordinator/plans/`:
- `active/`: Currently executing plans
- `archived/`: Successfully completed plans
- `failed/`: Failed plans for debugging

### Viewing Plans

```bash
# List all plans
ls -la .aida/coordinator/plans/archived/

# View specific plan
cat .aida/coordinator/plans/archived/plan_0001_*.json | jq
```

### Common Issues

1. **"No code provided for analysis"**
   - Check MCP filesystem connection
   - Verify file exists and is readable
   - Check response parsing for content extraction

2. **Worker not found**
   - Ensure worker started and registered
   - Check capability names match exactly
   - Verify A2A connection established

3. **Task timeout**
   - Increase timeout in configuration
   - Check worker health/performance
   - Review task complexity

## Best Practices

1. **Capability Design**
   - Keep capabilities focused and specific
   - Use clear, descriptive names
   - Document required parameters

2. **Error Handling**
   - Always return structured errors
   - Include context in error messages
   - Let coordinator handle retries

3. **Progress Reporting**
   - Report progress at meaningful intervals
   - Include descriptive status messages
   - Update at 10%, 30%, 60%, 90%

4. **Resource Management**
   - Configure appropriate sandbox limits
   - Clean up temporary resources
   - Handle cancellation gracefully

## Available Workers

### CodingWorker
Handles code-related tasks through MCP filesystem integration:
- **code_analysis**: Analyzes code structure, complexity, and quality
- **code_generation**: Generates code based on specifications
- **code_review**: Reviews code for best practices and issues
- **refactoring**: Suggests and implements code improvements

## Future Enhancements

1. **Dynamic Worker Scaling**
   - Auto-spawn workers based on load
   - Cloud-based worker pools
   - Resource-aware scheduling

2. **Advanced Planning**
   - Multi-step dependency resolution
   - Parallel execution optimization
   - Cost-aware planning

3. **Enhanced Observability**
   - OpenTelemetry integration
   - Distributed tracing
   - Performance metrics dashboard
