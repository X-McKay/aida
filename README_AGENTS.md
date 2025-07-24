# AIDA Agent System

AIDA's agent system provides a distributed architecture for complex task execution with specialized workers coordinated by a central orchestrator.

## Key Features

- **Distributed Execution**: Coordinator-worker architecture for parallel task processing
- **Specialized Workers**: Different worker types for specific capabilities (coding, research, data analysis)
- **MCP Integration**: Model Context Protocol support for enhanced tool access
- **Sandboxed Execution**: Secure task execution in Dagger containers
- **Automatic Retry**: Built-in retry logic with exponential backoff
- **Plan Storage**: Persistent storage of execution plans for debugging and recovery

## Quick Example

```python
import asyncio
from aida.agents.coordination.coordinator_agent import CoordinatorAgent
from aida.agents.worker.coding_worker import CodingWorker

async def analyze_code():
    # Start the system
    coordinator = CoordinatorAgent("coordinator_001")
    worker = CodingWorker("worker_001")

    await coordinator.start()
    await worker.start()

    # Wait for worker registration
    await asyncio.sleep(2)

    # Analyze code
    result = await coordinator.execute_request({
        "task_type": "code_analysis",
        "file_path": "example.py"
    })

    print(f"Analysis complete: {result}")

    # Cleanup
    await worker.stop()
    await coordinator.stop()

asyncio.run(analyze_code())
```

## Architecture Overview

```
┌─────────────────┐
│   User Request  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│   Coordinator   │────▶│   Plan Storage  │
│     Agent       │     │ (.aida/plans/)  │
└────────┬────────┘     └─────────────────┘
         │
         ├─── Discovers Workers ───┐
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│  CodingWorker   │       │ ResearchWorker  │
│   - analyze     │       │   - search      │
│   - generate    │       │   - summarize   │
│   - refactor    │       │   - extract     │
└─────────────────┘       └─────────────────┘
         │                         │
         └──── MCP Servers ────────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│   Filesystem    │   │     Search      │
│   MCP Server    │   │   MCP Server    │
└─────────────────┘   └─────────────────┘
```

## Components

### Coordinator Agent
- Creates execution plans from user requests
- Manages worker discovery and task delegation
- Monitors task progress and handles failures
- Stores plans for debugging and recovery

### Worker Agents
- **CodingWorker**: Code analysis, generation, refactoring, test generation
- **ResearchWorker**: (Planned) Web search, documentation analysis
- **DataWorker**: (Planned) Data processing and visualization

### Communication (A2A Protocol)
- WebSocket-based agent-to-agent communication
- Automatic reconnection and health monitoring
- Message type routing and capability discovery

### MCP Integration
- Filesystem access through MCP servers
- Extensible tool integration
- Sandboxed execution environment

## Documentation

- [Full Architecture Guide](docs/agents/README.md) - Detailed architecture and implementation
- [Quick Start Guide](docs/agents/QUICKSTART.md) - Get up and running quickly
- [Test Examples](aida/agents/tests/) - Integration test examples

## Testing

Run the comprehensive test suite:

```bash
uv run python aida/agents/tests/test_coordinator_worker_improved.py
```

Tests cover:
- Worker registration and discovery
- Code analysis with MCP filesystem
- Code generation with LLM integration
- Plan storage and retrieval
- Error handling and retry logic

## Development Status

✅ **Implemented**
- Coordinator-worker architecture
- CodingWorker with full capabilities
- MCP filesystem integration
- Plan storage and management
- Comprehensive test suite

🚧 **In Progress**
- Full Dagger sandbox execution
- ResearchWorker implementation
- DataWorker implementation

📋 **Planned**
- Dynamic worker scaling
- Cloud worker deployment
- Advanced planning strategies
- Distributed tracing

## Contributing

To add a new worker type:

1. Extend `WorkerAgent` base class
2. Define capabilities in `__init__`
3. Implement `execute_task_logic`
4. Add tests for new capabilities

See [CodingWorker](aida/agents/worker/coding_worker.py) for a complete example.
