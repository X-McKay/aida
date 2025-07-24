# Agent System Quick Start Guide

## Using the TodoOrchestrator (Recommended)

The easiest way to use AIDA is through the TodoOrchestrator wrapper:

```python
import asyncio
from aida.core.orchestrator import get_orchestrator

async def main():
    # Get the orchestrator instance
    orchestrator = get_orchestrator()

    # Execute a request (string-based for simplicity)
    result = await orchestrator.execute_request(
        "Analyze the file src/main.py and suggest improvements"
    )

    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Using the Coordinator-Worker System Directly

For more control, you can use the coordinator and workers directly:

### 1. Basic Example

```python
import asyncio
from aida.agents.base import AgentConfig
from aida.agents.coordination.coordinator_agent import CoordinatorAgent
from aida.agents.worker.coding_worker import CodingWorker

async def main():
    # Create coordinator
    coordinator_config = AgentConfig(
        agent_id="main_coordinator",
        agent_type="coordinator",
        capabilities=["planning", "task_delegation"]
    )
    coordinator = CoordinatorAgent(coordinator_config)
    await coordinator.start()

    # Start a coding worker
    worker = CodingWorker("worker_001")
    await worker.start()

    # Wait for registration
    await asyncio.sleep(2)

    # Submit a task as a string (natural language)
    result = await coordinator.execute_request(
        "Analyze the code in example.py and provide suggestions"
    )

    print(f"Task result: {result}")

    # Cleanup
    await worker.stop()
    await coordinator.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Running Multiple Workers

```python
# Start multiple workers for parallel execution
workers = []
for i in range(3):
    worker = CodingWorker(f"worker_{i:03d}")
    await worker.start()
    workers.append(worker)

# Coordinator automatically load balances between workers
```

## Common Use Cases

### Using the CLI (Recommended for Interactive Use)

```bash
# Start interactive chat
uv run aida chat

# Use the TUI for visual monitoring
uv run aida tui

# Run a specific command
uv run aida execute "Create unit tests for calculator.py"
```

### Code Analysis

```python
# Using TodoOrchestrator
orchestrator = get_orchestrator()
result = await orchestrator.execute_request(
    "Analyze src/calculator.py and report code quality metrics"
)
```

### Code Generation

```python
result = await orchestrator.execute_request(
    "Create a Python function that calculates fibonacci numbers with memoization"
)

# The result includes generated code and validation
print(result["result"]["generated_code"])
```

### Code Refactoring

```python
result = await orchestrator.execute_request(
    "Refactor legacy_code.py to improve readability and add type hints"
)
```

### Test Generation

```python
result = await orchestrator.execute_request(
    "Generate comprehensive unit tests for the Calculator class in calc.py"
)
```

## Monitoring and Debugging

### View Plans in the Filesystem

Plans are stored in `.aida/orchestrator/`:

```bash
# List active plans
ls -la .aida/orchestrator/active/

# View a specific plan
cat .aida/orchestrator/active/plan_*.json | jq

# Check archived (completed) plans
ls -la .aida/orchestrator/archived/
```

### Using the TUI for Real-time Monitoring

```bash
# Start the TUI
uv run aida tui

# You'll see:
# - Real-time task progress
# - System resource usage (CPU, Memory, GPU)
# - Active workers
# - Interactive chat
```

### Programmatic Plan Access

```python
# Access plan details
if "plan_id" in result:
    plan_id = result["plan_id"]
    # Plans are stored as JSON files
    import json
    with open(f".aida/orchestrator/active/plan_{plan_id}.json") as f:
        plan = json.load(f)

    print(f"Plan status: {plan.get('status')}")
    for step in plan.get("steps", []):
        print(f"  Step {step['id']}: {step['status']} - {step['description']}")
```

## Configuration

### Using Environment Variables

```bash
# Configure default LLM (Ollama)
export OLLAMA_HOST=http://localhost:11434

# Set working directory for file operations
export AIDA_WORKSPACE=/path/to/project
```

### Programmatic Configuration

```python
# Configure with specific storage location
from aida.core.orchestrator import TodoOrchestrator

orchestrator = TodoOrchestrator(storage_dir="/custom/path/plans")
```

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve

# Pull required model
ollama pull llama3.2
```

### Worker Registration Issues

If workers aren't processing tasks:

1. Check the logs for registration messages
2. Ensure both coordinator and worker are running
3. Verify no firewall blocking local connections

### Task Timeout

For long-running tasks, increase the timeout:

```python
# The orchestrator handles timeouts automatically
# But you can configure them in the worker
worker_config = WorkerConfig(
    agent_id="worker_001",
    task_timeout_seconds=600  # 10 minutes
)
```

### Debugging Failed Tasks

```python
# Check why a task failed
result = await orchestrator.execute_request("Some task")

if result.get("status") == "failed":
    print(f"Error: {result.get('error')}")
    # Check the plan file for detailed step failures
```

## Best Practices

1. **Use Natural Language Requests**
   - The orchestrator understands natural language
   - Be specific about what you want
   - Example: "Create a REST API endpoint for user management with FastAPI"

2. **Monitor with TUI**
   - Use `uv run aida tui` for visual feedback
   - Watch task progress in real-time
   - See system resource usage

3. **Check Plan Storage**
   - Plans are saved in `.aida/orchestrator/`
   - Review completed plans for debugging
   - Archive old plans periodically

4. **Start Simple**
   - Begin with simple requests
   - Gradually increase complexity
   - Use the chat interface for exploration

## Next Steps

1. Try the [interactive chat](../../README.md#quick-start): `uv run aida chat`
2. Explore the [TUI interface](../tui/README.md): `uv run aida tui`
3. Read about [creating custom workers](README.md#creating-new-workers)
4. Learn about the [hybrid tool system](../tools/README.md)
