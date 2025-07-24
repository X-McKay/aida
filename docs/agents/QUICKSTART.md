# Agent System Quick Start Guide

## Running the Coordinator-Worker System

### 1. Basic Example

```python
import asyncio
from aida.agents.coordination.coordinator_agent import CoordinatorAgent
from aida.agents.worker.coding_worker import CodingWorker

async def main():
    # Start coordinator
    coordinator = CoordinatorAgent("main_coordinator")
    await coordinator.start()

    # Start a coding worker
    worker = CodingWorker("worker_001")
    await worker.start()

    # Wait for registration
    await asyncio.sleep(2)

    # Submit a task
    response = await coordinator.execute_request({
        "task_type": "code_analysis",
        "file_path": "example.py"
    })

    print(f"Task result: {response}")

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

# Coordinator automatically load balances
```

### 3. Using the CLI

```bash
# Start coordinator (in one terminal)
uv run aida agent coordinator start

# Start workers (in other terminals)
uv run aida agent worker start --type coding --id worker_001
uv run aida agent worker start --type coding --id worker_002

# Submit tasks via CLI
uv run aida agent task submit --type code_analysis --file src/main.py
```

## Common Tasks

### Code Analysis

```python
response = await coordinator.execute_request({
    "task_type": "code_analysis",
    "file_path": "src/calculator.py",
    "detailed_analysis": True
})

# Response includes:
# - Language detection
# - Code metrics (lines, complexity)
# - Structure analysis (classes, functions)
# - Quality assessment (if detailed=True)
```

### Code Generation

```python
response = await coordinator.execute_request({
    "task_type": "code_generation",
    "specification": "Create a Python function that calculates fibonacci numbers with memoization",
    "language": "python",
    "style": "clean"
})

# Response includes:
# - Generated code
# - Validation results
# - Code analysis metrics
```

### Refactoring

```python
response = await coordinator.execute_request({
    "task_type": "refactoring",
    "file_path": "src/legacy_code.py",
    "objectives": ["improve readability", "reduce complexity", "add type hints"]
})

# Response includes:
# - Original code
# - Refactored code
# - Improvements summary
# - Before/after metrics
```

## Monitoring

### View Active Plans

```python
# Get plan status
plan_id = response["plan_id"]
plan = coordinator.storage.get_plan(plan_id)

print(f"Plan status: {plan.status}")
for step in plan.steps:
    print(f"  Step {step.id}: {step.status} - {step.description}")
```

### Worker Health

```python
# Check worker status
health = await worker.health_check()
print(f"Worker {health['agent_id']}:")
print(f"  State: {health['state']}")
print(f"  Tasks completed: {health['worker_state']['completed_tasks']}")
print(f"  Success rate: {health['worker_state']['success_rate']:.1%}")
```

### Coordinator Stats

```python
# Get coordinator statistics
stats = coordinator.get_statistics()
print(f"Total plans: {stats['total_plans']}")
print(f"Active plans: {stats['active_plans']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Available workers: {len(stats['workers'])}")
```

## Configuration Examples

### High-Performance Setup

```python
# Coordinator with short timeouts
coordinator_config = CoordinatorConfig(
    planning_timeout=10.0,
    execution_timeout=60.0,
    max_retries=1
)

# Workers with high concurrency
worker_config = WorkerConfig(
    max_concurrent_tasks=5,
    task_timeout_multiplier=0.8
)
```

### Reliable Setup

```python
# Coordinator with retries
coordinator_config = CoordinatorConfig(
    planning_timeout=60.0,
    execution_timeout=600.0,
    max_retries=5,
    retry_delay=2.0
)

# Workers with resource limits
worker_config = WorkerConfig(
    max_concurrent_tasks=1,
    sandbox_config=SandboxConfig(
        resource_limits=ResourceLimits(
            cpu_cores=1,
            memory_mb=512,
            timeout_seconds=300
        )
    )
)
```

## Troubleshooting

### Worker Not Registering

```python
# Check A2A connection
print(f"Coordinator A2A: {coordinator.a2a_protocol._server}")
print(f"Worker A2A: {worker.a2a_protocol._client}")

# Verify coordinator endpoint
print(f"Worker connecting to: {worker.worker_config.coordinator_endpoint}")
```

### Task Failures

```python
# Check failed plan details
failed_plans = coordinator.storage.list_plans(status="failed")
for plan_path in failed_plans:
    plan = coordinator.storage.load_plan(plan_path)
    print(f"\nPlan {plan.id} failed:")
    for step in plan.steps:
        if step.status == "failed":
            print(f"  Step {step.id}: {step.error}")
```

### MCP Connection Issues

```python
# Test MCP filesystem directly
from aida.providers.mcp.filesystem_client import MCPFilesystemClient

client = MCPFilesystemClient()
await client.connect()
tools = await client.list_tools()
print(f"Available tools: {[t.get('name') for t in tools]}")

# Test file reading
result = await client.call_tool("read_file", {"path": "test.py"})
print(f"Read result: {result}")
```

## Next Steps

1. Review the [full architecture documentation](README.md)
2. Explore the [test examples](../../aida/agents/tests/)
3. Create custom workers for your use cases
4. Integrate with your existing tools via MCP
