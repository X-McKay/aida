# TODO-Based Workflow Orchestrator

The new TODO-based orchestrator provides a more intuitive and transparent approach to workflow management in AIDA. Instead of complex workflow graphs, it uses a familiar TODO list format that progressively gets checked off as steps complete.

## Key Features

- **Progressive Checking**: Steps are visually tracked with checkboxes (⬜ → 🔄 → ✅)
- **Plan Re-evaluation**: Plans automatically adapt when steps fail or new information becomes available
- **Dependency Management**: Steps can depend on other steps, ensuring proper execution order
- **Retry Logic**: Failed steps can be automatically retried with exponential backoff
- **Markdown Output**: Plans are displayed as readable TODO.md style lists
- **Live Updates**: Real-time progress tracking with Rich console integration

## Core Components

### TodoStep

Represents a single actionable step in the workflow:

```python
@dataclass
class TodoStep:
    id: str                           # Unique identifier
    description: str                  # Human-readable description
    tool_name: str                   # Tool to execute
    parameters: Dict[str, Any]       # Tool parameters
    status: TodoStatus               # PENDING, IN_PROGRESS, COMPLETED, FAILED, SKIPPED
    dependencies: List[str]          # IDs of prerequisite steps
    retry_count: int                 # Current retry attempt
    max_retries: int                 # Maximum retry attempts
```

### TodoPlan

Represents the complete workflow plan:

```python
@dataclass
class TodoPlan:
    id: str                          # Unique plan identifier
    user_request: str                # Original user request
    analysis: str                    # LLM analysis of the request
    expected_outcome: str            # What should be achieved
    steps: List[TodoStep]            # Ordered list of steps
    plan_version: int                # Version number (increments on replan)
    replan_history: List[Dict]       # History of replanning events
```

## Usage Examples

### Basic Usage

```python
from aida.core.orchestrator import get_todo_orchestrator

# Get orchestrator instance
orchestrator = get_todo_orchestrator()

# Create a plan
plan = await orchestrator.create_plan(
    "Create a Python script that calculates fibonacci numbers"
)

# Execute the plan
result = await orchestrator.execute_plan(plan)

# View as markdown
print(plan.to_markdown())
```

### With Progress Callbacks

```python
def progress_callback(plan: TodoPlan, step: TodoStep):
    print(f"🔄 Starting: {step.description}")

def replan_callback(plan: TodoPlan, reason: ReplanReason) -> bool:
    print(f"🔄 Replanning needed: {reason.value}")
    return True  # Auto-approve replanning

result = await orchestrator.execute_plan(
    plan,
    progress_callback=progress_callback,
    replan_callback=replan_callback
)
```

### CLI Integration

```bash
# Run the demo command
uv run aida todo demo "Analyze sorting algorithm performance"

# List active plans
uv run aida todo list

# Show specific plan
uv run aida todo show plan_123456
```

## Plan Re-evaluation

Plans are automatically re-evaluated when:

1. **Critical Steps Fail**: Steps with no dependencies that fail trigger replanning
2. **User Clarification**: New information from the user requires plan updates
3. **Periodic Checks**: Every 5 completed steps or 10 minutes, whichever comes first
4. **Dependency Changes**: When step dependencies change due to failures

### Replan Process

1. **Trigger Detection**: System detects need for replanning
2. **Context Gathering**: Collect current progress, failures, and new information
3. **LLM Re-analysis**: Send replanning prompt to LLM with current state
4. **Plan Update**: Replace pending steps with new plan, keep completed steps
5. **Version Increment**: Increment plan version and log the replanning event

## Output Format

Plans are displayed as markdown TODO lists:

```markdown
# Workflow Plan: Create a Python script that calculates fibonacci numbers

**Analysis:** Need to create a Python script with fibonacci calculation logic
**Expected Outcome:** Working Python script that can calculate fibonacci sequence
**Plan Version:** 1
**Last Updated:** 2024-01-15 14:30:22

## TODO List:

- ✅ Analyze the problem and design approach
- 🔄 Write Python script with fibonacci function (In Progress)
- ⬜ Test the script with sample inputs
- ⬜ Add error handling and input validation

**Progress:** 1/4 steps completed (25.0%)
```

## Error Handling

The orchestrator includes robust error handling:

- **Automatic Retries**: Steps retry on transient errors (network, timeout)
- **Error Classification**: Different error types trigger different responses
- **Graceful Degradation**: Plans continue executing non-dependent steps
- **Error Reporting**: Clear error messages in the TODO list format

## Testing

Comprehensive test suite covers:

- Unit tests for TodoStep and TodoPlan classes
- Integration tests for the full orchestrator workflow
- Mock LLM and tool interactions
- Error scenarios and recovery
- Dependency resolution
- Replanning logic

Run tests with:

```bash
pytest aida/tests/unit/test_todo_orchestrator.py -v
```

## Comparison with Legacy Orchestrator

| Feature | Legacy Orchestrator | TODO Orchestrator |
|---------|-------------------|------------------|
| Progress Tracking | Complex state machine | Simple TODO checkboxes |
| Plan Visualization | JSON objects | Markdown TODO lists |
| Re-evaluation | Manual only | Automatic + manual triggers |
| Dependencies | Implicit | Explicit step dependencies |
| Error Recovery | Basic retry | Intelligent replanning |
| User Experience | Technical | Intuitive and visual |
| Testing | Monolithic | Modular and testable |

## Migration Guide

To migrate from the legacy orchestrator:

1. **Replace Imports**:
   ```python
   # Old
   from aida.core.orchestrator import get_orchestrator
   
   # New
   from aida.core.orchestrator import get_todo_orchestrator
   ```

2. **Update Method Calls**:
   ```python
   # Old
   result = await orchestrator.execute_request(message)
   
   # New
   plan = await orchestrator.create_plan(message)
   result = await orchestrator.execute_plan(plan)
   ```

3. **Adapt Progress Tracking**:
   ```python
   # Old
   workflow.progress  # Float percentage
   
   # New
   plan.get_progress()  # Dict with detailed stats
   ```

## Future Enhancements

Planned improvements include:

- **Parallel Execution**: Run independent steps concurrently
- **Plan Templates**: Reusable plan templates for common workflows
- **Step Conditions**: Conditional step execution based on previous results
- **Plan Sharing**: Save and share successful plans
- **Metrics Collection**: Detailed analytics on plan execution patterns
- **Interactive Planning**: Web UI for plan creation and monitoring

## Best Practices

1. **Write Clear Descriptions**: Step descriptions should be actionable and specific
2. **Minimize Dependencies**: Only add dependencies when truly necessary
3. **Handle Failures Gracefully**: Design plans that can recover from partial failures
4. **Use Progress Callbacks**: Provide user feedback during long-running plans
5. **Test Plans Thoroughly**: Use the test framework to validate plan logic
6. **Monitor Re-planning**: Track when and why plans get re-evaluated