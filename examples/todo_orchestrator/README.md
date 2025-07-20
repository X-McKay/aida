# TODO Orchestrator Examples

This directory contains examples and demos for the new TODO-based workflow orchestrator in AIDA.

## Files

- **`cli_demo.py`** - CLI integration demo using typer
- **`standalone_example.py`** - Standalone orchestrator example with mock dependencies
- **`README.md`** - This documentation file

## CLI Demo (`cli_demo.py`)

A typer-based CLI application that demonstrates the TODO orchestrator integrated with AIDA's full infrastructure.

### Commands

#### Demo Command
Run a workflow with live progress updates:

```bash
python cli_demo.py demo "Create a Python script that calculates fibonacci numbers"
python cli_demo.py demo "Analyze sorting algorithm performance" --no-auto-replan
python cli_demo.py demo "Build a web scraper" --no-show-live
```

Options:
- `--auto-replan/--no-auto-replan` - Automatically replan on failures (default: enabled)
- `--show-live/--no-show-live` - Show live progress updates (default: enabled)

#### List Plans
View all active plans:

```bash
python cli_demo.py list
python cli_demo.py list --format json
python cli_demo.py list --format markdown
```

#### Show Plan
View a specific plan in detail:

```bash
python cli_demo.py show plan_123456
```

## Standalone Example (`standalone_example.py`)

A completely standalone example that demonstrates the TODO orchestrator without requiring the full AIDA environment. Uses mock LLM and tool implementations.

### Commands

#### Run Example
Execute a sample workflow:

```bash
# Default fibonacci example
python standalone_example.py run

# Custom request
python standalone_example.py run --request "Analyze data structures"

# Quiet mode
python standalone_example.py run --quiet
```

#### Test Features
Run comprehensive feature tests:

```bash
python standalone_example.py test
```

#### Interactive Mode
Interactive demo allowing custom requests:

```bash
python standalone_example.py interactive
```

## Key Features Demonstrated

### 1. TODO-Style Progress Tracking

Plans are displayed as familiar markdown TODO lists:

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

### 2. Progressive Execution

Steps are executed in order, respecting dependencies:

```python
# Step dependencies ensure proper execution order
step_2 = TodoStep(
    id="step_002",
    description="Write implementation",
    tool_name="execution",
    parameters={"code": "..."},
    dependencies=["step_001"]  # Must wait for step_001 to complete
)
```

### 3. Automatic Replanning

Plans automatically adapt to changing conditions:

- **Step Failures**: Critical failures trigger replanning
- **User Input**: New information updates the plan
- **Periodic Checks**: Regular re-evaluation ensures plan remains relevant

### 4. Rich Progress Display

Live updating console display with:

- Real-time TODO list updates
- Progress percentages
- Step status indicators
- Plan version tracking
- Error reporting

## Usage Patterns

### Basic Workflow

```python
from aida.core.orchestrator import get_todo_orchestrator

# Get orchestrator
orchestrator = get_todo_orchestrator()

# Create plan
plan = await orchestrator.create_plan("Your request here")

# Execute with callbacks
result = await orchestrator.execute_plan(
    plan,
    progress_callback=lambda p, s: print(f"Starting: {s.description}"),
    replan_callback=lambda p, r: True  # Auto-approve replanning
)

# View results
print(plan.to_markdown())
```

### Custom Progress Handling

```python
def progress_callback(plan: TodoPlan, step: TodoStep):
    """Called when a step starts executing."""
    print(f"🔄 Executing: {step.description}")
    # Log to file, update UI, etc.

def replan_callback(plan: TodoPlan, reason: ReplanReason) -> bool:
    """Called when replanning is needed."""
    if reason == ReplanReason.USER_CLARIFICATION:
        return True  # Always accept user clarifications
    
    # Ask user for other types
    return input(f"Replan needed ({reason.value}). Continue? (y/n): ").lower() == 'y'
```

### Standalone Testing

The standalone example shows how to test the orchestrator without full AIDA setup:

```python
# Mock dependencies
orchestrator = StandaloneOrchestrator()

# Create and execute plan
plan = await orchestrator.create_plan("Test request")
result = await orchestrator.execute_plan(plan)

# Validate results
assert result["status"] == "completed"
assert len(plan.steps) > 0
```

## Dependencies

### CLI Demo
- Full AIDA installation
- Access to LLM providers
- Tool registry with real tools

### Standalone Example  
- Python 3.8+
- typer
- rich (for console output)
- aida.core.orchestrator (imports path adjusted automatically)

## Integration Notes

### Adding to AIDA CLI

To integrate the TODO commands into the main AIDA CLI:

```python
# In aida/cli/main.py
from examples.todo_orchestrator.cli_demo import todo_app

# Add to main app
app.add_typer(todo_app, name="todo")
```

### Custom Tool Integration

The orchestrator works with any tool that implements the AIDA tool interface:

```python
# Custom tool example
class MyCustomTool:
    def get_capability(self) -> ToolCapability:
        return ToolCapability(
            name="my_tool",
            description="Custom tool description",
            parameters=[...]
        )
    
    async def execute_async(self, **kwargs) -> ToolResult:
        # Tool implementation
        return ToolResult(status="success", result={...})
```

## Best Practices

1. **Clear Descriptions**: Write step descriptions that are actionable and specific
2. **Minimal Dependencies**: Only add dependencies when truly necessary
3. **Error Handling**: Design plans that can recover from partial failures
4. **Progress Feedback**: Use callbacks to provide user feedback during long-running plans
5. **Testing**: Use the standalone example pattern for testing plan logic

## Troubleshooting

### Common Issues

**LLM Not Available**: The standalone example includes fallback logic for when LLM providers aren't configured.

**Tool Not Found**: Ensure tools are properly registered in the tool registry before plan execution.

**Permission Errors**: Some tools may require specific permissions or environment setup.

**Plan Stuck**: If a plan stops progressing, check for dependency cycles or missing prerequisites.

### Debug Mode

Enable debug logging to see detailed execution information:

```python
import logging
logging.getLogger('aida.core.orchestrator').setLevel(logging.DEBUG)
```

## Future Enhancements

- **Parallel Execution**: Run independent steps concurrently
- **Plan Templates**: Reusable plan templates for common workflows  
- **Web UI**: Browser-based plan monitoring and control
- **Plan Persistence**: Save and restore plans across sessions
- **Metrics Collection**: Detailed analytics on plan execution patterns