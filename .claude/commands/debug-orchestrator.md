# Debug Orchestrator Issues

Debug issues with the AIDA orchestrator system.

## Problem
$ARGUMENTS

## Steps

1. **Check Current State**:
   ```python
   # Verify orchestrator type
   from aida.core.orchestrator import get_orchestrator
   orch = get_orchestrator()
   print(f"Orchestrator type: {type(orch)}")
   print(f"Has execute_request: {hasattr(orch, 'execute_request')}")
   print(f"Has create_plan: {hasattr(orch, 'create_plan')}")
   ```

2. **Common Issues**:
   - Old orchestrator.py vs new orchestrator/ directory conflict
   - Missing execute_request compatibility method
   - Async/sync callback mismatches
   - Import path issues

3. **Check File Structure**:
   ```bash
   ls -la aida/core/orchestrator*
   find . -name "*orchestrator*.py" -type f | grep -v __pycache__
   ```

4. **Test Basic Functionality**:
   ```python
   # Test plan creation
   plan = await orch.create_plan("What is 2+2?")
   print(f"Plan created: {plan.id}")
   
   # Test execution
   result = await orch.execute_request("What is 2+2?")
   print(f"Result: {result['status']}")
   ```

5. **Fix Common Problems**:
   - Rename old files to avoid conflicts
   - Add missing compatibility methods
   - Fix import statements
   - Update callback handling

6. **Verify Fix**:
   ```bash
   uv run python smoke_test.py
   ```

## Key Files
- `aida/core/orchestrator/__init__.py`
- `aida/core/orchestrator/orchestrator.py`
- `aida/cli/commands/chat.py` (uses orchestrator)