# Coordinator-Worker Protocol Analysis

## Issue Summary
The TaskDispatcher and WorkerAgent are using incompatible message formats, causing all task executions to fail with "Agent does not support task execution" error.

## Root Cause Analysis

### 1. Message Format Mismatch
**Dispatcher sends:**
```python
worker_payload = {
    "task_id": task_data.get("task_id", "unknown"),
    "capability_required": task_data.get("tool_name", "unknown"),
    "task_data": task_data,
    "step": task_data,
}
```

**Worker expects:**
```python
{
    "task_id": str,
    "plan_id": str,
    "step_description": str,
    "capability_required": str,
    "parameters": dict,
    "timeout_seconds": int,
    "priority": int (optional)
}
```

### 2. Missing Fields
- `plan_id`: Worker needs this to track tasks by plan
- `step_description`: Human-readable description
- `parameters`: Direct parameters dict (not nested in task_data)
- `timeout_seconds`: Required for task timeout

### 3. Response Handling
- Worker sends response via `create_response()` method
- Dispatcher expects response payload to have success/error fields
- Need to ensure response format matches expectations

## Fix Strategy

### Option 1: Update Dispatcher (Recommended)
Fix the dispatcher to send the correct message format that workers expect.

**Pros:**
- Maintains existing worker implementation
- Aligns with A2A protocol patterns
- Clear separation of concerns

**Cons:**
- Need to update dispatcher logic

### Option 2: Update Workers
Change workers to accept the dispatcher's format.

**Pros:**
- Minimal changes to new code

**Cons:**
- Would break existing worker implementations
- Not aligned with A2A patterns

## Implementation Plan

1. **Update dispatcher.py** to construct proper worker payload
2. **Add proper field mapping** from TodoStep to worker format
3. **Test with existing integration tests**
4. **Update simple test to verify fix**

## Verification Steps

1. Run existing coordinator-worker test
2. Check that "does not support task execution" error is gone
3. Verify tasks are accepted and processed
4. Confirm revision logic still works on actual failures
