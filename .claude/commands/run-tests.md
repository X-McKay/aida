# Run AIDA Tests

Run specific AIDA test suites or all tests.

## Test Target
$ARGUMENTS

## Available Test Suites

- `llm` - LLM system with PydanticAI integration
- `orchestrator` - Todo Orchestrator with real LLM calls
- `hybrid_files` - Hybrid FileOperationsTool
- `hybrid_system` - Hybrid SystemTool
- `hybrid_execution` - Hybrid ExecutionTool
- `hybrid_context` - Hybrid ContextTool
- `chat_cli` - Chat CLI integration tests

## Steps

1. **Run Smoke Tests First**:
   ```bash
   uv run python smoke_test.py
   ```

2. **Run Specific Suite** (if specified):
   ```bash
   uv run python -m aida.cli.main test run --suite $ARGUMENTS
   ```
   
   Or with verbose output:
   ```bash
   uv run python -m aida.cli.main test run --suite $ARGUMENTS --verbose
   ```

3. **Run All Tests** (if no specific suite):
   ```bash
   uv run python -m aida.cli.main test run --verbose
   ```

4. **For Chat Testing**:
   Also manually test: `uv run aida chat`

5. **Review Results**:
   - Check success rate
   - Identify any failures
   - Note performance metrics
   - Verify tool usage patterns

## Quick Commands

- List available suites: `uv run python -m aida.cli.main test list`
- Quick smoke test: `uv run python -m aida.cli.main test quick`
- Chat CLI tests: `uv run python aida/tests/integration/chat_test_runner.py`