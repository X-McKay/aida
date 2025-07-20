# Run Smoke Tests

Run the AIDA smoke tests to verify core functionality before making changes or committing.

## Steps

1. Run `uv run python smoke_test.py` to execute all smoke tests
2. Review the test results:
   - Import verification
   - CLI command availability
   - LLM system initialization
   - Orchestrator functionality
   - Chat session creation
3. If any tests fail:
   - Identify the specific failure
   - Check recent changes that might have caused it
   - Fix the issue before proceeding
4. Provide a clear summary of the test results

## Success Criteria

All 5 smoke tests must pass before proceeding with any commits or major changes.