# Test Chat Functionality

Thoroughly test the AIDA chat interface to ensure it's working correctly.

## Steps

1. First run smoke tests with `uv run python smoke_test.py`
2. If smoke tests pass, start the chat interface:
   ```bash
   uv run aida chat
   ```
3. Test these critical scenarios:
   - Simple question: "What is 2+2?"
   - File operation: "List all Python files in the current directory"
   - Multi-tool: "Create a hello.py script and run it"
   - Tool shortcut: "@file_operations operation=list path=."
   - Help command: "/help"
4. Verify each test works correctly
5. Exit with ".exit"
6. Report any issues found

## Arguments

If specific functionality is mentioned in $ARGUMENTS, focus testing on that area.