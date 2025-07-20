# Add New Tool to AIDA

Create a new tool for AIDA with the hybrid architecture pattern.

## Tool Name
$ARGUMENTS

## Steps

1. Create the tool file in `aida/tools/<tool_name>.py`
2. Implement the hybrid architecture with:
   - Core AIDA interface (execute/execute_async methods)
   - PydanticAI compatibility (`to_pydantic_tools()`, `register_with_pydantic_agent()`)
   - MCP server interface (`get_mcp_server()`)
   - OpenTelemetry observability (`enable_observability()`)
3. Follow the pattern from existing tools like `FileOperationsTool` or `SystemTool`
4. Create integration tests in `aida/tests/integration/test_hybrid_<tool_name>.py`
5. Test all four interfaces:
   - AIDA native
   - PydanticAI
   - MCP
   - Observability
6. Add the tool to the registry in `aida/tools/base.py`
7. Update documentation in `aida/docs/`
8. Run integration tests:
   ```bash
   uv run python -m aida.cli.main test run --suite hybrid_<tool_name>
   ```
9. Update CHANGELOG.md

## Key Requirements

- No hardcoding or special cases
- Generic functionality only
- Comprehensive error handling
- Clear documentation
- All tests must pass
