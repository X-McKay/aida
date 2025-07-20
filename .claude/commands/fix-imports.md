# Fix Import Issues

Diagnose and fix Python import errors in the AIDA codebase.

## Steps

1. Identify the import error from: $ARGUMENTS
2. Check if the issue is:
   - Missing `__init__.py` file
   - Incorrect import path
   - Circular import
   - Module vs package naming conflict (e.g., both `module.py` and `module/` exist)
3. Use `grep` to find where the module is imported:
   ```bash
   grep -r "from.*<module>.*import\|import.*<module>" --include="*.py"
   ```
4. Verify the actual file structure exists
5. Fix the issue:
   - Add missing `__init__.py` files
   - Update import statements
   - Rename conflicting files
   - Resolve circular dependencies
6. Run smoke tests to verify the fix:
   ```bash
   uv run python smoke_test.py
   ```
7. Test the specific functionality that was broken

## Common Patterns

- `aida.core.orchestrator` vs `aida.core.orchestrator.orchestrator`
- Missing config module imports
- Tool registry initialization issues