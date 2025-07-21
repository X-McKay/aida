# Quick Fix Common Issues

Quickly fix the most common AIDA development issues.

## Issue Type
$ARGUMENTS

## Common Fixes

### "Module not found" Error
1. Check if `__init__.py` exists in all directories
2. Clear Python cache: `find . -name "*.pyc" -delete`
3. Verify import paths match file structure
4. Run: `uv run python -c "import <module>"`

### "Method not found" Error
1. Check if method was renamed or moved
2. Verify you're importing from correct module
3. Look for old vs new API differences
4. Add compatibility method if needed

### Chat Not Working
1. Ensure Ollama is running: `ollama serve`
2. Check LLM: `uv run python -c "from aida.llm import get_llm; print(len(get_llm().list_purposes()))"`
3. Test orchestrator: `/debug-orchestrator`
4. Run smoke tests: `uv run python smoke_test.py`

### Import Conflicts
1. Check for duplicate files (e.g., both `module.py` and `module/`)
2. Rename old files: `mv module.py module_old.py`
3. Update imports throughout codebase
4. Test affected functionality

### Async/Sync Issues
1. Check if callback should be async or sync
2. Use: `asyncio.iscoroutinefunction(callback)`
3. Handle both cases in code
4. Look for "was never awaited" warnings

## Verification
After any fix:
1. Run smoke tests
2. Test specific functionality
3. Check no new issues introduced
