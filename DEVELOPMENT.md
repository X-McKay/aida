# AIDA Development Guidelines

## Before You Code

1. **Run smoke tests first**:
   ```bash
   uv run python smoke_test.py
   ```
   If these fail, the codebase is already broken. Fix it first.

2. **Test your specific feature**:
   ```bash
   # For chat changes
   uv run aida chat
   
   # For specific tool changes
   uv run python -m aida.cli.main test run --suite <suite_name>
   ```

## Making Changes

### CRITICAL RULES:
1. **NO HARDCODING** - Every function must be generic
2. **NO UNNECESSARY COMPLEXITY** - If you can't explain it simply, it's too complex
3. **TEST IMPORTS FIRST** - Before any big change, verify your imports work

### Safe Change Process:

1. **Small Changes**: Make one change at a time
2. **Test Immediately**: After each change, run:
   ```bash
   uv run python smoke_test.py
   ```
3. **Test the Feature**: Test the specific feature you changed
4. **Commit Often**: Small, working commits are better than large broken ones

## Common Pitfalls to Avoid

### Import Conflicts
- **Problem**: Having both `module.py` and `module/` directory
- **Solution**: Rename old files before creating directories

### API Changes
- **Problem**: Changing method signatures breaks existing code
- **Solution**: Add compatibility methods, don't change existing ones

### Async/Sync Mismatch
- **Problem**: Making sync callbacks async or vice versa
- **Solution**: Check if callback is async before calling:
  ```python
  if asyncio.iscoroutinefunction(callback):
      await callback()
  else:
      callback()
  ```

## Quick Fixes for Common Issues

### "Module not found" errors
1. Check if `__init__.py` exists in all directories
2. Verify imports match actual file structure
3. Run: `find . -name "*.pyc" -delete` to clear cache

### "Method not found" errors
1. Check if you're importing from the right module
2. Verify the method name hasn't changed
3. Look for old vs new API differences

### Chat not working
1. Ensure Ollama is running: `ollama serve`
2. Check LLM system: `uv run python -c "from aida.llm import get_llm; print(get_llm().list_purposes())"`
3. Verify orchestrator: `uv run python -c "from aida.core.orchestrator import get_orchestrator; print(type(get_orchestrator()))"`

## Enable Git Hooks

Run once to enable automatic pre-commit testing:
```bash
./enable_hooks.sh
```

This will run smoke tests before every commit, preventing broken code from being committed.

## Emergency Recovery

If everything is broken:
1. Check recent commits: `git log --oneline -10`
2. Find last working commit
3. Create recovery branch: `git checkout -b recovery <commit-hash>`
4. Fix issues one at a time
5. Run smoke tests after each fix

## Remember

- **Every line of code is potential tech debt**
- **Simple is better than clever**
- **Working is better than perfect**
- **Test early, test often**

The smoke tests are your safety net. Use them.