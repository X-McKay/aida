# Safe Commit Process

Safely commit changes after running all necessary checks.

## Steps

1. **Pre-flight Checks**:
   ```bash
   # Run smoke tests
   uv run python smoke_test.py
   ```
   If any fail, fix them first.

2. **Review Changes**:
   ```bash
   git status
   git diff
   ```
   Ensure no unwanted files or debug code

3. **Run Specific Tests** (if applicable):
   - For tool changes: `uv run python -m aida.cli.main test run --suite <tool_name>`
   - For chat changes: Test chat manually
   - For LLM changes: `uv run python -m aida.cli.main test run --suite llm`

4. **Stage Changes**:
   ```bash
   git add -A  # Or selectively add files
   ```

5. **Create Commit**:
   - Use conventional commit format (feat:, fix:, docs:, test:, refactor:)
   - Write clear, descriptive message
   - Include "why" not just "what"

6. **Final Verification**:
   - Commit will trigger pre-commit hook automatically
   - If it fails, fix issues and try again

7. **Push Changes**:
   ```bash
   git push origin <branch-name>
   ```

## Context
Working on: $ARGUMENTS

## Remember
- Every commit should leave the codebase in a working state
- Small, focused commits are better than large ones
- Run smoke tests BEFORE and AFTER changes
