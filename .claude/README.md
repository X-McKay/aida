# Claude Code Configuration for AIDA

This directory contains Claude Code configurations to make development easier and prevent common issues.

## Available Slash Commands

Type `/` in Claude Code to see these commands:

### Testing & Validation
- `/smoke-test` - Run smoke tests before commits
- `/test-chat` - Test chat functionality
- `/run-tests [suite]` - Run specific test suite or all tests

### Development Workflow
- `/commit-safe` - Safe commit process with all checks
- `/add-tool <name>` - Create new hybrid architecture tool
- `/fix-imports` - Fix Python import issues
- `/quick-fix <issue>` - Quick fixes for common problems

### Debugging
- `/debug-orchestrator` - Debug orchestrator issues

## How to Use

1. **In VSCode with Claude Code**:
   ```
   You: /smoke-test
   Claude: [Runs smoke tests and reports results]
   ```

2. **With Arguments**:
   ```
   You: /run-tests hybrid_files
   Claude: [Runs specific test suite]

   You: /add-tool WebSearchTool
   Claude: [Creates new tool with hybrid architecture]
   ```

3. **Natural Language**:
   You can also use natural language that Claude will map to commands:
   - "run the smoke tests" → `/smoke-test`
   - "test the chat" → `/test-chat`
   - "help me commit safely" → `/commit-safe`

## Configuration Files

### CLAUDE.md
Project context and memory - Claude reads this to understand the project structure, common issues, and development guidelines.

### settings.json
- **Hooks**: Automatic syntax checking before edits
- **Allowed Tools**: Pre-approved tools for Claude to use
- **Quick Actions**: Common commands for easy access

### commands/
Each `.md` file is a slash command. Add your own by creating new files here.

## Best Practices

1. **Always run `/smoke-test` before major changes**
2. **Use `/commit-safe` instead of manual commits**
3. **Run `/test-chat` after any chat-related changes**
4. **Use `/quick-fix` for common issues**

## Adding Custom Commands

Create a new file in `commands/` directory:

```markdown
# Command Name

Description of what this command does.

## Arguments
$ARGUMENTS

## Steps
1. First step
2. Second step
3. etc.
```

The `$ARGUMENTS` placeholder will be replaced with whatever you type after the command.
