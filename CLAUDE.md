# AIDA Project Context

## Overview
AIDA (Advanced Intelligent Distributed Agent System) is a comprehensive agentic system for complex multi-agent coordination and task execution. It uses a hybrid architecture supporting multiple AI frameworks.

## Key Architecture Decisions

### LLM System
- **Default Provider**: Ollama (no API keys required)
- **Configuration**: `aida/config/llm_profiles.py`
- **Purposes**: DEFAULT, CODING, REASONING, MULTIMODAL, QUICK
- **Manager**: `aida.llm.get_llm()` returns singleton

### Orchestrator
- **Location**: `aida/core/orchestrator/` (NOT orchestrator.py)
- **Main Class**: `TodoOrchestrator`
- **Key Methods**: `create_plan()`, `execute_plan()`, `execute_request()` (compatibility)
- **Import**: `from aida.core.orchestrator import get_orchestrator`

### Tools
All tools follow hybrid architecture pattern supporting:
1. Native AIDA interface (`execute/execute_async`)
2. PydanticAI compatibility
3. MCP (Model Context Protocol) server
4. OpenTelemetry observability

### Chat Interface
- **Command**: `uv run aida chat`
- **Location**: `aida/cli/commands/chat.py`
- **Features**: Shortcuts (?, @, #), sessions, multiline input

## Critical Rules

1. **NO HARDCODING** - Every function must be generic
2. **NO UNNECESSARY COMPLEXITY** - Keep it simple
3. **TEST BEFORE COMMIT** - Always run smoke tests
4. **IMPORTS FIRST** - Verify imports before big changes

## Common Commands

### Development
```bash
# Run smoke tests (ALWAYS run these first)
uv run python tests/scripts/smoke_test.py

# Start chat
uv run aida chat

# Run specific tests
uv run python -m aida.cli.main test run --suite <name>

# List available test suites
uv run python -m aida.cli.main test list
```

### Testing Ollama
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Pull default model
ollama pull llama3.2:latest
```

## Common Issues & Fixes

### Import Errors
- Check for `__init__.py` in all directories
- Look for file vs directory conflicts (e.g., both `module.py` and `module/`)
- Clear cache: `find . -name "*.pyc" -delete`

### Chat Not Working
1. Ensure Ollama is running
2. Check orchestrator has `execute_request` method
3. Verify LLM system initialized
4. Run smoke tests

### Method Not Found
- Check if using old vs new API
- Add compatibility methods if needed
- Don't change existing method signatures

## Project Structure
```
aida/
├── cli/              # CLI interface
│   └── commands/     # CLI commands (chat, test, etc.)
├── core/
│   └── orchestrator/ # TodoOrchestrator (main brain)
├── tools/           # Hybrid architecture tools
├── llm/             # LLM management
├── config/          # Configuration (llm_profiles.py)
└── tests/           # Integration tests
```

## Slash Commands
Use `/command` in Claude Code:
- `/smoke-test` - Run smoke tests
- `/test-chat` - Test chat functionality
- `/fix-imports` - Fix import issues
- `/commit-safe` - Safe commit process
- `/add-tool` - Create new hybrid tool
- `/debug-orchestrator` - Debug orchestrator
- `/quick-fix` - Fix common issues
- `/run-tests` - Run test suites

## Remember
- Smoke tests are your safety net
- Every line is potential tech debt
- Simple is better than clever
- Test early, test often
