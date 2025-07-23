# AIDA - Advanced Intelligent Distributed Agent System

A comprehensive, production-ready agentic system designed for complex multi-agent applications with enterprise-grade security, scalability, and extensibility.

## Architecture Overview

AIDA implements a modular, plugin-based architecture with the following core components:

- **Agent-to-Agent (A2A) Communication Protocol** - Distributed agent coordination
- **Model Context Protocol (MCP) Integration** - Efficient context management
- **Containerized Execution Environment** - Secure task execution via Dagger.io
- **Multi-LLM Provider Support** - Flexible AI model integration
- **Rich CLI Interface** - Interactive and batch execution modes
- **Comprehensive Tool Suite** - File operations, system execution, thinking tools
- **Advanced Security** - Sandboxing, audit logging, and access controls

## Quick Start

```bash
# Install with uv
uv add aida

# Initialize a new project
aida init my-project

# Start interactive mode
aida interactive

# Execute a task
aida execute "Analyze the codebase and suggest improvements"
```

## Testing & Quality Assurance

AIDA maintains high code quality with comprehensive testing:
- **510 passing unit tests** with **54.21% code coverage**
- Comprehensive protocol test coverage (A2A: 63%, MCP: 49%)
- Integration tests for LLM and orchestrator systems
- Real functionality testing without mocks
- Automated CI/CD with coverage requirements

Run tests:
```bash
# Run all unit tests with coverage
uv run pytest aida/tests/unit/ --cov=aida

# Run integration tests
uv run python -m aida.cli.main test run --suite integration

# List available test suites
uv run python -m aida.cli.main test list
```

## Directory Structure

```
aida/
├── core/              # Core system components
│   ├── orchestrator/  # Modular orchestration system
│   ├── protocols/     # A2A and MCP protocols
│   ├── events/        # Event system and messaging
│   ├── state/         # State management
│   └── memory/        # Context and memory management
├── tools/             # Hybrid tool implementations (v2.0.0)
│   ├── execution/     # Task execution (Dagger.io)
│   ├── thinking/      # Reasoning and analysis tools
│   ├── files/         # File operations (native + MCP support)
│   ├── system/        # System execution tools
│   ├── context/       # Context management tools
│   └── llm_response/  # LLM response formatting tools
├── llm/               # LLM system with PydanticAI integration
├── cli/               # Enhanced CLI with chat interface
│   ├── commands/      # CLI commands (chat, test, etc.)
│   └── ui/            # Rich terminal UI components
├── config/            # Configuration and model profiles
├── providers/         # MCP provider implementations
├── templates/         # Project templates
└── tests/             # Comprehensive test suite
    ├── unit/          # 510+ unit tests (54.21% coverage)
    ├── integration/   # Real LLM and orchestrator tests
    └── scripts/       # Test utilities and smoke tests
```

## Features

### Core Capabilities
- **Multi-Agent Coordination** - Distributed agent networks with A2A communication
- **Secure Execution** - Containerized task execution with Dagger.io
- **Context Management** - Intelligent memory and conversation state management
- **Tool Ecosystem** - Comprehensive suite of built-in tools
- **MCP Filesystem Integration** - Optional support for official MCP filesystem server

### Security & Safety
- **Sandboxed Execution** - Isolated execution environments
- **Principle of Least Privilege** - Minimal access rights
- **Audit Logging** - Comprehensive operation tracking
- **Input Validation** - Rigorous security controls

### Development Experience
- **Modern Python Toolchain** - uv, ruff, mypy, pre-commit hooks
- **Rich CLI Interface** - Interactive terminal UI with progress tracking
- **Hot-Reloading** - Dynamic configuration updates
- **Plugin System** - Extensible architecture

### LLM Integration
- **Multi-Provider Support** - Ollama, OpenAI, Anthropic, Cohere, VLLM
- **Intelligent Routing** - Model selection and fallback
- **Local & Cloud** - Support for both local and remote models

## Configuration

AIDA uses YAML/TOML configuration with schema validation:

```yaml
# aida.config.yaml
system:
  name: "aida"
  version: "1.0.0"

security:
  sandbox_enabled: true
  audit_logging: true

providers:
  llm:
    default: "ollama"
    fallback: ["openai", "anthropic"]

agents:
  max_concurrent: 10
  timeout: 300
```

## License

MIT License - see LICENSE file for details.
