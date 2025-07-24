# AIDA - Advanced Intelligent Distributed Agent System

A modern agentic system featuring a coordinator-worker architecture with enterprise-grade security, scalability, and extensibility.

## Architecture Overview

AIDA implements a **coordinator-worker architecture** where specialized agents collaborate to complete complex tasks:

- **CoordinatorAgent** - Plans tasks, delegates work, and manages execution flow
- **Worker Agents** - Specialized agents (CodingWorker, ResearchWorker, etc.) that execute specific tasks
- **Agent-to-Agent (A2A) Protocol** - Enables distributed agent communication
- **Model Context Protocol (MCP)** - Standardized tool integration
- **Hybrid Tool System** - Tools work across native AIDA, PydanticAI, and MCP interfaces
- **Multi-LLM Support** - Ollama (default), OpenAI, Anthropic, and more

## Quick Start

```bash
# Install AIDA
pip install aida-agent

# Start interactive chat
uv run aida chat

# Start the Text User Interface (TUI)
uv run aida tui

# Run a specific test suite
uv run aida test run --suite integration

# Check system status
uv run aida status
```

## Core Components

### 1. Coordinator-Worker Architecture

The system uses a distributed architecture where:
- **CoordinatorAgent** creates execution plans and delegates tasks
- **Worker Agents** execute specialized tasks based on their capabilities
- **WorkerProxy** enables seamless remote worker integration
- **TaskDispatcher** intelligently routes tasks to appropriate workers

### 2. Hybrid Tool System

All tools support three interfaces:
- **Native AIDA** - Direct tool execution via `execute()` method
- **PydanticAI** - Integration with PydanticAI agents
- **MCP Server** - Model Context Protocol for standardized tool access

Example tools:
- `FileOperationsTool` - File I/O via MCP filesystem server
- `SystemExecutionTool` - Secure command execution
- `WebSearchTool` - Web search via MCP SearXNG
- `ThinkingTool` - Structured reasoning and analysis

### 3. LLM System

Flexible LLM integration with purpose-based routing:
- **DEFAULT** - General tasks (Ollama llama3.2)
- **CODING** - Code generation (DeepSeek Coder)
- **REASONING** - Complex analysis (Ollama llama3.2)
- **MULTIMODAL** - Image/text tasks
- **QUICK** - Fast responses

Configuration: `aida/config/llm_profiles.py`

## Directory Structure

```
aida/
├── agents/               # Agent implementations
│   ├── base/            # Base agent classes and protocols
│   ├── coordination/    # CoordinatorAgent and task management
│   ├── worker/          # Worker agents (CodingWorker, etc.)
│   └── mcp/             # MCP server integration
├── core/                # Core infrastructure
│   ├── orchestrator/    # TodoOrchestrator (compatibility wrapper)
│   ├── protocols/       # A2A and MCP protocol implementations
│   └── events/          # Event bus system
├── tools/               # Hybrid tool implementations
│   ├── base.py         # Base tool classes
│   ├── file_operations/ # File I/O tool
│   ├── system/         # System execution tool
│   ├── websearch/      # Web search tool
│   └── thinking/       # Analysis and reasoning tool
├── llm/                 # LLM management system
├── cli/                 # Command-line interface
│   ├── commands/       # CLI commands
│   └── tui/            # Text User Interface
└── tests/              # Comprehensive test suite
```

## Key Features

### 1. Text User Interface (TUI)
- Real-time system monitoring (CPU, Memory, GPU)
- Live task tracking and progress updates
- Interactive chat with visual feedback
- Available agents display

### 2. Plan-Based Execution
- Automatic task decomposition
- Progress tracking and monitoring
- Persistent plan storage in `.aida/orchestrator/`
- Support for replanning and error recovery

### 3. Flexible Tool Integration
- Tools work across multiple AI frameworks
- MCP server integration for standardized access
- Easy tool creation with hybrid support
- Built-in observability via OpenTelemetry

### 4. Production Features
- Comprehensive error handling
- Structured logging
- State persistence
- Graceful shutdown
- Resource management

## Development

### Running Tests

```bash
# Run smoke tests (quick validation)
uv run python tests/scripts/smoke_test.py

# Run unit tests with coverage
uv run pytest aida/tests/unit/ --cov=aida

# Run integration tests
uv run aida test run --suite integration

# Run pre-commit hooks
pre-commit run --all-files
```

### Creating New Tools

Tools inherit from `ToolBase` and implement three interfaces:

```python
from aida.tools.base import ToolBase

class MyTool(ToolBase):
    async def execute(self, **kwargs):
        """Native AIDA interface"""
        pass

    def to_pydantic_tool(self):
        """PydanticAI compatibility"""
        pass

    async def to_mcp_tool(self):
        """MCP server interface"""
        pass
```

### Creating New Workers

Workers inherit from `WorkerAgent`:

```python
from aida.agents.base import WorkerAgent

class MyWorker(WorkerAgent):
    async def execute_task(self, task_data):
        """Execute assigned task"""
        pass
```

## Configuration

### LLM Configuration
Edit `aida/config/llm_profiles.py` to configure LLM providers and models.

### System Configuration
AIDA uses sensible defaults but can be configured via:
- Environment variables
- Configuration files in `.aida/`
- Runtime parameters

## Common Workflows

### 1. Interactive Development
```bash
uv run aida chat
> Create a Python function to calculate fibonacci numbers
> Add unit tests for the function
> Optimize the implementation
```

### 2. TUI Monitoring
```bash
uv run aida tui
# Watch real-time task execution and system resources
```

### 3. Batch Processing
```bash
uv run aida execute "Analyze all Python files in src/ and suggest improvements"
```

## Troubleshooting

### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Pull required model
ollama pull llama3.2
```

### Task Execution Issues
- Check worker logs in `.aida/logs/`
- Verify plans in `.aida/orchestrator/active/`
- Run smoke tests: `uv run python tests/scripts/smoke_test.py`

## License

MIT License - see LICENSE file for details.
