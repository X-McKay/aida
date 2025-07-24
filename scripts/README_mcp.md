# AIDA MCP Server Scripts

This directory contains scripts and configuration for MCP (Model Context Protocol) servers with AIDA.

## Quick Start

**Important**: The MCP filesystem server is automatically spawned by AIDA agents when needed. You don't need to start it manually!

```python
# The coding worker will automatically start the MCP server
from aida.agents.worker.coding_worker import CodingWorker

worker = CodingWorker("my_worker")
await worker.start()  # This spawns the MCP filesystem server
```

To test the integration:

```bash
# Run the MCP filesystem test
uv run python examples/test_mcp_filesystem.py
```

## Available Scripts

### 1. `start_mcp_filesystem.sh` (Manual Testing Only)

This script is for manual testing of the MCP filesystem server. **You don't need this for normal AIDA usage** - the server is automatically spawned.

**When to use:**
- Debugging MCP communication
- Testing the filesystem server standalone
- Understanding how MCP works

**Usage:**
```bash
# Start server for manual testing
./scripts/start_mcp_filesystem.sh

# With custom directories
WORKSPACE_DIR=/my/project TEMP_DIR=/var/tmp ./scripts/start_mcp_filesystem.sh
```

**Note**: This runs the server in stdio mode (standard input/output), which is how MCP clients communicate with it.

### 2. `start_mcp_servers.sh`

Starts all AIDA-specific MCP servers.

**Available Servers:**
- **Files** (port 8001) - File operations
- **Execution** (port 8002) - Code execution
- **System** (port 8003) - System information
- **Context** (port 8004) - Context management
- **Thinking** (port 8005) - Structured reasoning
- **LLMResponse** (port 8006) - LLM interactions
- **WebSearch** (port 8007) - Web search (requires SearXNG)

**Features:**
- Automatic prerequisite checking (Ollama, SearXNG)
- Background process management
- PID tracking and logging
- Status checking
- Graceful shutdown

**Commands:**
```bash
# Start all servers
./scripts/start_mcp_servers.sh

# Check status of all servers
./scripts/start_mcp_servers.sh status

# Stop all servers
./scripts/start_mcp_servers.sh stop

# View logs
tail -f logs/mcp/<server_name>.log
```

## Prerequisites

### For Basic File Operations

1. **Node.js and npm**
   ```bash
   # Check if installed
   node --version
   npm --version

   # Install if needed
   # Visit: https://nodejs.org/
   ```

### For Full AIDA MCP Suite

1. **Python with uv**
   ```bash
   # AIDA uses uv for dependency management
   uv --version
   ```

2. **Ollama** (for LLM operations)
   ```bash
   # Start Ollama
   ollama serve

   # Pull a model
   ollama pull llama3.2:latest
   ```

3. **Docker** (for WebSearch with SearXNG)
   ```bash
   # Start SearXNG
   docker-compose up -d searxng
   ```

## Integration with Coding Worker

Once MCP servers are running, the coding worker can use them:

```python
from aida.agents.worker.coding_worker import CodingWorker

# Create worker with MCP
worker = CodingWorker("my_worker")

# Start (will connect to MCP servers)
await worker.start()

# The worker will handle tasks delegated by the coordinator
# File operations are handled through the MCP filesystem client
```

## Troubleshooting

### "Port already in use"
```bash
# Find what's using the port
lsof -i :3000

# Kill the process if needed
kill <PID>
```

### "MCP executor not connected"
1. Ensure MCP servers are running
2. Check the logs: `tail -f logs/mcp/*.log`
3. Verify network connectivity

### "npx not found"
Install Node.js from https://nodejs.org/

### "Permission denied"
```bash
# Make scripts executable
chmod +x scripts/*.sh
```

## Environment Variables

### For `start_mcp_filesystem.sh`:
- `MCP_PORT` - Server port (default: 3000)
- `ALLOWED_PATHS` - Comma-separated paths (default: /tmp,AIDA_DIR)

### For `start_mcp_servers.sh`:
- `MCP_LOG_DIR` - Log directory (default: ./logs/mcp)
- `MCP_PID_DIR` - PID file directory (default: ./logs/mcp/pids)
- `PYTHON_CMD` - Python command (default: uv run python)

## Architecture

The MCP servers provide a protocol layer that allows AIDA agents to:
1. Access filesystem operations safely
2. Execute code in controlled environments
3. Perform system operations
4. Manage context and state
5. Interface with LLMs
6. Search the web

Each server wraps an AIDA tool with the MCP protocol, enabling:
- Language-agnostic communication
- Network-based tool access
- Standardized interfaces
- Security boundaries

## Next Steps

1. Start with `start_mcp_filesystem.sh` for basic file operations
2. Use `start_mcp_servers.sh` for full AIDA capabilities
3. Configure the coding agent to use MCP providers
4. Monitor logs for any issues
5. Extend with custom MCP servers as needed
