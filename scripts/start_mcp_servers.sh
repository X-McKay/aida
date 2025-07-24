#!/bin/bash

# AIDA MCP Servers Startup Script
# This script starts all available MCP servers for AIDA

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MCP_LOG_DIR="${MCP_LOG_DIR:-./logs/mcp}"
MCP_PID_DIR="${MCP_PID_DIR:-./logs/mcp/pids}"
PYTHON_CMD="${PYTHON_CMD:-uv run python}"

# Create directories
mkdir -p "$MCP_LOG_DIR" "$MCP_PID_DIR"

echo -e "${GREEN}=== AIDA MCP Servers Startup ===${NC}\n"

# Function to start an MCP server
start_mcp_server() {
    local server_name=$1
    local module_path=$2
    local port=$3
    local extra_args="${4:-}"

    echo -e "${YELLOW}Starting $server_name MCP Server...${NC}"

    # Check if already running
    local pid_file="$MCP_PID_DIR/${server_name}.pid"
    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo -e "${YELLOW}  $server_name is already running (PID: $(cat "$pid_file"))${NC}"
        return 0
    fi

    # Start the server
    local log_file="$MCP_LOG_DIR/${server_name}.log"

    # Create a Python script to run the MCP server
    cat > "$MCP_LOG_DIR/${server_name}_runner.py" << EOF
import asyncio
import sys
sys.path.insert(0, '.')

async def main():
    try:
        # Import the server module
        module = __import__('$module_path', fromlist=['${server_name}MCPServer'])
        server_class = getattr(module, '${server_name}MCPServer')

        # Create and run server
        server = server_class()
        print(f"Starting {server_class.__name__} on port $port...")

        # For servers that have a run method
        if hasattr(server, 'run'):
            await server.run()
        else:
            # Keep the server running
            print(f"{server_class.__name__} initialized and ready")
            await asyncio.Event().wait()

    except Exception as e:
        print(f"Error starting $server_name: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
EOF

    # Run the server in background
    # shellcheck disable=SC2086
    # extra_args is intentionally unquoted to allow multiple arguments
    nohup "$PYTHON_CMD" "$MCP_LOG_DIR/${server_name}_runner.py" $extra_args > "$log_file" 2>&1 &
    local pid=$!

    # Save PID
    echo $pid > "$pid_file"

    # Wait a moment to check if it started successfully
    sleep 2

    if kill -0 "$pid" 2>/dev/null; then
        echo -e "${GREEN}  ✓ $server_name started successfully (PID: $pid)${NC}"
        echo -e "  Log: $log_file"
    else
        echo -e "${RED}  ✗ Failed to start $server_name${NC}"
        echo -e "  Check log: $log_file"
        tail -n 10 "$log_file"
        return 1
    fi
}

# Function to check Docker for SearXNG
check_searxng() {
    echo -e "${YELLOW}Checking SearXNG for WebSearch MCP Server...${NC}"

    if command -v docker &> /dev/null; then
        if docker ps | grep -q searxng; then
            echo -e "${GREEN}  ✓ SearXNG is running${NC}"
            return 0
        else
            echo -e "${YELLOW}  ! SearXNG is not running${NC}"
            echo -e "  To start SearXNG: ${GREEN}docker-compose up -d searxng${NC}"
            echo -e "  WebSearch MCP Server will have limited functionality"
            return 1
        fi
    else
        echo -e "${YELLOW}  ! Docker not found${NC}"
        echo -e "  WebSearch MCP Server requires SearXNG running in Docker"
        return 1
    fi
}

# Function to check if Ollama is running
check_ollama() {
    echo -e "${YELLOW}Checking Ollama for LLM operations...${NC}"

    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}  ✓ Ollama is running${NC}"
        return 0
    else
        echo -e "${YELLOW}  ! Ollama is not running${NC}"
        echo -e "  To start Ollama: ${GREEN}ollama serve${NC}"
        echo -e "  LLM-based MCP operations will not work"
        return 1
    fi
}

# Main startup sequence
echo "1. Checking prerequisites..."
echo

check_ollama
echo
check_searxng
echo

echo "2. Starting MCP Servers..."
echo

# Start each MCP server
# Format: start_mcp_server "ServerName" "module.path" port "extra_args"

# Files MCP Server - Basic file operations
start_mcp_server "Files" "aida.tools.files.mcp_server" 8001

# Execution MCP Server - Code execution capabilities
start_mcp_server "Execution" "aida.tools.execution.mcp_server" 8002

# System MCP Server - System information and operations
start_mcp_server "System" "aida.tools.system.mcp_server" 8003

# Context MCP Server - Context management
start_mcp_server "Context" "aida.tools.context.mcp_server" 8004

# Thinking MCP Server - Structured reasoning
start_mcp_server "Thinking" "aida.tools.thinking.mcp_server" 8005

# LLMResponse MCP Server - LLM interactions
start_mcp_server "LLMResponse" "aida.tools.llm_response.mcp_server" 8006

# WebSearch MCP Server - Web search capabilities (requires SearXNG)
if check_searxng > /dev/null 2>&1; then
    start_mcp_server "WebSearch" "aida.tools.websearch.mcp_server" 8007
else
    echo -e "${YELLOW}Skipping WebSearch MCP Server (SearXNG not available)${NC}"
fi

echo
echo -e "${GREEN}=== MCP Servers Startup Complete ===${NC}"
echo
echo "Summary:"
echo "--------"

# Count running servers
running_count=$(find "$MCP_PID_DIR" -name "*.pid" -type f 2>/dev/null | wc -l)
echo -e "Running MCP Servers: ${GREEN}$running_count${NC}"

# List running servers
if [ "$running_count" -gt 0 ]; then
    echo
    echo "Active servers:"
    for pid_file in "$MCP_PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            server_name=$(basename "$pid_file" .pid)
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "  - ${GREEN}$server_name${NC} (PID: $pid)"
            fi
        fi
    done
fi

echo
echo "Useful commands:"
echo "  View logs:        tail -f $MCP_LOG_DIR/<server_name>.log"
echo "  Stop all servers: $0 stop"
echo "  Server status:    $0 status"
echo

# Handle stop command
if [ "$1" = "stop" ]; then
    echo -e "${YELLOW}Stopping all MCP servers...${NC}"
    for pid_file in "$MCP_PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            server_name=$(basename "$pid_file" .pid)
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "  Stopping $server_name (PID: $pid)..."
                kill "$pid"
                rm "$pid_file"
            fi
        fi
    done
    echo -e "${GREEN}All MCP servers stopped${NC}"
    exit 0
fi

# Handle status command
if [ "$1" = "status" ]; then
    echo -e "${GREEN}MCP Server Status${NC}"
    echo "=================="
    for pid_file in "$MCP_PID_DIR"/*.pid; do
        if [ -f "$pid_file" ]; then
            server_name=$(basename "$pid_file" .pid)
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "  ${GREEN}✓${NC} $server_name (PID: $pid) - Running"
            else
                echo -e "  ${RED}✗${NC} $server_name (PID: $pid) - Not running"
                rm "$pid_file"
            fi
        fi
    done

    # Check for servers that should be running
    expected_servers=("Files" "Execution" "System" "Context" "Thinking" "LLMResponse" "WebSearch")
    for server in "${expected_servers[@]}"; do
        if [ ! -f "$MCP_PID_DIR/${server}.pid" ]; then
            echo -e "  ${YELLOW}?${NC} $server - Not started"
        fi
    done
    exit 0
fi
