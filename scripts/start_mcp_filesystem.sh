#!/bin/bash

# Start MCP Filesystem Server for AIDA Development
# This script starts the standard MCP filesystem server that AIDA can connect to

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== MCP Filesystem Server Startup ===${NC}\n"

# Configuration
WORKSPACE_DIR="${WORKSPACE_DIR:-$(pwd)}"
TEMP_DIR=".aida/"

# Check if npx is available
if ! command -v npx &> /dev/null; then
    echo -e "${RED}Error: npx not found. Please install Node.js and npm first.${NC}"
    echo "Visit: https://nodejs.org/"
    exit 1
fi

# Main execution
echo "1. Configuration:"
echo "   Workspace: $WORKSPACE_DIR"
echo "   Temp directory: $TEMP_DIR"
echo

echo "2. Starting MCP Filesystem Server..."
echo "   The server will have access to:"
echo "   - $WORKSPACE_DIR (your AIDA directory)"
echo "   - $TEMP_DIR (for temporary files)"
echo

echo -e "${YELLOW}Starting server...${NC}"
echo "The server will run using stdio interface (standard MCP protocol)"
echo
echo "To use with AIDA:"
echo "1. Keep this terminal open"
echo "2. In another terminal, run your AIDA commands"
echo "3. Press Ctrl+C here to stop the server"
echo
echo "----------------------------------------"

# Start the MCP filesystem server with the allowed directories
# The server expects directory paths as arguments
exec npx -y @modelcontextprotocol/server-filesystem "$WORKSPACE_DIR" "$TEMP_DIR"

# Note: The above command will run in foreground
# For background operation, you could use:
# nohup npx -y @modelcontextprotocol/server-filesystem \
#     --allowed-paths "$ALLOWED_PATHS" \
#     --port "$MCP_PORT" > mcp-filesystem.log 2>&1 &
