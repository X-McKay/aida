# Setting up SearXNG for AIDA WebSearch

SearXNG is required for the WebSearch MCP server and web search functionality in AIDA.

## Quick Start

### 1. Start SearXNG with Docker Compose

```bash
# Start only SearXNG
docker-compose up -d searxng

# Or start all AIDA services
docker-compose up -d
```

### 2. Verify SearXNG is Running

```bash
# Check if container is running
docker ps | grep searxng

# Test the API
curl -s http://localhost:8888/search?q=test&format=json | jq .
```

### 3. Access SearXNG Web Interface

Open http://localhost:8888 in your browser.

## Configuration

### Default Settings

- **Port**: 8888 (mapped from container's 8080)
- **API URL**: http://localhost:8888/
- **Container Name**: aida-searxng

### Custom Configuration

To customize SearXNG settings, create a configuration directory:

```bash
# Create config directory
mkdir -p searxng

# Create settings file
cat > searxng/settings.yml << 'EOF'
use_default_settings: true

general:
  instance_name: "AIDA SearXNG"

search:
  safe_search: 0
  autocomplete: ""
  formats:
    - html
    - json

server:
  secret_key: "change-this-secret-key"  # pragma: allowlist secret
  bind_address: "0.0.0.0"
  port: 8080

ui:
  static_use_hash: true

enabled_plugins:
  - 'Hash plugin'
  - 'Search on category select'
  - 'Self Information'
  - 'Tracker URL remover'
EOF
```

Then restart SearXNG:

```bash
docker-compose restart searxng
```

## Using with AIDA

### 1. With WebSearch Tool

The WebSearch tool will automatically use SearXNG:

```python
from aida.tools.websearch import WebSearchTool

tool = WebSearchTool()
results = await tool.search("Python programming")
```

### 2. With WebSearch MCP Server

If using the MCP server approach:

```bash
# Start the WebSearch MCP server (if SearXNG is running)
./scripts/start_mcp_servers.sh
```

### 3. Configuration in AIDA

The WebSearch tool looks for SearXNG at `http://localhost:8888` by default. To use a different URL:

```python
# Set environment variable
export SEARXNG_API_BASE_URL="http://your-searxng:8888"

# Or configure in code
from aida.tools.websearch import WebSearchConfig

config = WebSearchConfig(
    SEARXNG_API_BASE_URL="http://your-searxng:8888"
)
```

## Troubleshooting

### SearXNG Container Won't Start

```bash
# Check logs
docker-compose logs searxng

# Common issues:
# 1. Port 8888 already in use
# 2. Permission issues with searxng directory
```

### Fix Permission Issues

```bash
# Set proper permissions
sudo chown -R 100:100 searxng/
```

### Connection Refused

```bash
# Ensure container is running
docker-compose ps searxng

# Check if port is accessible
nc -zv localhost 8888

# Check Docker network
docker network ls
```

### Search Returns No Results

1. Check if SearXNG can access search engines
2. Some engines may be blocked or rate-limited
3. Try different search queries

## Security Considerations

1. **For Development Only**: The default configuration is for development use
2. **Change Secret Key**: Update the secret_key in settings.yml for production
3. **Network Isolation**: SearXNG runs on the aida-network Docker network
4. **No External Access**: By default, only accessible from localhost

## Advanced Configuration

### Enable Specific Search Engines

Edit `searxng/settings.yml`:

```yaml
engines:
  - name: google
    engine: google
    shortcut: g
    disabled: false

  - name: duckduckgo
    engine: duckduckgo
    shortcut: ddg
    disabled: false
```

### Rate Limiting

Add to `searxng/settings.yml`:

```yaml
outgoing:
  request_timeout: 3.0
  useragent_suffix: ""
  pool_connections: 100
  pool_maxsize: 20
```

### Caching

The AIDA WebSearch tool includes built-in caching. SearXNG itself doesn't cache by default.

## Stopping SearXNG

```bash
# Stop only SearXNG
docker-compose stop searxng

# Remove container
docker-compose rm -f searxng

# Remove with volumes
docker-compose down -v searxng
```
