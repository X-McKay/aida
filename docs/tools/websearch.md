# Web Search Tool

The Web Search Tool provides comprehensive web search capabilities using the MCP SearXNG Enhanced server. It supports multiple search categories, website content scraping, and datetime operations.

## Overview

The Web Search Tool integrates with the open-source [MCP SearXNG Enhanced server](https://github.com/OvertliDS/mcp-searxng-enhanced) to provide:

- Web search across multiple categories (general, images, videos, files, maps, social media)
- Website content extraction and scraping
- Current datetime retrieval with timezone support
- Rate limiting and caching capabilities
- Full observability with OpenTelemetry

## Installation

The tool requires Docker to run the MCP SearXNG Enhanced server:

```bash
# Pull the SearXNG image
docker pull overtlids/mcp-searxng-enhanced:latest
```

## Usage

### Basic Search

```python
from aida.tools.websearch import WebSearchTool, SearchOperation, SearchCategory

# Initialize the tool
tool = WebSearchTool()

# Perform a basic web search
result = await tool.execute(
    operation=SearchOperation.SEARCH,
    query="Python programming best practices",
    category=SearchCategory.GENERAL,
    max_results=10
)

# Search results are in result.result["results"]
for item in result.result["results"]:
    print(f"Title: {item['title']}")
    print(f"URL: {item['url']}")
    print(f"Snippet: {item['snippet']}")
```

### Search with Content Scraping

```python
# Search and scrape top results
result = await tool.execute(
    operation=SearchOperation.SEARCH,
    query="machine learning tutorials",
    category=SearchCategory.GENERAL,
    max_results=5,
    scrape_content=True  # Enable content scraping
)

# Access scraped content
scraped_content = result.result["details"]["scraped_content"]
for page in scraped_content:
    print(f"Page: {page['title']}")
    print(f"Content: {page['content'][:500]}...")
    print(f"Word count: {page['word_count']}")
```

### Category-Specific Searches

```python
# Image search
image_results = await tool.execute(
    operation=SearchOperation.SEARCH,
    query="landscape photography",
    category=SearchCategory.IMAGES,
    max_results=20
)

# Video search
video_results = await tool.execute(
    operation=SearchOperation.SEARCH,
    query="Python tutorial",
    category=SearchCategory.VIDEOS,
    max_results=10
)

# File search (PDFs, documents, etc.)
file_results = await tool.execute(
    operation=SearchOperation.SEARCH,
    query="python cheat sheet pdf",
    category=SearchCategory.FILES,
    max_results=5
)
```

### Website Content Extraction

```python
# Extract content from a specific website
result = await tool.execute(
    operation=SearchOperation.GET_WEBSITE,
    url="https://example.com/article"
)

website_content = result.result["website_content"]
print(f"Title: {website_content['title']}")
print(f"Content: {website_content['content']}")
print(f"Word count: {website_content['word_count']}")
print(f"Citations: {website_content['citations']}")
```

### Get Current DateTime

```python
# Get current datetime in specific timezone
result = await tool.execute(
    operation=SearchOperation.GET_DATETIME,
    timezone="America/New_York"
)

datetime_info = result.result["datetime_info"]
print(f"Current time in New York: {datetime_info}")
```

## Configuration

The Web Search Tool can be configured through environment variables:

```bash
# SearXNG API configuration
export SEARXNG_ENGINE_API_BASE_URL="http://127.0.0.1:8080/search"

# Timezone
export DESIRED_TIMEZONE="UTC"

# Scraping settings
export SCRAPPED_PAGES_NO="5"              # Number of pages to scrape
export RETURNED_SCRAPPED_PAGES_NO="3"     # Number of scraped pages to return
export PAGE_CONTENT_WORDS_LIMIT="5000"    # Maximum words per page
export CITATION_LINKS="True"              # Include citation links

# Category limits
export MAX_IMAGE_RESULTS="10"
export MAX_VIDEO_RESULTS="10"
export MAX_FILE_RESULTS="5"
export MAX_MAP_RESULTS="5"
export MAX_SOCIAL_RESULTS="5"

# Timeouts
export TRAFILATURA_TIMEOUT="15"           # Content extraction timeout
export SCRAPING_TIMEOUT="20"              # Page scraping timeout

# Cache settings
export CACHE_MAXSIZE="100"
export CACHE_TTL_MINUTES="5"
export CACHE_MAX_AGE_MINUTES="30"

# Rate limiting
export RATE_LIMIT_REQUESTS_PER_MINUTE="10"
export RATE_LIMIT_TIMEOUT_SECONDS="60"

# Ignored websites (comma-separated)
export IGNORED_WEBSITES="example.com,spam-site.com"
```

## PydanticAI Integration

The tool provides PydanticAI-compatible functions:

```python
from pydantic_ai import Agent

# Get PydanticAI tools
tools = tool._create_pydantic_tools()

# Use with PydanticAI agent
agent = Agent(
    model="gpt-4",
    tools=[
        tools["search_web"],
        tools["get_website_content"],
        tools["get_current_datetime"]
    ]
)

# The agent can now use web search capabilities
result = await agent.run("Find the latest news about AI")
```

## MCP Server Mode

The tool can run as an MCP server:

```python
from aida.tools.websearch.mcp_server import WebSearchMCPServer

# Create and run MCP server
server = WebSearchMCPServer(tool)
await server.run()
```

## Observability

The tool includes comprehensive observability with OpenTelemetry:

- **Traces**: Track search operations, durations, and errors
- **Metrics**:
  - Total searches by category
  - Error counts
  - Result counts per search
  - Scraped page counts
  - Content sizes

## Error Handling

The tool handles various error scenarios:

- Missing required parameters
- Invalid URLs
- Network timeouts
- Rate limiting
- Docker/MCP server connection issues

## Testing

Run the test script to verify functionality:

```bash
# Run web search tests
uv run python tests/tools/test_websearch.py
```

## Troubleshooting

### Docker Connection Issues

If the MCP server fails to start:

1. Ensure Docker is installed and running
2. Check if the image is downloaded: `docker images | grep searxng`
3. Verify network connectivity
4. Check Docker permissions

### Rate Limiting

If you encounter rate limiting:

1. Reduce `RATE_LIMIT_REQUESTS_PER_MINUTE`
2. Increase `RATE_LIMIT_TIMEOUT_SECONDS`
3. Enable caching to reduce duplicate requests

### Content Extraction Failures

If website content extraction fails:

1. Check if the website is in `IGNORED_WEBSITES`
2. Increase `TRAFILATURA_TIMEOUT`
3. Verify the URL is accessible
4. Some websites may block automated access

## Security Considerations

- The tool runs SearXNG in a Docker container for isolation
- No API keys are required (uses public SearXNG instances)
- Content scraping respects robots.txt when configured
- Rate limiting prevents abuse
- Ignored websites list prevents accessing known problematic sites
