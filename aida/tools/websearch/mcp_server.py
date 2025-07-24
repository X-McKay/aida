"""MCP server for web search tool."""

from mcp import Resource, Tool
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListToolsRequest,
    ListToolsResult,
    ReadResourceRequest,
    ReadResourceResult,
    TextContent,
)
from pydantic import AnyUrl

from .models import SearchCategory, SearchOperation
from .websearch import WebSearchTool


class WebSearchMCPServer:
    """MCP server wrapper for web search tool."""

    def __init__(self, tool: WebSearchTool):
        """Initialize MCP server with web search tool."""
        self.tool = tool
        self.server = Server("aida-websearch")

        # Register handlers
        self.server.list_tools.register(self.handle_list_tools)
        self.server.call_tool.register(self.handle_call_tool)
        self.server.list_resources.register(self.handle_list_resources)
        self.server.read_resource.register(self.handle_read_resource)

    async def handle_list_tools(self, request: ListToolsRequest) -> ListToolsResult:
        """List available web search tools."""
        tools = [
            Tool(
                name="search",
                description="Search the web using SearXNG",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "category": {
                            "type": "string",
                            "enum": [cat.value for cat in SearchCategory],
                            "default": "general",
                            "description": "Search category",
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 10,
                            "description": "Maximum results",
                        },
                        "scrape_content": {
                            "type": "boolean",
                            "default": False,
                            "description": "Scrape website content",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="get_website_content",
                description="Get content from a specific website",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Website URL"},
                        "max_words": {
                            "type": "integer",
                            "default": 5000,
                            "description": "Maximum words to extract",
                        },
                        "include_citations": {
                            "type": "boolean",
                            "default": True,
                            "description": "Include citation links",
                        },
                    },
                    "required": ["url"],
                },
            ),
            Tool(
                name="get_datetime",
                description="Get current date and time in specified timezone",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "default": "UTC",
                            "description": "Timezone (e.g., UTC, America/New_York)",
                        }
                    },
                },
            ),
        ]

        return ListToolsResult(tools=tools)

    async def handle_call_tool(self, request: CallToolRequest) -> CallToolResult:
        """Handle tool calls."""
        tool_name = request.params.name
        arguments = request.params.arguments or {}

        try:
            if tool_name == "search":
                result = await self.tool.execute(
                    operation=SearchOperation.SEARCH,
                    query=arguments.get("query"),
                    category=arguments.get("category", SearchCategory.GENERAL),
                    max_results=arguments.get("max_results", 10),
                    scrape_content=arguments.get("scrape_content", False),
                )

            elif tool_name == "get_website_content":
                result = await self.tool.execute(
                    operation=SearchOperation.GET_WEBSITE, url=arguments.get("url")
                )

            elif tool_name == "get_datetime":
                result = await self.tool.execute(
                    operation=SearchOperation.GET_DATETIME,
                    timezone=arguments.get("timezone", "UTC"),
                )

            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {tool_name}")],
                    isError=True,
                )

            if result.status.value == "completed":
                return CallToolResult(
                    content=[TextContent(type="text", text=str(result.result))], isError=False
                )
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {result.error}")], isError=True
                )

        except Exception as e:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error executing tool: {str(e)}")],
                isError=True,
            )

    async def handle_list_resources(self, request: ListResourcesRequest) -> ListResourcesResult:
        """List available resources."""
        resources = [
            Resource(
                uri=AnyUrl("websearch://config"),
                name="Web Search Configuration",
                description="Current web search configuration",
                mimeType="application/json",
            ),
            Resource(
                uri=AnyUrl("websearch://stats"),
                name="Search Statistics",
                description="Web search usage statistics",
                mimeType="application/json",
            ),
        ]

        return ListResourcesResult(resources=resources)

    async def handle_read_resource(self, request: ReadResourceRequest) -> ReadResourceResult:
        """Read resource content."""
        uri = request.params.uri

        if uri == "websearch://config":
            config_data = {
                "searxng_api_url": self.tool.config.SEARXNG_API_BASE_URL,
                "default_timezone": self.tool.config.DEFAULT_TIMEZONE,
                "max_search_results": self.tool.config.MAX_SEARCH_RESULTS,
                "scrapped_pages": self.tool.config.SCRAPPED_PAGES_NO,
                "page_content_limit": self.tool.config.PAGE_CONTENT_WORDS_LIMIT,
                "cache_settings": {
                    "maxsize": self.tool.config.CACHE_MAXSIZE,
                    "ttl_minutes": self.tool.config.CACHE_TTL_MINUTES,
                    "max_age_minutes": self.tool.config.CACHE_MAX_AGE_MINUTES,
                },
                "rate_limits": {
                    "requests_per_minute": self.tool.config.RATE_LIMIT_REQUESTS_PER_MINUTE,
                    "timeout_seconds": self.tool.config.RATE_LIMIT_TIMEOUT_SECONDS,
                },
            }

            return ReadResourceResult(
                contents=[TextContent(type="text", text=str(config_data))],
            )

        elif uri == "websearch://stats":
            # TODO: Implement statistics tracking
            stats = {
                "total_searches": 0,
                "categories": {},
                "scraped_pages": 0,
                "cache_hits": 0,
                "cache_misses": 0,
            }

            return ReadResourceResult(
                contents=[TextContent(type="text", text=str(stats))],
            )

        else:
            return ReadResourceResult(
                contents=[TextContent(type="text", text=f"Unknown resource: {uri}")],
            )

    async def run(self):
        """Run the MCP server."""
        async with self.server:
            options = InitializationOptions(
                server_name="aida-websearch",
                server_version="1.0.0",
                capabilities=self.server.get_capabilities(
                    notification_options=None, experimental_capabilities={}
                ),
            )

            await self.server.run(
                lambda: print("Web Search MCP Server running..."),
                options,
                raise_exceptions=True,
            )
