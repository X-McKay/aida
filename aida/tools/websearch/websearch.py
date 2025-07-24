"""Web search tool using MCP SearXNG Enhanced server."""

from datetime import datetime
import logging
from typing import Any
import uuid

from aida.providers.mcp.base import MCPProvider
from aida.tools.base import ToolCapability, ToolParameter, ToolResult, ToolStatus
from aida.tools.base_tool import BaseModularTool

from .config import WebSearchConfig
from .models import (
    SearchCategory,
    SearchOperation,
    SearchResult,
    WebSearchRequest,
    WebSearchResponse,
    WebsiteContent,
)

logger = logging.getLogger(__name__)


class MCPSearXNGClient(MCPProvider):
    """Client for connecting to MCP SearXNG Enhanced server."""

    def __init__(self, config: WebSearchConfig | None = None):
        """Initialize MCP SearXNG client.

        Args:
            config: Web search configuration. If None, uses default config.
        """
        self.config = config or WebSearchConfig()
        super().__init__("searxng", {"config": self.config})

        self._process = None
        self._reader = None
        self._writer = None
        self._read_task = None
        self._pending_responses = {}

    async def connect(self) -> bool:
        """Connect to MCP SearXNG server by spawning Docker container."""
        try:
            import asyncio
            import subprocess

            # Prepare Docker command
            cmd = ["docker"] + self.config.get_docker_args()

            logger.info(f"Starting MCP SearXNG server with command: {' '.join(cmd)}")

            # Start the server process
            self._process = await asyncio.create_subprocess_exec(
                *cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            self._reader = self._process.stdout
            self._writer = self._process.stdin

            # Start reading messages from server
            self._read_task = asyncio.create_task(self._read_messages())

            # Initialize the session
            await self.initialize_session()

            self._connected = True
            logger.info("Successfully connected to MCP SearXNG server")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP SearXNG server: {e}")
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Disconnect from MCP server and cleanup process."""

        if self._read_task and not self._read_task.done():
            self._read_task.cancel()

        if self._process:
            self._process.terminate()
            await self._process.wait()

        self._process = None
        self._reader = None
        self._writer = None
        self._connected = False

        await super().disconnect()

    async def _read_messages(self):
        """Read messages from the server process."""
        import asyncio
        import json

        while self._connected and self._reader:
            try:
                # Read line from server
                line = await self._reader.readline()
                if not line:
                    break

                # Parse JSON-RPC message
                try:
                    message_data = json.loads(line.decode("utf-8"))
                    from aida.providers.mcp.base import MCPMessage

                    message = MCPMessage(**message_data)

                    # Handle response
                    if message.id is not None and message.id in self._pending_responses:
                        self._pending_responses[message.id].set_result(message)

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error reading message: {e}")
                break

    async def send_message(self, method: str, params: dict[str, Any] | None = None):
        """Send MCP message and get response."""
        import asyncio

        if not self._connected or not self._writer:
            raise RuntimeError("MCP SearXNG client not connected")

        from aida.providers.mcp.base import MCPMessage

        self._message_id += 1
        message = MCPMessage(id=self._message_id, method=method, params=params or {})

        # Create future for response
        response_future = asyncio.Future()
        self._pending_responses[message.id] = response_future

        # Send message
        message_json = message.model_dump_json(exclude_none=True) + "\n"
        self._writer.write(message_json.encode("utf-8"))
        await self._writer.drain()

        # Wait for response
        try:
            response = await asyncio.wait_for(response_future, timeout=self.config.MCP_TIMEOUT)
            return response
        except TimeoutError:
            del self._pending_responses[message.id]
            raise TimeoutError(f"Timeout waiting for response to {method}")
        finally:
            if message.id in self._pending_responses:
                del self._pending_responses[message.id]


class WebSearchTool(BaseModularTool[WebSearchRequest, WebSearchResponse, WebSearchConfig]):
    """Web search tool using MCP SearXNG Enhanced server."""

    def __init__(self, config: WebSearchConfig | None = None):
        """Initialize web search tool.

        Args:
            config: Web search configuration. If None, uses default config.
        """
        super().__init__()
        self.config = config or WebSearchConfig()
        self._client = None
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure MCP client is initialized and connected."""
        if not self._initialized:
            self._client = MCPSearXNGClient(self.config)
            if await self._client.connect():
                self._initialized = True
            else:
                raise RuntimeError("Failed to connect to MCP SearXNG server")

    def _get_tool_name(self) -> str:
        return "web_search"

    def _get_tool_version(self) -> str:
        return "1.0.0"

    def _get_tool_description(self) -> str:
        return "Web search using MCP SearXNG Enhanced server"

    def _get_default_config(self):
        return WebSearchConfig

    def get_capability(self) -> ToolCapability:
        """Get tool capability descriptor."""
        return ToolCapability(
            name=self.name,
            version=self.version,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="operation",
                    type="str",
                    description="Search operation to perform",
                    required=True,
                    choices=[op.value for op in SearchOperation],
                ),
                ToolParameter(
                    name="query",
                    type="str",
                    description="Search query",
                    required=False,
                ),
                ToolParameter(
                    name="category",
                    type="str",
                    description="Search category",
                    required=False,
                    choices=[cat.value for cat in SearchCategory],
                    default=SearchCategory.GENERAL.value,
                ),
                ToolParameter(
                    name="url",
                    type="str",
                    description="URL for website content retrieval",
                    required=False,
                ),
                ToolParameter(
                    name="timezone",
                    type="str",
                    description="Timezone for datetime operation",
                    required=False,
                ),
                ToolParameter(
                    name="max_results",
                    type="int",
                    description="Maximum number of results",
                    required=False,
                    default=10,
                ),
                ToolParameter(
                    name="scrape_content",
                    type="bool",
                    description="Whether to scrape website content",
                    required=False,
                    default=False,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolResult:
        """Execute web search operation using MCP SearXNG server."""
        start_time = datetime.utcnow()

        try:
            # Ensure MCP client is connected
            await self._ensure_initialized()

            # Validate operation
            if "operation" not in kwargs:
                return ToolResult(
                    tool_name=self.name,
                    execution_id=str(uuid.uuid4()),
                    status=ToolStatus.FAILED,
                    error="Missing required parameter: operation",
                    started_at=start_time,
                    completed_at=datetime.utcnow(),
                    metadata={"error_type": "validation_error"},
                )

            # Create request model
            request = WebSearchRequest(**kwargs)

            # Map operations to MCP tool names
            operation_map = {
                SearchOperation.SEARCH: "search",
                SearchOperation.GET_WEBSITE: "get_website_content",
                SearchOperation.GET_DATETIME: "get_datetime",
            }

            mcp_tool = operation_map.get(request.operation)

            if not mcp_tool:
                return ToolResult(
                    tool_name=self.name,
                    execution_id=str(uuid.uuid4()),
                    status=ToolStatus.FAILED,
                    error=f"Unsupported operation: {request.operation}",
                    started_at=start_time,
                    completed_at=datetime.utcnow(),
                )

            # Prepare arguments for MCP tool
            mcp_args = {}

            if request.operation == SearchOperation.SEARCH:
                if not request.query:
                    return ToolResult(
                        tool_name=self.name,
                        execution_id=str(uuid.uuid4()),
                        status=ToolStatus.FAILED,
                        error="Search operation requires 'query' parameter",
                        started_at=start_time,
                        completed_at=datetime.utcnow(),
                    )

                mcp_args["query"] = request.query
                mcp_args["category"] = request.category.value
                mcp_args["max_results"] = min(request.max_results, self.config.MAX_SEARCH_RESULTS)

                if request.scrape_content:
                    mcp_args["scrape_top_n"] = self.config.SCRAPPED_PAGES_NO

            elif request.operation == SearchOperation.GET_WEBSITE:
                if not request.url:
                    return ToolResult(
                        tool_name=self.name,
                        execution_id=str(uuid.uuid4()),
                        status=ToolStatus.FAILED,
                        error="Get website operation requires 'url' parameter",
                        started_at=start_time,
                        completed_at=datetime.utcnow(),
                    )

                mcp_args["url"] = request.url
                mcp_args["max_words"] = self.config.PAGE_CONTENT_WORDS_LIMIT
                mcp_args["include_citations"] = self.config.CITATION_LINKS

            elif request.operation == SearchOperation.GET_DATETIME:
                mcp_args["timezone"] = request.timezone or self.config.DEFAULT_TIMEZONE

            # Call MCP tool
            result = await self._client.call_tool(mcp_tool, mcp_args)

            # Process and format response
            response = await self._process_mcp_result(request, result)

            return ToolResult(
                tool_name=self.name,
                execution_id=response.request_id,
                status=ToolStatus.COMPLETED,
                result=response.model_dump(),
                started_at=start_time,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                metadata={
                    "operation": request.operation.value,
                    "category": request.category.value,
                    "success": response.success,
                    "total_results": response.total_results,
                },
            )

        except Exception as e:
            logger.error(f"MCP web search operation failed: {e}")
            return ToolResult(
                tool_name=self.name,
                execution_id=str(uuid.uuid4()),
                status=ToolStatus.FAILED,
                error=str(e),
                started_at=start_time,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            )

    async def _process_mcp_result(
        self, request: WebSearchRequest, mcp_result: dict[str, Any]
    ) -> WebSearchResponse:
        """Process MCP result into WebSearchResponse."""
        response = WebSearchResponse(
            operation=request.operation,
            success=True,
            timestamp=datetime.utcnow(),
        )

        if request.operation == SearchOperation.SEARCH:
            # Process search results
            results = []
            for item in mcp_result.get("results", []):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=item.get("snippet", ""),
                        category=request.category.value,
                        metadata=item.get("metadata", {}),
                    )
                )

            response.results = results
            response.total_results = len(results)

            # Process scraped content if available
            if "scraped_content" in mcp_result:
                scraped = []
                for content in mcp_result["scraped_content"]:
                    scraped.append(
                        WebsiteContent(
                            url=content.get("url", ""),
                            title=content.get("title", ""),
                            content=content.get("content", ""),
                            word_count=content.get("word_count", 0),
                            extracted_at=datetime.utcnow(),
                            citations=content.get("citations", []),
                        )
                    )
                response.details["scraped_content"] = scraped

        elif request.operation == SearchOperation.GET_WEBSITE:
            # Process website content
            content_data = mcp_result.get("content", {})
            response.website_content = WebsiteContent(
                url=request.url or "",
                title=content_data.get("title", ""),
                content=content_data.get("text", ""),
                word_count=content_data.get("word_count", 0),
                extracted_at=datetime.utcnow(),
                citations=content_data.get("citations", []),
            )

        elif request.operation == SearchOperation.GET_DATETIME:
            # Process datetime info
            response.datetime_info = mcp_result

        # Store raw result for compatibility
        response.result = mcp_result

        return response

    async def cleanup(self):
        """Cleanup MCP client connection."""
        if self._client and self._initialized:
            await self._client.disconnect()
            self._initialized = False

    def _create_pydantic_tools(self) -> dict[str, Any]:
        """Create PydanticAI-compatible tool functions."""

        async def search_web(
            query: str,
            category: str = "general",
            max_results: int = 10,
            scrape_content: bool = False,
        ) -> list[dict[str, Any]]:
            """Search the web."""
            result = await self.execute(
                operation="search",
                query=query,
                category=category,
                max_results=max_results,
                scrape_content=scrape_content,
            )
            return result.result.get("results", [])

        async def get_website_content(
            url: str, max_words: int = 5000, include_citations: bool = True
        ) -> dict[str, Any]:
            """Get website content."""
            result = await self.execute(
                operation="get_website",
                url=url,
            )
            return result.result.get("website_content", {})

        async def get_current_datetime(timezone: str = "UTC") -> dict[str, Any]:
            """Get current date and time."""
            result = await self.execute(operation="get_datetime", timezone=timezone)
            return result.result.get("datetime_info", {})

        return {
            "search_web": search_web,
            "get_website_content": get_website_content,
            "get_current_datetime": get_current_datetime,
        }

    def _create_mcp_server(self):
        """Create MCP server instance."""
        from .mcp_server import WebSearchMCPServer

        return WebSearchMCPServer(self)

    def _create_observability(self, config: dict[str, Any]):
        """Create observability instance."""
        from .observability import WebSearchObservability

        return WebSearchObservability(self, config)
