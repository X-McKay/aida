"""Tests for web search tool using MCP SearXNG server."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aida.tools.base import ToolCapability, ToolStatus
from aida.tools.websearch import (
    SearchCategory,
    SearchOperation,
    WebSearchConfig,
    WebSearchTool,
)
from aida.tools.websearch.websearch import MCPSearXNGClient


class TestWebSearchTool:
    """Test WebSearchTool class with MCP SearXNG integration."""

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP SearXNG client."""
        client = AsyncMock(spec=MCPSearXNGClient)
        client.connect = AsyncMock(return_value=True)
        client.disconnect = AsyncMock()
        client.call_tool = AsyncMock()
        return client

    @pytest.fixture
    def tool(self, mock_mcp_client):
        """Create a web search tool with mocked MCP client."""
        tool = WebSearchTool()
        # Mock the client initialization
        with patch.object(tool, "_ensure_initialized", new=AsyncMock()):
            tool._client = mock_mcp_client
            tool._initialized = True
        return tool

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return WebSearchConfig()

    def test_initialization(self):
        """Test tool initialization."""
        tool = WebSearchTool()
        assert tool.name == "web_search"
        assert tool.version == "1.0.0"
        assert tool.description == "Web search using MCP SearXNG Enhanced server"
        assert tool.config is not None
        assert not tool._initialized

    def test_initialization_with_config(self):
        """Test tool initialization with custom config."""
        config = WebSearchConfig()
        config.MAX_SEARCH_RESULTS = 100
        tool = WebSearchTool(config=config)
        assert tool.config == config
        assert tool.config.MAX_SEARCH_RESULTS == 100

    def test_get_capability(self):
        """Test getting tool capability."""
        tool = WebSearchTool()
        capability = tool.get_capability()

        assert isinstance(capability, ToolCapability)
        assert capability.name == "web_search"
        assert capability.version == "1.0.0"
        assert len(capability.parameters) == 7

        # Check required parameter
        operation_param = next(p for p in capability.parameters if p.name == "operation")
        assert operation_param.required is True
        assert operation_param.type == "str"
        assert len(operation_param.choices) == len(SearchOperation)

        # Check optional parameters
        query_param = next(p for p in capability.parameters if p.name == "query")
        assert query_param.required is False

        category_param = next(p for p in capability.parameters if p.name == "category")
        assert category_param.default == SearchCategory.GENERAL.value
        assert len(category_param.choices) == len(SearchCategory)

        max_results_param = next(p for p in capability.parameters if p.name == "max_results")
        assert max_results_param.default == 10

    @pytest.mark.asyncio
    async def test_ensure_initialized(self):
        """Test MCP client initialization."""
        tool = WebSearchTool()
        mock_client = AsyncMock(spec=MCPSearXNGClient)
        mock_client.connect = AsyncMock(return_value=True)

        with patch("aida.tools.websearch.websearch.MCPSearXNGClient", return_value=mock_client):
            await tool._ensure_initialized()

        assert tool._initialized
        assert tool._client == mock_client
        mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_initialized_failure(self):
        """Test MCP client initialization failure."""
        tool = WebSearchTool()
        mock_client = AsyncMock(spec=MCPSearXNGClient)
        mock_client.connect = AsyncMock(return_value=False)

        with patch("aida.tools.websearch.websearch.MCPSearXNGClient", return_value=mock_client):
            with pytest.raises(RuntimeError, match="Failed to connect to MCP SearXNG server"):
                await tool._ensure_initialized()

    @pytest.mark.asyncio
    async def test_execute_missing_operation(self, tool):
        """Test execute with missing operation parameter."""
        result = await tool.execute(query="test search")

        assert result.status == ToolStatus.FAILED
        assert "Missing required parameter: operation" in result.error
        assert result.metadata["error_type"] == "validation_error"

    @pytest.mark.asyncio
    async def test_search_success(self, tool):
        """Test successful web search via MCP."""
        # Mock MCP response
        tool._client.call_tool.return_value = {
            "results": [
                {
                    "title": "Python Programming",
                    "url": "https://python.org",
                    "snippet": "Official Python website",
                    "metadata": {"source": "web"},
                },
                {
                    "title": "Learn Python",
                    "url": "https://learnpython.org",
                    "snippet": "Interactive Python tutorial",
                    "metadata": {"source": "web"},
                },
            ]
        }

        result = await tool.execute(
            operation=SearchOperation.SEARCH.value,
            query="Python programming",
            category=SearchCategory.GENERAL.value,
            max_results=10,
        )

        assert result.status == ToolStatus.COMPLETED
        assert len(result.result["results"]) == 2
        assert result.result["results"][0]["title"] == "Python Programming"
        assert result.metadata["operation"] == "search"
        assert result.metadata["category"] == "general"
        assert result.metadata["total_results"] == 2

        # Verify MCP call
        tool._client.call_tool.assert_called_once_with(
            "search", {"query": "Python programming", "category": "general", "max_results": 10}
        )

    @pytest.mark.asyncio
    async def test_search_with_scraping(self, tool):
        """Test search with content scraping."""
        # Mock MCP response with scraped content
        tool._client.call_tool.return_value = {
            "results": [
                {"title": "Test Result", "url": "https://example.com", "snippet": "Test snippet"}
            ],
            "scraped_content": [
                {
                    "url": "https://example.com",
                    "title": "Example Page",
                    "content": "Full page content here...",
                    "word_count": 500,
                    "citations": ["https://example.com/ref1"],
                }
            ],
        }

        result = await tool.execute(
            operation=SearchOperation.SEARCH.value, query="test query", scrape_content=True
        )

        assert result.status == ToolStatus.COMPLETED
        # Process the response to get the proper format
        response_data = result.result
        assert "scraped_content" in response_data["details"] or response_data.get("scraped_content")
        scraped = response_data["details"].get("scraped_content", [])
        assert len(scraped) == 1
        assert scraped[0].url == "https://example.com"
        assert scraped[0].word_count == 500

        # Verify MCP call includes scraping parameter
        tool._client.call_tool.assert_called_once()
        call_args = tool._client.call_tool.call_args[0]
        assert call_args[1]["scrape_top_n"] == tool.config.SCRAPPED_PAGES_NO

    @pytest.mark.asyncio
    async def test_search_missing_query(self, tool):
        """Test search operation without query."""
        result = await tool.execute(
            operation=SearchOperation.SEARCH.value, category=SearchCategory.GENERAL.value
        )

        assert result.status == ToolStatus.FAILED
        assert "Search operation requires 'query' parameter" in result.error

    @pytest.mark.asyncio
    async def test_image_search(self, tool):
        """Test image search category."""
        tool._client.call_tool.return_value = {
            "results": [
                {
                    "title": "Cat Image",
                    "url": "https://example.com/cat.jpg",
                    "snippet": "Cute cat photo",
                    "metadata": {"type": "image", "size": "1024x768"},
                }
            ]
        }

        result = await tool.execute(
            operation=SearchOperation.SEARCH.value,
            query="cute cats",
            category=SearchCategory.IMAGES.value,
            max_results=5,
        )

        assert result.status == ToolStatus.COMPLETED
        assert result.metadata["category"] == "images"

        # Verify category passed to MCP
        tool._client.call_tool.assert_called_once()
        call_args = tool._client.call_tool.call_args[0]
        assert call_args[1]["category"] == "images"

    @pytest.mark.asyncio
    async def test_get_website_content(self, tool):
        """Test website content retrieval."""
        tool._client.call_tool.return_value = {
            "content": {
                "title": "Example Website",
                "text": "This is the website content...",
                "word_count": 1500,
                "citations": ["https://example.com/source1", "https://example.com/source2"],
            }
        }

        result = await tool.execute(
            operation=SearchOperation.GET_WEBSITE.value, url="https://example.com/article"
        )

        assert result.status == ToolStatus.COMPLETED
        # Access the website_content from the result
        response_data = result.result
        assert "website_content" in response_data
        content = response_data["website_content"]
        assert content.title == "Example Website"
        assert content.content == "This is the website content..."
        assert content.word_count == 1500
        assert len(content.citations) == 2

        # Verify MCP call
        tool._client.call_tool.assert_called_once_with(
            "get_website_content",
            {
                "url": "https://example.com/article",
                "max_words": tool.config.PAGE_CONTENT_WORDS_LIMIT,
                "include_citations": tool.config.CITATION_LINKS,
            },
        )

    @pytest.mark.asyncio
    async def test_get_website_missing_url(self, tool):
        """Test get website operation without URL."""
        result = await tool.execute(operation=SearchOperation.GET_WEBSITE.value)

        assert result.status == ToolStatus.FAILED
        assert "Get website operation requires 'url' parameter" in result.error

    @pytest.mark.asyncio
    async def test_get_datetime(self, tool):
        """Test datetime retrieval."""
        tool._client.call_tool.return_value = {
            "datetime": "2024-01-15 10:30:00",
            "timezone": "America/New_York",
            "utc_offset": "-05:00",
            "dst": False,
        }

        result = await tool.execute(
            operation=SearchOperation.GET_DATETIME.value, timezone="America/New_York"
        )

        assert result.status == ToolStatus.COMPLETED
        # Access datetime_info from the result
        response_data = result.result
        assert "datetime_info" in response_data
        datetime_info = response_data["datetime_info"]
        assert datetime_info["timezone"] == "America/New_York"
        assert datetime_info["utc_offset"] == "-05:00"

        # Verify MCP call
        tool._client.call_tool.assert_called_once_with(
            "get_datetime", {"timezone": "America/New_York"}
        )

    @pytest.mark.asyncio
    async def test_get_datetime_default_timezone(self, tool):
        """Test datetime with default timezone."""
        tool._client.call_tool.return_value = {"datetime": "2024-01-15 15:30:00", "timezone": "UTC"}

        result = await tool.execute(operation=SearchOperation.GET_DATETIME.value)

        assert result.status == ToolStatus.COMPLETED

        # Verify default timezone used
        tool._client.call_tool.assert_called_once_with(
            "get_datetime", {"timezone": tool.config.DEFAULT_TIMEZONE}
        )

    @pytest.mark.asyncio
    async def test_max_results_limit(self, tool):
        """Test that max_results is limited by config."""
        tool._client.call_tool.return_value = {"results": []}

        # Request more than allowed
        result = await tool.execute(
            operation=SearchOperation.SEARCH.value,
            query="test",
            max_results=1000,  # Way more than MAX_SEARCH_RESULTS
        )

        assert result.status == ToolStatus.COMPLETED

        # Verify limited by config
        tool._client.call_tool.assert_called_once()
        call_args = tool._client.call_tool.call_args[0]
        assert call_args[1]["max_results"] == tool.config.MAX_SEARCH_RESULTS

    @pytest.mark.asyncio
    async def test_unsupported_operation(self, tool):
        """Test unsupported operation."""
        result = await tool.execute(operation="unsupported_op", query="test")

        assert result.status == ToolStatus.FAILED
        assert "Unsupported operation" in str(result.error) or "Invalid" in str(result.error)

    @pytest.mark.asyncio
    async def test_mcp_error_handling(self, tool):
        """Test MCP client error handling."""
        tool._client.call_tool.side_effect = Exception("MCP connection error")

        result = await tool.execute(operation=SearchOperation.SEARCH.value, query="test search")

        assert result.status == ToolStatus.FAILED
        assert "MCP web search operation failed" in str(result.error)

    @pytest.mark.asyncio
    async def test_process_mcp_result_search(self, tool):
        """Test processing MCP search results."""
        from aida.tools.websearch.models import SearchResult, WebSearchRequest

        request = WebSearchRequest(
            operation=SearchOperation.SEARCH, query="test", category=SearchCategory.GENERAL
        )

        mcp_result = {
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example1.com",
                    "snippet": "First result",
                    "metadata": {"rank": 1},
                },
                {
                    "title": "Result 2",
                    "url": "https://example2.com",
                    "snippet": "Second result",
                    "metadata": {"rank": 2},
                },
            ]
        }

        response = await tool._process_mcp_result(request, mcp_result)

        assert response.operation == SearchOperation.SEARCH
        assert response.success is True
        assert len(response.results) == 2
        assert response.total_results == 2
        assert isinstance(response.results[0], SearchResult)
        assert response.results[0].title == "Result 1"
        assert response.results[0].category == "general"

    @pytest.mark.asyncio
    async def test_process_mcp_result_website(self, tool):
        """Test processing MCP website content result."""
        from aida.tools.websearch.models import WebSearchRequest, WebsiteContent

        request = WebSearchRequest(operation=SearchOperation.GET_WEBSITE, url="https://example.com")

        mcp_result = {
            "content": {
                "title": "Test Page",
                "text": "Page content here",
                "word_count": 250,
                "citations": ["https://ref1.com"],
            }
        }

        response = await tool._process_mcp_result(request, mcp_result)

        assert response.operation == SearchOperation.GET_WEBSITE
        assert response.success is True
        assert isinstance(response.website_content, WebsiteContent)
        assert response.website_content.title == "Test Page"
        assert response.website_content.word_count == 250
        assert len(response.website_content.citations) == 1

    @pytest.mark.asyncio
    async def test_process_mcp_result_datetime(self, tool):
        """Test processing MCP datetime result."""
        from aida.tools.websearch.models import WebSearchRequest

        request = WebSearchRequest(operation=SearchOperation.GET_DATETIME, timezone="UTC")

        mcp_result = {
            "datetime": "2024-01-15 12:00:00",
            "timezone": "UTC",
            "unix_timestamp": 1705320000,
        }

        response = await tool._process_mcp_result(request, mcp_result)

        assert response.operation == SearchOperation.GET_DATETIME
        assert response.success is True
        assert response.datetime_info == mcp_result

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup of MCP client."""
        tool = WebSearchTool()
        mock_client = AsyncMock(spec=MCPSearXNGClient)
        tool._client = mock_client
        tool._initialized = True

        await tool.cleanup()

        mock_client.disconnect.assert_called_once()
        assert not tool._initialized

    def test_create_pydantic_tools(self, tool):
        """Test creating PydanticAI-compatible tools."""
        pydantic_tools = tool._create_pydantic_tools()

        expected_tools = [
            "search_web",
            "get_website_content",
            "get_current_datetime",
        ]

        for tool_name in expected_tools:
            assert tool_name in pydantic_tools
            assert callable(pydantic_tools[tool_name])

    @pytest.mark.asyncio
    async def test_pydantic_search_web(self, tool):
        """Test PydanticAI search_web function."""
        tool._client.call_tool.return_value = {
            "results": [
                {"title": "Result 1", "url": "https://example1.com", "snippet": "Test 1"},
                {"title": "Result 2", "url": "https://example2.com", "snippet": "Test 2"},
            ]
        }

        pydantic_tools = tool._create_pydantic_tools()
        search_web = pydantic_tools["search_web"]

        results = await search_web("test query", category="general", max_results=5)

        assert len(results) == 2
        assert results[0]["title"] == "Result 1"

    @pytest.mark.asyncio
    async def test_pydantic_get_website_content(self, tool):
        """Test PydanticAI get_website_content function."""
        with patch.object(tool, "execute") as mock_execute:
            mock_execute.return_value = MagicMock(
                result={
                    "website_content": {
                        "title": "Test Page",
                        "content": "Page content",
                        "word_count": 100,
                    }
                }
            )

            pydantic_tools = tool._create_pydantic_tools()
            get_website = pydantic_tools["get_website_content"]

            result = await get_website("https://example.com")

            assert result["title"] == "Test Page"
            assert result["word_count"] == 100

    @pytest.mark.asyncio
    async def test_pydantic_get_current_datetime(self, tool):
        """Test PydanticAI get_current_datetime function."""
        with patch.object(tool, "execute") as mock_execute:
            mock_execute.return_value = MagicMock(
                result={"datetime_info": {"datetime": "2024-01-15 12:00:00", "timezone": "UTC"}}
            )

            pydantic_tools = tool._create_pydantic_tools()
            get_datetime = pydantic_tools["get_current_datetime"]

            result = await get_datetime(timezone="UTC")

            assert result["datetime"] == "2024-01-15 12:00:00"
            assert result["timezone"] == "UTC"

    def test_create_mcp_server(self, tool):
        """Test creating MCP server."""
        with patch("aida.tools.websearch.mcp_server.WebSearchMCPServer") as mock_mcp:
            tool._create_mcp_server()
            mock_mcp.assert_called_once_with(tool)

    def test_create_observability(self, tool):
        """Test creating observability."""
        config = {"trace_enabled": True}

        with patch("aida.tools.websearch.observability.WebSearchObservability") as mock_obs:
            tool._create_observability(config)
            mock_obs.assert_called_once_with(tool, config)

    def test_config_docker_args(self):
        """Test WebSearchConfig.get_docker_args()."""
        config = WebSearchConfig()
        args = config.get_docker_args()

        assert "run" in args
        assert "-i" in args
        assert "--rm" in args
        assert "--network=host" in args
        assert config.SEARXNG_IMAGE in args

        # Check environment variables
        env_vars = []
        for i, arg in enumerate(args):
            if arg == "-e" and i + 1 < len(args):
                env_vars.append(args[i + 1])

        expected_vars = [
            f"SEARXNG_ENGINE_API_BASE_URL={config.SEARXNG_API_BASE_URL}",
            f"DESIRED_TIMEZONE={config.DEFAULT_TIMEZONE}",
            f"SCRAPPED_PAGES_NO={config.SCRAPPED_PAGES_NO}",
        ]

        for var in expected_vars:
            assert any(var in env for env in env_vars)

    def test_config_docker_args_with_additional_env(self):
        """Test get_docker_args with additional environment variables."""
        config = WebSearchConfig()
        additional_env = {"CUSTOM_VAR": "custom_value", "ANOTHER_VAR": "another_value"}

        args = config.get_docker_args(additional_env)

        # Check additional environment variables are included
        env_pairs = []
        for i, arg in enumerate(args):
            if arg == "-e" and i + 1 < len(args):
                env_pairs.append(args[i + 1])

        assert "CUSTOM_VAR=custom_value" in env_pairs
        assert "ANOTHER_VAR=another_value" in env_pairs


class TestMCPSearXNGClient:
    """Test MCPSearXNGClient class."""

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Test MCP client initialization."""
        config = WebSearchConfig()
        client = MCPSearXNGClient(config)

        assert client.config == config
        assert client._process is None
        # Check inherited _connected attribute
        assert hasattr(client, "_connected")

    @pytest.mark.asyncio
    async def test_client_connect(self):
        """Test client connection."""
        client = MCPSearXNGClient()

        # Mock subprocess and asyncio
        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stdin = MagicMock()

        # Need to import asyncio in the websearch module context
        with patch(
            "aida.tools.websearch.websearch.asyncio.create_subprocess_exec",
            return_value=mock_process,
        ) as mock_create:
            with patch.object(client, "initialize_session", new=AsyncMock()) as mock_init:
                with patch("aida.tools.websearch.websearch.asyncio.create_task") as mock_task:
                    connected = await client.connect()

        assert connected is True
        assert client._connected is True
        assert client._process == mock_process

        # Verify Docker command
        mock_create.assert_called_once()
        call_args = mock_create.call_args[0]
        assert call_args[0] == "docker"
        assert "run" in call_args
        assert client.config.SEARXNG_IMAGE in call_args

    @pytest.mark.asyncio
    async def test_client_connect_failure(self):
        """Test client connection failure."""
        client = MCPSearXNGClient()

        with patch(
            "aida.tools.websearch.websearch.asyncio.create_subprocess_exec",
            side_effect=Exception("Docker error"),
        ):
            connected = await client.connect()

        assert connected is False

    @pytest.mark.asyncio
    async def test_client_disconnect(self):
        """Test client disconnection."""
        client = MCPSearXNGClient()
        client._connected = True
        client._process = MagicMock()
        client._process.terminate = MagicMock()
        client._process.wait = AsyncMock()
        client._read_task = MagicMock()
        client._read_task.done = MagicMock(return_value=False)
        client._read_task.cancel = MagicMock()

        await client.disconnect()

        assert not client._connected
        assert client._process is None
        client._read_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending message to MCP server."""
        from aida.providers.mcp.base import MCPMessage

        client = MCPSearXNGClient()
        client._connected = True
        client._message_id = 0
        client._writer = MagicMock()
        client._writer.write = MagicMock()
        client._writer.drain = AsyncMock()
        client._pending_responses = {}

        # Mock response - set up the future before calling send_message
        async def mock_send():
            # Simulate the response being set after message is sent
            await asyncio.sleep(0.01)
            if 1 in client._pending_responses:
                response = MCPMessage(id=1, result={"test": "data"})
                client._pending_responses[1].set_result(response)

        # Start the mock response task
        asyncio.create_task(mock_send())

        # Now send the message
        result = await client.send_message("test_method", {"param": "value"})

        # Verify the result
        assert result.result == {"test": "data"}
        client._writer.write.assert_called_once()
        client._writer.drain.assert_called_once()

        # Check message format
        written_data = client._writer.write.call_args[0][0]
        assert b"test_method" in written_data
        assert b'"param"' in written_data
        assert b'"value"' in written_data

    @pytest.mark.asyncio
    async def test_send_message_timeout(self):
        """Test message timeout handling."""
        client = MCPSearXNGClient()
        client._connected = True
        client._message_id = 0
        client._writer = MagicMock()
        client._writer.write = MagicMock()
        client._writer.drain = AsyncMock()
        client._pending_responses = {}
        client.config.MCP_TIMEOUT = 0.1  # Short timeout

        with pytest.raises(TimeoutError, match="Timeout waiting for response"):
            await client.send_message("test_method", {})


if __name__ == "__main__":
    pytest.main([__file__])
