"""Shared pytest fixtures for unit tests."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aida.providers.mcp.base import MCPMessage
from aida.providers.mcp.filesystem_client import MCPFilesystemClient
from aida.tools.websearch.websearch import MCPSearXNGClient


@pytest.fixture
def mock_mcp_filesystem_client():
    """Create a mock MCP filesystem client."""
    client = AsyncMock(spec=MCPFilesystemClient)
    client.connect = AsyncMock(return_value=True)
    client.disconnect = AsyncMock()
    client.call_tool = AsyncMock()
    client._connected = True
    client._message_id = 0

    # Default responses for common operations
    client.call_tool.return_value = {"success": True}

    return client


@pytest.fixture
def mock_mcp_searxng_client():
    """Create a mock MCP SearXNG client."""
    client = AsyncMock(spec=MCPSearXNGClient)
    client.connect = AsyncMock(return_value=True)
    client.disconnect = AsyncMock()
    client.call_tool = AsyncMock()
    client._connected = True
    client._message_id = 0

    # Default responses for common operations
    client.call_tool.return_value = {"results": []}

    return client


@pytest.fixture
def mock_mcp_message():
    """Create a mock MCP message."""

    def _create_message(id=1, method=None, params=None, result=None, error=None):
        return MCPMessage(id=id, method=method, params=params or {}, result=result, error=error)

    return _create_message


@pytest.fixture
def mock_subprocess():
    """Create a mock subprocess for MCP server processes."""
    process = MagicMock()
    process.stdout = MagicMock()
    process.stdin = MagicMock()
    process.stderr = MagicMock()
    process.terminate = MagicMock()
    process.wait = AsyncMock()
    process.returncode = None

    # Mock readline for stdout
    async def mock_readline():
        return b'{"jsonrpc": "2.0", "id": 1, "result": {"success": true}}\n'

    process.stdout.readline = mock_readline

    return process


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Tool-specific fixtures
@pytest.fixture
def file_operation_response():
    """Common file operation responses."""
    return {
        "read": {"content": "Test file content", "encoding": "utf-8", "size": 17},
        "write": {"bytes_written": 17, "path": "/tmp/test.txt"},
        "delete": {"deleted": True},
        "list": {
            "entries": [
                {"name": "file1.txt", "path": "/tmp/file1.txt", "type": "file", "size": 100},
                {"name": "dir1", "path": "/tmp/dir1", "type": "directory"},
            ]
        },
        "info": {
            "name": "test.txt",
            "path": "/tmp/test.txt",
            "size": 1024,
            "type": "file",
            "modified": "2024-01-01T00:00:00Z",
        },
    }


@pytest.fixture
def web_search_response():
    """Common web search responses."""
    return {
        "search": {
            "results": [
                {
                    "title": "Search Result 1",
                    "url": "https://example1.com",
                    "snippet": "This is the first search result",
                    "metadata": {"rank": 1},
                },
                {
                    "title": "Search Result 2",
                    "url": "https://example2.com",
                    "snippet": "This is the second search result",
                    "metadata": {"rank": 2},
                },
            ]
        },
        "website": {
            "content": {
                "title": "Example Website",
                "text": "This is the website content that was extracted.",
                "word_count": 10,
                "citations": ["https://example.com/source"],
            }
        },
        "datetime": {
            "datetime": "2024-01-15 12:00:00",
            "timezone": "UTC",
            "unix_timestamp": 1705320000,
            "utc_offset": "+00:00",
        },
    }


# Async helpers
@pytest.fixture
def async_mock():
    """Factory for creating async mocks."""

    def _create_async_mock(**kwargs):
        mock = AsyncMock(**kwargs)
        return mock

    return _create_async_mock


@pytest.fixture
def mock_asyncio_subprocess():
    """Mock asyncio.create_subprocess_exec."""

    async def _mock_create_subprocess(*args, **kwargs):
        process = MagicMock()
        process.stdout = MagicMock()
        process.stdin = MagicMock()
        process.stderr = MagicMock()
        process.terminate = MagicMock()
        process.wait = AsyncMock()
        process.returncode = 0
        return process

    return _mock_create_subprocess


# Test data fixtures
@pytest.fixture
def sample_file_paths():
    """Sample file paths for testing."""
    return {
        "file": "/tmp/test_file.txt",
        "directory": "/tmp/test_dir",
        "nested": "/tmp/test_dir/subdir/file.txt",
        "python": "/tmp/script.py",
        "json": "/tmp/data.json",
        "binary": "/tmp/image.png",
    }


@pytest.fixture
def sample_search_queries():
    """Sample search queries for testing."""
    return {
        "general": "Python programming tutorial",
        "images": "cute cats",
        "videos": "machine learning course",
        "files": "python cheat sheet pdf",
        "map": "restaurants near me",
        "social": "AI news Twitter",
    }


@pytest.fixture
def sample_urls():
    """Sample URLs for testing."""
    return {
        "article": "https://example.com/article/python-best-practices",
        "blog": "https://blog.example.com/2024/01/ai-trends",
        "docs": "https://docs.python.org/3/tutorial/",
        "github": "https://github.com/example/repo",
        "invalid": "not-a-valid-url",
    }
