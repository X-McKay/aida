"""Models for web search tool."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SearchCategory(str, Enum):
    """Available search categories."""

    GENERAL = "general"
    IMAGES = "images"
    VIDEOS = "videos"
    FILES = "files"
    MAP = "map"
    SOCIAL = "social_media"


class SearchOperation(str, Enum):
    """Web search operations."""

    SEARCH = "search"
    GET_WEBSITE = "get_website"
    GET_DATETIME = "get_datetime"


class SearchResult(BaseModel):
    """Individual search result."""

    title: str
    url: str
    snippet: str
    category: str = "general"
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebsiteContent(BaseModel):
    """Scraped website content."""

    url: str
    title: str
    content: str
    word_count: int
    extracted_at: datetime
    citations: list[str] = Field(default_factory=list)


class WebSearchRequest(BaseModel):
    """Request for web search operations."""

    operation: SearchOperation
    query: str | None = None
    category: SearchCategory = SearchCategory.GENERAL
    url: str | None = None
    timezone: str | None = None
    max_results: int = 10
    scrape_content: bool = False


class WebSearchResponse(BaseModel):
    """Response from web search operations."""

    operation: SearchOperation
    success: bool = True
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Search results
    results: list[SearchResult] | None = None
    total_results: int | None = None

    # Website content
    website_content: WebsiteContent | None = None

    # DateTime result
    datetime_info: dict[str, Any] | None = None

    # General result field
    result: Any | None = None

    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


# Import uuid for request ID generation
import uuid
