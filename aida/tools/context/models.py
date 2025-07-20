"""Data models for context tool."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class ContextOperation(str, Enum):
    """Types of context operations."""

    COMPRESS = "compress"
    SUMMARIZE = "summarize"
    EXTRACT_KEY_POINTS = "extract_key_points"
    MERGE = "merge"
    SPLIT = "split"
    ANALYZE_RELEVANCE = "analyze_relevance"
    OPTIMIZE_TOKENS = "optimize_tokens"
    SNAPSHOT = "snapshot"
    RESTORE = "restore"
    SEARCH = "search"
    EXPORT = "export"
    IMPORT = "import"


class CompressionLevel(float, Enum):
    """Compression level settings."""

    MINIMAL = 0.9  # Keep 90% of content
    LIGHT = 0.7  # Keep 70% of content
    MODERATE = 0.5  # Keep 50% of content
    AGGRESSIVE = 0.3  # Keep 30% of content
    EXTREME = 0.1  # Keep 10% of content


class ContextPriority(str, Enum):
    """Priority settings for context operations."""

    RECENCY = "recency"
    RELEVANCE = "relevance"
    IMPORTANCE = "importance"
    BALANCED = "balanced"


class ContextFormat(str, Enum):
    """Export/import formats."""

    JSON = "json"
    MARKDOWN = "markdown"
    YAML = "yaml"
    TEXT = "text"


class OutputFormat(str, Enum):
    """Output formatting options."""

    STRUCTURED = "structured"
    NARRATIVE = "narrative"
    BULLET_POINTS = "bullet_points"


class ContextRequest(BaseModel):
    """Request model for context operations."""

    operation: ContextOperation
    content: str | dict[str, Any] | list[str] | None = Field(
        None, description="Content to process (string, dict, or list)"
    )

    # Operation-specific parameters
    compression_level: CompressionLevel | None = Field(
        CompressionLevel.MODERATE, description="Level of compression to apply"
    )
    max_tokens: int | None = Field(2000, ge=100, le=50000, description="Maximum tokens in output")
    priority: ContextPriority | None = Field(
        ContextPriority.BALANCED, description="Priority for content selection"
    )
    output_format: OutputFormat | None = Field(
        OutputFormat.STRUCTURED, description="Output format preference"
    )

    # Additional parameters
    query: str | None = Field(None, description="Search query for relevance/search operations")
    file_path: str | None = Field(None, description="File path for snapshot/export/import")
    format_type: ContextFormat | None = Field(None, description="Format for export/import")
    max_results: int | None = Field(10, description="Maximum search results")
    preserve_priority: str | None = Field("balanced", description="Priority for merge operations")

    @validator("content")
    def validate_content(self, v, values):
        operation = values.get("operation")
        if operation == ContextOperation.RESTORE:
            # Restore doesn't need content
            return v
        if v is None:
            raise ValueError(f"Content is required for {operation} operation")
        if operation == ContextOperation.MERGE and not isinstance(v, dict | list):
            raise ValueError("Merge operation requires dict or list content")
        return v


class ContextSection(BaseModel):
    """A section of processed context."""

    title: str
    content: str
    importance_score: float = Field(0.5, ge=0.0, le=1.0)
    word_count: int = 0
    token_estimate: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextSnapshot(BaseModel):
    """Snapshot of context state."""

    id: str = Field(default_factory=lambda: f"snapshot_{datetime.utcnow().timestamp()}")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    content: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    file_path: str | None = None


class ContextSearchResult(BaseModel):
    """Search result within context."""

    content: str
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    location: str | None = None
    context_before: str | None = None
    context_after: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextResponse(BaseModel):
    """Response model for context operations."""

    request_id: str = Field(default_factory=lambda: f"ctx_{datetime.utcnow().timestamp()}")
    operation: ContextOperation
    status: str = "success"

    # Results based on operation type
    compressed_content: str | None = None
    summary: str | None = None
    key_points: list[str | dict[str, Any]] | None = None
    sections: dict[str, ContextSection] | None = None
    search_results: list[ContextSearchResult] | None = None

    # Metrics
    original_length: int | None = None
    compressed_length: int | None = None
    compression_ratio: float | None = None
    token_count: int | None = None
    processing_time: float | None = None

    # Operation-specific results
    snapshot_id: str | None = None
    file_path: str | None = None
    export_format: ContextFormat | None = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
