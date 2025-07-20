"""Data models for context tool."""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
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
    LIGHT = 0.7    # Keep 70% of content
    MODERATE = 0.5  # Keep 50% of content
    AGGRESSIVE = 0.3  # Keep 30% of content
    EXTREME = 0.1   # Keep 10% of content


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
    content: Optional[Union[str, Dict[str, Any], List[str]]] = Field(
        None, 
        description="Content to process (string, dict, or list)"
    )
    
    # Operation-specific parameters
    compression_level: Optional[CompressionLevel] = Field(
        CompressionLevel.MODERATE,
        description="Level of compression to apply"
    )
    max_tokens: Optional[int] = Field(
        2000,
        ge=100,
        le=50000,
        description="Maximum tokens in output"
    )
    priority: Optional[ContextPriority] = Field(
        ContextPriority.BALANCED,
        description="Priority for content selection"
    )
    output_format: Optional[OutputFormat] = Field(
        OutputFormat.STRUCTURED,
        description="Output format preference"
    )
    
    # Additional parameters
    query: Optional[str] = Field(None, description="Search query for relevance/search operations")
    file_path: Optional[str] = Field(None, description="File path for snapshot/export/import")
    format_type: Optional[ContextFormat] = Field(None, description="Format for export/import")
    max_results: Optional[int] = Field(10, description="Maximum search results")
    preserve_priority: Optional[str] = Field("balanced", description="Priority for merge operations")
    
    @validator('content')
    def validate_content(cls, v, values):
        operation = values.get('operation')
        if operation == ContextOperation.RESTORE:
            # Restore doesn't need content
            return v
        if v is None:
            raise ValueError(f"Content is required for {operation} operation")
        if operation == ContextOperation.MERGE and not isinstance(v, (dict, list)):
            raise ValueError("Merge operation requires dict or list content")
        return v


class ContextSection(BaseModel):
    """A section of processed context."""
    title: str
    content: str
    importance_score: float = Field(0.5, ge=0.0, le=1.0)
    word_count: int = 0
    token_estimate: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextSnapshot(BaseModel):
    """Snapshot of context state."""
    id: str = Field(default_factory=lambda: f"snapshot_{datetime.utcnow().timestamp()}")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    content: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    file_path: Optional[str] = None


class ContextSearchResult(BaseModel):
    """Search result within context."""
    content: str
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    location: Optional[str] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextResponse(BaseModel):
    """Response model for context operations."""
    request_id: str = Field(default_factory=lambda: f"ctx_{datetime.utcnow().timestamp()}")
    operation: ContextOperation
    status: str = "success"
    
    # Results based on operation type
    compressed_content: Optional[str] = None
    summary: Optional[str] = None
    key_points: Optional[List[Union[str, Dict[str, Any]]]] = None
    sections: Optional[Dict[str, ContextSection]] = None
    search_results: Optional[List[ContextSearchResult]] = None
    
    # Metrics
    original_length: Optional[int] = None
    compressed_length: Optional[int] = None
    compression_ratio: Optional[float] = None
    token_count: Optional[int] = None
    processing_time: Optional[float] = None
    
    # Operation-specific results
    snapshot_id: Optional[str] = None
    file_path: Optional[str] = None
    export_format: Optional[ContextFormat] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }