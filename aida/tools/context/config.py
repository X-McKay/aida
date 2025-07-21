"""Configuration for context tool."""

from typing import Any

from aida.config.llm_profiles import Purpose


class ContextConfig:
    """Configuration for context tool operations."""

    # LLM Configuration
    LLM_PURPOSE = Purpose.DEFAULT

    # Compression Configuration
    DEFAULT_COMPRESSION_LEVEL = 0.5  # Keep 50% by default
    MIN_CONTENT_LENGTH = 100  # Minimum length to apply compression
    COMPRESSION_CHUNK_SIZE = 1000  # Process in chunks for large content

    # Token Configuration
    DEFAULT_MAX_TOKENS = 2000
    TOKEN_ESTIMATION_RATIO = 0.75  # Rough estimate: 1 word ≈ 0.75 tokens

    # Summary Configuration
    SUMMARY_MIN_LENGTH = 50
    SUMMARY_MAX_LENGTH = 500
    SUMMARY_EXTRACT_SENTENCES = 5

    # Key Points Configuration
    DEFAULT_MAX_KEY_POINTS = 10
    MIN_KEY_POINTS = 3
    KEY_POINT_MIN_LENGTH = 20

    # Search Configuration
    DEFAULT_MAX_SEARCH_RESULTS = 10
    SEARCH_CONTEXT_WINDOW = 50  # Characters before/after match
    MIN_RELEVANCE_SCORE = 0.3

    # Snapshot Configuration
    SNAPSHOT_DIR = ".aida/context_snapshots"
    SNAPSHOT_RETENTION_DAYS = 30
    MAX_SNAPSHOT_SIZE_MB = 10

    # Export/Import Configuration
    SUPPORTED_FORMATS = ["json", "markdown", "yaml", "text"]
    DEFAULT_EXPORT_FORMAT = "json"

    # Content Scoring Weights
    SCORING_WEIGHTS = {
        "recency": {"position": 0.7, "keywords": 0.2, "structure": 0.1},
        "relevance": {"position": 0.2, "keywords": 0.6, "structure": 0.2},
        "importance": {"position": 0.3, "keywords": 0.4, "structure": 0.3},
        "balanced": {"position": 0.4, "keywords": 0.4, "structure": 0.2},
    }

    # Important Keywords for Scoring
    IMPORTANCE_KEYWORDS = [
        "important",
        "critical",
        "key",
        "essential",
        "must",
        "required",
        "priority",
        "urgent",
        "significant",
        "vital",
        "core",
        "fundamental",
    ]

    # Action Keywords
    ACTION_KEYWORDS = [
        "todo",
        "task",
        "action",
        "need to",
        "must",
        "should",
        "will",
        "plan to",
        "implement",
        "fix",
        "create",
        "update",
        "review",
    ]

    # Structural Markers
    STRUCTURAL_MARKERS = [
        "summary:",
        "conclusion:",
        "key points:",
        "overview:",
        "requirements:",
        "objectives:",
        "goals:",
        "results:",
    ]

    @classmethod
    def get_compression_prompt(cls, priority: str) -> str:
        """Get compression prompt based on priority."""
        prompts = {
            "recency": "Compress this content while preserving the most recent information:",
            "relevance": "Compress this content while preserving the most relevant information to the query:",
            "importance": "Compress this content while preserving the most important information:",
            "balanced": "Compress this content while balancing recency, relevance, and importance:",
        }
        return prompts.get(priority, prompts["balanced"])

    @classmethod
    def get_summary_prompt(cls, max_length: int) -> str:
        """Get summary prompt with length constraint."""
        return f"""
Provide a concise summary of the following content.
Maximum length: {max_length} characters.
Focus on key information, decisions, and outcomes.
"""

    @classmethod
    def get_key_points_prompt(cls, max_points: int) -> str:
        """Get key points extraction prompt."""
        return f"""
Extract the {max_points} most important key points from this content.
Each point should be:
- Self-contained and clear
- Action-oriented when applicable
- No longer than 100 characters
"""

    @classmethod
    def get_mcp_config(cls) -> dict[str, Any]:
        """Get MCP server configuration."""
        return {
            "server_name": "aida-context",
            "max_concurrent_requests": 10,
            "timeout_seconds": 30,
            "max_content_size_mb": 5,
        }

    @classmethod
    def get_observability_config(cls) -> dict[str, Any]:
        """Get OpenTelemetry configuration."""
        return {
            "service_name": "aida-context-tool",
            "trace_enabled": True,
            "metrics_enabled": True,
            "export_endpoint": "http://localhost:4317",
        }
