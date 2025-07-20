"""PydanticAI tool functions for context tool."""

from collections.abc import Callable
from typing import Any

from .config import ContextConfig


def create_pydantic_tools(context_tool) -> dict[str, Callable]:
    """Create PydanticAI-compatible tool functions."""

    async def compress_context(
        content: str,
        compression_level: float = ContextConfig.DEFAULT_COMPRESSION_LEVEL,
        priority: str = "balanced",
    ) -> dict[str, Any]:
        """Compress context to reduce size while preserving important information."""
        result = await context_tool.execute(
            operation="compress",
            content=content,
            compression_level=compression_level,
            priority=priority,
        )
        # Return the full result dict for PydanticAI
        if isinstance(result.result, dict):
            return result.result
        else:
            return {"compressed_content": result.result}

    async def summarize_context(
        content: str,
        max_tokens: int = ContextConfig.DEFAULT_MAX_TOKENS,
        output_format: str = "structured",
    ) -> dict[str, Any]:
        """Create a summary of the context content."""
        result = await context_tool.execute(
            operation="summarize",
            content=content,
            max_tokens=max_tokens,
            output_format=output_format,
        )
        if isinstance(result.result, dict):
            return result.result
        else:
            return {"summary": result.result}

    async def extract_key_points(
        content: str, max_points: int = ContextConfig.DEFAULT_MAX_KEY_POINTS
    ) -> list[dict[str, Any]]:
        """Extract key points from the context."""
        result = await context_tool.execute(
            operation="extract_key_points", content=content, max_results=max_points
        )
        if isinstance(result.result, dict) and "key_points" in result.result:
            return result.result["key_points"]
        else:
            return result.result

    async def search_context(
        content: str, query: str, max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Search for specific information within the context."""
        result = await context_tool.execute(
            operation="search", content=content, query=query, max_results=max_results
        )
        return result.result

    async def optimize_tokens(
        content: str, max_tokens: int = ContextConfig.DEFAULT_MAX_TOKENS
    ) -> dict[str, Any]:
        """Optimize context to fit within token limits."""
        result = await context_tool.execute(
            operation="optimize_tokens", content=content, max_tokens=max_tokens
        )
        if isinstance(result.result, dict):
            return result.result
        else:
            return {"optimized_content": result.result}

    async def create_context_snapshot(content: str, file_path: str = None) -> dict[str, str]:
        """Create a snapshot of the current context."""
        result = await context_tool.execute(
            operation="snapshot", content=content, file_path=file_path
        )
        return {"snapshot_id": result.result.get("snapshot_id", "unknown")}

    async def export_context(content: str, file_path: str, format_type: str = "json") -> str:
        """Export context to a file."""
        result = await context_tool.execute(
            operation="export", content=content, file_path=file_path, format_type=format_type
        )
        return result.result.get("file_path", file_path)

    return {
        "compress_context": compress_context,
        "summarize_context": summarize_context,
        "extract_key_points": extract_key_points,
        "search_context": search_context,
        "optimize_tokens": optimize_tokens,
        "create_context_snapshot": create_context_snapshot,
        "export_context": export_context,
    }
