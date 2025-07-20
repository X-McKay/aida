"""Context processing logic."""

import json
import logging
from pathlib import Path
import re
from typing import Any

import yaml

from aida.llm import chat

from .config import ContextConfig

logger = logging.getLogger(__name__)


class ContextProcessor:
    """Handles context processing operations."""

    async def compress(
        self, content: str, compression_level: float, priority: str
    ) -> dict[str, Any]:
        """Compress content based on priority and level."""
        if len(content) < ContextConfig.MIN_CONTENT_LENGTH:
            return {
                "content": content,
                "original_length": len(content),
                "compressed_length": len(content),
                "ratio": 1.0,
                "metadata": {"skipped": "content_too_short"},
            }

        # Score content importance
        scored_content = self._score_content_importance(content, priority)

        # Calculate target length
        target_length = int(len(content) * compression_level)

        # Apply compression
        compressed = self._apply_compression(scored_content, target_length, priority)

        # If still too long, use LLM compression
        if len(compressed) > target_length * 1.2:  # 20% tolerance
            prompt = f"{ContextConfig.get_compression_prompt(priority)}\n\nTarget length: approximately {target_length} characters\n\nContent:\n{compressed}"
            compressed = await chat(prompt, purpose=ContextConfig.LLM_PURPOSE)

        return {
            "content": compressed,
            "original_length": len(content),
            "compressed_length": len(compressed),
            "ratio": len(compressed) / len(content) if len(content) > 0 else 1.0,
            "metadata": {
                "compression_level": compression_level,
                "priority": priority,
                "method": "hybrid",  # scoring + LLM
            },
        }

    async def summarize(self, content: str, max_tokens: int, output_format: str) -> dict[str, Any]:
        """Summarize content with format preference."""
        # Estimate character limit from token limit
        char_limit = int(max_tokens * 4)  # Rough estimate: 1 token ≈ 4 characters

        prompt = f"{ContextConfig.get_summary_prompt(char_limit)}\n\nOutput format: {output_format}\n\nContent:\n{content}"

        summary = await chat(prompt, purpose=ContextConfig.LLM_PURPOSE)

        # Extract key points if structured format
        key_points = []
        if output_format == "structured":
            key_points = self._extract_bullet_points(summary)

        return {
            "summary": summary,
            "key_points": key_points,
            "metadata": {
                "format": output_format,
                "estimated_tokens": self._estimate_tokens(summary),
            },
        }

    async def extract_key_points(self, content: str, max_points: int) -> dict[str, Any]:
        """Extract key points from content."""
        prompt = ContextConfig.get_key_points_prompt(max_points)
        prompt += f"\n\nContent:\n{content}"

        response = await chat(prompt, purpose=ContextConfig.LLM_PURPOSE)

        # Parse key points from response
        raw_points = self._extract_bullet_points(response)

        # Ensure we don't exceed max_points
        if len(raw_points) > max_points:
            raw_points = raw_points[:max_points]

        # Categorize key points
        key_points = []
        categories = set()

        for point in raw_points:
            # Attempt to categorize based on content
            category = self._categorize_point(point)
            categories.add(category)

            key_points.append(
                {
                    "content": point,
                    "category": category,
                    "importance": self._calculate_importance(point),
                }
            )

        return {
            "key_points": key_points,
            "metadata": {
                "requested_points": max_points,
                "extracted_points": len(key_points),
                "categories": list(categories),
            },
        }

    async def search(self, content: str, query: str, max_results: int) -> dict[str, Any]:
        """Search for query within content."""
        results = []

        # Split content into searchable chunks
        chunks = self._split_into_chunks(content, ContextConfig.COMPRESSION_CHUNK_SIZE)

        # Search each chunk
        for i, chunk in enumerate(chunks):
            # Simple keyword search
            matches = self._find_matches(chunk, query)

            for match in matches:
                # Calculate relevance score
                relevance = self._calculate_relevance(match["context"], query)

                if relevance >= ContextConfig.MIN_RELEVANCE_SCORE:
                    results.append(
                        {
                            "content": match["text"],
                            "relevance_score": relevance,
                            "location": f"chunk_{i}",
                            "context_before": match["before"],
                            "context_after": match["after"],
                            "metadata": {"chunk_index": i},
                        }
                    )

        # Sort by relevance and limit results
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        results = results[:max_results]

        return {
            "results": results,
            "metadata": {
                "query": query,
                "total_matches": len(results),
                "chunks_searched": len(chunks),
            },
        }

    async def export_context(
        self, content: Any, file_path: str, format_type: str
    ) -> dict[str, Any]:
        """Export context to file in specified format."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert content based on format
        if format_type == "json":
            content_data = {"content": content} if isinstance(content, str) else content
            path.write_text(json.dumps(content_data, indent=2))

        elif format_type == "yaml":
            content_data = {"content": content} if isinstance(content, str) else content
            path.write_text(yaml.dump(content_data, default_flow_style=False))

        elif format_type == "markdown":
            if isinstance(content, dict):
                md_content = self._dict_to_markdown(content)
            else:
                md_content = f"# Context\n\n{content}"
            path.write_text(md_content)

        else:  # text
            path.write_text(str(content))

        return {
            "file_path": str(path.absolute()),
            "metadata": {"format": format_type, "size_bytes": path.stat().st_size},
        }

    def _score_content_importance(self, content: str, priority: str) -> dict[str, Any]:
        """Score content sections by importance."""
        lines = content.split("\n")
        scored_lines = []

        weights = ContextConfig.SCORING_WEIGHTS.get(
            priority, ContextConfig.SCORING_WEIGHTS["balanced"]
        )

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            # Position score (recent content scores higher for recency priority)
            position_score = (i + 1) / len(lines) if priority == "recency" else 1 - (i / len(lines))

            # Keyword score
            keyword_score = self._calculate_keyword_score(line)

            # Structure score
            structure_score = self._calculate_structure_score(line)

            # Combined score
            total_score = (
                weights["position"] * position_score
                + weights["keywords"] * keyword_score
                + weights["structure"] * structure_score
            )

            scored_lines.append({"line": line, "score": total_score, "index": i})

        return {"lines": scored_lines, "total_lines": len(lines), "weights": weights}

    def _calculate_keyword_score(self, text: str) -> float:
        """Calculate keyword importance score."""
        text_lower = text.lower()

        importance_count = sum(
            1 for keyword in ContextConfig.IMPORTANCE_KEYWORDS if keyword in text_lower
        )

        action_count = sum(1 for keyword in ContextConfig.ACTION_KEYWORDS if keyword in text_lower)

        total_keywords = importance_count + action_count
        len(ContextConfig.IMPORTANCE_KEYWORDS) + len(ContextConfig.ACTION_KEYWORDS)

        return min(total_keywords / 3, 1.0)  # Normalize to 0-1

    def _calculate_structure_score(self, text: str) -> float:
        """Calculate structural importance score."""
        text_lower = text.lower()

        # Check for structural markers
        for marker in ContextConfig.STRUCTURAL_MARKERS:
            if marker in text_lower:
                return 1.0

        # Check for list items
        if re.match(r"^\s*[-*•]\s+", text) or re.match(r"^\s*\d+\.\s+", text):
            return 0.7

        # Check for headers (markdown style)
        if text.strip().startswith("#"):
            return 0.9

        return 0.3

    def _apply_compression(
        self, scored_content: dict[str, Any], target_length: int, priority: str
    ) -> str:
        """Apply compression based on scores."""
        lines = scored_content["lines"]

        # Sort by score (descending)
        lines.sort(key=lambda x: x["score"], reverse=True)

        # Keep lines until we reach target length
        kept_lines = []
        current_length = 0

        for line_data in lines:
            line_length = len(line_data["line"])
            if current_length + line_length <= target_length:
                kept_lines.append(line_data)
                current_length += line_length + 1  # +1 for newline
            else:
                break

        # Sort back by original index to maintain order
        kept_lines.sort(key=lambda x: x["index"])

        # Reconstruct content
        compressed = "\n".join(line["line"] for line in kept_lines)

        return compressed

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text."""
        word_count = len(text.split())
        return int(word_count * ContextConfig.TOKEN_ESTIMATION_RATIO)

    def _extract_bullet_points(self, text: str) -> list[str]:
        """Extract bullet points from text."""
        points = []

        lines = text.split("\n")
        for line in lines:
            # Match various bullet point formats
            if re.match(r"^\s*[-*•]\s+", line):
                point = re.sub(r"^\s*[-*•]\s+", "", line).strip()
                if point:
                    points.append(point)
            elif re.match(r"^\s*\d+\.\s+", line):
                point = re.sub(r"^\s*\d+\.\s+", "", line).strip()
                if point:
                    points.append(point)

        return points

    def _categorize_point(self, point: str) -> str:
        """Categorize a key point based on its content."""
        point_lower = point.lower()

        if any(word in point_lower for word in ["requirement", "must", "need", "critical"]):
            return "requirement"
        elif any(word in point_lower for word in ["decision", "chose", "selected", "decided"]):
            return "decision"
        elif any(word in point_lower for word in ["action", "task", "todo", "implement"]):
            return "action"
        elif any(word in point_lower for word in ["info", "note", "update", "status"]):
            return "information"
        else:
            return "general"

    def _calculate_importance(self, point: str) -> float:
        """Calculate importance score for a key point."""
        score = 0.5  # Base score

        # Boost for certain keywords
        importance_keywords = ["critical", "important", "must", "essential", "key"]
        for keyword in importance_keywords:
            if keyword in point.lower():
                score += 0.1

        # Cap at 1.0
        return min(score, 1.0)

    def _split_into_chunks(self, content: str, chunk_size: int) -> list[str]:
        """Split content into chunks for processing."""
        chunks = []
        current_chunk = []
        current_size = 0

        for line in content.split("\n"):
            line_size = len(line)
            if current_size + line_size > chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def _find_matches(self, text: str, query: str) -> list[dict[str, Any]]:
        """Find query matches in text."""
        matches = []
        query_lower = query.lower()
        text_lower = text.lower()

        # Find all occurrences
        start = 0
        while True:
            pos = text_lower.find(query_lower, start)
            if pos == -1:
                break

            # Extract context
            context_start = max(0, pos - ContextConfig.SEARCH_CONTEXT_WINDOW)
            context_end = min(len(text), pos + len(query) + ContextConfig.SEARCH_CONTEXT_WINDOW)

            matches.append(
                {
                    "text": text[pos : pos + len(query)],
                    "before": text[context_start:pos],
                    "after": text[pos + len(query) : context_end],
                    "context": text[context_start:context_end],
                    "position": pos,
                }
            )

            start = pos + 1

        return matches

    def _calculate_relevance(self, context: str, query: str) -> float:
        """Calculate relevance score for search result."""
        context_lower = context.lower()
        query_lower = query.lower()

        # Exact match bonus
        score = 0.5 if query_lower in context_lower else 0.0

        # Word overlap bonus
        query_words = set(query_lower.split())
        context_words = set(context_lower.split())
        overlap = query_words & context_words

        if query_words:
            overlap_ratio = len(overlap) / len(query_words)
            score += overlap_ratio * 0.5

        return min(score, 1.0)

    def _dict_to_markdown(self, data: dict[str, Any], level: int = 1) -> str:
        """Convert dictionary to markdown format."""
        md_lines = []

        for key, value in data.items():
            header = "#" * level + f" {key.replace('_', ' ').title()}"
            md_lines.append(header)

            if isinstance(value, dict):
                md_lines.append(self._dict_to_markdown(value, level + 1))
            elif isinstance(value, list):
                for item in value:
                    md_lines.append(f"- {item}")
            else:
                md_lines.append(str(value))

            md_lines.append("")  # Empty line

        return "\n".join(md_lines)
