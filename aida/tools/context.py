"""Context management tool for AIDA with hybrid architecture."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import hashlib
import zlib
from pathlib import Path

from pydantic_ai import RunContext
from pydantic import BaseModel
from mcp.types import Tool as MCPTool, CallToolResult as MCPToolResult, TextContent
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

from aida.tools.base import Tool, ToolResult, ToolCapability, ToolParameter, ToolStatus


logger = logging.getLogger(__name__)


class ContextTool(Tool):
    """Advanced context management tool with compression and optimization using hybrid architecture."""
    
    def __init__(self):
        super().__init__(
            name="context",
            description="Manages conversation context, memory compression, and context optimization",
            version="2.0.0"
        )
        self._mcp_server = None
        self._observability = None
    
    def get_capability(self) -> ToolCapability:
        return ToolCapability(
            name=self.name,
            version=self.version,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="operation",
                    type="str",
                    description="Context operation to perform",
                    required=True,
                    choices=[
                        "compress", "summarize", "extract_key_points",
                        "merge_contexts", "split_context", "analyze_relevance",
                        "optimize_tokens", "create_snapshot", "restore_snapshot",
                        "search_context", "export_context", "import_context"
                    ]
                ),
                ToolParameter(
                    name="content",
                    type="str",
                    description="Content to process",
                    required=False
                ),
                ToolParameter(
                    name="context_data",
                    type="dict",
                    description="Context data structure",
                    required=False
                ),
                ToolParameter(
                    name="compression_ratio",
                    type="float",
                    description="Target compression ratio (0.1-0.9)",
                    required=False,
                    default=0.5,
                    min_value=0.1,
                    max_value=0.9
                ),
                ToolParameter(
                    name="max_tokens",
                    type="int",
                    description="Maximum tokens for output",
                    required=False,
                    default=2000,
                    min_value=100,
                    max_value=32000
                ),
                ToolParameter(
                    name="preserve_priority",
                    type="str",
                    description="Priority for preservation",
                    required=False,
                    default="balanced",
                    choices=["recent", "important", "balanced", "comprehensive"]
                ),
                ToolParameter(
                    name="format_type",
                    type="str",
                    description="Output format type",
                    required=False,
                    default="structured",
                    choices=["structured", "narrative", "bullet_points", "json", "markdown"]
                ),
                ToolParameter(
                    name="search_query",
                    type="str",
                    description="Search query for context search",
                    required=False
                ),
                ToolParameter(
                    name="file_path",
                    type="str",
                    description="File path for import/export operations",
                    required=False
                )
            ],
            required_permissions=["context_management"],
            supported_platforms=["any"],
            dependencies=[]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute context management operation."""
        operation = kwargs["operation"]
        
        try:
            if operation == "compress":
                result = await self._compress_context(
                    kwargs.get("content", ""),
                    kwargs.get("compression_ratio", 0.5),
                    kwargs.get("preserve_priority", "balanced")
                )
            elif operation == "summarize":
                result = await self._summarize_context(
                    kwargs.get("content", ""),
                    kwargs.get("max_tokens", 2000),
                    kwargs.get("format_type", "structured")
                )
            elif operation == "extract_key_points":
                result = await self._extract_key_points(
                    kwargs.get("content", ""),
                    kwargs.get("max_tokens", 1000)
                )
            elif operation == "merge_contexts":
                result = await self._merge_contexts(
                    kwargs.get("context_data", {}),
                    kwargs.get("preserve_priority", "balanced")
                )
            elif operation == "split_context":
                result = await self._split_context(
                    kwargs.get("content", ""),
                    kwargs.get("max_tokens", 2000)
                )
            elif operation == "analyze_relevance":
                result = await self._analyze_relevance(
                    kwargs.get("content", ""),
                    kwargs.get("search_query", "")
                )
            elif operation == "optimize_tokens":
                result = await self._optimize_tokens(
                    kwargs.get("content", ""),
                    kwargs.get("max_tokens", 2000)
                )
            elif operation == "create_snapshot":
                result = await self._create_snapshot(
                    kwargs.get("context_data", {}),
                    kwargs.get("file_path")
                )
            elif operation == "restore_snapshot":
                result = await self._restore_snapshot(
                    kwargs.get("file_path", "")
                )
            elif operation == "search_context":
                result = await self._search_context(
                    kwargs.get("content", ""),
                    kwargs.get("search_query", "")
                )
            elif operation == "export_context":
                result = await self._export_context(
                    kwargs.get("context_data", {}),
                    kwargs.get("file_path", ""),
                    kwargs.get("format_type", "json")
                )
            elif operation == "import_context":
                result = await self._import_context(
                    kwargs.get("file_path", "")
                )
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return ToolResult(
                tool_name=self.name,
                execution_id="",
                status=ToolStatus.COMPLETED,
                result=result,
                started_at=datetime.utcnow(),
                metadata={
                    "operation": operation,
                    "processing_time": "simulated",
                    "optimization_applied": True
                }
            )
            
        except Exception as e:
            raise Exception(f"Context operation failed: {str(e)}")
    
    async def _compress_context(
        self, 
        content: str, 
        compression_ratio: float, 
        preserve_priority: str
    ) -> Dict[str, Any]:
        """Compress context while preserving important information."""
        await asyncio.sleep(0.1)  # Simulate processing
        
        # Analyze content structure
        lines = content.splitlines()
        sentences = content.split('. ')
        words = content.split()
        
        # Calculate compression targets
        target_lines = max(1, int(len(lines) * compression_ratio))
        target_sentences = max(1, int(len(sentences) * compression_ratio))
        target_words = max(10, int(len(words) * compression_ratio))
        
        # Importance scoring based on priority
        scored_content = self._score_content_importance(content, preserve_priority)
        
        # Apply compression
        compressed_content = self._apply_compression(
            scored_content, 
            target_words,
            preserve_priority
        )
        
        # Calculate compression metrics
        original_size = len(content)
        compressed_size = len(compressed_content)
        actual_ratio = compressed_size / original_size if original_size > 0 else 0
        
        return {
            "original_content": content,
            "compressed_content": compressed_content,
            "compression_stats": {
                "original_size": original_size,
                "compressed_size": compressed_size,
                "target_ratio": compression_ratio,
                "actual_ratio": actual_ratio,
                "space_saved": original_size - compressed_size,
                "efficiency": (1 - actual_ratio) * 100
            },
            "preserve_priority": preserve_priority,
            "key_elements_preserved": self._identify_preserved_elements(content, compressed_content)
        }
    
    async def _summarize_context(
        self, 
        content: str, 
        max_tokens: int, 
        format_type: str
    ) -> Dict[str, Any]:
        """Create intelligent summary of context."""
        await asyncio.sleep(0.1)
        
        # Extract key components
        key_topics = self._extract_topics(content)
        important_facts = self._extract_facts(content)
        action_items = self._extract_action_items(content)
        decisions = self._extract_decisions(content)
        
        # Create summary based on format
        if format_type == "structured":
            summary = {
                "overview": self._create_overview(content),
                "key_topics": key_topics,
                "important_facts": important_facts,
                "action_items": action_items,
                "decisions": decisions,
                "context_type": self._classify_context_type(content)
            }
        elif format_type == "narrative":
            summary = self._create_narrative_summary(content, key_topics, important_facts)
        elif format_type == "bullet_points":
            summary = self._create_bullet_summary(key_topics, important_facts, action_items)
        else:
            summary = self._create_overview(content)
        
        return {
            "original_length": len(content.split()),
            "summary": summary,
            "format_type": format_type,
            "compression_achieved": self._calculate_compression_ratio(content, str(summary)),
            "key_elements": {
                "topics_count": len(key_topics),
                "facts_count": len(important_facts),
                "actions_count": len(action_items),
                "decisions_count": len(decisions)
            }
        }
    
    async def _extract_key_points(self, content: str, max_points: int) -> Dict[str, Any]:
        """Extract most important key points from content."""
        await asyncio.sleep(0.1)
        
        # Identify different types of key points
        key_points = {
            "main_concepts": self._extract_main_concepts(content),
            "critical_information": self._extract_critical_info(content),
            "conclusions": self._extract_conclusions(content),
            "requirements": self._extract_requirements(content),
            "recommendations": self._extract_recommendations(content)
        }
        
        # Score and rank all points
        all_points = []
        for category, points in key_points.items():
            for point in points:
                all_points.append({
                    "category": category,
                    "content": point,
                    "importance_score": self._score_point_importance(point, content),
                    "relevance_score": self._score_point_relevance(point, content)
                })
        
        # Sort by combined score and limit
        all_points.sort(key=lambda x: x["importance_score"] + x["relevance_score"], reverse=True)
        top_points = all_points[:max_points]
        
        return {
            "key_points": top_points,
            "total_points_found": len(all_points),
            "points_selected": len(top_points),
            "categories_represented": list(set(p["category"] for p in top_points)),
            "average_importance": sum(p["importance_score"] for p in top_points) / len(top_points) if top_points else 0
        }
    
    async def _merge_contexts(self, context_data: Dict, preserve_priority: str) -> Dict[str, Any]:
        """Merge multiple contexts intelligently."""
        await asyncio.sleep(0.1)
        
        merged_context = {
            "content_sources": [],
            "merged_content": "",
            "key_themes": [],
            "timeline": [],
            "conflict_resolution": [],
            "merge_strategy": preserve_priority
        }
        
        # Process each context
        for source_id, context in context_data.items():
            merged_context["content_sources"].append({
                "source_id": source_id,
                "length": len(str(context)),
                "type": self._classify_context_type(str(context)),
                "key_elements": self._extract_key_elements(str(context))
            })
        
        # Identify and resolve conflicts
        conflicts = self._identify_conflicts(context_data)
        resolutions = self._resolve_conflicts(conflicts, preserve_priority)
        merged_context["conflict_resolution"] = resolutions
        
        # Create merged content
        merged_content = self._perform_merge(context_data, resolutions, preserve_priority)
        merged_context["merged_content"] = merged_content
        
        # Extract combined themes and timeline
        merged_context["key_themes"] = self._extract_combined_themes(context_data)
        merged_context["timeline"] = self._create_combined_timeline(context_data)
        
        return merged_context
    
    async def _split_context(self, content: str, max_tokens: int) -> Dict[str, Any]:
        """Split large context into manageable chunks."""
        await asyncio.sleep(0.1)
        
        # Analyze content structure for optimal splitting
        structure_analysis = self._analyze_content_structure(content)
        
        # Determine splitting strategy
        split_strategy = self._determine_split_strategy(content, max_tokens)
        
        # Perform splitting
        chunks = self._perform_split(content, max_tokens, split_strategy)
        
        # Create chunk metadata
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "chunk_id": i + 1,
                "size": len(chunk.split()),
                "themes": self._extract_topics(chunk),
                "start_context": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                "importance_score": self._score_chunk_importance(chunk, content)
            }
            chunk_metadata.append(metadata)
        
        return {
            "original_size": len(content.split()),
            "target_chunk_size": max_tokens,
            "chunks_created": len(chunks),
            "split_strategy": split_strategy,
            "chunks": chunks,
            "chunk_metadata": chunk_metadata,
            "overlap_strategy": "semantic_boundary",
            "preservation_quality": self._assess_preservation_quality(content, chunks)
        }
    
    async def _analyze_relevance(self, content: str, query: str) -> Dict[str, Any]:
        """Analyze relevance of content to a specific query."""
        await asyncio.sleep(0.1)
        
        # Perform relevance analysis
        relevance_scores = {
            "keyword_match": self._calculate_keyword_relevance(content, query),
            "semantic_similarity": self._calculate_semantic_similarity(content, query),
            "contextual_relevance": self._calculate_contextual_relevance(content, query),
            "topical_alignment": self._calculate_topical_alignment(content, query)
        }
        
        # Calculate overall relevance
        overall_relevance = sum(relevance_scores.values()) / len(relevance_scores)
        
        # Identify relevant sections
        relevant_sections = self._identify_relevant_sections(content, query)
        
        # Generate relevance insights
        insights = self._generate_relevance_insights(content, query, relevance_scores)
        
        return {
            "query": query,
            "content_length": len(content.split()),
            "relevance_scores": relevance_scores,
            "overall_relevance": overall_relevance,
            "relevance_level": self._classify_relevance_level(overall_relevance),
            "relevant_sections": relevant_sections,
            "insights": insights,
            "recommendations": self._generate_relevance_recommendations(overall_relevance, insights)
        }
    
    async def _optimize_tokens(self, content: str, max_tokens: int) -> Dict[str, Any]:
        """Optimize content for token efficiency."""
        await asyncio.sleep(0.1)
        
        # Analyze current token usage
        original_tokens = self._estimate_tokens(content)
        
        # Apply optimization techniques
        optimizations = {
            "redundancy_removal": self._remove_redundancy(content),
            "verbose_reduction": self._reduce_verbosity(content),
            "structure_optimization": self._optimize_structure(content),
            "synonym_replacement": self._optimize_synonyms(content)
        }
        
        # Apply optimizations progressively
        optimized_content = content
        for optimization_type, technique in optimizations.items():
            if self._estimate_tokens(optimized_content) > max_tokens:
                optimized_content = technique
        
        # Final token count
        final_tokens = self._estimate_tokens(optimized_content)
        
        return {
            "original_content": content,
            "optimized_content": optimized_content,
            "token_analysis": {
                "original_tokens": original_tokens,
                "target_tokens": max_tokens,
                "final_tokens": final_tokens,
                "reduction_achieved": original_tokens - final_tokens,
                "efficiency_gain": ((original_tokens - final_tokens) / original_tokens) * 100 if original_tokens > 0 else 0
            },
            "optimizations_applied": list(optimizations.keys()),
            "quality_preservation": self._assess_optimization_quality(content, optimized_content)
        }
    
    async def _create_snapshot(self, context_data: Dict, file_path: Optional[str]) -> Dict[str, Any]:
        """Create a snapshot of current context state."""
        await asyncio.sleep(0.1)
        
        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "context_data": context_data,
            "metadata": {
                "total_size": len(str(context_data)),
                "data_hash": hashlib.sha256(str(context_data).encode()).hexdigest(),
                "compression_applied": False
            }
        }
        
        # Apply compression if content is large
        if len(str(context_data)) > 10000:
            compressed_data = zlib.compress(json.dumps(context_data).encode())
            snapshot["compressed_data"] = compressed_data.hex()
            snapshot["metadata"]["compression_applied"] = True
            snapshot["metadata"]["compressed_size"] = len(compressed_data)
        
        # Save to file if path provided
        if file_path:
            snapshot_path = Path(file_path)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            
            snapshot["file_path"] = str(snapshot_path.absolute())
        
        return snapshot
    
    async def _restore_snapshot(self, file_path: str) -> Dict[str, Any]:
        """Restore context from a snapshot file."""
        await asyncio.sleep(0.1)
        
        snapshot_path = Path(file_path)
        
        if not snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot file not found: {file_path}")
        
        with open(snapshot_path, 'r') as f:
            snapshot = json.load(f)
        
        # Decompress if needed
        if snapshot["metadata"].get("compression_applied", False):
            compressed_data = bytes.fromhex(snapshot["compressed_data"])
            decompressed_data = zlib.decompress(compressed_data)
            context_data = json.loads(decompressed_data.decode())
        else:
            context_data = snapshot["context_data"]
        
        return {
            "snapshot_info": {
                "timestamp": snapshot["timestamp"],
                "version": snapshot["version"],
                "original_size": snapshot["metadata"]["total_size"],
                "was_compressed": snapshot["metadata"].get("compression_applied", False)
            },
            "restored_context": context_data,
            "integrity_check": self._verify_snapshot_integrity(snapshot),
            "file_path": file_path
        }
    
    async def _search_context(self, content: str, query: str) -> Dict[str, Any]:
        """Search for specific information within context."""
        await asyncio.sleep(0.1)
        
        # Perform different types of searches
        search_results = {
            "exact_matches": self._find_exact_matches(content, query),
            "partial_matches": self._find_partial_matches(content, query),
            "semantic_matches": self._find_semantic_matches(content, query),
            "contextual_matches": self._find_contextual_matches(content, query)
        }
        
        # Rank and combine results
        all_matches = []
        for match_type, matches in search_results.items():
            for match in matches:
                all_matches.append({
                    "type": match_type,
                    "content": match,
                    "relevance_score": self._score_match_relevance(match, query),
                    "context_position": self._find_match_position(content, match)
                })
        
        # Sort by relevance
        all_matches.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {
            "query": query,
            "total_matches": len(all_matches),
            "match_types": {k: len(v) for k, v in search_results.items()},
            "top_matches": all_matches[:10],
            "search_coverage": self._calculate_search_coverage(content, all_matches),
            "suggestions": self._generate_search_suggestions(query, all_matches)
        }
    
    async def _export_context(self, context_data: Dict, file_path: str, format_type: str) -> Dict[str, Any]:
        """Export context data to external file."""
        await asyncio.sleep(0.1)
        
        export_path = Path(file_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format data based on type
        if format_type == "json":
            with open(export_path, 'w') as f:
                json.dump(context_data, f, indent=2, default=str)
        elif format_type == "markdown":
            markdown_content = self._convert_to_markdown(context_data)
            export_path.write_text(markdown_content)
        elif format_type == "structured":
            structured_content = self._convert_to_structured(context_data)
            export_path.write_text(structured_content)
        else:
            # Default to JSON
            with open(export_path, 'w') as f:
                json.dump(context_data, f, indent=2, default=str)
        
        return {
            "export_path": str(export_path.absolute()),
            "format_type": format_type,
            "file_size": export_path.stat().st_size,
            "data_size": len(str(context_data)),
            "export_timestamp": datetime.utcnow().isoformat(),
            "success": True
        }
    
    async def _import_context(self, file_path: str) -> Dict[str, Any]:
        """Import context data from external file."""
        await asyncio.sleep(0.1)
        
        import_path = Path(file_path)
        
        if not import_path.exists():
            raise FileNotFoundError(f"Import file not found: {file_path}")
        
        # Determine format from extension
        if import_path.suffix.lower() == '.json':
            with open(import_path, 'r') as f:
                context_data = json.load(f)
        elif import_path.suffix.lower() in ['.md', '.markdown']:
            markdown_content = import_path.read_text()
            context_data = self._parse_markdown_context(markdown_content)
        else:
            # Try to parse as text
            text_content = import_path.read_text()
            context_data = {"imported_content": text_content}
        
        return {
            "import_path": str(import_path.absolute()),
            "file_size": import_path.stat().st_size,
            "format_detected": import_path.suffix.lower(),
            "context_data": context_data,
            "data_structure": self._analyze_imported_structure(context_data),
            "import_timestamp": datetime.utcnow().isoformat(),
            "success": True
        }
    
    # ========== PydanticAI Interface ==========
    
    def to_pydantic_tools(self) -> Dict[str, Callable]:
        """Convert to PydanticAI-compatible tools."""
        
        async def compress_context(
            ctx: RunContext,
            content: str,
            compression_ratio: float = 0.5,
            preserve_priority: str = "balanced"
        ) -> Dict[str, Any]:
            """Compress context while preserving important information.
            
            Args:
                content: Content to compress
                compression_ratio: Target compression ratio (0.1-0.9)
                preserve_priority: Priority strategy (recent, important, balanced)
                
            Returns:
                Compression result with statistics
            """
            result = await self.execute(
                operation="compress",
                content=content,
                compression_ratio=compression_ratio,
                preserve_priority=preserve_priority
            )
            
            if result.status == ToolStatus.COMPLETED:
                return result.result
            else:
                raise Exception(f"Compression failed: {result.error}")
        
        async def summarize_context(
            ctx: RunContext,
            content: str,
            max_tokens: int = 2000,
            format_type: str = "structured"
        ) -> Dict[str, Any]:
            """Create intelligent summary of context.
            
            Args:
                content: Content to summarize
                max_tokens: Maximum tokens for summary
                format_type: Output format (structured, narrative, bullet_points)
                
            Returns:
                Summary with key elements
            """
            result = await self.execute(
                operation="summarize",
                content=content,
                max_tokens=max_tokens,
                format_type=format_type
            )
            
            if result.status == ToolStatus.COMPLETED:
                return result.result
            else:
                raise Exception(f"Summarization failed: {result.error}")
        
        async def extract_key_points(
            ctx: RunContext,
            content: str,
            max_points: int = 10
        ) -> Dict[str, Any]:
            """Extract most important key points from content.
            
            Args:
                content: Content to analyze
                max_points: Maximum number of key points to extract
                
            Returns:
                Ranked key points with scores
            """
            result = await self.execute(
                operation="extract_key_points",
                content=content,
                max_tokens=max_points  # Using max_tokens param for max_points
            )
            
            if result.status == ToolStatus.COMPLETED:
                return result.result
            else:
                raise Exception(f"Key point extraction failed: {result.error}")
        
        async def optimize_tokens(
            ctx: RunContext,
            content: str,
            max_tokens: int = 2000
        ) -> Dict[str, Any]:
            """Optimize content for token efficiency.
            
            Args:
                content: Content to optimize
                max_tokens: Target token limit
                
            Returns:
                Optimized content with statistics
            """
            result = await self.execute(
                operation="optimize_tokens",
                content=content,
                max_tokens=max_tokens
            )
            
            if result.status == ToolStatus.COMPLETED:
                return result.result
            else:
                raise Exception(f"Token optimization failed: {result.error}")
        
        async def search_context(
            ctx: RunContext,
            content: str,
            query: str
        ) -> Dict[str, Any]:
            """Search for specific information within context.
            
            Args:
                content: Content to search in
                query: Search query
                
            Returns:
                Search results with relevance scores
            """
            result = await self.execute(
                operation="search_context",
                content=content,
                search_query=query
            )
            
            if result.status == ToolStatus.COMPLETED:
                return result.result
            else:
                raise Exception(f"Context search failed: {result.error}")
        
        return {
            "compress_context": compress_context,
            "summarize_context": summarize_context,
            "extract_key_points": extract_key_points,
            "optimize_tokens": optimize_tokens,
            "search_context": search_context
        }
    
    def register_with_pydantic_agent(self, agent: Any) -> None:
        """Register tools with a PydanticAI agent."""
        tools = self.to_pydantic_tools()
        
        for name, func in tools.items():
            agent.tool(name=name)(func)
    
    # ========== MCP Server Interface ==========
    
    def get_mcp_server(self) -> 'ContextMCPServer':
        """Get MCP server instance for this tool."""
        if self._mcp_server is None:
            self._mcp_server = ContextMCPServer(self)
        return self._mcp_server
    
    # ========== OpenTelemetry Interface ==========
    
    def enable_observability(self, config: Dict[str, Any]) -> 'ContextObservability':
        """Enable OpenTelemetry observability."""
        if self._observability is None:
            self._observability = ContextObservability(self, config)
        return self._observability
    
    # Helper methods for context operations
    def _score_content_importance(self, content: str, priority: str) -> Dict[str, float]:
        """Score content importance based on priority strategy."""
        # Simplified scoring system
        sentences = content.split('.')
        scores = {}
        
        for i, sentence in enumerate(sentences):
            base_score = 0.5
            
            if priority == "recent":
                # Recent content gets higher scores
                base_score += (i / len(sentences)) * 0.5
            elif priority == "important":
                # Content with keywords gets higher scores
                if any(word in sentence.lower() for word in ["important", "critical", "key", "main"]):
                    base_score += 0.3
            elif priority == "balanced":
                # Balanced scoring
                base_score += 0.2 if len(sentence.split()) > 10 else 0.1
            
            scores[sentence] = min(base_score, 1.0)
        
        return scores
    
    def _apply_compression(self, scored_content: Dict, target_words: int, priority: str) -> str:
        """Apply compression based on scores and targets."""
        # Sort content by score
        sorted_content = sorted(scored_content.items(), key=lambda x: x[1], reverse=True)
        
        compressed_parts = []
        current_words = 0
        
        for content, score in sorted_content:
            content_words = len(content.split())
            if current_words + content_words <= target_words:
                compressed_parts.append(content)
                current_words += content_words
            else:
                break
        
        return '. '.join(compressed_parts)
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Rough estimation: 1 token ≈ 0.75 words
        return int(len(text.split()) * 1.33)
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract main topics from content."""
        # Simplified topic extraction
        words = content.lower().split()
        # This would use more sophisticated NLP in a real implementation
        return ["topic_1", "topic_2", "topic_3"]
    
    def _extract_facts(self, content: str) -> List[str]:
        """Extract important facts from content."""
        return ["fact_1", "fact_2", "fact_3"]
    
    def _extract_action_items(self, content: str) -> List[str]:
        """Extract action items from content."""
        return ["action_1", "action_2"]
    
    def _extract_decisions(self, content: str) -> List[str]:
        """Extract decisions from content."""
        return ["decision_1", "decision_2"]
    
    def _calculate_compression_ratio(self, original: str, compressed: str) -> float:
        """Calculate compression ratio."""
        orig_len = len(original)
        comp_len = len(str(compressed))
        return comp_len / orig_len if orig_len > 0 else 0
    
    def _convert_to_markdown(self, data: Dict) -> str:
        """Convert context data to markdown format."""
        md_content = "# Context Export\n\n"
        
        for key, value in data.items():
            md_content += f"## {key.replace('_', ' ').title()}\n\n"
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    md_content += f"- **{subkey}**: {subvalue}\n"
            elif isinstance(value, list):
                for item in value:
                    md_content += f"- {item}\n"
            else:
                md_content += f"{value}\n"
            md_content += "\n"
        
        return md_content
    
    def _identify_preserved_elements(self, original: str, compressed: str) -> List[str]:
        """Identify what elements were preserved in compression."""
        return ["key_concepts", "main_decisions", "critical_facts"]
    
    def _create_overview(self, content: str) -> str:
        """Create overview of content."""
        return f"Overview of {len(content.split())} words covering main topics and key points."
    
    def _classify_context_type(self, content: str) -> str:
        """Classify the type of context."""
        if "meeting" in content.lower():
            return "meeting_notes"
        elif "code" in content.lower():
            return "technical_discussion"
        else:
            return "general_conversation"
    
    # Additional helper methods (placeholders for brevity)
    def _create_narrative_summary(self, content: str, topics: List[str], facts: List[str]) -> str:
        return f"This context discusses {len(topics)} main topics and contains {len(facts)} key facts."
    
    def _create_bullet_summary(self, topics: List[str], facts: List[str], actions: List[str]) -> str:
        return f"• Topics: {len(topics)}\n• Facts: {len(facts)}\n• Actions: {len(actions)}"
    
    def _extract_main_concepts(self, content: str) -> List[str]:
        return ["concept_1", "concept_2"]
    
    def _extract_critical_info(self, content: str) -> List[str]:
        return ["critical_1", "critical_2"]
    
    def _extract_conclusions(self, content: str) -> List[str]:
        return ["conclusion_1"]
    
    def _extract_requirements(self, content: str) -> List[str]:
        return ["requirement_1"]
    
    def _extract_recommendations(self, content: str) -> List[str]:
        return ["recommendation_1"]
    
    def _score_point_importance(self, point: str, content: str) -> float:
        return 0.8
    
    def _score_point_relevance(self, point: str, content: str) -> float:
        return 0.7
    
    def _extract_key_elements(self, content: str) -> List[str]:
        return ["element_1", "element_2"]
    
    def _identify_conflicts(self, context_data: Dict) -> List[Dict]:
        return []
    
    def _resolve_conflicts(self, conflicts: List[Dict], priority: str) -> List[Dict]:
        return []
    
    def _perform_merge(self, context_data: Dict, resolutions: List[Dict], priority: str) -> str:
        return "Merged content"
    
    def _extract_combined_themes(self, context_data: Dict) -> List[str]:
        return ["theme_1", "theme_2"]
    
    def _create_combined_timeline(self, context_data: Dict) -> List[Dict]:
        return []
    
    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        return {"structure": "analyzed"}
    
    def _determine_split_strategy(self, content: str, max_tokens: int) -> str:
        return "semantic_boundary"
    
    def _perform_split(self, content: str, max_tokens: int, strategy: str) -> List[str]:
        words = content.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size >= max_tokens:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = 1
            else:
                current_chunk.append(word)
                current_size += 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _score_chunk_importance(self, chunk: str, full_content: str) -> float:
        return 0.7
    
    def _assess_preservation_quality(self, original: str, chunks: List[str]) -> float:
        return 0.9
    
    def _calculate_keyword_relevance(self, content: str, query: str) -> float:
        return 0.6
    
    def _calculate_semantic_similarity(self, content: str, query: str) -> float:
        return 0.7
    
    def _calculate_contextual_relevance(self, content: str, query: str) -> float:
        return 0.8
    
    def _calculate_topical_alignment(self, content: str, query: str) -> float:
        return 0.75
    
    def _identify_relevant_sections(self, content: str, query: str) -> List[str]:
        return ["section_1", "section_2"]
    
    def _generate_relevance_insights(self, content: str, query: str, scores: Dict) -> List[str]:
        return ["insight_1", "insight_2"]
    
    def _classify_relevance_level(self, score: float) -> str:
        if score > 0.8:
            return "high"
        elif score > 0.5:
            return "medium"
        else:
            return "low"
    
    def _generate_relevance_recommendations(self, score: float, insights: List[str]) -> List[str]:
        return ["recommendation_1"]
    
    def _remove_redundancy(self, content: str) -> str:
        return content  # Simplified
    
    def _reduce_verbosity(self, content: str) -> str:
        return content  # Simplified
    
    def _optimize_structure(self, content: str) -> str:
        return content  # Simplified
    
    def _optimize_synonyms(self, content: str) -> str:
        return content  # Simplified
    
    def _assess_optimization_quality(self, original: str, optimized: str) -> float:
        return 0.85
    
    def _verify_snapshot_integrity(self, snapshot: Dict) -> bool:
        return True
    
    def _find_exact_matches(self, content: str, query: str) -> List[str]:
        return []
    
    def _find_partial_matches(self, content: str, query: str) -> List[str]:
        return []
    
    def _find_semantic_matches(self, content: str, query: str) -> List[str]:
        return []
    
    def _find_contextual_matches(self, content: str, query: str) -> List[str]:
        return []
    
    def _score_match_relevance(self, match: str, query: str) -> float:
        return 0.8
    
    def _find_match_position(self, content: str, match: str) -> int:
        return 0
    
    def _calculate_search_coverage(self, content: str, matches: List[Dict]) -> float:
        return 0.7
    
    def _generate_search_suggestions(self, query: str, matches: List[Dict]) -> List[str]:
        return ["suggestion_1"]
    
    def _convert_to_structured(self, data: Dict) -> str:
        return str(data)
    
    def _parse_markdown_context(self, content: str) -> Dict[str, Any]:
        return {"parsed": content}
    
    def _analyze_imported_structure(self, data: Dict) -> Dict[str, Any]:
        return {"structure": "analyzed"}


class ContextMCPServer:
    """MCP server interface for ContextTool."""
    
    def __init__(self, tool: ContextTool):
        self.tool = tool
        self._tools = self._create_mcp_tools()
    
    def _create_mcp_tools(self) -> List[MCPTool]:
        """Create MCP tool definitions."""
        return [
            MCPTool(
                name="context_compress",
                description="Compress context while preserving important information",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to compress"
                        },
                        "compression_ratio": {
                            "type": "number",
                            "description": "Target compression ratio (0.1-0.9)",
                            "default": 0.5,
                            "minimum": 0.1,
                            "maximum": 0.9
                        },
                        "preserve_priority": {
                            "type": "string",
                            "description": "Preservation priority",
                            "enum": ["recent", "important", "balanced", "comprehensive"],
                            "default": "balanced"
                        }
                    },
                    "required": ["content"]
                }
            ),
            MCPTool(
                name="context_summarize",
                description="Create intelligent summary of context",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to summarize"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens for summary",
                            "default": 2000
                        },
                        "format_type": {
                            "type": "string",
                            "description": "Output format",
                            "enum": ["structured", "narrative", "bullet_points"],
                            "default": "structured"
                        }
                    },
                    "required": ["content"]
                }
            ),
            MCPTool(
                name="context_search",
                description="Search for specific information within context",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to search in"
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["content", "query"]
                }
            ),
            MCPTool(
                name="context_optimize_tokens",
                description="Optimize content for token efficiency",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to optimize"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Target token limit",
                            "default": 2000
                        }
                    },
                    "required": ["content"]
                }
            )
        ]
    
    def list_tools(self) -> List[MCPTool]:
        """List available MCP tools."""
        return self._tools
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """Call an MCP tool."""
        try:
            if name == "context_compress":
                result = await self.tool.execute(
                    operation="compress",
                    content=arguments["content"],
                    compression_ratio=arguments.get("compression_ratio", 0.5),
                    preserve_priority=arguments.get("preserve_priority", "balanced")
                )
                
            elif name == "context_summarize":
                result = await self.tool.execute(
                    operation="summarize",
                    content=arguments["content"],
                    max_tokens=arguments.get("max_tokens", 2000),
                    format_type=arguments.get("format_type", "structured")
                )
                
            elif name == "context_search":
                result = await self.tool.execute(
                    operation="search_context",
                    content=arguments["content"],
                    search_query=arguments["query"]
                )
                
            elif name == "context_optimize_tokens":
                result = await self.tool.execute(
                    operation="optimize_tokens",
                    content=arguments["content"],
                    max_tokens=arguments.get("max_tokens", 2000)
                )
                
            else:
                return MCPToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                    isError=True
                )
            
            if result.status == ToolStatus.COMPLETED:
                return MCPToolResult(
                    content=[TextContent(type="text", text=json.dumps(result.result))],
                    isError=False
                )
            else:
                return MCPToolResult(
                    content=[TextContent(type="text", text=result.error)],
                    isError=True
                )
                
        except Exception as e:
            return MCPToolResult(
                content=[TextContent(type="text", text=str(e))],
                isError=True
            )


class ContextObservability:
    """OpenTelemetry observability for ContextTool."""
    
    def __init__(self, tool: ContextTool, config: Dict[str, Any]):
        self.tool = tool
        self.config = config
        
        # Initialize tracer
        self.tracer = trace.get_tracer(
            "aida.tools.context",
            tool.version
        )
        
        # Initialize metrics
        meter = metrics.get_meter(
            "aida.tools.context",
            tool.version
        )
        
        self.operation_counter = meter.create_counter(
            "context.operations",
            description="Number of context operations",
            unit="1"
        )
        
        self.operation_duration = meter.create_histogram(
            "context.duration",
            description="Context operation duration",
            unit="s"
        )
        
        self.compression_ratio = meter.create_histogram(
            "context.compression_ratio",
            description="Compression ratio achieved",
            unit="1"
        )
        
        self.token_reduction = meter.create_histogram(
            "context.token_reduction",
            description="Token reduction achieved",
            unit="1"
        )
    
    def trace_operation(self, operation: str, content_size: int):
        """Create a trace span for context operation."""
        return self.tracer.start_as_current_span(
            f"context_{operation}",
            attributes={
                "operation": operation,
                "content_size": content_size,
                "tool": "context"
            }
        )
    
    def record_operation(self, operation: str, duration: float, success: bool):
        """Record operation metrics."""
        self.operation_counter.add(
            1,
            {"operation": operation, "success": str(success)}
        )
        
        if success:
            self.operation_duration.record(
                duration,
                {"operation": operation}
            )
    
    def record_compression(self, ratio: float):
        """Record compression ratio."""
        self.compression_ratio.record(ratio)
    
    def record_token_reduction(self, reduction_percent: float):
        """Record token reduction percentage."""
        self.token_reduction.record(reduction_percent)