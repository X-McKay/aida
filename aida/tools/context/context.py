"""Main context tool implementation."""

import os
import json
import yaml
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from pathlib import Path

from aida.tools.base import Tool, ToolResult, ToolCapability, ToolParameter
from aida.llm import chat

from .models import (
    ContextOperation,
    ContextRequest,
    ContextResponse,
    ContextSection,
    ContextSnapshot,
    ContextSearchResult,
    CompressionLevel,
    ContextPriority,
    ContextFormat
)
from .config import ContextConfig
from .processors import ContextProcessor
from .storage import SnapshotManager

logger = logging.getLogger(__name__)


class ContextTool(Tool):
    """Tool for managing and processing context in conversations and workflows."""
    
    def __init__(self):
        super().__init__(
            name="context",
            description="Manages conversation context with compression, summarization, and search capabilities",
            version="2.0.0"
        )
        self._pydantic_tools_cache = {}
        self._mcp_server = None
        self._observability = None
        
        # Initialize components
        self.processor = ContextProcessor()
        self.snapshot_manager = SnapshotManager(ContextConfig.SNAPSHOT_DIR)
    
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
                    description="The context operation to perform",
                    required=True,
                    choices=[op.value for op in ContextOperation]
                ),
                ToolParameter(
                    name="content",
                    type="str",
                    description="The content to process",
                    required=True
                ),
                ToolParameter(
                    name="compression_level",
                    type="float",
                    description="Compression level (0.1-0.9, where 0.1 keeps 10% of content)",
                    required=False,
                    default=ContextConfig.DEFAULT_COMPRESSION_LEVEL,
                    min_value=0.1,
                    max_value=0.9
                ),
                ToolParameter(
                    name="max_tokens",
                    type="int",
                    description="Maximum tokens in output",
                    required=False,
                    default=ContextConfig.DEFAULT_MAX_TOKENS,
                    min_value=100,
                    max_value=50000
                ),
                ToolParameter(
                    name="priority", 
                    type="str",
                    description="Priority for content selection",
                    required=False,
                    default="balanced",
                    choices=[p.value for p in ContextPriority]
                ),
                ToolParameter(
                    name="output_format",
                    type="str",
                    description="Output format preference",
                    required=False,
                    default="structured",
                    choices=["structured", "narrative", "bullet_points"]
                ),
                ToolParameter(
                    name="query",
                    type="str",
                    description="Search query for relevance/search operations",
                    required=False
                ),
                ToolParameter(
                    name="file_path",
                    type="str",
                    description="File path for snapshot/export/import operations",
                    required=False
                ),
                ToolParameter(
                    name="format_type",
                    type="str",
                    description="Format for export/import operations",
                    required=False,
                    choices=ContextConfig.SUPPORTED_FORMATS
                )
            ]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute context operation."""
        start_time = datetime.utcnow()
        
        try:
            # Handle compression_level conversion
            if 'compression_level' in kwargs:
                level = kwargs['compression_level']
                if isinstance(level, str):
                    # Convert string names to enum values
                    level_map = {
                        'minimal': CompressionLevel.MINIMAL,
                        'light': CompressionLevel.LIGHT,
                        'moderate': CompressionLevel.MODERATE,
                        'aggressive': CompressionLevel.AGGRESSIVE,
                        'extreme': CompressionLevel.EXTREME
                    }
                    kwargs['compression_level'] = level_map.get(level.lower(), CompressionLevel.MODERATE)
                elif isinstance(level, (int, float)):
                    # Convert float to nearest CompressionLevel enum
                    if level >= 0.9:
                        kwargs['compression_level'] = CompressionLevel.MINIMAL
                    elif level >= 0.7:
                        kwargs['compression_level'] = CompressionLevel.LIGHT
                    elif level >= 0.5:
                        kwargs['compression_level'] = CompressionLevel.MODERATE
                    elif level >= 0.3:
                        kwargs['compression_level'] = CompressionLevel.AGGRESSIVE
                    else:
                        kwargs['compression_level'] = CompressionLevel.EXTREME
            
            # Create request model
            request = ContextRequest(**kwargs)
            
            # Route to appropriate operation
            if request.operation == ContextOperation.COMPRESS:
                response = await self._compress_context(request)
            elif request.operation == ContextOperation.SUMMARIZE:
                response = await self._summarize_context(request)
            elif request.operation == ContextOperation.EXTRACT_KEY_POINTS:
                response = await self._extract_key_points(request)
            elif request.operation == ContextOperation.MERGE:
                response = await self._merge_contexts(request)
            elif request.operation == ContextOperation.SPLIT:
                response = await self._split_context(request)
            elif request.operation == ContextOperation.ANALYZE_RELEVANCE:
                response = await self._analyze_relevance(request)
            elif request.operation == ContextOperation.OPTIMIZE_TOKENS:
                response = await self._optimize_tokens(request)
            elif request.operation == ContextOperation.SNAPSHOT:
                response = await self._create_snapshot(request)
            elif request.operation == ContextOperation.RESTORE:
                response = await self._restore_snapshot(request)
            elif request.operation == ContextOperation.SEARCH:
                response = await self._search_context(request)
            elif request.operation == ContextOperation.EXPORT:
                response = await self._export_context(request)
            elif request.operation == ContextOperation.IMPORT:
                response = await self._import_context(request)
            else:
                raise ValueError(f"Unknown operation: {request.operation}")
            
            # Update processing time
            response.processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create tool result
            return self._create_tool_result(response, start_time)
            
        except Exception as e:
            logger.error(f"Context operation failed: {e}")
            return ToolResult(
                tool_name=self.name,
                execution_id="",
                status="failed",
                error=str(e),
                started_at=start_time,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds()
            )
    
    async def _compress_context(self, request: ContextRequest) -> ContextResponse:
        """Compress context content."""
        content = self._normalize_content(request.content)
        
        # Use processor for compression
        compressed = await self.processor.compress(
            content,
            request.compression_level.value if request.compression_level else 0.5,
            request.priority.value if request.priority else "balanced"
        )
        
        return ContextResponse(
            operation=request.operation,
            compressed_content=compressed["content"],
            original_length=compressed["original_length"],
            compressed_length=compressed["compressed_length"],
            compression_ratio=compressed["ratio"],
            metadata=compressed.get("metadata", {})
        )
    
    async def _summarize_context(self, request: ContextRequest) -> ContextResponse:
        """Summarize context content."""
        content = self._normalize_content(request.content)
        
        # Use processor for summarization
        summary_data = await self.processor.summarize(
            content,
            request.max_tokens or ContextConfig.DEFAULT_MAX_TOKENS,
            request.output_format.value if request.output_format else "structured"
        )
        
        return ContextResponse(
            operation=request.operation,
            summary=summary_data["summary"],
            key_points=summary_data.get("key_points", []),
            original_length=len(content),
            compressed_length=len(summary_data["summary"]),
            metadata=summary_data.get("metadata", {})
        )
    
    async def _extract_key_points(self, request: ContextRequest) -> ContextResponse:
        """Extract key points from content."""
        content = self._normalize_content(request.content)
        max_points = min(
            request.max_results or ContextConfig.DEFAULT_MAX_KEY_POINTS,
            ContextConfig.DEFAULT_MAX_KEY_POINTS
        )
        
        # Use processor for key point extraction
        extraction = await self.processor.extract_key_points(content, max_points)
        
        return ContextResponse(
            operation=request.operation,
            key_points=extraction["key_points"],
            metadata=extraction.get("metadata", {})
        )
    
    async def _search_context(self, request: ContextRequest) -> ContextResponse:
        """Search within context."""
        content = self._normalize_content(request.content)
        
        if not request.query:
            raise ValueError("Search query is required for search operation")
        
        # Use processor for search
        search_results = await self.processor.search(
            content,
            request.query,
            request.max_results or ContextConfig.DEFAULT_MAX_SEARCH_RESULTS
        )
        
        # Convert to response models
        results = [
            ContextSearchResult(**result)
            for result in search_results["results"]
        ]
        
        return ContextResponse(
            operation=request.operation,
            search_results=results,
            metadata=search_results.get("metadata", {})
        )
    
    async def _create_snapshot(self, request: ContextRequest) -> ContextResponse:
        """Create a snapshot of context."""
        content = self._normalize_content(request.content)
        
        # Create snapshot
        snapshot = await self.snapshot_manager.create_snapshot(
            content,
            request.file_path
        )
        
        return ContextResponse(
            operation=request.operation,
            snapshot_id=snapshot.id,
            file_path=snapshot.file_path,
            metadata={"snapshot": snapshot.dict()}
        )
    
    async def _export_context(self, request: ContextRequest) -> ContextResponse:
        """Export context to file."""
        if not request.file_path:
            raise ValueError("File path is required for export operation")
        
        content = self._normalize_content(request.content)
        format_type = request.format_type or ContextFormat.JSON
        
        # Export based on format
        export_result = await self.processor.export_context(
            content,
            request.file_path,
            format_type.value
        )
        
        return ContextResponse(
            operation=request.operation,
            file_path=export_result["file_path"],
            export_format=format_type,
            metadata=export_result.get("metadata", {})
        )
    
    def _normalize_content(self, content: Union[str, Dict, List]) -> str:
        """Normalize content to string format."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content, indent=2)
        elif isinstance(content, list):
            return "\n".join(str(item) for item in content)
        else:
            return str(content)
    
    def _create_tool_result(
        self,
        response: ContextResponse,
        start_time: datetime
    ) -> ToolResult:
        """Create ToolResult from ContextResponse."""
        # Create a comprehensive result dict
        result = {
            "operation": response.operation.value,
            "status": response.status
        }
        
        # Add operation-specific results
        if response.compressed_content:
            result["compressed_content"] = response.compressed_content
            result["compression_stats"] = {
                "original_size": response.original_length or 0,
                "compressed_size": response.compressed_length or 0,
                "ratio": response.compression_ratio or 1.0,
                "efficiency": 1.0 - (response.compression_ratio or 1.0) if response.compression_ratio else 0,
                "actual_ratio": response.compression_ratio or 1.0
            }
        
        if response.summary:
            result["summary"] = response.summary
            if hasattr(response.summary, 'dict'):
                result["summary"] = response.summary.dict()
            elif isinstance(response.summary, str):
                # Create structured summary format
                result["summary"] = {
                    "overview": response.summary,
                    "format": response.metadata.get("output_format", "structured")
                }
            result["output_format"] = response.metadata.get("output_format", "structured")
        
        if response.key_points:
            result["key_points"] = response.key_points
            result["total_points_found"] = len(response.key_points)
            result["categories_represented"] = response.metadata.get("categories", [])
        
        if response.sections:
            result["sections"] = response.sections
            result["key_elements"] = {
                "sections": len(response.sections),
                "total_tokens": sum(s.token_estimate for s in response.sections.values())
            }
        
        if response.search_results:
            result["search_results"] = [r.dict() for r in response.search_results]
            result["total_matches"] = len(response.search_results)
            result["match_types"] = response.metadata.get("match_types", {})
            result["search_coverage"] = response.metadata.get("search_coverage", 0)
        
        if response.snapshot_id:
            result["snapshot_id"] = response.snapshot_id
            result["snapshot_info"] = response.metadata.get("snapshot_info", {})
        
        if response.file_path:
            result["file_path"] = response.file_path
            # Get restored_content from metadata
            restored = response.metadata.get("restored_content")
            # If restored content is wrapped in 'content' key, unwrap it
            if isinstance(restored, dict) and 'content' in restored and len(restored) == 1:
                try:
                    # Try to parse the JSON string
                    result["restored_context"] = json.loads(restored['content'])
                except (json.JSONDecodeError, TypeError):
                    result["restored_context"] = restored
            else:
                result["restored_context"] = restored
        
        # Add token analysis for optimization
        if response.operation == ContextOperation.OPTIMIZE_TOKENS:
            result["token_analysis"] = {
                "original_tokens": response.metadata.get("original_tokens", 0),
                "final_tokens": response.metadata.get("final_tokens", 0),
                "target_tokens": response.metadata.get("target_tokens", 0),
                "reduction_achieved": response.metadata.get("reduction_achieved", 0),
                "efficiency_gain": response.metadata.get("efficiency_gain", 0)
            }
        
        # Add split info
        if response.operation == ContextOperation.SPLIT:
            # Pull chunks from metadata to top level for compatibility
            chunks = response.metadata.get("chunks", [])
            result["chunks"] = chunks
            result["chunk_metadata"] = response.metadata.get("chunk_metadata", [])
            result["split_strategy"] = response.metadata.get("split_strategy", "sentence")
            result["preservation_quality"] = response.metadata.get("preservation_quality", 1.0)
        
        return ToolResult(
            tool_name=self.name,
            execution_id=response.request_id,
            status="completed",
            result=result,
            started_at=start_time,
            completed_at=datetime.utcnow(),
            duration_seconds=response.processing_time or 0,
            metadata=response.metadata
        )
    
    # Placeholder methods for operations not yet implemented
    async def _merge_contexts(self, request: ContextRequest) -> ContextResponse:
        """Merge multiple contexts."""
        # TODO: Implement merge logic
        return ContextResponse(
            operation=request.operation,
            summary="Merge operation not yet implemented",
            metadata={"status": "not_implemented"}
        )
    
    async def _split_context(self, request: ContextRequest) -> ContextResponse:
        """Split context into sections."""
        content = self._normalize_content(request.content)
        max_chunks = request.max_results or 10
        
        # Simple split by paragraphs or sentences
        paragraphs = content.split('\n\n')
        
        if len(paragraphs) <= max_chunks:
            chunks = paragraphs
        else:
            # Split into roughly equal chunks
            chunk_size = len(content) // max_chunks
            chunks = []
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_size = len(para)
                if current_size + para_size > chunk_size and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [para]
                    current_size = para_size
                else:
                    current_chunk.append(para)
                    current_size += para_size
            
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
        
        # Create metadata for each chunk
        chunk_metadata = [
            {
                "index": i,
                "size": len(chunk),
                "start_position": sum(len(c) for c in chunks[:i]),
                "end_position": sum(len(c) for c in chunks[:i+1])
            }
            for i, chunk in enumerate(chunks)
        ]
        
        return ContextResponse(
            operation=request.operation,
            metadata={
                "chunks": chunks,
                "chunk_metadata": chunk_metadata,
                "split_strategy": "paragraph",
                "preservation_quality": 1.0,
                "total_chunks": len(chunks)
            }
        )
    
    async def _analyze_relevance(self, request: ContextRequest) -> ContextResponse:
        """Analyze relevance of context."""
        # TODO: Implement relevance analysis
        return ContextResponse(
            operation=request.operation,
            summary="Relevance analysis not yet implemented",
            metadata={"status": "not_implemented"}
        )
    
    async def _optimize_tokens(self, request: ContextRequest) -> ContextResponse:
        """Optimize context for token limits."""
        content = self._normalize_content(request.content)
        max_tokens = request.max_tokens or ContextConfig.DEFAULT_MAX_TOKENS
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        original_tokens = len(content) // 4
        
        if original_tokens <= max_tokens:
            # Already within limits
            return ContextResponse(
                operation=request.operation,
                compressed_content=content,
                metadata={
                    "original_tokens": original_tokens,
                    "final_tokens": original_tokens,
                    "target_tokens": max_tokens,
                    "reduction_achieved": 0,
                    "efficiency_gain": 0
                }
            )
        
        # Calculate required compression
        compression_ratio = max_tokens / original_tokens
        
        # Use processor for compression with token target
        compressed = await self.processor.compress(
            content,
            compression_ratio,
            request.priority.value if request.priority else "balanced"
        )
        
        final_tokens = len(compressed["content"]) // 4
        reduction = ((original_tokens - final_tokens) / original_tokens) * 100
        
        return ContextResponse(
            operation=request.operation,
            compressed_content=compressed["content"],
            metadata={
                "original_tokens": original_tokens,
                "final_tokens": final_tokens,
                "target_tokens": max_tokens,
                "reduction_achieved": reduction,
                "efficiency_gain": reduction / 100
            }
        )
    
    async def _restore_snapshot(self, request: ContextRequest) -> ContextResponse:
        """Restore context from snapshot."""
        if not request.file_path:
            raise ValueError("File path is required for restore operation")
        
        # Load snapshot
        snapshot = await self.snapshot_manager.load_snapshot(request.file_path)
        
        # Parse the content
        if isinstance(snapshot.content, str):
            try:
                restored_content = json.loads(snapshot.content)
            except json.JSONDecodeError:
                restored_content = snapshot.content
        else:
            restored_content = snapshot.content
        
        return ContextResponse(
            operation=request.operation,
            file_path=request.file_path,
            metadata={
                "restored_content": restored_content,
                "snapshot_info": {
                    "id": snapshot.id,
                    "created_at": snapshot.created_at.isoformat(),
                    "metadata": snapshot.metadata
                }
            }
        )
    
    async def _import_context(self, request: ContextRequest) -> ContextResponse:
        """Import context from file."""
        # TODO: Implement import logic
        return ContextResponse(
            operation=request.operation,
            summary="Import operation not yet implemented",
            metadata={"status": "not_implemented"}
        )
    
    # ============================================================================
    # HYBRID ARCHITECTURE METHODS
    # ============================================================================
    
    def to_pydantic_tools(self, agent=None) -> Dict[str, Callable]:
        """Convert to PydanticAI-compatible tool functions."""
        if agent and id(agent) in self._pydantic_tools_cache:
            return self._pydantic_tools_cache[id(agent)]
        
        from .pydantic_tools import create_pydantic_tools
        tools = create_pydantic_tools(self)
        
        if agent:
            self._pydantic_tools_cache[id(agent)] = tools
        
        return tools
    
    def get_mcp_server(self):
        """Get or create MCP server wrapper."""
        if self._mcp_server is None:
            from .mcp_server import ContextMCPServer
            self._mcp_server = ContextMCPServer(self)
        return self._mcp_server
    
    def enable_observability(self, config: Optional[Dict[str, Any]] = None):
        """Enable OpenTelemetry observability."""
        if self._observability is None:
            from .observability import ContextObservability
            self._observability = ContextObservability(
                self,
                config or ContextConfig.get_observability_config()
            )
        return self._observability