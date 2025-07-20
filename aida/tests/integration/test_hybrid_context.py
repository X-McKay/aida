"""Integration tests for hybrid ContextTool."""

import asyncio
import json
import tempfile
from typing import Dict, Any, List
from pathlib import Path

from aida.tests.base import BaseTestSuite, TestResult, test_registry
from aida.tools.context import ContextTool
from aida.tools.base import ToolStatus
from pydantic_ai import Agent
import aida


class HybridContextTestSuite(BaseTestSuite):
    """Test suite for hybrid ContextTool functionality."""
    
    def __init__(self, verbose: bool = False, persist_files: bool = False):
        super().__init__("Hybrid ContextTool", verbose, persist_files)
        self.tool = ContextTool()
        self.test_content = """
        This is a long conversation about implementing AI tools and context management.
        We discussed important architectural decisions including the use of hybrid patterns.
        The main requirement is to support multiple frameworks like PydanticAI and MCP.
        Critical information: The system must maintain backward compatibility.
        Key decision: We will use a unified interface for all tools.
        Action item: Implement comprehensive testing for all interfaces.
        Recent update: Added support for OpenTelemetry observability.
        This additional content helps test compression and summarization features.
        """
    
    async def test_core_interface_compress(self) -> Dict[str, Any]:
        """Test core interface with compression operation."""
        self.log("Testing core interface with compression")
        
        result = await self.tool.execute(
            operation="compress",
            content=self.test_content,
            compression_ratio=0.5,
            preserve_priority="important"
        )
        
        if result.status != ToolStatus.COMPLETED:
            return {"success": False, "message": f"Compression failed: {result.error}"}
        
        compression_stats = result.result.get("compression_stats", {})
        
        if compression_stats.get("compressed_size", 0) < compression_stats.get("original_size", 1):
            return {
                "success": True,
                "message": "Compression successful",
                "efficiency": compression_stats.get("efficiency", 0),
                "actual_ratio": compression_stats.get("actual_ratio", 1)
            }
        else:
            return {
                "success": False,
                "message": "Compression did not reduce size"
            }
    
    async def test_core_interface_summarize(self) -> Dict[str, Any]:
        """Test core interface with summarization."""
        self.log("Testing core interface with summarization")
        
        result = await self.tool.execute(
            operation="summarize",
            content=self.test_content,
            max_tokens=500,
            format_type="structured"
        )
        
        if result.status != ToolStatus.COMPLETED:
            return {"success": False, "message": f"Summarization failed: {result.error}"}
        
        summary = result.result.get("summary", {})
        
        if isinstance(summary, dict) and "overview" in summary:
            return {
                "success": True,
                "message": "Summarization successful",
                "key_elements": result.result.get("key_elements", {}),
                "format": result.result.get("format_type")
            }
        else:
            return {
                "success": False,
                "message": "Summary not properly structured"
            }
    
    async def test_pydantic_ai_interface(self) -> Dict[str, Any]:
        """Test PydanticAI interface."""
        self.log("Testing PydanticAI interface")
        
        tools = self.tool.to_pydantic_tools()
        
        # Test compress_context function
        try:
            result = await tools["compress_context"](
                ctx=None,  # RunContext not needed for this test
                content=self.test_content,
                compression_ratio=0.3,
                preserve_priority="recent"
            )
            
            if "compressed_content" in result and "compression_stats" in result:
                return {
                    "success": True,
                    "message": "PydanticAI interface working",
                    "compressed_size": result["compression_stats"]["compressed_size"],
                    "efficiency": result["compression_stats"]["efficiency"]
                }
            else:
                return {
                    "success": False,
                    "message": "Unexpected result structure"
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"PydanticAI interface error: {str(e)}"
            }
    
    async def test_mcp_server_interface(self) -> Dict[str, Any]:
        """Test MCP server interface."""
        self.log("Testing MCP server interface")
        
        mcp_server = self.tool.get_mcp_server()
        
        # List available tools
        tools = mcp_server.list_tools()
        tool_names = [t.name for t in tools]
        
        if "context_compress" not in tool_names:
            return {
                "success": False,
                "message": "MCP tools not properly registered"
            }
        
        # Test summarization via MCP
        result = await mcp_server.call_tool(
            "context_summarize",
            {
                "content": self.test_content,
                "max_tokens": 300,
                "format_type": "bullet_points"
            }
        )
        
        if result.isError:
            error_text = result.content[0].text if result.content else "Unknown error"
            return {
                "success": False,
                "message": f"MCP execution error: {error_text}"
            }
        
        # Parse the JSON response from content
        import json
        response_text = result.content[0].text if result.content else "{}"
        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError:
            response_data = {}
        
        if "summary" in response_data:
            return {
                "success": True,
                "message": "MCP interface working",
                "format_type": response_data.get("format_type")
            }
        else:
            return {
                "success": False,
                "message": "Unexpected MCP result"
            }
    
    async def test_observability_interface(self) -> Dict[str, Any]:
        """Test OpenTelemetry observability interface."""
        self.log("Testing observability interface")
        
        try:
            observability = self.tool.enable_observability({
                "enabled": True,
                "service_name": "test-context",
                "endpoint": "http://localhost:4317"
            })
            
            # Check that observability components are created
            if not hasattr(observability, 'tracer'):
                return {
                    "success": False,
                    "message": "Tracer not initialized"
                }
            
            if not hasattr(observability, 'compression_ratio'):
                return {
                    "success": False,
                    "message": "Metrics not initialized"
                }
            
            # Test creating a span
            with observability.trace_operation("compress", len(self.test_content)):
                # Simulate operation
                pass
            
            # Record metrics
            observability.record_operation("compress", 0.5, True)
            observability.record_compression(0.6)
            observability.record_token_reduction(40.0)
            
            return {
                "success": True,
                "message": "Observability interface configured",
                "components": ["tracer", "metrics", "compression_ratio", "token_reduction"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Observability error: {str(e)}"
            }
    
    async def test_key_point_extraction(self) -> Dict[str, Any]:
        """Test key point extraction."""
        self.log("Testing key point extraction")
        
        result = await self.tool.execute(
            operation="extract_key_points",
            content=self.test_content,
            max_tokens=5  # Using max_tokens as max_points
        )
        
        if result.status != ToolStatus.COMPLETED:
            return {"success": False, "message": f"Extraction failed: {result.error}"}
        
        key_points = result.result.get("key_points", [])
        
        if len(key_points) > 0 and all("category" in kp for kp in key_points):
            return {
                "success": True,
                "message": "Key point extraction successful",
                "points_found": result.result.get("total_points_found", 0),
                "points_selected": len(key_points),
                "categories": result.result.get("categories_represented", [])
            }
        else:
            return {
                "success": False,
                "message": "No key points extracted"
            }
    
    async def test_token_optimization(self) -> Dict[str, Any]:
        """Test token optimization."""
        self.log("Testing token optimization")
        
        result = await self.tool.execute(
            operation="optimize_tokens",
            content=self.test_content,
            max_tokens=100
        )
        
        if result.status != ToolStatus.COMPLETED:
            return {"success": False, "message": f"Optimization failed: {result.error}"}
        
        token_analysis = result.result.get("token_analysis", {})
        
        if token_analysis.get("final_tokens", 0) <= token_analysis.get("target_tokens", 100):
            return {
                "success": True,
                "message": "Token optimization successful",
                "reduction": token_analysis.get("reduction_achieved", 0),
                "efficiency_gain": token_analysis.get("efficiency_gain", 0)
            }
        else:
            return {
                "success": False,
                "message": "Token optimization did not meet target"
            }
    
    async def test_context_search(self) -> Dict[str, Any]:
        """Test context search functionality."""
        self.log("Testing context search")
        
        result = await self.tool.execute(
            operation="search_context",
            content=self.test_content,
            search_query="hybrid patterns"
        )
        
        if result.status != ToolStatus.COMPLETED:
            return {"success": False, "message": f"Search failed: {result.error}"}
        
        total_matches = result.result.get("total_matches", 0)
        match_types = result.result.get("match_types", {})
        
        return {
            "success": True,
            "message": "Context search completed",
            "total_matches": total_matches,
            "match_types": match_types,
            "search_coverage": result.result.get("search_coverage", 0)
        }
    
    async def test_snapshot_operations(self) -> Dict[str, Any]:
        """Test snapshot create and restore."""
        self.log("Testing snapshot operations")
        
        # Create test context data
        context_data = {
            "conversation_id": "test123",
            "messages": ["message1", "message2"],
            "metadata": {"timestamp": "2024-01-01"}
        }
        
        # Create snapshot
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            snapshot_path = f.name
        
        try:
            # Create snapshot
            create_result = await self.tool.execute(
                operation="create_snapshot",
                context_data=context_data,
                file_path=snapshot_path
            )
            
            if create_result.status != ToolStatus.COMPLETED:
                return {"success": False, "message": f"Snapshot creation failed: {create_result.error}"}
            
            # Restore snapshot
            restore_result = await self.tool.execute(
                operation="restore_snapshot",
                file_path=snapshot_path
            )
            
            if restore_result.status != ToolStatus.COMPLETED:
                return {"success": False, "message": f"Snapshot restore failed: {restore_result.error}"}
            
            restored_data = restore_result.result.get("restored_context", {})
            
            if restored_data == context_data:
                return {
                    "success": True,
                    "message": "Snapshot operations successful",
                    "snapshot_info": restore_result.result.get("snapshot_info", {})
                }
            else:
                return {
                    "success": False,
                    "message": "Restored data doesn't match original"
                }
                
        finally:
            # Clean up
            Path(snapshot_path).unlink(missing_ok=True)
    
    async def test_context_split(self) -> Dict[str, Any]:
        """Test context splitting."""
        self.log("Testing context splitting")
        
        # Create a longer content for splitting
        long_content = self.test_content * 10  # Make it longer
        
        result = await self.tool.execute(
            operation="split_context",
            content=long_content,
            max_tokens=50
        )
        
        if result.status != ToolStatus.COMPLETED:
            return {"success": False, "message": f"Split failed: {result.error}"}
        
        chunks = result.result.get("chunks", [])
        chunk_metadata = result.result.get("chunk_metadata", [])
        
        if len(chunks) > 1 and len(chunks) == len(chunk_metadata):
            return {
                "success": True,
                "message": "Context split successful",
                "chunks_created": len(chunks),
                "split_strategy": result.result.get("split_strategy"),
                "preservation_quality": result.result.get("preservation_quality")
            }
        else:
            return {
                "success": False,
                "message": "Context not properly split"
            }
    
    async def run_all(self) -> List[TestResult]:
        """Run all context tests."""
        tests = [
            ("Core Interface - Compress", self.test_core_interface_compress),
            ("Core Interface - Summarize", self.test_core_interface_summarize),
            ("PydanticAI Interface", self.test_pydantic_ai_interface),
            ("MCP Server Interface", self.test_mcp_server_interface),
            ("Observability Interface", self.test_observability_interface),
            ("Key Point Extraction", self.test_key_point_extraction),
            ("Token Optimization", self.test_token_optimization),
            ("Context Search", self.test_context_search),
            ("Snapshot Operations", self.test_snapshot_operations),
            ("Context Split", self.test_context_split),
        ]
        
        for test_name, test_func in tests:
            result = await self.run_test(test_name, test_func)
            self.results.append(result)
        
        return self.results


# Register the test suite
test_registry.register("hybrid_context", HybridContextTestSuite)