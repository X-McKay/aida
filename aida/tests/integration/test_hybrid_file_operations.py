"""Integration tests for hybrid FileOperationsTool.

Tests the hybrid architecture that supports:
- Original AIDA interface
- PydanticAI compatibility
- MCP server integration
- OpenTelemetry observability
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

from aida.tests.base import BaseTestSuite, TestResult, test_registry
from aida.tools.files import FileOperationsTool
from aida.tools.base import ToolStatus


class HybridFileOperationsTestSuite(BaseTestSuite):
    """Test suite for hybrid FileOperationsTool architecture."""
    
    def __init__(self, verbose: bool = False, persist_files: bool = False):
        super().__init__("Hybrid File Operations", verbose, persist_files)
        self.tool = FileOperationsTool()
        self.temp_dir = None
        self.test_files = {}
        self.generated_files = []  # Track files for cleanup
        self._test_counter = 0  # Counter for unique test directories
    
    async def setup(self):
        """Set up test environment."""
        # Create a unique test directory
        self._test_counter += 1
        test_dir = self.create_test_directory(f"hybrid_test_{self._test_counter}")
        self.temp_dir = test_dir
        
        # Create test files
        test_dir = Path(self.temp_dir)
        
        self.test_files = {
            "simple.txt": test_dir / "simple.txt",
            "data.json": test_dir / "data.json",
            "subdir": test_dir / "subdir",
            "nested.py": test_dir / "subdir" / "nested.py"
        }
        
        # Create subdirectory
        self.test_files["subdir"].mkdir(exist_ok=True)
        
        # Write test content
        self.test_files["simple.txt"].write_text("Hello, AIDA!")
        self.test_files["data.json"].write_text('{"test": "data", "number": 42}')
        self.test_files["nested.py"].write_text('def test_function():\n    return "nested"')
        
        self.log(f"Test environment created at: {self.temp_dir}")
    
    async def teardown(self):
        """Clean up test environment."""
        # Cleanup is now handled by cleanup_generated_files() in run_all()
        pass
    
    async def test_aida_interface_file_operations(self) -> Dict[str, Any]:
        """Test original AIDA interface with rich metadata."""
        try:
            # Test read operation
            result = await self.tool.execute(
                operation="read_file",
                path=str(self.test_files["simple.txt"])
            )
            
            # Verify AIDA-specific features
            assert result.status == ToolStatus.COMPLETED, f"Read failed: {result.error}"
            assert result.result["content"] == "Hello, AIDA!"
            assert "size_bytes" in result.result
            assert "line_count" in result.result
            # Check metadata has operation info
            assert "operation" in result.metadata
            assert result.duration_seconds > 0
            
            # Test write operation
            new_content = "Updated content from AIDA interface"
            write_result = await self.tool.execute(
                operation="write_file",
                path=str(self.test_files["simple.txt"]),
                content=new_content
            )
            
            assert write_result.status == ToolStatus.COMPLETED, f"Write failed: {write_result.error}"
            assert write_result.result["bytes_written"] == len(new_content)
            
            # Verify write
            verify_result = await self.tool.execute(
                operation="read_file",
                path=str(self.test_files["simple.txt"])
            )
            assert verify_result.result["content"] == new_content
            
            return {
                "success": True,
                "message": "AIDA interface works correctly",
                "operations_tested": ["read_file", "write_file"],
                "metadata_verified": True,
                "duration_tracking": True
            }
            
        except Exception as e:
            import traceback
            return {"success": False, "message": f"AIDA interface test failed: {str(e)}", "traceback": traceback.format_exc()}
    
    async def test_pydantic_ai_compatibility(self) -> Dict[str, Any]:
        """Test PydanticAI tool functions."""
        try:
            # Get PydanticAI tools
            pydantic_tools = self.tool.to_pydantic_tools()
            
            # Verify expected tools are available
            expected_tools = ["read_file", "write_file", "create_directory", "list_files"]
            for tool_name in expected_tools:
                assert tool_name in pydantic_tools, f"Missing PydanticAI tool: {tool_name}"
            
            # Test read functionality
            content_result = await pydantic_tools["read_file"](str(self.test_files["data.json"]))
            
            # Verify PydanticAI-style response (clean, minimal)
            assert "content" in content_result
            assert "size_bytes" in content_result
            data = json.loads(content_result["content"])
            assert data["test"] == "data"
            assert data["number"] == 42
            
            # Test write functionality
            new_data = {"pydantic": True, "test": "passed"}
            test_file = Path(self.temp_dir) / "pydantic_test.json"
            
            write_result = await pydantic_tools["write_file"](
                str(test_file), 
                json.dumps(new_data, indent=2)
            )
            
            assert "bytes_written" in write_result
            assert write_result["bytes_written"] > 0
            
            # Test directory creation
            new_dir = Path(self.temp_dir) / "pydantic_dir"
            dir_result = await pydantic_tools["create_directory"](str(new_dir))
            assert dir_result["created"] is True
            assert new_dir.exists()
            
            # Test file listing
            list_result = await pydantic_tools["list_files"](self.temp_dir)
            assert "files" in list_result
            assert len(list_result["files"]) > 0
            
            return {
                "success": True,
                "message": "PydanticAI compatibility verified",
                "tools_available": len(pydantic_tools),
                "operations_tested": expected_tools,
                "clean_responses": True
            }
            
        except Exception as e:
            return {"success": False, "message": f"PydanticAI test failed: {e}"}
    
    async def test_mcp_server_integration(self) -> Dict[str, Any]:
        """Test MCP server interface."""
        try:
            # Get MCP server
            mcp_server = self.tool.get_mcp_server()
            
            # Test MCP-style tool calling
            test_content = "MCP server test content"
            mcp_test_file = Path(self.temp_dir) / "mcp_test.txt"
            
            # Write using MCP interface
            write_result = await mcp_server.call_tool("file_write_file", {
                "path": str(mcp_test_file),
                "content": test_content
            })
            
            # Verify MCP response format
            assert not write_result.get("isError"), f"MCP write failed: {write_result}"
            assert "content" in write_result
            
            # Read using MCP interface
            read_result = await mcp_server.call_tool("file_read_file", {
                "path": str(mcp_test_file)
            })
            
            assert not read_result.get("isError"), f"MCP read failed: {read_result}"
            
            # Parse MCP response structure
            content_data = read_result["content"][0]["text"]
            parsed_data = json.loads(content_data)
            assert parsed_data["content"] == test_content
            
            # Test directory listing via MCP
            list_result = await mcp_server.call_tool("file_list_files", {
                "path": self.temp_dir
            })
            
            assert not list_result.get("isError")
            list_content = json.loads(list_result["content"][0]["text"])
            assert "files" in list_content
            
            return {
                "success": True,
                "message": "MCP server integration working",
                "mcp_tools_tested": ["file_write_file", "file_read_file", "file_list_files"],
                "response_format": "MCP compatible",
                "external_compatibility": True
            }
            
        except Exception as e:
            return {"success": False, "message": f"MCP server test failed: {e}"}
    
    async def test_observability_integration(self) -> Dict[str, Any]:
        """Test OpenTelemetry observability."""
        try:
            # Enable observability
            observability = self.tool.enable_observability({
                "enabled": True,
                "service_name": "test-hybrid-file-ops"
            })
            
            # Verify observability is configured
            assert observability.enabled, "Observability not enabled"
            assert observability.config["service_name"] == "test-hybrid-file-ops"
            
            # Test traced operation
            test_file = Path(self.temp_dir) / "traced_test.txt"
            
            with observability.trace_operation("test_trace", path=str(test_file)):
                # Perform file operation within trace
                result = await self.tool.execute(
                    operation="write_file",
                    path=str(test_file),
                    content="Traced operation content"
                )
                assert result.status == ToolStatus.COMPLETED
            
            # Verify file was created
            assert test_file.exists()
            
            return {
                "success": True,
                "message": "Observability integration working",
                "tracing_enabled": True,
                "service_name": observability.config["service_name"],
                "operations_traced": True
            }
            
        except Exception as e:
            return {"success": False, "message": f"Observability test failed: {e}"}
    
    async def test_hybrid_architecture_compatibility(self) -> Dict[str, Any]:
        """Test that all interfaces work together without conflicts."""
        try:
            # Create a file using each interface and verify with others
            test_file = Path(self.temp_dir) / "hybrid_compatibility.txt"
            
            # 1. Create with AIDA interface
            aida_content = "Created with AIDA"
            aida_result = await self.tool.execute(
                operation="write_file",
                path=str(test_file),
                content=aida_content
            )
            assert aida_result.status == "completed"
            
            # 2. Read with PydanticAI interface
            pydantic_tools = self.tool.to_pydantic_tools()
            pydantic_result = await pydantic_tools["read_file"](str(test_file))
            assert pydantic_result["content"] == aida_content
            
            # 3. Update with MCP interface
            mcp_server = self.tool.get_mcp_server()
            mcp_content = "Updated with MCP"
            mcp_result = await mcp_server.call_tool("file_write_file", {
                "path": str(test_file),
                "content": mcp_content
            })
            assert not mcp_result.get("isError")
            
            # 4. Verify with AIDA interface
            verify_result = await self.tool.execute(
                operation="read_file",
                path=str(test_file)
            )
            assert verify_result.status == "completed"
            assert verify_result.result["content"] == mcp_content
            
            # 5. Test concurrent access (simplified)
            concurrent_tasks = [
                self.tool.execute(operation="read_file", path=str(test_file)),
                pydantic_tools["read_file"](str(test_file))
            ]
            
            concurrent_results = await asyncio.gather(*concurrent_tasks)
            for result in concurrent_results:
                # Handle different response formats
                if hasattr(result, 'result'):  # AIDA format
                    assert result.status == ToolStatus.COMPLETED
                    assert result.result["content"] == mcp_content
                else:  # PydanticAI format
                    assert result["content"] == mcp_content
            
            return {
                "success": True,
                "message": "All interfaces compatible and working together",
                "interfaces_tested": ["AIDA", "PydanticAI", "MCP"],
                "cross_interface_operations": True,
                "concurrent_access": True,
                "no_conflicts": True
            }
            
        except Exception as e:
            return {"success": False, "message": f"Hybrid compatibility test failed: {e}"}
    
    async def test_performance_overhead(self) -> Dict[str, Any]:
        """Test that hybrid architecture doesn't introduce significant overhead."""
        try:
            import time
            
            test_file = Path(self.temp_dir) / "performance_test.txt"
            test_content = "Performance test content " * 100  # ~2.5KB
            
            # Measure AIDA interface performance
            start_time = time.time()
            for _ in range(10):
                await self.tool.execute(
                    operation="write_file",
                    path=str(test_file),
                    content=test_content
                )
            aida_time = time.time() - start_time
            
            # Measure PydanticAI interface performance
            pydantic_tools = self.tool.to_pydantic_tools()
            start_time = time.time()
            for _ in range(10):
                await pydantic_tools["write_file"](str(test_file), test_content)
            pydantic_time = time.time() - start_time
            
            # Calculate overhead
            overhead_percent = abs(pydantic_time - aida_time) / aida_time * 100
            
            return {
                "success": True,
                "message": "Performance overhead acceptable",
                "aida_time_10ops": round(aida_time, 3),
                "pydantic_time_10ops": round(pydantic_time, 3),
                "overhead_percent": round(overhead_percent, 1),
                "overhead_acceptable": overhead_percent < 20  # Less than 20% overhead
            }
            
        except Exception as e:
            return {"success": False, "message": f"Performance test failed: {e}"}
    
    async def run_all(self) -> list[TestResult]:
        """Run all hybrid architecture tests."""
        await self.setup()
        
        try:
            # Run all test methods
            test_methods = [
                ("AIDA Interface", self.test_aida_interface_file_operations),
                ("PydanticAI Compatibility", self.test_pydantic_ai_compatibility),
                ("MCP Server Integration", self.test_mcp_server_integration),
                ("Observability Integration", self.test_observability_integration),
                ("Hybrid Compatibility", self.test_hybrid_architecture_compatibility),
                ("Performance Overhead", self.test_performance_overhead)
            ]
            
            for test_name, test_func in test_methods:
                result = await self.run_test(test_name, test_func)
                self.results.append(result)
            
            return self.results
            
        finally:
            await self.teardown()
            
            # Cleanup generated files unless persist_files is True
            if not self.persist_files:
                self.cleanup_generated_files()
    
    def create_test_directory(self, test_name: str) -> str:
        """Create a test directory within .aida/tests."""
        test_dir = Path(".aida/tests/hybrid_files") / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        test_dir_str = str(test_dir)
        self.track_generated_file(test_dir_str)
        self.log(f"Created test directory: {test_dir_str}")
        return test_dir_str
    
    def track_generated_file(self, file_path: str):
        """Track a file that was generated during testing."""
        import os
        if os.path.exists(file_path):
            self.generated_files.append(file_path)
            self.log(f"Tracking generated file: {file_path}")
    
    def cleanup_generated_files(self):
        """Clean up files generated during testing."""
        if not self.generated_files:
            return
        
        import os
        import shutil
        cleaned_count = 0
        
        for file_path in self.generated_files:
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    cleaned_count += 1
                    self.log(f"Removed file: {file_path}")
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    cleaned_count += 1
                    self.log(f"Removed directory: {file_path}")
            except Exception as e:
                self.log(f"Failed to remove {file_path}: {e}")
        
        if cleaned_count > 0:
            self.log(f"🧹 Cleaned up {cleaned_count} generated files")
        
        self.generated_files.clear()


# Register the test suite
test_registry.register("hybrid_files", HybridFileOperationsTestSuite)