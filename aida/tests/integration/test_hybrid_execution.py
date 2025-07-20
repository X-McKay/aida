"""Integration tests for hybrid ExecutionTool."""

import asyncio
import json
from typing import Dict, Any, List
from pathlib import Path

from aida.tests.base import BaseTestSuite, TestResult, test_registry
from aida.tools.execution import ExecutionTool
from aida.tools.base import ToolStatus
from pydantic_ai import Agent
import aida


class HybridExecutionTestSuite(BaseTestSuite):
    """Test suite for hybrid ExecutionTool functionality."""
    
    def __init__(self, verbose: bool = False, persist_files: bool = False):
        super().__init__("Hybrid ExecutionTool", verbose, persist_files)
        self.tool = ExecutionTool()
    
    async def test_core_interface_python(self) -> Dict[str, Any]:
        """Test core interface with Python execution."""
        self.log("Testing core interface with Python")
        
        result = await self.tool.execute(
            language="python",
            code="print('Hello from Python!')\nprint(2 + 2)",
            timeout=30
        )
        
        if result.status != ToolStatus.COMPLETED:
            return {"success": False, "message": f"Execution failed: {result.error}"}
        
        
        self.log(f"Result type: {type(result.result)}")
        self.log(f"Result: {result.result}")
        
        stdout = result.result.get("output", result.result.get("stdout", "")) if isinstance(result.result, dict) else ""
        expected_output = "Hello from Python!\n4"
        
        if expected_output in stdout:
            return {
                "success": True,
                "message": "Python execution successful",
                "output": stdout[:100]
            }
        else:
            return {
                "success": False,
                "message": f"Unexpected output: {stdout[:100]}"
            }
    
    async def test_core_interface_bash(self) -> Dict[str, Any]:
        """Test core interface with Bash execution."""
        self.log("Testing core interface with Bash")
        
        result = await self.tool.execute(
            language="bash",
            code="echo 'Hello from Bash!'\nls -la /workspace",
            timeout=30
        )
        
        if result.status != ToolStatus.COMPLETED:
            return {"success": False, "message": f"Execution failed: {result.error}"}
        
        stdout = result.result.get("output", result.result.get("stdout", ""))
        
        if "Hello from Bash!" in stdout:
            return {
                "success": True,
                "message": "Bash execution successful",
                "output": stdout[:100]
            }
        else:
            return {
                "success": False,
                "message": f"Unexpected output: {stdout[:100]}"
            }
    
    async def test_pydantic_ai_interface(self) -> Dict[str, Any]:
        """Test PydanticAI interface."""
        self.log("Testing PydanticAI interface")
        
        tools = self.tool.to_pydantic_tools()
        
        # Check available tools
        if "run_python" not in tools:
            return {
                "success": False,
                "message": f"run_python not in tools. Available: {list(tools.keys())}"
            }
        
        # Test run_python function
        try:
            result = await tools["run_python"](
                code="import sys\nprint(f'Python {sys.version.split()[0]}')"
            )
            
            # run_python returns a string directly
            stdout = result if isinstance(result, str) else result.get("stdout", "")
            
            if "Python" in stdout:
                return {
                    "success": True,
                    "message": "PydanticAI interface working",
                    "output": stdout[:100]
                }
            else:
                return {
                    "success": False,
                    "message": f"Unexpected output: {stdout[:100]}"
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
        tool_names = [t["name"] for t in tools]
        
        if "execution_execute" not in tool_names:
            return {
                "success": False,
                "message": f"MCP tools not properly registered. Available: {tool_names}"
            }
        
        # Execute Python via MCP
        result = await mcp_server.call_tool(
            "execution_run_python",
            {
                "code": "print('MCP execution test')\nprint(10 * 5)",
                "timeout": 30
            }
        )
        
        if result.get("isError", False):
            error_text = result.get("content", [{}])[0].get("text", "Unknown error")
            return {
                "success": False,
                "message": f"MCP execution error: {error_text}"
            }
        
        # Parse the JSON response from content
        import json
        response_text = result.get("content", [{}])[0].get("text", "{}") if isinstance(result, dict) else "{}"
        try:
            response_data = json.loads(response_text)
            stdout = response_data.get("output", response_data.get("stdout", ""))
        except json.JSONDecodeError:
            stdout = response_text
        
        if "MCP execution test" in stdout:
            return {
                "success": True,
                "message": "MCP interface working",
                "output": stdout[:100]
            }
        else:
            return {
                "success": False,
                "message": f"Unexpected output: {stdout[:100]}"
            }
    
    async def test_observability_interface(self) -> Dict[str, Any]:
        """Test OpenTelemetry observability interface."""
        self.log("Testing observability interface")
        
        try:
            observability = self.tool.enable_observability({
                "enabled": True,
                "service_name": "test-execution",
                "endpoint": "http://localhost:4317"
            })
            
            # Check that observability components are created
            if not hasattr(observability, 'tracer'):
                return {
                    "success": False,
                    "message": "Tracer not initialized"
                }
            
            if not hasattr(observability, 'execution_counter'):
                return {
                    "success": False,
                    "message": "Metrics not initialized"
                }
            
            # Test creating a span
            with observability.trace_execution("python", 100):
                # Simulate execution
                pass
            
            # Record metrics
            observability.record_execution("python", 1.5, True)
            
            return {
                "success": True,
                "message": "Observability interface configured",
                "components": ["tracer", "metrics"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Observability error: {str(e)}"
            }
    
    async def test_with_packages(self) -> Dict[str, Any]:
        """Test execution with package installation."""
        self.log("Testing execution with packages")
        
        result = await self.tool.execute(
            language="python",
            code="""
# Try to use a standard library module
import json
data = {"test": "package_test", "value": 42}
print(json.dumps(data, indent=2))
""",
            packages=["requests"],  # Include a package even if not used
            timeout=60
        )
        
        if result.status != ToolStatus.COMPLETED:
            return {"success": False, "message": f"Execution failed: {result.error}"}
        
        stdout = result.result.get("output", result.result.get("stdout", ""))
        
        if "package_test" in stdout:
            return {
                "success": True,
                "message": "Package execution successful",
                "output": stdout[:100]
            }
        else:
            return {
                "success": False,
                "message": f"Unexpected output: {stdout[:100]}"
            }
    
    async def test_with_files(self) -> Dict[str, Any]:
        """Test execution with additional files."""
        self.log("Testing execution with files")
        
        result = await self.tool.execute(
            language="python",
            code="""
with open('data.txt', 'r') as f:
    content = f.read()
print(f"File content: {content}")
""",
            files={
                "data.txt": "Hello from file!"
            },
            timeout=30
        )
        
        if result.status != ToolStatus.COMPLETED:
            return {"success": False, "message": f"Execution failed: {result.error}"}
        
        stdout = result.result.get("output", result.result.get("stdout", ""))
        
        if "Hello from file!" in stdout:
            return {
                "success": True,
                "message": "File execution successful",
                "output": stdout[:100]
            }
        else:
            return {
                "success": False,
                "message": f"Unexpected output: {stdout[:100]}"
            }
    
    async def test_timeout_handling(self) -> Dict[str, Any]:
        """Test timeout handling."""
        self.log("Testing timeout handling")
        
        result = await self.tool.execute(
            language="python",
            code="""
import time
print("Starting long operation...")
time.sleep(10)  # This should timeout
print("This should not print")
""",
            timeout=2  # Short timeout
        )
        
        if result.status == ToolStatus.FAILED:
            if result.error and ("timeout" in result.error.lower() or "timed out" in result.error.lower()):
                return {
                    "success": True,
                    "message": "Timeout handled correctly",
                    "error": result.error
                }
        
        return {
            "success": False,
            "message": "Timeout not handled properly",
            "status": result.status.value,
            "error": result.error
        }
    
    async def test_javascript_execution(self) -> Dict[str, Any]:
        """Test JavaScript execution."""
        self.log("Testing JavaScript execution")
        
        result = await self.tool.execute(
            language="javascript",
            code="""
console.log('Hello from JavaScript!');
const result = [1, 2, 3].map(x => x * 2);
console.log('Result:', result);
""",
            timeout=30
        )
        
        if result.status != ToolStatus.COMPLETED:
            return {"success": False, "message": f"Execution failed: {result.error}"}
        
        stdout = result.result.get("output", result.result.get("stdout", ""))
        
        if "Hello from JavaScript!" in stdout:
            return {
                "success": True,
                "message": "JavaScript execution successful",
                "output": stdout[:100]
            }
        else:
            return {
                "success": False,
                "message": f"Unexpected output: {stdout[:100]}"
            }
    
    async def run_all(self) -> List[TestResult]:
        """Run all execution tests."""
        tests = [
            ("Core Interface - Python", self.test_core_interface_python),
            ("Core Interface - Bash", self.test_core_interface_bash),
            ("PydanticAI Interface", self.test_pydantic_ai_interface),
            ("MCP Server Interface", self.test_mcp_server_interface),
            ("Observability Interface", self.test_observability_interface),
            ("Execution with Packages", self.test_with_packages),
            ("Execution with Files", self.test_with_files),
            ("Timeout Handling", self.test_timeout_handling),
            ("JavaScript Execution", self.test_javascript_execution),
        ]
        
        for test_name, test_func in tests:
            result = await self.run_test(test_name, test_func)
            self.results.append(result)
        
        return self.results


# Register the test suite
test_registry.register("hybrid_execution", HybridExecutionTestSuite)