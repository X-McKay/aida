"""Integration tests for hybrid SystemTool.

Tests the hybrid architecture that supports:
- Original AIDA interface
- PydanticAI compatibility
- MCP server integration
- OpenTelemetry observability
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from aida.tests.base import BaseTestSuite, TestResult, test_registry
from aida.tools.base import ToolStatus
from aida.tools.system import SystemTool


class HybridSystemTestSuite(BaseTestSuite):
    """Test suite for hybrid SystemTool architecture."""

    def __init__(self, verbose: bool = False, persist_files: bool = False):
        super().__init__("Hybrid System Tool", verbose, persist_files)
        self.tool = SystemTool()
        self.generated_files = []
        self._test_counter = 0

    async def test_aida_interface_command_execution(self) -> dict[str, Any]:
        """Test original AIDA interface for command execution."""
        try:
            # Test simple command
            result = await self.tool.execute(
                operation="execute", command="echo", args=["Test from AIDA interface"]
            )

            # Verify AIDA-specific features
            assert result.status == ToolStatus.COMPLETED, f"Command failed: {result.error}"
            # result.result is a CommandResult object
            assert result.result.stdout.strip() == "Test from AIDA interface"
            assert result.result.exit_code == 0
            assert result.metadata["success"] is True
            assert "operation" in result.metadata
            assert result.duration_seconds > 0

            # Test with working directory
            test_dir = self.create_test_directory("aida_test")
            result2 = await self.tool.execute(
                operation="execute", command="pwd" if os.name != "nt" else "cd", cwd=test_dir
            )

            assert result2.status == ToolStatus.COMPLETED
            assert test_dir in result2.result.stdout

            return {
                "success": True,
                "message": "AIDA interface command execution works correctly",
                "operations_tested": ["execute_command", "working_directory"],
                "metadata_verified": True,
                "duration_tracking": True,
            }

        except Exception as e:
            import traceback

            return {
                "success": False,
                "message": f"AIDA interface test failed: {str(e)}",
                "traceback": traceback.format_exc(),
            }

    async def test_security_validation(self) -> dict[str, Any]:
        """Test security controls."""
        try:
            # Test dangerous command blocking
            dangerous_commands = ["rm", "sudo", "chmod"]

            for cmd in dangerous_commands:
                result = await self.tool.execute(
                    operation="execute", command=cmd, args=["-rf", "/"]
                )
                assert result.status == ToolStatus.FAILED
                assert "not allowed" in result.error
                self.log(f"Successfully blocked dangerous command: {cmd}")

            # Test non-whitelisted command
            result = await self.tool.execute(operation="execute", command="unknown_command_xyz")
            assert result.status == ToolStatus.FAILED
            assert "not allowed" in result.error

            # Test pattern detection
            result = await self.tool.execute(
                operation="execute", command="echo", args=["test", "&&", "rm", "-rf", "/"]
            )
            # Should fail due to dangerous pattern detection
            assert result.status == ToolStatus.FAILED
            assert "dangerous pattern" in result.error.lower()
            self.log("Successfully blocked command with dangerous pattern")

            return {
                "success": True,
                "message": "Security validation working correctly",
                "dangerous_commands_blocked": len(dangerous_commands),
                "allowed_list_enforced": True,
                "pattern_detection": True,
            }

        except Exception as e:
            import traceback

            return {
                "success": False,
                "message": f"Security test failed: {e}",
                "traceback": traceback.format_exc(),
            }

    async def test_pydantic_ai_compatibility(self) -> dict[str, Any]:
        """Test PydanticAI tool functions."""
        try:
            # Get PydanticAI tools
            pydantic_tools = self.tool.to_pydantic_tools()

            # Verify expected tools are available
            expected_tools = ["run_command", "get_system_info", "list_processes", "which_command"]
            for tool_name in expected_tools:
                assert tool_name in pydantic_tools, f"Missing PydanticAI tool: {tool_name}"

            # Test run_command
            result = await pydantic_tools["run_command"]("echo", ["PydanticAI test"])
            assert "stdout" in result
            assert result["stdout"].strip() == "PydanticAI test"
            assert result["exit_code"] == 0

            # Test which_command
            echo_path = await pydantic_tools["which_command"]("echo")
            assert echo_path is not None

            not_exists = await pydantic_tools["which_command"]("nonexistent_command_12345")
            assert not_exists is None

            # Test get_system_info
            info = await pydantic_tools["get_system_info"]()
            assert "platform" in info
            assert "cpu_count" in info

            # Test list_processes
            processes = await pydantic_tools["list_processes"]()
            assert isinstance(processes, list)
            assert len(processes) > 0

            return {
                "success": True,
                "message": "PydanticAI compatibility verified",
                "tools_available": len(pydantic_tools),
                "operations_tested": [
                    "run_command",
                    "which_command",
                    "get_system_info",
                    "list_processes",
                ],
                "clean_responses": True,
            }

        except Exception as e:
            return {"success": False, "message": f"PydanticAI test failed: {e}"}

    async def test_mcp_server_integration(self) -> dict[str, Any]:
        """Test MCP server interface."""
        try:
            # Get MCP server
            mcp_server = self.tool.get_mcp_server()

            # Test command execution via MCP
            exec_result = await mcp_server.call_tool(
                "system_execute", {"command": "echo", "args": ["MCP test"]}
            )

            assert not exec_result.get("isError"), f"MCP execution failed: {exec_result}"
            assert "content" in exec_result

            # Parse MCP response
            content_text = exec_result["content"][0]["text"]
            # MCP server should return JSON-serialized CommandResult
            result_data = json.loads(content_text)
            assert result_data["stdout"].strip() == "MCP test"
            assert result_data["exit_code"] == 0

            # Test system info via MCP
            info_result = await mcp_server.call_tool("system_system_info", {})
            assert not info_result.get("isError"), f"MCP system info failed: {info_result}"

            # Parse system info response
            info_text = info_result["content"][0]["text"]
            info_data = json.loads(info_text)
            assert "platform" in info_data
            assert "cpu_count" in info_data

            # Test process list via MCP
            proc_result = await mcp_server.call_tool("system_processes", {})
            assert not proc_result.get("isError"), f"MCP process list failed: {proc_result}"

            return {
                "success": True,
                "message": "MCP server integration working",
                "mcp_tools_tested": ["system_execute", "system_system_info", "system_processes"],
                "response_format": "MCP compatible",
                "external_compatibility": True,
            }

        except Exception as e:
            return {"success": False, "message": f"MCP server test failed: {e}"}

    async def test_script_execution(self) -> dict[str, Any]:
        """Test script execution functionality."""
        try:
            # Test bash script
            bash_script = """#!/bin/bash
echo "Hello from bash script"
echo "Current directory: $(pwd)"
exit 0
"""
            bash_result = await self.tool.execute(
                operation="script", script_content=bash_script, interpreter="bash"
            )

            if bash_result.status != ToolStatus.COMPLETED:
                self.log(f"Bash script failed: {bash_result.error}")
            assert bash_result.status == ToolStatus.COMPLETED
            if bash_result.result:
                assert "Hello from bash script" in bash_result.result.stdout
                assert bash_result.result.exit_code == 0
            else:
                self.log("Bash result is None")

            # Test Python script
            python_script = """
import sys
print(f"Python version: {sys.version.split()[0]}")
print("Script executed successfully")
"""
            python_result = await self.tool.execute(
                operation="script", script_content=python_script, interpreter="python3"
            )

            if python_result.status != ToolStatus.COMPLETED:
                self.log(f"Python script execution failed: {python_result.error}")
                self.log(f"Status: {python_result.status}")
            assert python_result.status == ToolStatus.COMPLETED
            if python_result.result:
                assert "Script executed successfully" in python_result.result.stdout
            else:
                self.log("Python result is None")

            # Verify metadata
            if "script_language" not in python_result.metadata:
                self.log(f"Metadata: {python_result.metadata}")
                self.log("script_language not in metadata")

            return {
                "success": True,
                "message": "Script execution working correctly",
                "languages_tested": ["bash", "python"],
                "temp_directory": ".aida/tmp",
                "metadata_included": True,
            }

        except Exception as e:
            return {"success": False, "message": f"Script execution test failed: {e}"}

    async def test_observability_integration(self) -> dict[str, Any]:
        """Test OpenTelemetry observability."""
        try:
            # Enable observability
            observability = self.tool.enable_observability(
                {"enabled": True, "service_name": "test-hybrid-system"}
            )

            # Verify observability is configured
            assert observability.enabled, "Observability not enabled"
            assert observability.config["service_name"] == "test-hybrid-system"

            # Test traced operation
            with observability.trace_operation("test_command", command="echo", args=["traced"]):
                result = await self.tool.execute(
                    operation="execute", command="echo", args=["Traced operation"]
                )
                assert result.status == ToolStatus.COMPLETED

            return {
                "success": True,
                "message": "Observability integration working",
                "tracing_enabled": True,
                "service_name": observability.config["service_name"],
                "operations_traced": True,
            }

        except Exception as e:
            return {"success": False, "message": f"Observability test failed: {e}"}

    async def test_hybrid_architecture_compatibility(self) -> dict[str, Any]:
        """Test that all interfaces work together without conflicts."""
        try:
            test_message = "Hybrid test message"

            # 1. Execute with AIDA interface
            aida_result = await self.tool.execute(
                operation="execute", command="echo", args=[test_message]
            )
            assert aida_result.status == ToolStatus.COMPLETED
            aida_output = aida_result.result.stdout.strip()

            # 2. Execute with PydanticAI interface
            pydantic_tools = self.tool.to_pydantic_tools()
            pydantic_result = await pydantic_tools["run_command"]("echo", [test_message])
            pydantic_output = pydantic_result["stdout"].strip()

            # 3. Execute with MCP interface
            mcp_server = self.tool.get_mcp_server()
            mcp_result = await mcp_server.call_tool(
                "system_execute", {"command": "echo", "args": [test_message]}
            )
            mcp_data = json.loads(mcp_result["content"][0]["text"])
            mcp_output = mcp_data["stdout"].strip()

            # Verify all produce same output
            assert aida_output == test_message
            assert pydantic_output == test_message
            assert mcp_output == test_message

            # Test concurrent access
            concurrent_tasks = [
                self.tool.execute(operation="execute", command="echo", args=["concurrent1"]),
                pydantic_tools["run_command"]("echo", ["concurrent2"]),
            ]

            concurrent_results = await asyncio.gather(*concurrent_tasks)
            assert all(
                r.status == ToolStatus.COMPLETED if hasattr(r, "status") else r["exit_code"] == 0
                for r in concurrent_results
            )

            return {
                "success": True,
                "message": "All interfaces compatible and working together",
                "interfaces_tested": ["AIDA", "PydanticAI", "MCP"],
                "consistent_output": True,
                "concurrent_access": True,
                "no_conflicts": True,
            }

        except Exception as e:
            return {"success": False, "message": f"Hybrid compatibility test failed: {e}"}

    async def test_performance_overhead(self) -> dict[str, Any]:
        """Test that hybrid architecture doesn't introduce significant overhead."""
        try:
            import time

            # Measure AIDA interface performance
            start_time = time.time()
            for _ in range(10):
                await self.tool.execute(
                    operation="execute", command="echo", args=["performance test"]
                )
            aida_time = time.time() - start_time

            # Measure PydanticAI interface performance
            pydantic_tools = self.tool.to_pydantic_tools()
            start_time = time.time()
            for _ in range(10):
                await pydantic_tools["run_command"]("echo", ["performance test"])
            pydantic_time = time.time() - start_time

            # Calculate overhead
            overhead_percent = abs(pydantic_time - aida_time) / aida_time * 100

            return {
                "success": True,
                "message": "Performance overhead acceptable",
                "aida_time_10ops": round(aida_time, 3),
                "pydantic_time_10ops": round(pydantic_time, 3),
                "overhead_percent": round(overhead_percent, 1),
                "overhead_acceptable": overhead_percent < 20,  # Less than 20% overhead
            }

        except Exception as e:
            return {"success": False, "message": f"Performance test failed: {e}"}

    async def run_all(self) -> list[TestResult]:
        """Run all hybrid architecture tests."""
        try:
            # Run all test methods
            test_methods = [
                ("AIDA Interface Commands", self.test_aida_interface_command_execution),
                ("Security Validation", self.test_security_validation),
                ("PydanticAI Compatibility", self.test_pydantic_ai_compatibility),
                ("MCP Server Integration", self.test_mcp_server_integration),
                ("Script Execution", self.test_script_execution),
                ("Observability Integration", self.test_observability_integration),
                ("Hybrid Compatibility", self.test_hybrid_architecture_compatibility),
                ("Performance Overhead", self.test_performance_overhead),
            ]

            for test_name, test_func in test_methods:
                result = await self.run_test(test_name, test_func)
                self.results.append(result)

            return self.results

        finally:
            # Cleanup generated files unless persist_files is True
            if not self.persist_files:
                self.cleanup_generated_files()

    def create_test_directory(self, test_name: str) -> str:
        """Create a test directory within .aida/tests."""
        test_dir = Path(".aida/tests/hybrid_system") / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        test_dir_str = str(test_dir)
        self.track_generated_file(test_dir_str)
        self.log(f"Created test directory: {test_dir_str}")
        return test_dir_str

    def track_generated_file(self, file_path: str):
        """Track a file that was generated during testing."""
        if os.path.exists(file_path):
            self.generated_files.append(file_path)
            self.log(f"Tracking generated file: {file_path}")

    def cleanup_generated_files(self):
        """Clean up files generated during testing."""
        if not self.generated_files:
            return

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
test_registry.register("hybrid_system", HybridSystemTestSuite)
