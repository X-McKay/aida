#!/usr/bin/env python3
"""
Basic usage examples of the hybrid FileOperationsTool.

This demonstrates all three usage patterns:
1. Original AIDA tool interface
2. PydanticAI compatibility 
3. MCP server integration
"""

import asyncio
import tempfile
from pathlib import Path

# Import the hybrid file operations tool
from aida.tools.files import FileOperationsTool


async def demo_original_interface():
    """Demonstrate original AIDA tool interface."""
    print("🔧 ORIGINAL AIDA INTERFACE")
    print("=" * 50)
    
    file_tool = FileOperationsTool()
    
    # Create a temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.txt"
        
        # Write a file
        result = await file_tool.execute(
            operation="write_file",
            path=str(test_file),
            content="Hello from AIDA FileOperationsTool!"
        )
        print(f"✅ Write result: {result.status}")
        print(f"   Metadata: {result.metadata}")
        
        # Read the file back
        result = await file_tool.execute(
            operation="read_file",
            path=str(test_file)
        )
        print(f"✅ Read content: {result.result['content'][:50]}...")
        
        # List files
        result = await file_tool.execute(
            operation="list_files",
            path=str(temp_path)
        )
        print(f"✅ Found {result.result['total_files']} files")


async def demo_pydantic_compatibility():
    """Demonstrate PydanticAI compatibility."""
    print("\n🤖 PYDANTIC AI COMPATIBILITY")
    print("=" * 50)
    
    file_tool = FileOperationsTool()
    
    # Get PydanticAI-compatible tools
    pydantic_tools = file_tool.to_pydantic_tools()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "pydantic_test.txt"
        
        # Use the individual tool functions
        write_result = await pydantic_tools["write_file"](
            str(test_file), 
            "Hello from PydanticAI compatibility layer!"
        )
        print(f"✅ PydanticAI write: {write_result['bytes_written']} bytes")
        
        read_result = await pydantic_tools["read_file"](str(test_file))
        print(f"✅ PydanticAI read: {read_result['line_count']} lines")
        
        list_result = await pydantic_tools["list_files"](str(temp_path))
        print(f"✅ PydanticAI list: {list_result['total_files']} files")


async def demo_mcp_server():
    """Demonstrate MCP server interface."""
    print("\n🌐 MCP SERVER INTERFACE")
    print("=" * 50)
    
    file_tool = FileOperationsTool()
    mcp_server = file_tool.get_mcp_server()
    
    # List available MCP tools
    tools = await mcp_server.list_tools()
    print(f"✅ MCP server exposes {len(tools)} tools:")
    for tool in tools[:3]:  # Show first 3
        print(f"   - {tool['name']}: {tool['description']}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "mcp_test.txt"
        
        # Call MCP tools
        write_result = await mcp_server.call_tool(
            "file_write_file",
            {
                "path": str(test_file),
                "content": "Hello from MCP server interface!"
            }
        )
        print(f"✅ MCP write successful: {not write_result.get('isError', False)}")
        
        read_result = await mcp_server.call_tool(
            "file_read_file",
            {"path": str(test_file)}
        )
        print(f"✅ MCP read successful: {not read_result.get('isError', False)}")


async def demo_observability():
    """Demonstrate observability features."""
    print("\n📊 OBSERVABILITY FEATURES")
    print("=" * 50)
    
    file_tool = FileOperationsTool()
    
    # Enable observability (will warn if OpenTelemetry not installed)
    observability = file_tool.enable_observability({
        "enabled": True,
        "service_name": "aida-file-demo"
    })
    
    print(f"✅ Observability enabled: {observability.enabled}")
    
    # Demonstrate tracing
    if observability.enabled:
        span = observability.trace_operation("demo_operation", path="/demo/path")
        if span:
            print("✅ Created trace span for demo operation")
            span.end()
    else:
        print("⚠️  OpenTelemetry not available - install with:")
        print("   pip install opentelemetry-api opentelemetry-sdk")


async def main():
    """Run all demos."""
    print("🚀 HYBRID FILE OPERATIONS TOOL DEMO")
    print("=" * 60)
    
    await demo_original_interface()
    await demo_pydantic_compatibility()
    await demo_mcp_server()
    await demo_observability()
    
    print("\n✨ All demos completed successfully!")
    print("\nKey Benefits of Hybrid Architecture:")
    print("✅ Backward compatible with existing AIDA code")
    print("✅ PydanticAI integration for modern AI agents")
    print("✅ MCP server for interoperability with Claude, etc.")
    print("✅ OpenTelemetry for production observability")


if __name__ == "__main__":
    asyncio.run(main())