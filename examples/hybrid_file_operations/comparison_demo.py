#!/usr/bin/env python3
"""
Clear comparison showing the same file operations across all three interfaces.

This demonstrates how the SAME underlying tool can be used in completely 
different ways depending on your AI framework needs.
"""

import asyncio
import tempfile
from pathlib import Path

from aida.tools.files import FileOperationsTool


async def demo_same_task_three_ways():
    """
    Perform the EXACT SAME file operations using all three interfaces:
    1. Original AIDA interface
    2. PydanticAI interface  
    3. MCP interface
    
    Task: Create a file, write content, read it back, then clean up.
    """
    
    print("🎯 SAME TASK, THREE DIFFERENT INTERFACES")
    print("=" * 60)
    
    # Create temporary directory for all demos
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Initialize the hybrid tool once
        file_tool = FileOperationsTool()
        
        # ================================================================
        # METHOD 1: Original AIDA Interface (Existing Code Pattern)
        # ================================================================
        print("\n1️⃣  ORIGINAL AIDA INTERFACE")
        print("-" * 40)
        
        aida_file = temp_path / "aida_demo.txt"
        content = "Hello from Original AIDA interface!"
        
        # Write file using original interface
        result = await file_tool.execute(
            operation="write_file",
            path=str(aida_file),
            content=content
        )
        print(f"📝 Write: {result.status} ({result.metadata['files_processed']} file)")
        
        # Read file using original interface
        result = await file_tool.execute(
            operation="read_file", 
            path=str(aida_file)
        )
        print(f"📖 Read: {result.result['line_count']} lines, {result.result['character_count']} chars")
        print(f"   Content: '{result.result['content']}'")
        
        # ================================================================
        # METHOD 2: PydanticAI Interface (Modern AI Frameworks)
        # ================================================================
        print("\n2️⃣  PYDANTIC AI INTERFACE")
        print("-" * 40)
        
        pydantic_file = temp_path / "pydantic_demo.txt"
        content = "Hello from PydanticAI interface!"
        
        # Get PydanticAI tools
        tools = file_tool.to_pydantic_tools()
        
        # Write file using PydanticAI interface
        result = await tools["write_file"](str(pydantic_file), content)
        print(f"📝 Write: {result['bytes_written']} bytes written")
        
        # Read file using PydanticAI interface
        result = await tools["read_file"](str(pydantic_file))
        print(f"📖 Read: {result['line_count']} lines, {result['character_count']} chars")
        print(f"   Content: '{result['content']}'")
        
        # ================================================================
        # METHOD 3: MCP Interface (Claude, Universal AI Agents)
        # ================================================================
        print("\n3️⃣  MCP SERVER INTERFACE")
        print("-" * 40)
        
        mcp_file = temp_path / "mcp_demo.txt"
        content = "Hello from MCP interface!"
        
        # Get MCP server
        mcp_server = file_tool.get_mcp_server()
        
        # Write file using MCP interface
        result = await mcp_server.call_tool("file_write_file", {
            "path": str(mcp_file),
            "content": content
        })
        print(f"📝 Write: {'Success' if not result.get('isError') else 'Failed'}")
        if not result.get('isError'):
            # Parse the JSON response to get details
            import json
            write_data = json.loads(result['content'][0]['text'])
            print(f"   Details: {write_data['bytes_written']} bytes written")
        
        # Read file using MCP interface
        result = await mcp_server.call_tool("file_read_file", {
            "path": str(mcp_file)
        })
        print(f"📖 Read: {'Success' if not result.get('isError') else 'Failed'}")
        if not result.get('isError'):
            # Parse the JSON response to get content
            import json
            read_data = json.loads(result['content'][0]['text'])
            print(f"   Content: '{read_data['content']}'")
        
        # ================================================================
        # COMPARISON SUMMARY
        # ================================================================
        print("\n📊 INTERFACE COMPARISON")
        print("-" * 40)
        print("✅ All three interfaces performed the SAME operations")
        print("✅ Each interface has different strengths:")
        print("   🔧 AIDA: Rich metadata, execution tracking, statistics")
        print("   🤖 PydanticAI: Clean functions, type safety, modern AI frameworks") 
        print("   🌐 MCP: Universal compatibility, JSON-RPC, Claude integration")
        print("\n🎯 KEY INSIGHT: One tool, three ecosystems!")


async def demo_framework_integration():
    """
    Show how each interface would be used in real AI framework contexts.
    """
    
    print("\n\n🤖 FRAMEWORK INTEGRATION EXAMPLES")
    print("=" * 60)
    
    file_tool = FileOperationsTool()
    
    # ================================================================
    # AIDA Agent Usage (Original)
    # ================================================================
    print("\n📋 AIDA AGENT USAGE")
    print("-" * 30)
    print("```python")
    print("# In an AIDA agent workflow")
    print("from aida.tools.files import FileOperationsTool")
    print("")
    print("async def aida_agent_step():")
    print("    file_tool = FileOperationsTool()")
    print("    result = await file_tool.execute(")
    print("        operation='read_file',")
    print("        path='config.json'")
    print("    )")
    print("    return result.result")
    print("```")
    
    # ================================================================
    # PydanticAI Agent Usage
    # ================================================================
    print("\n📋 PYDANTIC AI AGENT USAGE")
    print("-" * 30)
    print("```python")
    print("# In a PydanticAI agent")
    print("from pydantic_ai import Agent")
    print("from aida.tools.files import FileOperationsTool")
    print("")
    print("agent = Agent(model='openai:gpt-4')")
    print("file_tool = FileOperationsTool()")
    print("")
    print("# Register file operations as tools")
    print("file_tool.register_with_pydantic_agent(agent)")
    print("")
    print("# Agent can now call: read_file, write_file, etc.")
    print("```")
    
    # ================================================================
    # MCP Client Usage
    # ================================================================
    print("\n📋 MCP CLIENT USAGE")
    print("-" * 30)
    print("```python")
    print("# In Claude Desktop or any MCP client")
    print("from aida.tools.files import FileOperationsTool")
    print("")
    print("# Expose as MCP server")
    print("file_tool = FileOperationsTool()")
    print("mcp_server = file_tool.get_mcp_server()")
    print("")
    print("# Claude can now discover and use:")
    print("# file_read_file, file_write_file, etc.")
    print("```")
    
    # ================================================================
    # Observability Usage
    # ================================================================
    print("\n📋 PRODUCTION OBSERVABILITY")
    print("-" * 30)
    print("```python")
    print("# In production deployment")
    print("file_tool = FileOperationsTool()")
    print("observability = file_tool.enable_observability({")
    print("    'enabled': True,")
    print("    'service_name': 'my-ai-agent'")
    print("})")
    print("")
    print("# All operations now traced with OpenTelemetry")
    print("```")


async def main():
    """Run all demonstration examples."""
    await demo_same_task_three_ways()
    await demo_framework_integration()
    
    print("\n\n🎉 HYBRID ARCHITECTURE BENEFITS")
    print("=" * 60)
    print("✅ ONE TOOL, MULTIPLE ECOSYSTEMS")
    print("✅ ZERO BREAKING CHANGES")
    print("✅ MAXIMUM INTEROPERABILITY") 
    print("✅ PRODUCTION READY")
    print("✅ FUTURE PROOF")


if __name__ == "__main__":
    asyncio.run(main())