#!/usr/bin/env python3
"""Test script for MCP filesystem integration."""

import asyncio
import os
from pathlib import Path
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aida.tools.files import FileOperation, FileOperationsTool


async def test_mcp_filesystem():
    """Test MCP filesystem operations."""
    print("🔧 Testing MCP Filesystem Integration")
    print("=" * 50)

    # Enable MCP backend
    os.environ["AIDA_FILES_USE_MCP"] = "true"

    # Create temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Test directory: {temp_dir}")

        # Initialize tool with test directory
        tool = FileOperationsTool(allowed_directories=[temp_dir])

        try:
            # Test 1: Write file
            print("\n📝 Test 1: Writing file...")
            test_file = os.path.join(temp_dir, "test.txt")
            write_result = await tool.execute(
                operation=FileOperation.WRITE.value,
                path=test_file,
                content="Hello from MCP filesystem!",
            )

            if write_result.status.value == "completed":
                print("✅ Write successful")
            else:
                print(f"❌ Write failed: {write_result.error}")
                return False

            # Test 2: Read file
            print("\n📖 Test 2: Reading file...")
            read_result = await tool.execute(operation=FileOperation.READ.value, path=test_file)

            if read_result.status.value == "completed":
                content = read_result.result.get("content", "")
                print(f"✅ Read successful: '{content}'")
                if content != "Hello from MCP filesystem!":
                    print("❌ Content mismatch!")
                    return False
            else:
                print(f"❌ Read failed: {read_result.error}")
                return False

            # Test 3: Create directory
            print("\n📂 Test 3: Creating directory...")
            test_dir = os.path.join(temp_dir, "subdir")
            mkdir_result = await tool.execute(
                operation=FileOperation.CREATE_DIR.value, path=test_dir
            )

            if mkdir_result.status.value == "completed":
                print("✅ Directory created")
            else:
                print(f"❌ Directory creation failed: {mkdir_result.error}")
                return False

            # Test 4: List directory
            print("\n📋 Test 4: Listing directory...")
            list_result = await tool.execute(operation=FileOperation.LIST_DIR.value, path=temp_dir)

            if list_result.status.value == "completed":
                entries = list_result.result
                print(f"✅ Directory listing: {len(entries)} entries")
                for entry in entries:
                    print(f"  - {entry.get('name')} ({'dir' if entry.get('is_dir') else 'file'})")
            else:
                print(f"❌ List failed: {list_result.error}")
                return False

            # Test 5: Move file
            print("\n🚚 Test 5: Moving file...")
            new_path = os.path.join(test_dir, "moved.txt")
            move_result = await tool.execute(
                operation=FileOperation.MOVE.value, path=test_file, destination=new_path
            )

            if move_result.status.value == "completed":
                print(f"✅ File moved to {new_path}")
            else:
                print(f"❌ Move failed: {move_result.error}")
                return False

            # Test 6: Search in files
            print("\n🔍 Test 6: Searching in files...")
            search_result = await tool.execute(
                operation=FileOperation.SEARCH.value,
                path=temp_dir,
                search_text="MCP",
                recursive=True,
            )

            if search_result.status.value == "completed":
                results = search_result.result.get("results", [])
                print(f"✅ Search found {len(results)} files")
                for result in results:
                    print(f"  - {result['file']}: {result['matches']} matches")
            else:
                print(f"❌ Search failed: {search_result.error}")
                return False

            # Test 7: Delete file
            print("\n🗑️  Test 7: Deleting file...")
            delete_result = await tool.execute(operation=FileOperation.DELETE.value, path=new_path)

            if delete_result.status.value == "completed":
                print("✅ File deleted")
            else:
                print(f"❌ Delete failed: {delete_result.error}")
                return False

            print("\n🎉 All tests passed!")
            return True

        except Exception as e:
            print(f"\n❌ Error during testing: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            # Cleanup
            if hasattr(tool, "cleanup"):
                await tool.cleanup()


async def test_native_filesystem():
    """Test native filesystem operations for comparison."""
    print("\n\n🔧 Testing Native Filesystem (for comparison)")
    print("=" * 50)

    # Disable MCP backend
    os.environ["AIDA_FILES_USE_MCP"] = "false"

    # Create temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Test directory: {temp_dir}")

        # Initialize tool
        tool = FileOperationsTool()

        # Test write and read
        print("\n📝 Testing native write/read...")
        test_file = os.path.join(temp_dir, "native_test.txt")

        write_result = await tool.execute(
            operation=FileOperation.WRITE.value,
            path=test_file,
            content="Hello from native filesystem!",
        )

        if write_result.status.value == "completed":
            print("✅ Native write successful")

            read_result = await tool.execute(operation=FileOperation.READ.value, path=test_file)

            if read_result.status.value == "completed":
                content = read_result.result.get("content", "")
                print(f"✅ Native read successful: '{content}'")
            else:
                print(f"❌ Native read failed: {read_result.error}")
        else:
            print(f"❌ Native write failed: {write_result.error}")


async def main():
    """Run all tests."""
    print("🚀 AIDA MCP Filesystem Integration Test")
    print("=" * 50)

    # Check if Node.js and npx are available
    import subprocess

    try:
        subprocess.run(["npx", "--version"], capture_output=True, check=True)
        print("✅ npx is available")
    except Exception:
        print("❌ npx is not available. Please install Node.js.")
        return

    # Run MCP tests
    mcp_success = await test_mcp_filesystem()

    # Run native tests for comparison
    await test_native_filesystem()

    if mcp_success:
        print("\n✨ MCP filesystem integration is working correctly!")
    else:
        print("\n❌ MCP filesystem integration has issues.")


if __name__ == "__main__":
    asyncio.run(main())
