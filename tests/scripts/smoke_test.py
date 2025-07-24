#!/usr/bin/env python
"""Smoke test - Run this before EVERY commit to catch breaking changes.

Simple, fast checks that ensure core functionality works.
No complex setup, no dependencies, just verification.
"""

import asyncio
import subprocess
import sys


def check_imports():
    """Verify all critical imports work."""
    print("1. Checking imports...", end=" ")
    critical_imports = [
        "from aida.cli.main import app",
        "from aida.cli.commands.chat import ChatSession",
        "from aida.core.orchestrator import get_orchestrator",
        "from aida.llm import get_llm, chat",
        "from aida.tools.base import get_tool_registry",
        "from aida.config.llm_profiles import Purpose",
        "from aida.agents.coordination.coordinator_agent import CoordinatorAgent",
        "from aida.agents.worker.coding_worker import CodingWorker",
        "from aida.core.protocols.a2a import A2AProtocol",
    ]

    for imp in critical_imports:
        try:
            exec(imp)
        except ImportError as e:
            print(f"\n   ❌ BROKEN: {imp}")
            print(f"   Error: {e}")
            return False

    print("✅")
    return True


async def check_llm_system():
    """Verify LLM system initializes."""
    print("2. Checking LLM system...", end=" ")
    try:
        from aida.llm import get_llm

        manager = get_llm()
        purposes = manager.list_purposes()
        if len(purposes) < 1:
            print("\n   ❌ No LLM purposes available")
            return False
        print("✅")
        return True
    except Exception as e:
        print(f"\n   ❌ LLM failed: {e}")
        return False


async def check_coordinator():
    """Verify coordinator works."""
    print("3. Checking coordinator...", end=" ")
    try:
        from aida.agents.coordination.coordinator_agent import CoordinatorAgent

        # Create a test coordinator (it has default config)
        coordinator = CoordinatorAgent()

        # Check critical methods exist
        if not hasattr(coordinator, "execute_plan"):
            print("\n   ❌ Missing execute_plan method")
            return False
        if not hasattr(coordinator, "handle_message"):
            print("\n   ❌ Missing handle_message method")
            return False

        print("✅")
        return True
    except Exception as e:
        print(f"\n   ❌ Coordinator failed: {e}")
        return False


async def check_chat_session():
    """Verify chat can be created."""
    print("4. Checking chat session...", end=" ")
    try:
        from aida.cli.commands.chat import ChatSession

        session = ChatSession()

        # Don't actually start it, just verify it creates
        if not hasattr(session, "_handle_chat_message"):
            print("\n   ❌ Missing _handle_chat_message method")
            return False

        print("✅")
        return True
    except Exception as e:
        print(f"\n   ❌ Chat session failed: {e}")
        return False


def check_cli_commands():
    """Verify CLI commands are accessible."""
    print("5. Checking CLI commands...", end=" ")

    # Test help for key commands
    commands = ["chat", "test", "tools"]
    for cmd in commands:
        result = subprocess.run(
            ["uv", "run", "python", "-m", "aida.cli.main", cmd, "--help"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"\n   ❌ Command '{cmd}' broken")
            return False

    print("✅")
    return True


async def run_smoke_tests():
    """Run all smoke tests."""
    print("=" * 50)
    print("AIDA SMOKE TESTS - Run before EVERY commit")
    print("=" * 50)

    all_passed = True

    # Synchronous checks
    all_passed &= check_imports()
    all_passed &= check_cli_commands()

    # Async checks
    all_passed &= await check_llm_system()
    all_passed &= await check_coordinator()
    all_passed &= await check_chat_session()

    print("=" * 50)

    if all_passed:
        print("✅ ALL TESTS PASSED - Safe to commit")
        return 0
    else:
        print("❌ TESTS FAILED - DO NOT COMMIT")
        print("\nFix the issues above before committing!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_smoke_tests())
    sys.exit(exit_code)
