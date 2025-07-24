"""Tests for simple modules to improve coverage."""

import pytest


# Test providers init
def test_providers_init():
    """Test providers package imports."""
    from aida import providers

    assert hasattr(providers, "MCPProvider")


# Test tools init
def test_tools_init():
    """Test tools package imports."""
    from aida import tools

    # Check that tools are registered
    assert hasattr(tools, "get_tool_registry")
    assert hasattr(tools, "initialize_default_tools")


# Test core init
def test_core_init():
    """Test core package imports."""
    from aida import core

    assert hasattr(core, "A2AProtocol")
    assert hasattr(core, "MCPProtocol")
    assert hasattr(core, "Event")
    assert hasattr(core, "EventBus")


# Test llm init
def test_llm_init():
    """Test llm package imports."""
    from aida import llm

    # Already tested in test_llm_manager.py but add simple checks
    assert hasattr(llm, "get_llm")
    assert hasattr(llm, "chat")


# Test execution tool init
def test_execution_tool_init():
    """Test execution tool package imports."""
    from aida.tools import execution

    assert hasattr(execution, "ExecutionTool")


# Test context tool init
def test_context_tool_init():
    """Test context tool package imports."""
    from aida.tools import context

    assert hasattr(context, "ContextTool")


# Test files tool init
def test_files_tool_init():
    """Test files tool package imports."""
    from aida.tools import files

    assert hasattr(files, "FileOperationsTool")


# Test system tool init
def test_system_tool_init():
    """Test system tool package imports."""
    from aida.tools import system

    assert hasattr(system, "SystemTool")


# Test thinking tool init
def test_thinking_tool_init():
    """Test thinking tool package imports."""
    from aida.tools import thinking

    assert hasattr(thinking, "ThinkingTool")


# Test llm_response tool init
def test_llm_response_tool_init():
    """Test llm_response tool package imports."""
    from aida.tools import llm_response

    assert hasattr(llm_response, "LLMResponseTool")


# Test orchestrator init
def test_orchestrator_init():
    """Test orchestrator package imports."""
    from aida.core import orchestrator

    assert hasattr(orchestrator, "get_orchestrator")
    assert hasattr(orchestrator, "TodoOrchestrator")


# Test protocol init
def test_protocols_init():
    """Test protocols package imports."""
    from aida.core import protocols

    assert protocols.__name__ == "aida.core.protocols"


# Test providers mcp init
def test_providers_mcp_init():
    """Test providers mcp package."""
    from aida.providers import mcp

    assert mcp.__name__ == "aida.providers.mcp"


# Test providers llm init
def test_providers_llm_init():
    """Test providers llm package."""
    from aida.providers import llm

    assert llm.__name__ == "aida.providers.llm"


if __name__ == "__main__":
    pytest.main([__file__])
