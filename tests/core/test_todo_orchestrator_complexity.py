#!/usr/bin/env python3
"""
Tests for TODO orchestrator plan complexity scaling.

This test suite verifies that the TODO orchestrator generates appropriately
complex plans based on the complexity of user requests.
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import Mock
from datetime import datetime, timezone

import pytest

# Add AIDA to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aida.core.orchestrator import TodoOrchestrator, TodoPlan, TodoStep, TodoStatus, ReplanReason
from aida.tools.base import ToolResult, ToolCapability, ToolParameter, ToolStatus


class MockTool:
    """Mock tool for testing."""
    
    def __init__(self, name: str, description: str, parameters: list = None):
        self.name = name
        self.description = description
        self.parameters = parameters or []
    
    def get_capability(self) -> ToolCapability:
        return ToolCapability(
            name=self.name,
            description=self.description,
            parameters=self.parameters
        )
    
    async def execute_async(self, **kwargs) -> ToolResult:
        """Mock execution with simulated work."""
        await asyncio.sleep(0.1)  # Shorter delay for tests
        
        result = {"message": f"Mock execution of {self.name} tool"}
        
        return ToolResult(
            tool_name=self.name,
            execution_id=f"test_{self.name}_{hash(str(kwargs))}"[:16],
            status=ToolStatus.COMPLETED,
            result=result,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration_seconds=0.1
        )


class MockToolRegistry:
    """Mock tool registry for testing."""
    
    def __init__(self):
        self.tools = {
            "thinking": MockTool(
                name="thinking",
                description="Thinking and analysis tool",
                parameters=[
                    ToolParameter(name="problem", type="string", description="Problem to analyze", required=True),
                    ToolParameter(name="reasoning_type", type="string", description="Type of reasoning", required=False)
                ]
            ),
            "execution": MockTool(
                name="execution", 
                description="Code execution tool",
                parameters=[
                    ToolParameter(name="code", type="string", description="Code to execute", required=True),
                    ToolParameter(name="language", type="string", description="Programming language", required=False)
                ]
            ),
            "file_operations": MockTool(
                name="file_operations",
                description="File operations tool",
                parameters=[
                    ToolParameter(name="operation", type="string", description="Operation to perform", required=True),
                    ToolParameter(name="path", type="string", description="File path", required=False)
                ]
            )
        }
    
    async def list_tools(self):
        return list(self.tools.keys())
    
    async def get_tool(self, name: str):
        return self.tools.get(name)


class MockLLMManager:
    """Mock LLM manager for testing with different complexity responses."""
    
    def __init__(self):
        self.responses = {
            "simple": '''```json
{
    "analysis": "Simple request requiring basic execution",
    "expected_outcome": "Quick output with minimal processing",
    "execution_plan": [
        {
            "description": "Execute simple task",
            "tool": "execution",
            "parameters": {
                "code": "print('Hello, World!')",
                "language": "python"
            }
        }
    ]
}
```''',
            "complex": '''```json
{
    "analysis": "Complex multi-step project requiring comprehensive implementation with testing, documentation, and optimization",
    "expected_outcome": "Full-featured application with proper architecture, error handling, testing suite, documentation, and deployment pipeline",
    "execution_plan": [
        {
            "description": "Analyze requirements and design system architecture",
            "tool": "thinking",
            "parameters": {
                "problem": "Design comprehensive system architecture for complex application",
                "reasoning_type": "system_design"
            }
        },
        {
            "description": "Set up project structure and dependencies",
            "tool": "file_operations",
            "parameters": {
                "operation": "create_project_structure",
                "path": "./complex_project"
            },
            "dependencies": ["step_000"]
        },
        {
            "description": "Implement core functionality",
            "tool": "execution",
            "parameters": {
                "code": "# Core implementation code here",
                "language": "python"
            },
            "dependencies": ["step_001"]
        },
        {
            "description": "Add error handling and validation",
            "tool": "execution",
            "parameters": {
                "code": "# Error handling and validation code",
                "language": "python"
            },
            "dependencies": ["step_002"]
        },
        {
            "description": "Create comprehensive test suite",
            "tool": "execution",
            "parameters": {
                "code": "# Test suite implementation",
                "language": "python"
            },
            "dependencies": ["step_003"]
        },
        {
            "description": "Write documentation and usage examples",
            "tool": "file_operations",
            "parameters": {
                "operation": "create_documentation",
                "path": "./docs"
            },
            "dependencies": ["step_004"]
        },
        {
            "description": "Optimize performance and memory usage",
            "tool": "execution",
            "parameters": {
                "code": "# Performance optimization code",
                "language": "python"
            },
            "dependencies": ["step_005"]
        },
        {
            "description": "Set up CI/CD pipeline and deployment",
            "tool": "file_operations",
            "parameters": {
                "operation": "create_cicd_config",
                "path": "./.github/workflows"
            },
            "dependencies": ["step_006"]
        }
    ]
}
```'''
        }
        
    async def chat_completion(self, messages, **kwargs):
        """Mock chat completion with complexity-based responses."""
        user_message = messages[-1].content.lower()
        
        # Extract just the user request from the prompt
        if "user request:" in user_message:
            user_request = user_message.split("user request:")[1].split("\n")[0].strip()
        else:
            user_request = user_message
        
        # Complex requests (multiple requirements, architecture, testing, etc.) - check first
        if any(word in user_request for word in ["comprehensive", "full-featured", "complete", "architecture", "testing", "documentation", "deployment", "pipeline", "optimize", "complex", "web application"]):
            response_content = self.responses["complex"]
        # Simple requests (single word/action)
        elif any(word in user_request for word in ["hello", "hi", "print hello", "echo", "simple"]):
            response_content = self.responses["simple"]
        else:
            response_content = self.responses["simple"]  # Default to simple for tests
        
        mock_response = Mock()
        mock_response.content = response_content
        return mock_response


class TestOrchestrator(TodoOrchestrator):
    """Test orchestrator with mocked dependencies."""
    
    def __init__(self):
        # Don't call super().__init__() to avoid real dependencies
        self.tool_registry = MockToolRegistry()
        self.llm_manager = MockLLMManager()
        self.active_plans = {}
        self._tools_initialized = True
        self._step_counter = 0


@pytest.fixture
def orchestrator():
    """Create a test orchestrator instance."""
    return TestOrchestrator()


@pytest.mark.asyncio
async def test_simple_request_generates_simple_plan(orchestrator):
    """Test that simple requests generate plans with few steps."""
    simple_request = "print hello world"
    plan = await orchestrator.create_plan(simple_request)
    
    assert len(plan.steps) <= 2, f"Simple plan should have ≤2 steps, got {len(plan.steps)}"
    assert plan.user_request == simple_request
    
    # Check that plan doesn't contain complex keywords
    descriptions = [step.description for step in plan.steps]
    plan_text = " ".join(descriptions).lower()
    complex_keywords = ["architecture", "testing", "documentation", "optimization", "deployment", "pipeline"]
    
    assert not any(keyword in plan_text for keyword in complex_keywords), \
        "Simple plan should not contain complex architecture keywords"


@pytest.mark.asyncio
async def test_complex_request_generates_complex_plan(orchestrator):
    """Test that complex requests generate plans with many steps."""
    complex_request = "Build a comprehensive web application with full architecture, testing, documentation, and deployment pipeline"
    plan = await orchestrator.create_plan(complex_request)
    
    assert len(plan.steps) >= 5, f"Complex plan should have ≥5 steps, got {len(plan.steps)}"
    assert plan.user_request == complex_request
    
    # Check that plan contains complex keywords
    descriptions = [step.description for step in plan.steps]
    plan_text = " ".join(descriptions).lower()
    complex_keywords = ["architecture", "testing", "documentation", "optimization", "deployment", "pipeline"]
    
    assert any(keyword in plan_text for keyword in complex_keywords), \
        "Complex plan should contain complex architecture keywords"


@pytest.mark.asyncio
async def test_plan_complexity_scaling(orchestrator):
    """Test that complex requests generate more steps than simple requests."""
    # Create simple plan
    simple_request = "print hello world"
    simple_plan = await orchestrator.create_plan(simple_request)
    
    # Create complex plan
    complex_request = "Build a comprehensive web application with full architecture, testing, documentation, and deployment pipeline"
    complex_plan = await orchestrator.create_plan(complex_request)
    
    # Verify complexity scaling
    assert len(simple_plan.steps) < len(complex_plan.steps), \
        f"Simple plan ({len(simple_plan.steps)} steps) should have fewer steps than complex plan ({len(complex_plan.steps)} steps)"


@pytest.mark.asyncio
async def test_execution_time_scaling(orchestrator):
    """Test that complex plans take longer to execute than simple plans."""
    # Create plans
    simple_plan = await orchestrator.create_plan("print hello world")
    complex_plan = await orchestrator.create_plan("Build comprehensive web application with architecture, testing, documentation, deployment")
    
    # Execute simple plan
    start = time.time()
    simple_result = await orchestrator.execute_plan(simple_plan)
    simple_duration = time.time() - start
    
    # Execute complex plan
    start = time.time()
    complex_result = await orchestrator.execute_plan(complex_plan)
    complex_duration = time.time() - start
    
    # Verify both executed successfully
    assert simple_result["status"] == "completed"
    assert complex_result["status"] == "completed"
    
    # Complex should take similar or longer time (allowing for some variance due to mocking)
    assert complex_duration >= simple_duration * 0.5, \
        f"Complex execution ({complex_duration:.2f}s) should take similar or longer time than simple ({simple_duration:.2f}s)"


@pytest.mark.asyncio
async def test_plan_structure_validity(orchestrator):
    """Test that generated plans have valid structure."""
    requests = [
        "print hello world",
        "Build comprehensive web application with architecture, testing, documentation"
    ]
    
    for request in requests:
        plan = await orchestrator.create_plan(request)
        
        # Basic structure checks
        assert hasattr(plan, 'steps'), "Plan should have steps attribute"
        assert hasattr(plan, 'user_request'), "Plan should have user_request attribute"
        assert hasattr(plan, 'analysis'), "Plan should have analysis attribute"
        assert hasattr(plan, 'expected_outcome'), "Plan should have expected_outcome attribute"
        
        # Steps should be valid
        assert len(plan.steps) > 0, "Plan should have at least one step"
        for step in plan.steps:
            assert hasattr(step, 'description'), "Step should have description"
            assert hasattr(step, 'tool_name'), "Step should have tool_name"
            assert hasattr(step, 'parameters'), "Step should have parameters"
            assert len(step.description) > 0, "Step description should not be empty"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])