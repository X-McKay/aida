#!/usr/bin/env python3
"""
Standalone example demonstrating the TODO-based orchestrator.

This example shows how to use the TODO orchestrator as a standalone component
without the full AIDA CLI infrastructure.

Usage:
    python standalone_example.py
    python standalone_example.py --request "Your custom request"
    python standalone_example.py --test-mode
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from typing import Optional

import typer

# Add AIDA to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aida.core.orchestrator import TodoOrchestrator, TodoPlan, TodoStep, TodoStatus, ReplanReason
from aida.tools.base import ToolResult, ToolCapability, ToolParameter, ToolStatus
from datetime import datetime, timezone


class MockTool:
    """Mock tool for standalone testing."""
    
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
        await asyncio.sleep(0.5)  # Simulate work
        
        if self.name == "thinking":
            result = {
                "analysis": f"Analyzed: {kwargs.get('problem', 'unknown problem')}",
                "reasoning": "Systematic analysis completed",
                "recommendations": ["Step 1", "Step 2", "Step 3"]
            }
        elif self.name == "execution":
            code = kwargs.get('code', 'print("Hello World")')
            result = {
                "output": f"Executed: {code}",
                "exit_code": 0,
                "stdout": "Hello World\n",
                "stderr": ""
            }
        elif self.name == "file_operations":
            operation = kwargs.get('operation', 'read')
            result = {
                "operation": operation,
                "result": f"File operation '{operation}' completed successfully"
            }
        else:
            result = {"message": f"Mock execution of {self.name} tool"}
        
        return ToolResult(
            tool_name=self.name,
            execution_id=f"mock_{self.name}_{hash(str(kwargs))}"[:16],
            status=ToolStatus.COMPLETED,
            result=result,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            duration_seconds=0.5
        )


class MockToolRegistry:
    """Mock tool registry for standalone testing."""
    
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
    """Mock LLM manager for standalone testing."""
    
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
```''',
            "fibonacci": '''```json
{
    "analysis": "Need to create a Python script that calculates Fibonacci numbers efficiently",
    "expected_outcome": "Working Python script with Fibonacci function and example usage",
    "execution_plan": [
        {
            "description": "Analyze Fibonacci algorithm requirements",
            "tool": "thinking",
            "parameters": {
                "problem": "Design efficient Fibonacci calculation algorithm",
                "reasoning_type": "algorithmic_analysis"
            }
        },
        {
            "description": "Write Python Fibonacci function",
            "tool": "execution",
            "parameters": {
                "code": "def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    return fibonacci(n-1) + fibonacci(n-2)\\n\\n# Test the function\\nfor i in range(10):\\n    print(f'fib({i}) = {fibonacci(i)}')",
                "language": "python"
            },
            "dependencies": ["step_000"]
        },
        {
            "description": "Create optimized version with memoization",
            "tool": "execution",
            "parameters": {
                "code": "def fibonacci_memo(n, memo={}):\\n    if n in memo:\\n        return memo[n]\\n    if n <= 1:\\n        return n\\n    memo[n] = fibonacci_memo(n-1, memo) + fibonacci_memo(n-2, memo)\\n    return memo[n]\\n\\n# Test optimized version\\nimport time\\nstart = time.time()\\nresult = fibonacci_memo(30)\\nend = time.time()\\nprint(f'fibonacci_memo(30) = {result} (took {end-start:.4f}s)')",
                "language": "python"
            },
            "dependencies": ["step_001"]
        }
    ]
}
```''',
            "sorting": '''```json
{
    "analysis": "Need to analyze and compare different sorting algorithms for performance characteristics",
    "expected_outcome": "Comprehensive analysis of sorting algorithms with performance benchmarks",
    "execution_plan": [
        {
            "description": "Research sorting algorithms and their complexities",
            "tool": "thinking",
            "parameters": {
                "problem": "Compare bubble sort, merge sort, quick sort, and heap sort algorithms",
                "reasoning_type": "comparative_analysis"
            }
        },
        {
            "description": "Implement sorting algorithms in Python",
            "tool": "execution",
            "parameters": {
                "code": "import time\\nimport random\\n\\ndef bubble_sort(arr):\\n    n = len(arr)\\n    for i in range(n):\\n        for j in range(0, n-i-1):\\n            if arr[j] > arr[j+1]:\\n                arr[j], arr[j+1] = arr[j+1], arr[j]\\n    return arr\\n\\ndef merge_sort(arr):\\n    if len(arr) <= 1:\\n        return arr\\n    mid = len(arr) // 2\\n    left = merge_sort(arr[:mid])\\n    right = merge_sort(arr[mid:])\\n    return merge(left, right)\\n\\ndef merge(left, right):\\n    result = []\\n    i = j = 0\\n    while i < len(left) and j < len(right):\\n        if left[i] <= right[j]:\\n            result.append(left[i])\\n            i += 1\\n        else:\\n            result.append(right[j])\\n            j += 1\\n    result.extend(left[i:])\\n    result.extend(right[j:])\\n    return result\\n\\nprint('Sorting algorithms implemented successfully')",
                "language": "python"
            },
            "dependencies": ["step_000"]
        },
        {
            "description": "Benchmark sorting algorithms with different data sizes",
            "tool": "execution",
            "parameters": {
                "code": "# Performance benchmarking\\ndata_sizes = [100, 1000, 5000]\\nresults = {}\\n\\nfor size in data_sizes:\\n    print(f'\\nTesting with {size} elements:')\\n    test_data = [random.randint(1, 1000) for _ in range(size)]\\n    \\n    # Bubble sort\\n    start = time.time()\\n    bubble_sort(test_data.copy())\\n    bubble_time = time.time() - start\\n    \\n    # Merge sort\\n    start = time.time()\\n    merge_sort(test_data.copy())\\n    merge_time = time.time() - start\\n    \\n    print(f'  Bubble sort: {bubble_time:.4f}s')\\n    print(f'  Merge sort: {merge_time:.4f}s')\\n    \\n    results[size] = {'bubble': bubble_time, 'merge': merge_time}\\n\\nprint('\\nBenchmarking completed:', results)",
                "language": "python"
            },
            "dependencies": ["step_001"]
        }
    ]
}
```'''
        }
        
    async def chat_completion(self, messages, **kwargs):
        """Mock chat completion with predefined responses."""
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
        elif "fibonacci" in user_request:
            response_content = self.responses["fibonacci"]
        elif "sort" in user_request:
            response_content = self.responses["sorting"]
        else:
            response_content = self.responses["fibonacci"]  # Default
        
        mock_response = Mock()
        mock_response.content = response_content
        return mock_response


class StandaloneOrchestrator(TodoOrchestrator):
    """Standalone orchestrator with mocked dependencies."""
    
    def __init__(self):
        # Don't call super().__init__() to avoid real dependencies
        self.tool_registry = MockToolRegistry()
        self.llm_manager = MockLLMManager()
        self.active_plans = {}
        self._tools_initialized = True
        self._step_counter = 0
        
        # Create a mock storage manager that doesn't actually save files
        from unittest.mock import Mock
        self.storage_manager = Mock()
        self.storage_manager.save_plan = Mock(return_value="mock_path")


async def run_example(request: str, show_progress: bool = True):
    """Run a complete example workflow."""
    print(f"🚀 Starting standalone TODO orchestrator example")
    print(f"📝 Request: {request}")
    print("="*60)
    
    # Create orchestrator
    orchestrator = StandaloneOrchestrator()
    
    try:
        # Create plan
        print("\n📋 Creating workflow plan...")
        plan = await orchestrator.create_plan(request)
        
        print(f"✅ Plan created with {len(plan.steps)} steps")
        print("\n" + plan.to_markdown())
        print("\n" + "="*60)
        
        # Progress tracking callbacks
        def progress_callback(plan: TodoPlan, step: TodoStep):
            if show_progress:
                print(f"\n🔄 Executing: {step.description}")
        
        def replan_callback(plan: TodoPlan, reason: ReplanReason) -> bool:
            print(f"\n🔄 Replanning needed: {reason.value}")
            return True  # Auto-approve for demo
        
        # Execute plan
        print("🎯 Executing workflow...")
        result = await orchestrator.execute_plan(
            plan,
            progress_callback=progress_callback,
            replan_callback=replan_callback
        )
        
        # Show results
        print("\n" + "="*60)
        print("📊 EXECUTION RESULTS")
        print("="*60)
        
        print(result["final_markdown"])
        
        # Summary
        progress = plan.get_progress()
        print(f"\n✅ Execution completed!")
        print(f"📈 Final Status: {result['status']}")
        print(f"📊 Steps Completed: {progress['completed']}/{progress['total']}")
        print(f"🔄 Plan Version: {plan.plan_version}")
        
        if plan.replan_history:
            print(f"🔄 Replanning Events: {len(plan.replan_history)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


async def test_orchestrator_features():
    """Test various orchestrator features."""
    print("🧪 Testing TODO Orchestrator Features")
    print("="*60)
    
    orchestrator = StandaloneOrchestrator()
    
    # Test 1: Basic plan creation
    print("\n1️⃣ Testing plan creation...")
    plan = await orchestrator.create_plan("Test basic functionality")
    assert len(plan.steps) > 0
    assert plan.user_request == "Test basic functionality"
    print(f"✅ Created plan with {len(plan.steps)} steps")
    
    # Test 2: Step execution
    print("\n2️⃣ Testing step execution...")
    if plan.steps:
        step = plan.steps[0]
        result = await orchestrator._execute_step(step, plan)
        assert result["success"] == True
        assert step.status == TodoStatus.COMPLETED
        print(f"✅ Step executed successfully: {step.description}")
    
    # Test 3: Progress tracking
    print("\n3️⃣ Testing progress tracking...")
    progress = plan.get_progress()
    print(f"✅ Progress: {progress['completed']}/{progress['total']} ({progress['percentage']:.1f}%)")
    
    # Test 4: Markdown generation
    print("\n4️⃣ Testing markdown generation...")
    markdown = plan.to_markdown()
    assert "# Workflow Plan" in markdown
    assert "TODO List" in markdown
    print("✅ Markdown generation working")
    
    # Test 5: Dependency resolution
    print("\n5️⃣ Testing dependency resolution...")
    next_step = plan.get_next_executable_step()
    if next_step:
        print(f"✅ Next executable step: {next_step.description}")
    else:
        print("✅ No more executable steps (plan complete)")
    
    print("\n🎉 All tests passed!")


async def test_plan_complexity():
    """Test that simple requests generate simpler plans than complex requests."""
    print("🧪 Testing Plan Complexity Scaling")
    print("="*60)
    
    orchestrator = StandaloneOrchestrator()
    
    # Test simple request
    print("\n1️⃣ Testing simple request complexity...")
    simple_request = "print hello world"
    simple_plan = await orchestrator.create_plan(simple_request)
    
    # Test complex request  
    print("2️⃣ Testing complex request complexity...")
    complex_request = "Build a comprehensive web application with full architecture, testing, documentation, and deployment pipeline"
    complex_plan = await orchestrator.create_plan(complex_request)
    
    # Verify complexity differences
    print(f"\n📊 Plan Complexity Comparison:")
    print(f"   Simple request steps: {len(simple_plan.steps)}")
    print(f"   Complex request steps: {len(complex_plan.steps)}")
    
    # Assertions
    assert len(simple_plan.steps) < len(complex_plan.steps), f"Simple plan should have fewer steps than complex plan"
    assert len(simple_plan.steps) <= 2, f"Simple plan should have ≤2 steps, got {len(simple_plan.steps)}"
    assert len(complex_plan.steps) >= 5, f"Complex plan should have ≥5 steps, got {len(complex_plan.steps)}"
    
    # Check step descriptions for complexity indicators
    simple_descriptions = [step.description for step in simple_plan.steps]
    complex_descriptions = [step.description for step in complex_plan.steps]
    
    # Simple plan should not have complex keywords
    simple_text = " ".join(simple_descriptions).lower()
    complex_keywords = ["architecture", "testing", "documentation", "optimization", "deployment", "pipeline"]
    simple_has_complex_keywords = any(keyword in simple_text for keyword in complex_keywords)
    
    # Complex plan should have complex keywords
    complex_text = " ".join(complex_descriptions).lower()
    complex_has_complex_keywords = any(keyword in complex_text for keyword in complex_keywords)
    
    assert not simple_has_complex_keywords, "Simple plan should not contain complex architecture keywords"
    assert complex_has_complex_keywords, "Complex plan should contain complex architecture keywords"
    
    print(f"✅ Simple plan appropriate: {len(simple_plan.steps)} steps, basic execution")
    print(f"✅ Complex plan appropriate: {len(complex_plan.steps)} steps, comprehensive workflow")
    print(f"✅ Complexity scaling verified")
    
    # Test execution time differences (complex should take longer)
    print(f"\n3️⃣ Testing execution complexity...")
    
    import time
    
    start = time.time()
    simple_result = await orchestrator.execute_plan(simple_plan)
    simple_duration = time.time() - start
    
    start = time.time()  
    complex_result = await orchestrator.execute_plan(complex_plan)
    complex_duration = time.time() - start
    
    print(f"   Simple execution time: {simple_duration:.2f}s")
    print(f"   Complex execution time: {complex_duration:.2f}s")
    
    # Complex should generally take longer (though with mocking, difference may be small)
    assert complex_duration >= simple_duration * 0.8, "Complex plan should take similar or longer time"
    
    print(f"✅ Execution time scaling appropriate")
    print("\n🎉 Plan complexity tests passed!")


async def interactive_demo():
    """Interactive demo allowing user input."""
    print("🎮 Interactive TODO Orchestrator Demo")
    print("="*60)
    print("Enter requests to see how the orchestrator handles them.")
    print("Type 'quit' to exit.\n")
    
    orchestrator = StandaloneOrchestrator()
    
    while True:
        try:
            request = input("📝 Enter your request: ").strip()
            
            if request.lower() in ['quit', 'exit', 'q']:
                break
            
            if not request:
                continue
            
            print(f"\n🚀 Processing: {request}")
            await run_example(request, show_progress=False)
            print("\n" + "-"*40 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 EOF reached, exiting!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


app = typer.Typer(help="Standalone TODO Orchestrator Example")


@app.command()
def run(
    request: str = typer.Option(
        "Create a Python script that calculates fibonacci numbers",
        "--request", "-r",
        help="Request to process"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output (no progress updates)")
):
    """Run an example workflow with the TODO orchestrator."""
    asyncio.run(run_example(request, show_progress=not quiet))


@app.command()
def test():
    """Run feature tests for the orchestrator."""
    asyncio.run(test_orchestrator_features())


@app.command()
def test_complexity():
    """Test plan complexity scaling between simple and complex requests."""
    asyncio.run(test_plan_complexity())


@app.command()
def interactive():
    """Run interactive demo allowing user input."""
    asyncio.run(interactive_demo())


if __name__ == "__main__":
    app()