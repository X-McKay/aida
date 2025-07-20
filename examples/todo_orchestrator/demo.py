#!/usr/bin/env python3
"""
Simple TODO orchestrator demo.

This example shows how to use the TODO orchestrator with the new simplified LLM system.
It automatically uses Ollama with fallback to cloud providers.

Prerequisites:
    1. Install PydanticAI: pip install pydantic-ai
    2. For local models: Install and start Ollama + pull models
    3. For cloud fallback: Set API keys in environment

Usage:
    python demo.py run --request "Your custom request"
    python demo.py interactive

Environment Variables (optional):
    OPENAI_API_KEY - For OpenAI fallback
    ANTHROPIC_API_KEY - For Anthropic fallback
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock
from typing import Optional

import typer

# Add AIDA to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aida.core.orchestrator import TodoOrchestrator, TodoPlan, TodoStep, TodoStatus, ReplanReason
from aida.tools.base import ToolResult, ToolCapability, ToolParameter, ToolStatus
# Using new simplified LLM system - no manual LLM setup needed!
from datetime import datetime, timezone
import os


class MockTool:
    """Mock tool for demo purposes."""
    
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
    """Mock tool registry for demo purposes."""
    
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


class DemoOrchestrator(TodoOrchestrator):
    """Demo orchestrator with simplified LLM system and mocked tools."""
    
    def __init__(self):
        # Initialize with new simplified LLM system - just override tool registry
        super().__init__()
        self.tool_registry = MockToolRegistry()
        self._tools_initialized = True
        print("✅ Using new simplified LLM system with automatic provider selection")


async def run_demo(request: str, show_progress: bool = True):
    """Run a complete demo workflow."""
    print(f"🚀 TODO Orchestrator Demo")
    print(f"📝 Request: {request}")
    print("="*60)
    
    # Create orchestrator
    orchestrator = DemoOrchestrator()
    
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


async def interactive_demo():
    """Interactive demo allowing user input."""
    print("🎮 Interactive TODO Orchestrator Demo")
    print("="*60)
    print("Enter requests to see how the orchestrator handles them.")
    print("Type 'quit' to exit.\n")
    
    orchestrator = DemoOrchestrator()
    
    while True:
        try:
            request = input("📝 Enter your request: ").strip()
            
            if request.lower() in ['quit', 'exit', 'q']:
                break
            
            if not request:
                continue
            
            print(f"\n🚀 Processing: {request}")
            await run_demo(request, show_progress=False)
            print("\n" + "-"*40 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 EOF reached, exiting!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


app = typer.Typer(help="TODO Orchestrator Demo")


@app.command()
def run(
    request: str = typer.Option(
        "Create a Python script that calculates fibonacci numbers",
        "--request", "-r",
        help="Request to process"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output (no progress updates)")
):
    """Run a demo workflow with the TODO orchestrator."""
    asyncio.run(run_demo(request, show_progress=not quiet))


@app.command()
def interactive():
    """Run interactive demo allowing user input."""
    asyncio.run(interactive_demo())


if __name__ == "__main__":
    app()