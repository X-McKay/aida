"""Test runner for chat CLI integration tests."""

import asyncio
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import tempfile
from pathlib import Path

from aida.core.orchestrator import get_orchestrator
from aida.tools.base import get_tool_registry, initialize_default_tools
from aida.llm import get_llm
from aida.config.llm_defaults import auto_configure_llm_providers


class ChatTestRunner:
    """Runs integration tests for chat functionality without hardcoding responses."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.orchestrator = None
        self.test_results = []
        self.test_dir = None
        
    async def setup(self):
        """Setup test environment."""
        # Create temporary test directory
        self.test_dir = tempfile.mkdtemp(prefix="aida_chat_test_")
        os.chdir(self.test_dir)
        
        # Initialize LLM providers
        try:
            await auto_configure_llm_providers()
            if self.verbose:
                print("✅ LLM providers configured")
        except Exception as e:
            print(f"⚠️  LLM setup warning: {e}")
        
        # Initialize tools
        try:
            await initialize_default_tools()
            if self.verbose:
                print("✅ Tools initialized")
        except Exception as e:
            print(f"⚠️  Tool initialization warning: {e}")
        
        # Get orchestrator
        self.orchestrator = get_orchestrator()
        
    async def cleanup(self):
        """Cleanup test environment."""
        if self.test_dir and os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)
    
    async def run_question(self, question: str, expected_tools: List[str] = None) -> Dict[str, Any]:
        """Run a single question through the chat system."""
        start_time = datetime.utcnow()
        
        try:
            # Execute through orchestrator
            result = await self.orchestrator.execute_request(
                question,
                context={"test_mode": True}
            )
            
            # Analyze tool usage
            tools_used = []
            if result.get("status") == "completed" and result.get("results"):
                for step_result in result["results"]:
                    if step_result.get("success"):
                        tool_name = step_result.get("step", {}).get("tool_name")
                        if tool_name and tool_name not in tools_used:
                            tools_used.append(tool_name)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            test_result = {
                "question": question,
                "success": result.get("status") == "completed",
                "tools_used": tools_used,
                "execution_time": execution_time,
                "error": result.get("error") if result.get("status") != "completed" else None
            }
            
            # Verify expected tools if provided
            if expected_tools is not None:
                expected_set = set(expected_tools)
                used_set = set(tools_used)
                test_result["tools_match"] = expected_set == used_set
                test_result["expected_tools"] = expected_tools
            
            return test_result
            
        except Exception as e:
            return {
                "question": question,
                "success": False,
                "error": str(e),
                "execution_time": (datetime.utcnow() - start_time).total_seconds()
            }
    
    async def run_category_tests(self, category: str, questions: List[str], expected_tool_count: int = None) -> Dict[str, Any]:
        """Run tests for a specific category."""
        print(f"\n🧪 Testing {category} questions...")
        
        results = []
        success_count = 0
        
        for i, question in enumerate(questions):
            if self.verbose:
                print(f"\n  [{i+1}/{len(questions)}] {question[:50]}...")
            
            result = await self.run_question(question)
            results.append(result)
            
            if result["success"]:
                success_count += 1
                
                # Verify tool count if specified
                if expected_tool_count is not None:
                    actual_count = len(result["tools_used"])
                    if actual_count == expected_tool_count:
                        if self.verbose:
                            print(f"    ✅ Success (used {actual_count} tools: {', '.join(result['tools_used'])})")
                    else:
                        if self.verbose:
                            print(f"    ⚠️  Tool count mismatch: expected {expected_tool_count}, got {actual_count}")
                else:
                    if self.verbose:
                        print(f"    ✅ Success (used: {', '.join(result['tools_used']) or 'no tools'})")
            else:
                if self.verbose:
                    print(f"    ❌ Failed: {result['error']}")
        
        category_result = {
            "category": category,
            "total_questions": len(questions),
            "successful": success_count,
            "success_rate": (success_count / len(questions)) * 100 if questions else 0,
            "average_time": sum(r["execution_time"] for r in results) / len(results) if results else 0,
            "results": results
        }
        
        return category_result
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        print("🚀 Starting Chat CLI Integration Tests")
        print("=" * 50)
        
        await self.setup()
        
        # Import test questions
        from aida.tests.integration.test_chat_cli import ChatCLITestSuite
        test_suite = ChatCLITestSuite()
        
        # Run tests by category
        test_categories = [
            ("No Tools Required", test_suite.no_tools_questions[:5], 0),  # Test subset
            ("Single Tool Required", test_suite.single_tool_questions[:5], 1),
            ("Multi Tool Required", test_suite.multi_tool_questions[:5], None),  # Don't enforce count
        ]
        
        all_results = []
        
        for category_name, questions, expected_tools in test_categories:
            category_result = await self.run_category_tests(
                category_name,
                questions,
                expected_tools
            )
            all_results.append(category_result)
            self.test_results.extend(category_result["results"])
        
        # Summary
        total_tests = sum(r["total_questions"] for r in all_results)
        total_success = sum(r["successful"] for r in all_results)
        overall_success_rate = (total_success / total_tests) * 100 if total_tests > 0 else 0
        
        summary = {
            "test_run": datetime.utcnow().isoformat(),
            "total_tests": total_tests,
            "successful": total_success,
            "failed": total_tests - total_success,
            "success_rate": overall_success_rate,
            "categories": all_results
        }
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 Test Summary")
        print("=" * 50)
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {total_success} ({overall_success_rate:.1f}%)")
        print(f"Failed: {total_tests - total_success}")
        
        print("\nBy Category:")
        for cat_result in all_results:
            print(f"  • {cat_result['category']}: {cat_result['successful']}/{cat_result['total_questions']} "
                  f"({cat_result['success_rate']:.1f}%) - Avg time: {cat_result['average_time']:.2f}s")
        
        # Tool usage statistics
        tool_usage = {}
        for test_result in self.test_results:
            if test_result.get("success"):
                for tool in test_result.get("tools_used", []):
                    tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        print("\nTool Usage:")
        for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {tool}: {count} times")
        
        await self.cleanup()
        
        return summary


async def main():
    """Run the chat integration tests."""
    runner = ChatTestRunner(verbose=True)
    
    try:
        summary = await runner.run_all_tests()
        
        # Save results
        with open("chat_test_results.json", "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n✅ Results saved to: chat_test_results.json")
        
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())