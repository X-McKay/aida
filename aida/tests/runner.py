"""AIDA integration test runner."""

import asyncio
import sys
from typing import Dict, List, Optional
from aida.tests.base import test_registry, TestResult

# Import test suites to register them
from aida.tests.integration.test_llm import LLMTestSuite
from aida.tests.integration.test_orchestrator import OrchestratorTestSuite
from aida.tests.integration.test_chat_cli import ChatCLITestSuite


class AIDATestRunner:
    """Main test runner for AIDA integration tests."""
    
    def __init__(self, verbose: bool = False, persist_files: bool = False):
        self.verbose = verbose
        self.persist_files = persist_files
        self.all_results: Dict[str, List[TestResult]] = {}
    
    async def run_specific_suite(self, suite_name: str) -> List[TestResult]:
        """Run a specific test suite."""
        if suite_name not in test_registry.list_suites():
            available = ", ".join(test_registry.list_suites())
            raise ValueError(f"Unknown test suite '{suite_name}'. Available: {available}")
        
        results = await test_registry.run_suite(suite_name, self.verbose, persist_files=self.persist_files)
        self.all_results[suite_name] = results
        return results
    
    async def run_all_suites(self) -> Dict[str, List[TestResult]]:
        """Run all registered test suites."""
        print("🔬 AIDA Integration Test Runner")
        print("=" * 60)
        print("Testing refactored components with real functionality")
        print("")
        
        results = await test_registry.run_all_suites(self.verbose, persist_files=self.persist_files)
        self.all_results = results
        return results
    
    def print_overall_summary(self):
        """Print overall test summary across all suites."""
        if not self.all_results:
            return
        
        print("\n" + "=" * 60)
        print("📊 OVERALL TEST SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        total_passed = 0
        total_duration = 0.0
        
        # Calculate totals
        for suite_name, results in self.all_results.items():
            suite_total = len(results)
            suite_passed = len([r for r in results if r.success])
            suite_duration = sum(r.duration for r in results)
            
            total_tests += suite_total
            total_passed += suite_passed
            total_duration += suite_duration
            
            success_rate = (suite_passed / suite_total * 100) if suite_total > 0 else 0
            print(f"{suite_name:20} {suite_passed:2d}/{suite_total:2d} ({success_rate:5.1f}%)")
        
        print("-" * 60)
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        print(f"{'TOTAL':20} {total_passed:2d}/{total_tests:2d} ({overall_success_rate:5.1f}%)")
        print(f"Total Duration: {total_duration:.2f}s")
        
        # Show failed tests
        failed_tests = []
        for suite_name, results in self.all_results.items():
            for result in results:
                if not result.success:
                    failed_tests.append((suite_name, result))
        
        if failed_tests:
            print(f"\n❌ Failed Tests ({len(failed_tests)}):")
            for suite_name, result in failed_tests:
                print(f"  {suite_name}/{result.name}: {result.message}")
        
        # Overall status
        if total_passed == total_tests and total_tests > 0:
            status = "✅ ALL TESTS PASSING"
        elif total_passed > 0:
            status = "⚠️ SOME TESTS FAILING"
        else:
            status = "❌ ALL TESTS FAILING"
        
        print(f"\n🎯 Refactored Components Status: {status}")
        
        return total_tests == total_passed
    
    def list_available_suites(self):
        """List all available test suites."""
        suites = test_registry.list_suites()
        print("Available test suites:")
        for suite in suites:
            print(f"  - {suite}")


async def run_tests(
    suite_name: Optional[str] = None,
    verbose: bool = False,
    persist_files: bool = False
) -> bool:
    """
    Run AIDA integration tests.
    
    Args:
        suite_name: Specific test suite to run (None for all)
        verbose: Enable verbose output
        persist_files: Keep generated test files instead of cleaning them up
    
    Returns:
        True if all tests passed, False otherwise
    """
    runner = AIDATestRunner(verbose=verbose, persist_files=persist_files)
    
    try:
        if suite_name:
            await runner.run_specific_suite(suite_name)
        else:
            await runner.run_all_suites()
        
        success = runner.print_overall_summary()
        return success
        
    except ValueError as e:
        print(f"❌ Error: {e}")
        runner.list_available_suites()
        return False
    except KeyboardInterrupt:
        print("\n\n⏹️ Tests interrupted by user")
        return False
    except Exception as e:
        print(f"\n\n💥 Test runner crashed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main entry point for standalone test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIDA Integration Test Runner")
    parser.add_argument("--suite", "-s", 
                       choices=test_registry.list_suites() + [None],
                       help="Test specific suite only")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available test suites")
    
    args = parser.parse_args()
    
    if args.list:
        runner = AIDATestRunner()
        runner.list_available_suites()
        return
    
    # Run tests
    success = asyncio.run(run_tests(args.suite, args.verbose))
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()