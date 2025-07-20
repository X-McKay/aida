"""Base test framework for AIDA integration tests."""

import asyncio
import time
import traceback
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class TestResult:
    """Test result with details."""
    name: str
    success: bool
    message: str
    duration: float
    details: Dict[str, Any]


class BaseTestSuite:
    """Base class for test suites."""
    
    def __init__(self, name: str, verbose: bool = False, persist_files: bool = False):
        self.name = name
        self.verbose = verbose
        self.persist_files = persist_files
        self.results: List[TestResult] = []
    
    def log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(f"  {message}")
    
    async def run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and capture results."""
        print(f"🧪 {test_name}")
        start_time = time.time()
        
        try:
            result = await test_func()
            duration = time.time() - start_time
            
            if result.get("success", True):
                print(f"✅ {test_name} - {duration:.2f}s")
                return TestResult(
                    name=test_name,
                    success=True, 
                    message=result.get("message", "Success"),
                    duration=duration,
                    details=result
                )
            else:
                print(f"❌ {test_name} - {result.get('message', 'Failed')}")
                return TestResult(
                    name=test_name,
                    success=False,
                    message=result.get("message", "Failed"),
                    duration=duration,
                    details=result
                )
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Exception: {str(e)}"
            print(f"❌ {test_name} - {error_msg}")
            if self.verbose:
                print(f"  Traceback: {traceback.format_exc()}")
            return TestResult(
                name=test_name,
                success=False,
                message=error_msg,
                duration=duration,
                details={"exception": str(e)}
            )
    
    async def run_all(self) -> List[TestResult]:
        """Run all tests in this suite. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement run_all()")
    
    def get_test_methods(self) -> List[str]:
        """Get all test method names."""
        return [method for method in dir(self) if method.startswith('test_')]
    
    def print_results(self):
        """Print test results summary."""
        if not self.results:
            return
            
        print(f"\n📊 {self.name} Results:")
        print("-" * 40)
        
        passed = len([r for r in self.results if r.success])
        total = len(self.results)
        
        print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
        print(f"Total time: {sum(r.duration for r in self.results):.2f}s")
        
        failed_tests = [r for r in self.results if not r.success]
        if failed_tests:
            print("Failed tests:")
            for result in failed_tests:
                print(f"  - {result.name}: {result.message}")


class TestRegistry:
    """Registry for test suites."""
    
    def __init__(self):
        self._suites: Dict[str, BaseTestSuite] = {}
    
    def register(self, name: str, suite_class: type):
        """Register a test suite."""
        self._suites[name] = suite_class
    
    def get_suite(self, name: str, verbose: bool = False, persist_files: bool = False) -> Optional[BaseTestSuite]:
        """Get a test suite instance."""
        if name in self._suites:
            return self._suites[name](verbose=verbose, persist_files=persist_files)
        return None
    
    def list_suites(self) -> List[str]:
        """List available test suites."""
        return list(self._suites.keys())
    
    async def run_suite(self, name: str, verbose: bool = False, persist_files: bool = False) -> List[TestResult]:
        """Run a specific test suite."""
        suite = self.get_suite(name, verbose, persist_files=persist_files)
        if not suite:
            raise ValueError(f"Unknown test suite: {name}")
        
        print(f"\n🚀 Running {suite.name} Tests")
        print("=" * 50)
        
        results = await suite.run_all()
        suite.print_results()
        return results
    
    async def run_all_suites(self, verbose: bool = False, persist_files: bool = False) -> Dict[str, List[TestResult]]:
        """Run all registered test suites."""
        all_results = {}
        
        for suite_name in self._suites.keys():
            results = await self.run_suite(suite_name, verbose, persist_files=persist_files)
            all_results[suite_name] = results
        
        return all_results


# Global test registry
test_registry = TestRegistry()