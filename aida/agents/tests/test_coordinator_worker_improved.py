"""Improved tests for coordinator-worker communication with proper assertions.

This test suite validates:
1. Worker registration and capability discovery
2. Task delegation and execution
3. Result accuracy and error handling
4. Plan storage and retrieval

Features:
- Automatic cleanup of test artifacts before and after test runs
- Fresh test file creation for consistent test environment
- Detailed assertions with clear error messages
- Optional cleanup control with --no-cleanup flag

Usage:
    # Run with automatic cleanup
    python test_coordinator_worker_improved.py

    # Run without cleanup (for debugging)
    python test_coordinator_worker_improved.py --no-cleanup
"""

import asyncio
import logging
import os
from pathlib import Path
import shutil
from typing import Any

from aida.agents.base import AgentConfig, WorkerConfig
from aida.agents.coordination import CoordinatorAgent
from aida.agents.coordination.plan_models import TodoStatus
from aida.agents.worker import CodingWorker
from aida.core.protocols.a2a import A2AMessage

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestResult:
    """Container for test results with assertions."""

    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.errors = []
        self.warnings = []

    def assert_true(self, condition: bool, message: str):
        """Assert a condition is true."""
        if not condition:
            self.passed = False
            self.errors.append(f"Assertion failed: {message}")
            logger.error(f"[{self.name}] [FAIL] {message}")
        else:
            logger.info(f"[{self.name}] [OK] {message}")

    def assert_equal(self, actual: Any, expected: Any, message: str):
        """Assert two values are equal."""
        if actual != expected:
            self.passed = False
            self.errors.append(f"Assertion failed: {message} (expected {expected}, got {actual})")
            logger.error(f"[{self.name}] [FAIL] {message} (expected {expected}, got {actual})")
        else:
            logger.info(f"[{self.name}] [OK] {message}")

    def assert_not_none(self, value: Any, message: str):
        """Assert a value is not None."""
        if value is None:
            self.passed = False
            self.errors.append(f"Assertion failed: {message} (value is None)")
            logger.error(f"[{self.name}] [FAIL] {message} (value is None)")
        else:
            logger.info(f"[{self.name}] [OK] {message}")

    def assert_in(self, item: Any, container: Any, message: str):
        """Assert an item is in a container."""
        if item not in container:
            self.passed = False
            self.errors.append(f"Assertion failed: {message} ({item} not in {container})")
            logger.error(f"[{self.name}] [FAIL] {message}")
        else:
            logger.info(f"[{self.name}] [OK] {message}")

    def warn(self, message: str):
        """Add a warning."""
        self.warnings.append(message)
        logger.warning(f"[{self.name}] [WARN] {message}")

    def report(self) -> bool:
        """Print test report and return pass/fail status."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Test: {self.name}")
        logger.info(f"{'=' * 60}")

        if self.passed:
            logger.info("[PASSED]")
        else:
            logger.error("[FAILED]")
            logger.error(f"Errors: {len(self.errors)}")
            for error in self.errors:
                logger.error(f"  - {error}")

        if self.warnings:
            logger.warning(f"Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        return self.passed


async def create_test_file() -> str:
    """Create a test Python file for analysis."""
    test_code = '''"""Test module for code analysis."""

class Calculator:
    """Simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    def divide(self, a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''

    # Create test directory
    test_dir = Path(".aida/test_files")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Write test file
    test_file = test_dir / "calculator.py"
    test_file.write_text(test_code)

    return str(test_file)


async def test_worker_registration(result: TestResult) -> tuple[CoordinatorAgent, CodingWorker]:
    """Test worker registration with coordinator."""
    # Create coordinator
    coordinator_config = AgentConfig(
        agent_id="test_coordinator",
        agent_type="coordinator",
        port=8100,
        capabilities=["planning", "task_delegation"],
    )
    coordinator = CoordinatorAgent(coordinator_config)

    # Start coordinator
    await coordinator.start()
    result.assert_true(coordinator._running, "Coordinator started successfully")

    # Create and start worker
    worker_config = WorkerConfig(
        agent_id="test_worker",
        agent_type="coding_worker",
        capabilities=["code_analysis", "code_generation"],
        coordinator_endpoint="ws://localhost:8100",
        # Enable filesystem MCP server for real testing
        allowed_mcp_servers=["filesystem"],
    )
    worker = CodingWorker("test_worker", worker_config)

    await worker.start()
    result.assert_true(worker._running, "Worker started successfully")

    # Wait for registration
    await asyncio.sleep(3)

    # Verify registration
    result.assert_in(
        "test_worker", coordinator._known_workers, "Worker registered with coordinator"
    )
    result.assert_in(
        "code_analysis", coordinator._capability_index, "Code analysis capability indexed"
    )
    result.assert_in(
        "code_generation", coordinator._capability_index, "Code generation capability indexed"
    )

    return coordinator, worker


async def test_code_analysis(coordinator: CoordinatorAgent, result: TestResult) -> str | None:
    """Test code analysis functionality."""
    # Create test file
    test_file = await create_test_file()
    result.assert_true(os.path.exists(test_file), f"Test file created: {test_file}")

    # Create analysis request
    analysis_request = {
        "request": f"Analyze the Python code in {test_file}",
        "context": {
            "task_type": "code_analysis",
            "file_path": test_file,
            "required_capability": "code_analysis",
        },
    }

    # Send request
    message = A2AMessage(
        sender_id="test_client",
        recipient_id=coordinator.agent_id,
        message_type=A2AMessage.MessageTypes.TASK_REQUEST,
        payload=analysis_request,
    )

    response = await coordinator.handle_message(message)
    result.assert_not_none(response, "Coordinator responded to analysis request")

    if response:
        result.assert_equal(response.payload.get("status"), "accepted", "Analysis request accepted")
        plan_id = response.payload.get("plan_id")
        result.assert_not_none(plan_id, "Plan ID returned")

        # Wait for execution
        await asyncio.sleep(5)

        # Check plan completion
        plan = coordinator.load_plan(plan_id)
        result.assert_not_none(plan, "Plan loaded successfully")

        if plan:
            progress = plan.get_progress()
            result.assert_equal(plan.status, "completed", "Plan completed successfully")
            result.assert_equal(progress["failed"], 0, "No failed steps")

            # Validate results
            if plan.steps:
                step = plan.steps[0]
                result.assert_equal(step.status, TodoStatus.COMPLETED, "Step completed")
                result.assert_not_none(step.result, "Step has results")

                if step.result:
                    # Validate analysis results
                    result.assert_in("language", step.result, "Language detected")
                    result.assert_equal(
                        step.result.get("language"), "python", "Correct language detected"
                    )

                    result.assert_in("metrics", step.result, "Code metrics calculated")
                    metrics = step.result.get("metrics", {})
                    result.assert_true(metrics.get("total_lines", 0) > 0, "Line count calculated")

                    result.assert_in("structure", step.result, "Code structure analyzed")
                    structure = step.result.get("structure", {})
                    result.assert_equal(structure.get("classes"), 1, "Correct class count")
                    result.assert_equal(structure.get("functions"), 5, "Correct function count")

                    result.assert_in("complexity_score", step.result, "Complexity calculated")

        return plan_id

    return None


async def test_code_generation(coordinator: CoordinatorAgent, result: TestResult) -> str | None:
    """Test code generation functionality."""
    generation_request = {
        "request": "Generate a Python function that calculates the nth Fibonacci number using memoization",
        "context": {
            "task_type": "code_generation",
            "language": "python",
            "specification": "Create an efficient fibonacci function with memoization",
            "required_capability": "code_generation",
        },
    }

    # Send request
    message = A2AMessage(
        sender_id="test_client",
        recipient_id=coordinator.agent_id,
        message_type=A2AMessage.MessageTypes.TASK_REQUEST,
        payload=generation_request,
    )

    response = await coordinator.handle_message(message)
    result.assert_not_none(response, "Coordinator responded to generation request")

    if response:
        result.assert_equal(
            response.payload.get("status"), "accepted", "Generation request accepted"
        )
        plan_id = response.payload.get("plan_id")
        result.assert_not_none(plan_id, "Plan ID returned")

        # Wait for execution
        await asyncio.sleep(10)  # Generation takes longer

        # Check results
        plan = coordinator.load_plan(plan_id)
        result.assert_not_none(plan, "Plan loaded successfully")

        if plan:
            progress = plan.get_progress()
            result.assert_equal(plan.status, "completed", "Plan completed successfully")

            if plan.steps and plan.steps[0].result:
                gen_result = plan.steps[0].result
                result.assert_in("generated_code", gen_result, "Code generated")
                result.assert_in("validation", gen_result, "Code validated")

                # Check if generated code contains expected elements
                code = gen_result.get("generated_code", "")
                result.assert_true("def " in code, "Function definition present")
                result.assert_true("fibonacci" in code.lower(), "Fibonacci function created")

                # Validation should pass
                validation = gen_result.get("validation", {})
                result.assert_equal(
                    validation.get("valid"), True, "Generated code is syntactically valid"
                )

        return plan_id

    return None


async def test_plan_storage(coordinator: CoordinatorAgent, plan_ids: list[str], result: TestResult):
    """Test plan storage functionality."""
    # Test listing plans
    all_plans = coordinator.list_stored_plans()
    result.assert_true(len(all_plans) >= len(plan_ids), "All plans listed")

    # Test storage stats
    stats = coordinator.get_storage_stats()
    result.assert_in("total_plans", stats, "Storage stats include total plans")
    result.assert_true(stats.get("total_plans", 0) >= len(plan_ids), "Correct plan count")

    # Test loading specific plan
    if plan_ids:
        loaded_plan = coordinator.load_plan(plan_ids[0])
        result.assert_not_none(loaded_plan, "Plan loaded from storage")

        if loaded_plan:
            result.assert_equal(loaded_plan.id, plan_ids[0], "Correct plan loaded")

    # Test archiving
    archived_count = coordinator.archive_completed_plans()
    result.assert_true(archived_count >= 0, f"Archived {archived_count} plans")

    # Test export
    report_file = coordinator.export_plan_report()
    result.assert_true(os.path.exists(report_file), f"Report exported to {report_file}")


async def test_error_handling(coordinator: CoordinatorAgent, result: TestResult):
    """Test error handling scenarios."""
    # Test invalid capability request
    invalid_request = {
        "request": "Do something impossible",
        "context": {"task_type": "invalid_task", "required_capability": "nonexistent_capability"},
    }

    message = A2AMessage(
        sender_id="test_client",
        recipient_id=coordinator.agent_id,
        message_type=A2AMessage.MessageTypes.TASK_REQUEST,
        payload=invalid_request,
    )

    response = await coordinator.handle_message(message)

    # Should still get a response, but plan creation might fail
    if response and response.payload.get("status") == "accepted":
        plan_id = response.payload.get("plan_id")
        await asyncio.sleep(3)

        plan = coordinator.load_plan(plan_id)
        if plan:
            # Check if plan handles unknown capability gracefully
            result.warn("Plan created for invalid capability - check if this is expected behavior")


async def cleanup_test_artifacts():
    """Clean up test artifacts after test run."""
    logger.info("Cleaning up test artifacts...")

    cleanup_paths = [
        # Test files
        ".aida/test_files/",
        ".aida/coordinator/plans/",
        # Test reports
        "coordinator_plan_summary_*.txt",
        # MCP server logs/artifacts
        ".aida/mcp_logs/",
        # Temporary test files at root
        "test_mcp_*.py",
        "test_*.log",
    ]

    for path_pattern in cleanup_paths:
        path = Path(path_pattern)

        # Handle glob patterns
        if "*" in path_pattern:
            for match in Path(".").glob(path_pattern):
                try:
                    if match.is_file():
                        match.unlink()
                        logger.debug(f"Removed file: {match}")
                    elif match.is_dir():
                        shutil.rmtree(match)
                        logger.debug(f"Removed directory: {match}")
                except Exception as e:
                    logger.warning(f"Failed to remove {match}: {e}")
        else:
            # Handle specific paths
            if path.exists():
                try:
                    if path.is_file():
                        path.unlink()
                        logger.debug(f"Removed file: {path}")
                    elif path.is_dir():
                        # Only remove if it's a test directory
                        if "test" in str(path) or "plans" in str(path):
                            shutil.rmtree(path)
                            logger.debug(f"Removed directory: {path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {e}")

    # Recreate necessary directories
    test_dirs = [
        ".aida/test_files",
        ".aida/coordinator/plans/active",
        ".aida/coordinator/plans/archived",
        ".aida/coordinator/plans/failed",
    ]

    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {dir_path}")

    # Create calculator.py test file for next run
    calculator_content = '''"""Test module for code analysis."""

class Calculator:
    """Simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    def divide(self, a: float, b: float) -> float:
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''

    calc_path = Path(".aida/test_files/calculator.py")
    calc_path.write_text(calculator_content)
    logger.debug("Created fresh calculator.py test file")

    logger.info("Test artifacts cleaned up successfully")


async def run_all_tests():
    """Run all coordinator-worker tests."""
    logger.info("Starting coordinator-worker integration tests")
    logger.info("=" * 80)

    # Clean up before starting tests to ensure clean state
    await cleanup_test_artifacts()
    logger.info("Pre-test cleanup completed")

    all_passed = True
    coordinator = None
    worker = None

    try:
        # Test 1: Worker Registration
        test1 = TestResult("Worker Registration")
        coordinator, worker = await test_worker_registration(test1)
        all_passed &= test1.report()

        plan_ids = []

        # Test 2: Code Analysis
        test2 = TestResult("Code Analysis")
        plan_id = await test_code_analysis(coordinator, test2)
        if plan_id:
            plan_ids.append(plan_id)
        all_passed &= test2.report()

        # Test 3: Code Generation
        test3 = TestResult("Code Generation")
        plan_id = await test_code_generation(coordinator, test3)
        if plan_id:
            plan_ids.append(plan_id)
        all_passed &= test3.report()

        # Test 4: Plan Storage
        test4 = TestResult("Plan Storage")
        await test_plan_storage(coordinator, plan_ids, test4)
        all_passed &= test4.report()

        # Test 5: Error Handling
        test5 = TestResult("Error Handling")
        await test_error_handling(coordinator, test5)
        all_passed &= test5.report()

    except Exception as e:
        logger.error(f"Test suite failed with exception: {e}", exc_info=True)
        all_passed = False

    finally:
        # Cleanup
        logger.info("\nCleaning up...")
        if worker:
            await worker.stop()
        if coordinator:
            await coordinator.stop()

        # Clean up test artifacts
        await cleanup_test_artifacts()

    # Final report
    logger.info("\n" + "=" * 80)
    if all_passed:
        logger.info("ALL TESTS PASSED!")
    else:
        logger.error("SOME TESTS FAILED")
    logger.info("=" * 80)

    return all_passed


async def main():
    """Main entry point."""
    import sys

    # Check for --no-cleanup flag
    no_cleanup = "--no-cleanup" in sys.argv
    if no_cleanup:
        logger.info("Cleanup disabled by --no-cleanup flag")
        # Temporarily disable cleanup
        global cleanup_test_artifacts
        original_cleanup = cleanup_test_artifacts

        async def noop_cleanup():
            logger.info("Skipping cleanup (--no-cleanup flag set)")

        cleanup_test_artifacts = noop_cleanup

    success = await run_all_tests()

    # Restore original cleanup function
    if no_cleanup:
        cleanup_test_artifacts = original_cleanup

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
