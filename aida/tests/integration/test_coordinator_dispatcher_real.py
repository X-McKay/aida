#!/usr/bin/env python3
"""Integration test for coordinator with dispatcher using real tools.

This test demonstrates:
1. Real file operations with error handling
2. Task retry on failures
3. LLM-based task revision
4. Capability-based agent routing
"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import tempfile

from aida.agents.base import AgentConfig, BaseAgent
from aida.agents.coordination import CoordinatorAgent, TodoPlan, TodoStep
from aida.core.protocols.a2a import A2AMessage
from aida.llm import get_llm
from aida.tools.files.files import FileOperationsTool
from aida.tools.system.system import SystemTool

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FileWorkerAgent(BaseAgent):
    """Agent specialized in file operations."""

    def __init__(self):
        config = AgentConfig(
            agent_id="file_worker",
            agent_type="worker",
            capabilities=["file_operations", "file_management"],
        )
        super().__init__(config)
        self.file_tool = FileOperationsTool()

    async def handle_message(self, message: A2AMessage) -> A2AMessage | None:
        """Handle incoming messages."""
        if message.message_type == "task_request":
            return await self._handle_task_request(message)
        elif message.message_type == "task_assignment":
            return await self._handle_task_assignment(message)
        return None

    async def _handle_task_request(self, message: A2AMessage) -> A2AMessage:
        """Handle task execution."""
        task_data = message.payload
        tool_name = task_data.get("tool_name")
        parameters = task_data.get("parameters", {})

        logger.info(f"FileWorker executing {tool_name} with params: {parameters}")

        try:
            if tool_name == "file_operations":
                # Map parameters to file tool format
                operation = parameters.get("operation", "read")

                if operation == "write":
                    result = await self.file_tool.execute(
                        operation="write",
                        path=parameters.get("path"),
                        content=parameters.get("content", ""),
                    )
                elif operation == "read":
                    result = await self.file_tool.execute(
                        operation="read", path=parameters.get("path")
                    )
                elif operation == "create_directory":
                    result = await self.file_tool.execute(
                        operation="create_directory", path=parameters.get("path")
                    )
                else:
                    raise ValueError(f"Unknown operation: {operation}")

                return A2AMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="task_response",
                    payload={
                        "success": result.status == "success",
                        "result": result.result,
                        "error": result.error,
                    },
                )
            else:
                return self._error_response(message, f"Unknown tool: {tool_name}")

        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return self._error_response(message, str(e))

    def _error_response(self, message: A2AMessage, error: str) -> A2AMessage:
        """Create error response."""
        return A2AMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type="task_response",
            payload={
                "success": False,
                "error": error,
                "retriable": True,
            },
        )

    async def execute_task(self, task_data: dict) -> dict:
        """Direct task execution for dispatcher."""
        message = A2AMessage(
            sender_id="dispatcher",
            recipient_id=self.agent_id,
            message_type="task_request",
            payload=task_data,
        )

        response = await self._handle_task_request(message)
        return response.payload if response else {"success": False, "error": "No response"}

    async def _handle_task_assignment(self, message: A2AMessage) -> A2AMessage:
        """Handle task assignment from coordinator."""
        payload = message.payload

        # Extract task data from assignment payload
        task_data = {
            "tool_name": payload.get("capability_required", "file_operations"),
            "parameters": payload.get("parameters", {}),
            "description": payload.get("step_description", ""),
            "task_id": payload.get("task_id"),
        }

        # Create task request message and handle it
        task_request = A2AMessage(
            sender_id=message.sender_id,
            recipient_id=self.agent_id,
            message_type="task_request",
            payload=task_data,
        )

        # Execute the task
        response = await self._handle_task_request(task_request)

        # Convert response to task completion format
        if response:
            return A2AMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type="task_completion",
                payload={
                    "task_id": payload.get("task_id"),
                    "worker_id": self.agent_id,
                    "success": response.payload.get("success", False),
                    "result": response.payload.get("result"),
                    "error": response.payload.get("error"),
                    "execution_time": 0.1,  # Mock execution time
                },
            )
        else:
            return self._error_response(message, "Failed to execute task")


class SystemWorkerAgent(BaseAgent):
    """Agent specialized in system operations."""

    def __init__(self):
        config = AgentConfig(
            agent_id="system_worker",
            agent_type="worker",
            capabilities=["system_info", "system_operations"],
        )
        super().__init__(config)
        self.system_tool = SystemTool()

    async def handle_message(self, message: A2AMessage) -> A2AMessage | None:
        """Handle incoming messages."""
        if message.message_type == "task_request":
            return await self._handle_task_request(message)
        elif message.message_type == "task_assignment":
            return await self._handle_task_assignment(message)
        return None

    async def _handle_task_request(self, message: A2AMessage) -> A2AMessage:
        """Handle task execution."""
        task_data = message.payload
        tool_name = task_data.get("tool_name")
        parameters = task_data.get("parameters", {})

        logger.info(f"SystemWorker executing {tool_name} with params: {parameters}")

        try:
            if tool_name == "system_info":
                operation = parameters.get("operation", "get_info")

                if operation == "get_info":
                    result = await self.system_tool.execute(operation="system_info")
                elif operation == "check_path":
                    result = await self.system_tool.execute(
                        operation="check_path", path=parameters.get("path", "")
                    )
                else:
                    raise ValueError(f"Unknown operation: {operation}")

                return A2AMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="task_response",
                    payload={
                        "success": result.status == "success",
                        "result": result.result,
                        "error": result.error,
                    },
                )
            else:
                return self._error_response(message, f"Unknown tool: {tool_name}")

        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return self._error_response(message, str(e))

    def _error_response(self, message: A2AMessage, error: str) -> A2AMessage:
        """Create error response."""
        return A2AMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type="task_response",
            payload={
                "success": False,
                "error": error,
                "retriable": True,
            },
        )

    async def execute_task(self, task_data: dict) -> dict:
        """Direct task execution for dispatcher."""
        message = A2AMessage(
            sender_id="dispatcher",
            recipient_id=self.agent_id,
            message_type="task_request",
            payload=task_data,
        )

        response = await self._handle_task_request(message)
        return response.payload if response else {"success": False, "error": "No response"}

    async def _handle_task_assignment(self, message: A2AMessage) -> A2AMessage:
        """Handle task assignment from coordinator."""
        payload = message.payload

        # Extract task data from assignment payload
        task_data = {
            "tool_name": payload.get("capability_required", "file_operations"),
            "parameters": payload.get("parameters", {}),
            "description": payload.get("step_description", ""),
            "task_id": payload.get("task_id"),
        }

        # Create task request message and handle it
        task_request = A2AMessage(
            sender_id=message.sender_id,
            recipient_id=self.agent_id,
            message_type="task_request",
            payload=task_data,
        )

        # Execute the task
        response = await self._handle_task_request(task_request)

        # Convert response to task completion format
        if response:
            return A2AMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type="task_completion",
                payload={
                    "task_id": payload.get("task_id"),
                    "worker_id": self.agent_id,
                    "success": response.payload.get("success", False),
                    "result": response.payload.get("result"),
                    "error": response.payload.get("error"),
                    "execution_time": 0.1,  # Mock execution time
                },
            )
        else:
            return self._error_response(message, "Failed to execute task")


async def test_real_scenario():
    """Test real scenario with file operations and task revision."""
    logger.info("=== Real Integration Test: Coordinator with Dispatcher ===")

    # Check LLM
    logger.info("Checking LLM availability...")
    try:
        _ = get_llm()
        logger.info("LLM initialized successfully")
    except Exception as e:
        logger.error(f"LLM initialization failed: {e}")
        logger.info("Test will proceed but revision features may not work")

    # Create coordinator
    coordinator_config = AgentConfig(
        agent_id="test_coordinator",
        agent_type="coordinator",
        capabilities=["planning", "task_delegation"],
    )
    coordinator = CoordinatorAgent(config=coordinator_config)

    # Create worker agents
    file_worker = FileWorkerAgent()
    system_worker = SystemWorkerAgent()

    # Start all agents
    logger.info("Starting agents...")
    await coordinator.start()
    await file_worker.start()
    await system_worker.start()

    # Register workers with coordinator
    coordinator._known_workers[file_worker.agent_id] = file_worker
    coordinator._known_workers[system_worker.agent_id] = system_worker
    await coordinator._update_dispatcher()

    # Create test directory
    test_dir = Path(tempfile.mkdtemp(prefix="aida_test_"))
    logger.info(f"Test directory: {test_dir}")

    # Create test plan with intentional failures
    plan = TodoPlan(
        id="real_test_001",
        user_request="Create project structure with config files",
        analysis="Testing real file operations with error scenarios",
        expected_outcome="Successfully create project structure after handling errors",
        steps=[
            TodoStep(
                id="check_system",
                description="Check system information",
                tool_name="system_info",
                parameters={
                    "operation": "get_info",
                },
            ),
            TodoStep(
                id="create_project_dir",
                description="Create project directory",
                tool_name="file_operations",
                parameters={
                    "operation": "create_directory",
                    "path": str(test_dir / "my_project"),
                },
                dependencies=["check_system"],
            ),
            TodoStep(
                id="write_config",
                description="Write configuration file",
                tool_name="file_operations",
                parameters={
                    "operation": "write",
                    # Intentionally missing 'path' to trigger revision
                    "content": '{"name": "my_project", "version": "1.0.0"}',
                },
                dependencies=["create_project_dir"],
            ),
            TodoStep(
                id="write_readme",
                description="Write README file",
                tool_name="file_operations",
                parameters={
                    "operation": "write",
                    "path": str(test_dir / "my_project" / "README.md"),
                    "content": "# My Project\n\nThis is a test project.",
                },
                dependencies=["create_project_dir"],
            ),
            TodoStep(
                id="verify_structure",
                description="Read back the config to verify",
                tool_name="file_operations",
                parameters={
                    "operation": "read",
                    "path": str(test_dir / "my_project" / "config.json"),
                },
                dependencies=["write_config"],
            ),
        ],
    )

    # Execute the plan
    logger.info(f"\nExecuting plan: {plan.id}")
    logger.info("Watch for task failures and revisions...\n")

    start_time = datetime.utcnow()
    result = await coordinator.execute_plan(plan)
    end_time = datetime.utcnow()

    # Display results
    logger.info("\n=== Execution Results ===")
    logger.info(f"Plan Status: {result['status']}")
    logger.info(f"Execution Time: {(end_time - start_time).total_seconds():.2f}s")

    # Show step results
    logger.info("\nStep Results:")
    for step in plan.steps:
        logger.info(f"\n{step.id}:")
        logger.info(f"  Status: {step.status}")
        if step.error:
            logger.info(f"  Error: {step.error}")
        if step.metadata.get("revised"):
            logger.info("  Revised: YES")
            logger.info(f"  Revision: {step.metadata.get('revision_reasoning', 'N/A')}")

    # Verify files were created
    logger.info("\n=== Verification ===")
    project_dir = test_dir / "my_project"
    if project_dir.exists():
        logger.info(f"✓ Project directory created: {project_dir}")

        config_file = project_dir / "config.json"
        if config_file.exists():
            logger.info(f"✓ Config file created: {config_file}")
            logger.info(f"  Content: {config_file.read_text()}")
        else:
            logger.info("✗ Config file missing")

        readme_file = project_dir / "README.md"
        if readme_file.exists():
            logger.info(f"✓ README file created: {readme_file}")
        else:
            logger.info("✗ README file missing")
    else:
        logger.info("✗ Project directory not created")

    # Cleanup
    logger.info(f"\nCleaning up test directory: {test_dir}")
    import shutil

    shutil.rmtree(test_dir, ignore_errors=True)

    # Stop agents
    await coordinator.stop()
    await file_worker.stop()
    await system_worker.stop()

    logger.info("\n=== Test Complete ===")
    return result


async def main():
    """Run the integration test."""
    try:
        result = await test_real_scenario()
        # Check if test passed
        if result["status"] == "completed":
            logger.info("\n✅ Integration test PASSED")
            return 0
        else:
            logger.error("\n❌ Integration test FAILED")
            return 1
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
