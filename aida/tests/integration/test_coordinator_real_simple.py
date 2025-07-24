#!/usr/bin/env python3
"""Simple real integration test for coordinator with dispatcher.

This test demonstrates the working TaskDispatcher with retry and revision features.
"""

import asyncio
import logging
from pathlib import Path
import tempfile

from aida.agents.base import AgentConfig, BaseAgent
from aida.agents.coordination import CoordinatorAgent, TodoPlan, TodoStep
from aida.core.protocols.a2a import A2AMessage
from aida.llm import get_llm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FileAgent(BaseAgent):
    """Simple agent that handles file operations."""

    def __init__(self):
        config = AgentConfig(
            agent_id="file_agent",
            agent_type="worker",
            capabilities=["file_write", "file_read"],
        )
        super().__init__(config)
        self.executions = []

    async def handle_message(self, message: A2AMessage) -> A2AMessage | None:
        """Handle incoming messages."""
        if message.message_type == "task_request":
            return await self._handle_task(message)
        return None

    async def _handle_task(self, message: A2AMessage) -> A2AMessage:
        """Handle task execution."""
        task_data = message.payload
        tool_name = task_data.get("tool_name")
        parameters = task_data.get("parameters", {})

        self.executions.append({"tool": tool_name, "params": parameters})

        logger.info(f"FileAgent executing {tool_name} with params: {parameters}")

        try:
            if tool_name == "file_write":
                path = parameters.get("path")
                content = parameters.get("content", "")

                # Simulate error if path is missing
                if not path:
                    return A2AMessage(
                        sender_id=self.agent_id,
                        recipient_id=message.sender_id,
                        message_type="task_response",
                        payload={
                            "success": False,
                            "error": "Missing required parameter: 'path'",
                            "retriable": True,
                        },
                    )

                # Write the file
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(content)

                return A2AMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="task_response",
                    payload={
                        "success": True,
                        "result": {"written": len(content), "path": path},
                    },
                )

            elif tool_name == "file_read":
                path = parameters.get("path")

                if not path or not Path(path).exists():
                    return A2AMessage(
                        sender_id=self.agent_id,
                        recipient_id=message.sender_id,
                        message_type="task_response",
                        payload={
                            "success": False,
                            "error": f"File not found: {path}",
                            "retriable": False,
                        },
                    )

                content = Path(path).read_text()
                return A2AMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="task_response",
                    payload={
                        "success": True,
                        "result": {"content": content},
                    },
                )

            else:
                return A2AMessage(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    message_type="task_response",
                    payload={
                        "success": False,
                        "error": f"Unknown tool: {tool_name}",
                        "retriable": False,
                    },
                )

        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return A2AMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type="task_response",
                payload={
                    "success": False,
                    "error": str(e),
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

        response = await self._handle_task(message)
        return response.payload if response else {"success": False, "error": "No response"}


async def run_test():
    """Run the simple integration test."""
    logger.info("=== Simple Real Integration Test ===")

    # Check LLM
    logger.info("Checking LLM...")
    try:
        _ = get_llm()
        logger.info("✓ LLM ready")
    except Exception as e:
        logger.warning(f"⚠ LLM not available: {e}")

    # Create coordinator and agent
    coordinator = CoordinatorAgent(
        config=AgentConfig(
            agent_id="test_coordinator",
            agent_type="coordinator",
            capabilities=["planning", "task_delegation"],
        )
    )

    file_agent = FileAgent()

    # Start agents
    await coordinator.start()
    await file_agent.start()

    # Register agent
    coordinator._known_workers[file_agent.agent_id] = file_agent
    await coordinator._update_dispatcher()

    # Create test directory
    test_dir = Path(tempfile.mkdtemp(prefix="aida_test_"))
    logger.info(f"Test directory: {test_dir}")

    # Create plan with intentional failure
    plan = TodoPlan(
        id="simple_test_001",
        user_request="Write configuration files",
        analysis="Testing file operations with retry",
        expected_outcome="Files created after handling errors",
        context={"test_directory": str(test_dir)},  # Provide context for LLM
        steps=[
            TodoStep(
                id="write_config",
                description="Write configuration file",
                tool_name="file_write",
                parameters={
                    # Missing 'path' to trigger error and revision
                    "content": '{"name": "test", "version": "1.0.0"}',
                },
            ),
            TodoStep(
                id="write_readme",
                description="Write README file",
                tool_name="file_write",
                parameters={
                    "path": str(test_dir / "README.md"),
                    "content": "# Test Project\n\nThis is a test.",
                },
                dependencies=["write_config"],
            ),
            TodoStep(
                id="verify_config",
                description="Read config to verify",
                tool_name="file_read",
                parameters={
                    "path": str(test_dir / "config.json"),
                },
                dependencies=["write_config"],
            ),
        ],
    )

    # Execute plan
    logger.info("\nExecuting plan...")
    result = await coordinator.execute_plan(plan)

    # Show results
    logger.info("\n=== Results ===")
    logger.info(f"Status: {result['status']}")

    # Check steps
    for step in plan.steps:
        logger.info(f"\nStep {step.id}:")
        logger.info(f"  Status: {step.status}")
        if step.metadata.get("revised"):
            logger.info("  ✓ Revised by LLM")
        if step.error:
            logger.info(f"  Error: {step.error}")

    # Check executions
    logger.info(f"\nAgent executions: {len(file_agent.executions)}")
    for i, exec_data in enumerate(file_agent.executions):
        logger.info(f"  {i + 1}. {exec_data['tool']} - {exec_data['params']}")

    # Verify files
    logger.info("\n=== File Verification ===")
    config_path = test_dir / "config.json"
    readme_path = test_dir / "README.md"

    if config_path.exists():
        logger.info(f"✓ Config file created: {config_path}")
    else:
        logger.info("✗ Config file missing")

    if readme_path.exists():
        logger.info(f"✓ README file created: {readme_path}")
    else:
        logger.info("✗ README file missing")

    # Cleanup
    import shutil

    shutil.rmtree(test_dir, ignore_errors=True)

    # Stop agents
    await coordinator.stop()
    await file_agent.stop()

    # Return success/failure
    success = result["status"] == "completed" and readme_path.exists()
    logger.info(f"\n{'✅ TEST PASSED' if success else '❌ TEST FAILED'}")
    return success


async def main():
    """Run the test."""
    try:
        success = await run_test()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
