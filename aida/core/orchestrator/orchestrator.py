"""Main TODO orchestrator implementation."""

import asyncio
from collections.abc import Callable
from datetime import datetime
import json
import logging
from typing import Any

from aida.llm import chat
from aida.tools.base import get_tool_registry

from .config import OrchestratorConfig
from .models import ReplanReason, TodoPlan, TodoStatus, TodoStep
from .storage import PlanStorageManager

logger = logging.getLogger(__name__)


class TodoOrchestrator:
    """TODO-based workflow orchestrator with progressive checking and re-evaluation."""

    def __init__(self, storage_dir: str | None = None):
        """Initialize the TODO orchestrator.

        Args:
            storage_dir: Optional directory path for storing workflow plans.
                If not provided, uses the default storage directory from config.
                Plans are persisted here for recovery and historical analysis.
        """
        self.tool_registry = get_tool_registry()
        self.active_plans: dict[str, TodoPlan] = {}
        self.storage_manager = PlanStorageManager(
            storage_dir or OrchestratorConfig.DEFAULT_STORAGE_DIR
        )
        self._tools_initialized = False
        self._step_counter = 0

    async def create_plan(
        self, user_request: str, context: dict[str, Any] | None = None
    ) -> TodoPlan:
        """Create a new TODO-based workflow plan."""
        if not self._tools_initialized:
            await self._ensure_tools_initialized()

        try:
            # Generate plan using LLM
            plan_data = await self._generate_initial_plan(user_request, context or {})

            # Validate the plan data structure
            self._validate_plan_data(plan_data)

            # Create TodoStep objects
            steps = self._create_steps_from_plan_data(plan_data)

            # Create plan
            plan = TodoPlan(
                id=f"plan_{datetime.utcnow().timestamp()}",
                user_request=user_request,
                analysis=plan_data.get("analysis", ""),
                expected_outcome=plan_data.get("expected_outcome", ""),
                steps=steps,
                context=context or {},
            )

            # Store plan in memory and persistent storage
            self.active_plans[plan.id] = plan
            self.storage_manager.save_plan(plan)

            logger.info(f"Successfully created TODO plan with {len(steps)} steps")
            logger.debug(f"Plan markdown:\n{plan.to_markdown()}")

            return plan

        except Exception as e:
            # Clear error message for any plan creation failure
            error_msg = f"Failed to create workflow plan for '{user_request}': {str(e)}"
            logger.error(error_msg)

            # Don't create a fallback plan - let the error propagate with a clear message
            raise RuntimeError(error_msg)

    def _validate_plan_data(self, plan_data: dict[str, Any]) -> None:
        """Validate the plan data structure."""
        if not isinstance(plan_data, dict):
            raise ValueError("Plan data is not a valid dictionary")

        for field in OrchestratorConfig.REQUIRED_PLAN_FIELDS:
            if field not in plan_data or not plan_data[field]:
                raise ValueError(f"Plan data missing required '{field}' field")

        if not isinstance(plan_data["execution_plan"], list):
            raise ValueError("Plan data 'execution_plan' field is not a list")

        if len(plan_data["execution_plan"]) == 0:
            raise ValueError("Plan contains no execution steps")

        if len(plan_data["execution_plan"]) > OrchestratorConfig.MAX_STEPS_PER_PLAN:
            raise ValueError(
                f"Plan contains too many steps (max: {OrchestratorConfig.MAX_STEPS_PER_PLAN})"
            )

    def _create_steps_from_plan_data(self, plan_data: dict[str, Any]) -> list[TodoStep]:
        """Create TodoStep objects from plan data."""
        steps = []

        for i, step_data in enumerate(plan_data.get("execution_plan", [])):
            if not isinstance(step_data, dict):
                raise ValueError(f"Step {i + 1} is not a valid dictionary")

            for field in OrchestratorConfig.REQUIRED_STEP_FIELDS:
                if field not in step_data or not step_data[field]:
                    raise ValueError(f"Step {i + 1} missing required '{field}' field")

            step_id = f"step_{self._step_counter:03d}"
            self._step_counter += 1

            # Handle dependencies - they should be step IDs (strings)
            raw_dependencies = step_data.get("dependencies", [])
            dependencies = []
            for dep in raw_dependencies:
                if isinstance(dep, str):
                    dependencies.append(dep)
                elif isinstance(dep, dict) and "id" in dep:
                    # If LLM returns dict objects, extract the ID
                    dependencies.append(dep["id"])
                # Skip invalid dependencies

            step = TodoStep(
                id=step_id,
                description=step_data.get("description", f"Step {i + 1}"),
                tool_name=step_data.get("tool_name")
                or step_data.get("tool", OrchestratorConfig.DEFAULT_TOOL_NAME),
                parameters=step_data.get("parameters", {}),
                dependencies=dependencies,
                max_retries=OrchestratorConfig.DEFAULT_MAX_RETRIES,
            )
            steps.append(step)

        return steps

    async def execute_plan(
        self,
        plan: TodoPlan,
        progress_callback: Callable[[TodoPlan, TodoStep], None] | None = None,
        replan_callback: Callable[[TodoPlan, ReplanReason], bool] | None = None,
    ) -> dict[str, Any]:
        """Execute a TODO plan with progressive checking and re-evaluation."""
        results = []

        while True:
            # Check if we should replan
            should_replan, reason = plan.should_replan()
            if should_replan and reason != ReplanReason.PERIODIC_CHECK:
                logger.info(f"Replanning needed: {reason.value if reason else 'unknown'}")

                # Ask callback if we should replan
                if replan_callback and not replan_callback(plan, reason):
                    logger.info("Replanning declined by callback")
                else:
                    await self._replan(plan, reason)

            # Get next executable step
            next_step = plan.get_next_executable_step()
            if not next_step:
                # Check if we're done or stuck
                progress = plan.get_progress()
                if progress["status"] == "completed":
                    logger.info("All steps completed!")
                    break
                elif progress["status"] in ["failed", "partial_failure"]:
                    logger.error("Plan execution failed")
                    break
                else:
                    logger.warning("No executable steps available but plan not complete")
                    break

            # Execute step
            if progress_callback:
                if asyncio.iscoroutinefunction(progress_callback):
                    await progress_callback(plan, next_step)
                else:
                    progress_callback(plan, next_step)

            step_result = await self._execute_step(next_step, plan)
            results.append(step_result)

            # Update plan timestamps and save to storage
            plan.last_updated = datetime.utcnow()
            self.storage_manager.save_plan(plan)

        return {
            "status": plan.get_progress()["status"],
            "plan": plan,
            "results": results,
            "final_markdown": plan.to_markdown(),
        }

    # Plan management methods
    def load_plan(self, plan_id: str) -> TodoPlan | None:
        """Load a plan from storage."""
        # Check if already in memory
        if plan_id in self.active_plans:
            return self.active_plans[plan_id]

        # Load from storage
        plan = self.storage_manager.load_plan(plan_id)
        if plan:
            self.active_plans[plan_id] = plan

        return plan

    def list_stored_plans(self) -> list[dict[str, Any]]:
        """List all stored plans."""
        return self.storage_manager.list_plans()

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan from both memory and storage."""
        # Remove from memory
        self.active_plans.pop(plan_id, None)

        # Remove from storage
        return self.storage_manager.delete_plan(plan_id)

    def get_plan_summary(self, plan_id: str) -> dict[str, Any] | None:
        """Get a summary of a specific plan."""
        plan = self.load_plan(plan_id)
        if not plan:
            return None

        return plan.get_summary_stats()

    def display_plan(self, plan_id: str) -> str | None:
        """Get terminal display format for a plan."""
        plan = self.load_plan(plan_id)
        if not plan:
            return None

        return plan.to_terminal_display()

    def archive_completed_plans(self) -> int:
        """Archive all completed plans."""
        return self.storage_manager.archive_completed_plans()

    def cleanup_old_plans(self, days_old: int = 30) -> int:
        """Clean up old plans."""
        return self.storage_manager.cleanup_old_plans(days_old)

    def export_summary_report(self, output_file: str | None = None) -> str:
        """Export a summary report of all plans."""
        return self.storage_manager.export_plan_summary(output_file)

    # Internal methods
    async def _execute_step(self, step: TodoStep, plan: TodoPlan) -> dict[str, Any]:
        """Execute a single TODO step."""
        step.status = TodoStatus.IN_PROGRESS
        step.started_at = datetime.utcnow()

        logger.info(f"Executing: {step.description}")

        try:
            # Get tool and execute
            tool = await self.tool_registry.get_tool(step.tool_name)
            if not tool:
                raise ValueError(f"Tool '{step.tool_name}' not found")

            # Validate parameters
            capability = tool.get_capability()
            required_params = [p.name for p in capability.parameters if p.required]
            missing_params = set(required_params) - set(step.parameters.keys())

            if missing_params:
                raise ValueError(f"Missing required parameters: {missing_params}")

            # Execute tool
            result = await tool.execute_async(**step.parameters)
            step.result = result
            step.completed_at = datetime.utcnow()

            # Check if tool execution was successful
            if (
                result
                and hasattr(result, "status")
                and str(result.status).lower() in ["failed", "error"]
            ):
                step.status = TodoStatus.FAILED
                step.error = result.error if hasattr(result, "error") else "Tool execution failed"
                logger.error(f"Failed: {step.description} - {step.error}")
            else:
                step.status = TodoStatus.COMPLETED
                logger.info(f"Completed: {step.description}")

            # Save plan after step completion
            self.storage_manager.save_plan(plan)

            return {
                "step_id": step.id,
                "success": step.status == TodoStatus.COMPLETED,
                "result": (
                    result.model_dump()
                    if result and hasattr(result, "model_dump")
                    else (result.dict() if result and hasattr(result, "dict") else result)
                ),
            }

        except Exception as e:
            step.error = str(e)
            step.completed_at = datetime.utcnow()

            # Decide if we should retry or fail
            if step.retry_count < step.max_retries and self._should_retry(step, e):
                step.retry_count += 1
                step.status = TodoStatus.PENDING  # Reset for retry
                step.started_at = None
                step.completed_at = None
                logger.warning(
                    f"Retrying step {step.retry_count}/{step.max_retries}: {step.description}"
                )
            else:
                step.status = TodoStatus.FAILED
                logger.error(f"Failed: {step.description} - {e}")

            # Save plan after step failure/retry
            self.storage_manager.save_plan(plan)

            return {
                "step_id": step.id,
                "success": False,
                "error": str(e),
                "retry_count": step.retry_count,
            }

    async def _generate_initial_plan(
        self, user_request: str, context: dict[str, Any]
    ) -> dict[str, Any]:  # ty: ignore[invalid-return-type]
        """Generate initial plan using LLM."""
        # Get available tools
        available_tools = await self.tool_registry.list_tools()
        tool_specs = await self._get_tool_specifications(available_tools)

        # Create planning prompt
        planning_prompt = self._create_todo_planning_prompt(user_request, tool_specs, context)

        # Try up to 2 times to get a valid response
        for attempt in range(2):
            try:
                # Use configured LLM model
                response = await chat(planning_prompt, purpose=OrchestratorConfig.LLM_PURPOSE)

                # Debug: Log the raw response
                logger.debug(f"Raw LLM response: {response[:500]}...")

                # Parse JSON response
                json_content = self._extract_json_from_response(response)
                plan_data = json.loads(json_content)

                # Validate the plan data
                self._validate_plan_data(plan_data)

                return plan_data

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse plan on attempt {attempt + 1}: {e}")
                logger.debug(
                    f"Raw LLM response causing error: {response[:1000] if 'response' in locals() else 'No response'}"
                )

                if attempt == 0:
                    # Add more explicit instructions for retry
                    planning_prompt = (
                        planning_prompt
                        + "\n\nIMPORTANT: Respond ONLY with the JSON object. Do not include any explanatory text before or after the JSON."
                    )
                else:
                    # Final attempt failed
                    raise RuntimeError(
                        f"Plan generation failed: {e}. The LLM did not return a valid JSON response. Please check your LLM configuration and try again."
                    )
            except Exception as e:
                # Non-parsing errors should fail immediately
                logger.error(f"Failed to generate plan: {e}")
                raise RuntimeError(
                    f"Plan generation failed: {e}. Please check your LLM configuration and try again."
                )

    def _create_todo_planning_prompt(
        self, user_request: str, tool_specs: dict, context: dict[str, Any]
    ) -> str:
        """Create a planning prompt focused on TODO-style output."""
        tools_info = []
        for name, spec in tool_specs.items():
            params = []
            for param in spec.get("parameters", []):
                param_desc = f"{param['name']} ({param['type']})"
                if param.get("required"):
                    param_desc += " [REQUIRED]"
                params.append(f"    - {param_desc}: {param['description']}")

            tool_desc = f"- {name}: {spec['description']}"
            if params:
                tool_desc += "\n  Parameters:\n" + "\n".join(params)
            tools_info.append(tool_desc)

        template = OrchestratorConfig.get_planning_prompt_template()
        return template.format(
            user_request=user_request,
            tools_info="\n".join(tools_info),
            context=json.dumps(context, indent=2) if context else "None",
        )

    async def _get_tool_specifications(self, tool_names: list[str]) -> dict[str, dict]:
        """Get detailed specifications for tools."""
        specs = {}
        for tool_name in tool_names:
            tool = await self.tool_registry.get_tool(tool_name)
            if tool:
                capability = tool.get_capability()
                specs[tool_name] = {
                    "description": capability.description,
                    "parameters": [
                        {
                            "name": param.name,
                            "type": param.type,
                            "description": param.description,
                            "required": param.required,
                        }
                        for param in capability.parameters
                    ],
                }
        return specs

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON content from LLM response."""
        # Handle case where response is already a dictionary/object
        if hasattr(response, "content"):
            response_text = response.content
        elif isinstance(response, dict):
            # If it's already parsed, convert back to string for consistency
            return json.dumps(response)
        else:
            response_text = str(response)

        # Look for JSON code blocks first
        if isinstance(response_text, str) and "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                json_content = response_text[start:end].strip()
                # Validate that it's actually JSON
                try:
                    json.loads(json_content)
                    return json_content
                except json.JSONDecodeError:
                    pass

        # Look for JSON code blocks without language specification
        if isinstance(response_text, str) and "```" in response_text:
            lines = response_text.split("\n")
            in_code_block = False
            json_lines = []

            for line in lines:
                if line.strip() == "```" and not in_code_block:
                    in_code_block = True
                    continue
                elif line.strip() == "```" and in_code_block:
                    if json_lines:
                        json_content = "\n".join(json_lines).strip()
                        try:
                            json.loads(json_content)
                            return json_content
                        except json.JSONDecodeError:
                            pass
                    json_lines = []
                    in_code_block = False
                elif in_code_block:
                    json_lines.append(line)

        # Look for JSON objects with proper bracket matching
        if isinstance(response_text, str):
            brace_count = 0
            start_pos = -1

            for i, char in enumerate(response_text):
                if char == "{":
                    if start_pos == -1:
                        start_pos = i
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0 and start_pos != -1:
                        json_content = response_text[start_pos : i + 1]
                        try:
                            json.loads(json_content)
                            return json_content
                        except json.JSONDecodeError:
                            # Reset and continue looking
                            start_pos = -1
                            brace_count = 0

        # Debug: Log more details about what we tried to parse
        if isinstance(response_text, str):
            logger.debug(f"JSON extraction failed. Response length: {len(response_text)}")
            logger.debug(f"Response starts with: {response_text[:100]}...")
            logger.debug(f"Response ends with: ...{response_text[-100:]}")
        else:
            logger.debug(f"JSON extraction failed. Response type: {type(response_text)}")

        # Try one more approach: look for any JSON-like structure
        if isinstance(response_text, str):
            import re

            json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
            matches = re.findall(json_pattern, response_text, re.DOTALL)

            for match in matches:
                try:
                    json.loads(match)
                    logger.debug(f"Found valid JSON with regex: {len(match)} chars")
                    return match
                except json.JSONDecodeError:
                    continue

            raise ValueError(
                f"No valid JSON found in response. Response length: {len(response_text)}, preview: {response_text[:200]}..."
            )
        else:
            raise ValueError(
                f"No valid JSON found in response. Response type: {type(response_text)}"
            )

    async def _replan(self, plan: TodoPlan, reason: ReplanReason):
        """Re-evaluate and update the plan."""
        logger.info(f"Re-evaluating plan due to: {reason.value}")

        # Add replan to history
        plan.replan_history.append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "reason": reason.value,
                "version": plan.plan_version,
            }
        )

        # For now, just mark timestamp - full replanning would be more complex
        plan.last_evaluated = datetime.utcnow()
        plan.plan_version += 1

    def _should_retry(self, step: TodoStep, error: Exception) -> bool:
        """Determine if a step should be retried based on the error."""
        error_str = str(error).lower()
        return any(retry_error in error_str for retry_error in OrchestratorConfig.RETRY_ERRORS)

    async def _ensure_tools_initialized(self):
        """Ensure tools are properly initialized."""
        if not self._tools_initialized:
            # Initialize default tools
            from aida.tools.base import initialize_default_tools

            await initialize_default_tools()
            self._tools_initialized = True

    async def execute_request(
        self,
        user_message: str,
        context: dict[str, Any] | None = None,
        progress_callback: Callable | None = None,
    ) -> dict[str, Any]:
        """Execute a user request - compatibility method for chat interface."""
        try:
            # Create a plan
            plan = await self.create_plan(user_message, context)

            # Create a progress wrapper if callback provided
            async def wrapped_progress(plan, step):
                if progress_callback:
                    # Convert to workflow format for compatibility
                    # Create a simple object with the attributes expected by chat
                    class WorkflowAdapter:
                        def __init__(self, plan, current_step_obj):
                            self.steps = [
                                {
                                    "tool_name": s.tool_name,
                                    "parameters": s.parameters,
                                    "purpose": s.description,
                                }
                                for s in plan.steps
                            ]
                            self.current_step = (
                                plan.steps.index(current_step_obj)
                                if current_step_obj in plan.steps
                                else 0
                            )

                    workflow = WorkflowAdapter(plan, step)

                    # Create step adapter with expected attributes
                    class StepAdapter:
                        def __init__(self, step_obj):
                            self.tool_name = step_obj.tool_name
                            self.parameters = step_obj.parameters
                            self.purpose = step_obj.description
                            self.status = (
                                "running"
                                if step_obj.status.value == "in_progress"
                                else step_obj.status.value
                            )

                    step_adapter = StepAdapter(step)
                    await progress_callback(workflow, step_adapter)

            # Execute the plan
            result = await self.execute_plan(
                plan, progress_callback=wrapped_progress if progress_callback else None
            )

            # Convert to expected format
            if result["status"] == "completed":
                return {
                    "status": "completed",
                    "workflow": {
                        "analysis": plan.analysis,
                        "expected_outcome": plan.expected_outcome,
                        "steps": [s.to_dict() for s in plan.steps],
                    },
                    "results": [
                        {
                            "step": {
                                "tool_name": step.tool_name,
                                "parameters": step.parameters,
                                "purpose": step.description,
                            },
                            "success": step.status == TodoStatus.COMPLETED,
                            "result": (
                                step.result.model_dump()
                                if step.result and hasattr(step.result, "model_dump")
                                else step.result
                            ),
                        }
                        for step in plan.steps
                    ],
                    "execution_summary": result.get("execution_summary", {}),
                }
            else:
                return {
                    "status": "failed",
                    "error": result.get("message", "Unknown error"),
                    "workflow": {},
                    "results": [],
                }

        except Exception as e:
            import traceback

            logger.error(f"Error in execute_request: {e}")
            logger.error(traceback.format_exc())
            return {"status": "failed", "error": str(e), "workflow": {}, "results": []}
