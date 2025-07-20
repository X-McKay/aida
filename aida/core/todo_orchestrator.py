"""TODO-based workflow orchestration with progressive checking and plan re-evaluation."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from aida.llm import chat
from aida.config.llm_profiles import Purpose
from aida.tools.base import get_tool_registry, ToolResult


logger = logging.getLogger(__name__)


class TodoStatus(Enum):
    """Status of a TODO item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ReplanReason(Enum):
    """Reasons for replanning."""
    STEP_FAILED = "step_failed"
    USER_CLARIFICATION = "user_clarification"
    NEW_INFORMATION = "new_information"
    DEPENDENCY_CHANGED = "dependency_changed"
    PERIODIC_CHECK = "periodic_check"


@dataclass
class TodoStep:
    """A single TODO step in the workflow."""
    id: str
    description: str
    tool_name: str
    parameters: Dict[str, Any]
    status: TodoStatus = TodoStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[ToolResult] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # IDs of steps this depends on
    retry_count: int = 0
    max_retries: int = 2
    
    def to_markdown_line(self) -> str:
        """Convert to markdown TODO line."""
        checkbox = "✅" if self.status == TodoStatus.COMPLETED else "❌" if self.status == TodoStatus.FAILED else "🔄" if self.status == TodoStatus.IN_PROGRESS else "⏸️" if self.status == TodoStatus.SKIPPED else "⬜"
        
        suffix = ""
        if self.status == TodoStatus.FAILED and self.error:
            suffix = f" (Failed: {self.error[:50]}...)" if len(self.error) > 50 else f" (Failed: {self.error})"
        elif self.status == TodoStatus.IN_PROGRESS:
            suffix = " (In Progress)"
        elif self.retry_count > 0:
            suffix = f" (Retry {self.retry_count}/{self.max_retries})"
            
        return f"- {checkbox} {self.description}{suffix}"
    
    def can_execute(self, completed_steps: set[str]) -> bool:
        """Check if this step can be executed based on dependencies."""
        return all(dep_id in completed_steps for dep_id in self.dependencies)
    
    def is_terminal(self) -> bool:
        """Check if this step is in a terminal state."""
        return self.status in {TodoStatus.COMPLETED, TodoStatus.FAILED, TodoStatus.SKIPPED}


@dataclass 
class TodoPlan:
    """A TODO-style workflow plan with progressive checking."""
    id: str
    user_request: str
    analysis: str
    expected_outcome: str
    steps: List[TodoStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    last_evaluated: datetime = field(default_factory=datetime.utcnow)
    plan_version: int = 1
    replan_history: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_markdown(self) -> str:
        """Convert plan to markdown TODO format."""
        lines = [
            f"# Workflow Plan: {self.user_request}",
            "",
            f"**Analysis:** {self.analysis}",
            f"**Expected Outcome:** {self.expected_outcome}",
            f"**Created:** {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Version:** {self.plan_version}",
            "",
            "## TODO List",
            ""
        ]
        
        for step in self.steps:
            lines.append(step.to_markdown_line())
        
        # Add progress summary
        progress = self.get_progress()
        lines.extend([
            "",
            "## Progress Summary",
            f"- **Status:** {progress['status'].title()}",
            f"- **Completed:** {progress['completed']}/{progress['total']} steps",
            f"- **Progress:** {progress['percentage']:.1f}%"
        ])
        
        if progress['failed'] > 0:
            lines.append(f"- **Failed:** {progress['failed']} steps")
        
        return "\n".join(lines)
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress statistics."""
        total = len(self.steps)
        completed = sum(1 for step in self.steps if step.status == TodoStatus.COMPLETED)
        failed = sum(1 for step in self.steps if step.status == TodoStatus.FAILED)
        in_progress = sum(1 for step in self.steps if step.status == TodoStatus.IN_PROGRESS)
        pending = sum(1 for step in self.steps if step.status == TodoStatus.PENDING)
        
        if total == 0:
            percentage = 100
            status = "completed"
        else:
            percentage = (completed / total) * 100
            
            if failed > 0 and completed + failed == total:
                status = "failed" if completed == 0 else "partial_failure"
            elif completed == total:
                status = "completed"
            elif in_progress > 0:
                status = "in_progress"
            else:
                status = "pending"
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": pending,
            "percentage": percentage,
            "status": status
        }
    
    def get_next_executable_step(self) -> Optional[TodoStep]:
        """Get the next step that can be executed."""
        completed_step_ids = {step.id for step in self.steps if step.status == TodoStatus.COMPLETED}
        
        for step in self.steps:
            if (step.status == TodoStatus.PENDING and 
                step.can_execute(completed_step_ids)):
                return step
        
        return None
    
    def get_failed_steps(self) -> List[TodoStep]:
        """Get list of failed steps."""
        return [step for step in self.steps if step.status == TodoStatus.FAILED]
    
    def should_replan(self) -> tuple[bool, Optional[ReplanReason]]:
        """Check if the plan should be re-evaluated."""
        now = datetime.utcnow()
        
        # Check for failed steps
        failed_steps = self.get_failed_steps()
        if failed_steps:
            return True, ReplanReason.STEP_FAILED
        
        # Check for periodic re-evaluation
        completed_since_last = sum(
            1 for step in self.steps 
            if (step.status == TodoStatus.COMPLETED and 
                step.completed_at and 
                step.completed_at > self.last_evaluated)
        )
        
        time_since_evaluation = (now - self.last_evaluated).total_seconds()
        
        if completed_since_last >= 5 or time_since_evaluation > 600:  # 10 minutes
            return True, ReplanReason.PERIODIC_CHECK
        
        return False, None


class TodoOrchestrator:
    """TODO-based workflow orchestrator with progressive checking and re-evaluation."""
    
    def __init__(self):
        self.tool_registry = get_tool_registry()
        self.active_plans: Dict[str, TodoPlan] = {}
        self._tools_initialized = False
        self._step_counter = 0
    
    async def create_plan(
        self, 
        user_request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TodoPlan:
        """Create a new TODO-based workflow plan."""
        
        if not self._tools_initialized:
            await self._ensure_tools_initialized()
        
        # Generate plan using LLM
        plan_data = await self._generate_initial_plan(user_request, context or {})
        
        # Create TodoStep objects
        steps = []
        for i, step_data in enumerate(plan_data.get("execution_plan", [])):
            step_id = f"step_{self._step_counter:03d}"
            self._step_counter += 1
            
            step = TodoStep(
                id=step_id,
                description=step_data.get("description", f"Step {i+1}"),
                tool_name=step_data.get("tool", "thinking"),  # Default to thinking tool
                parameters=step_data.get("parameters", {}),
                dependencies=step_data.get("dependencies", [])
            )
            steps.append(step)
        
        # Create plan
        plan = TodoPlan(
            id=f"plan_{datetime.utcnow().timestamp()}",
            user_request=user_request,
            analysis=plan_data.get("analysis", ""),
            expected_outcome=plan_data.get("expected_outcome", ""),
            steps=steps,
            context=context or {}
        )
        
        # Store plan
        self.active_plans[plan.id] = plan
        
        logger.info(f"📋 Created TODO plan with {len(steps)} steps")
        logger.info(f"Plan markdown:\n{plan.to_markdown()}")
        
        return plan
    
    async def execute_plan(
        self, 
        plan: TodoPlan,
        progress_callback: Optional[Callable[[TodoPlan, TodoStep], None]] = None,
        replan_callback: Optional[Callable[[TodoPlan, ReplanReason], bool]] = None
    ) -> Dict[str, Any]:
        """Execute a TODO plan with progressive checking and re-evaluation."""
        
        results = []
        
        while True:
            # Check if we should replan
            should_replan, reason = plan.should_replan()
            if should_replan and reason != ReplanReason.PERIODIC_CHECK:
                logger.info(f"🔄 Replanning needed: {reason.value}")
                
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
                    logger.info("✅ All steps completed!")
                    break
                elif progress["status"] in ["failed", "partial_failure"]:
                    logger.error("❌ Plan execution failed")
                    break
                else:
                    logger.warning("⚠️ No executable steps available but plan not complete")
                    break
            
            # Execute step
            if progress_callback:
                progress_callback(plan, next_step)
            
            step_result = await self._execute_step(next_step, plan)
            results.append(step_result)
            
            # Update plan timestamps
            plan.last_updated = datetime.utcnow()
        
        return {
            "status": plan.get_progress()["status"],
            "plan": plan,
            "results": results,
            "final_markdown": plan.to_markdown()
        }
    
    async def _execute_step(self, step: TodoStep, plan: TodoPlan) -> Dict[str, Any]:
        """Execute a single TODO step."""
        step.status = TodoStatus.IN_PROGRESS
        step.started_at = datetime.utcnow()
        
        logger.info(f"🔄 Executing: {step.description}")
        
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
            step.status = TodoStatus.COMPLETED
            
            logger.info(f"✅ Completed: {step.description}")
            
            return {
                "step_id": step.id,
                "success": True,
                "result": result.dict() if result else None
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
                logger.warning(f"🔄 Retrying step {step.retry_count}/{step.max_retries}: {step.description}")
            else:
                step.status = TodoStatus.FAILED
                logger.error(f"❌ Failed: {step.description} - {e}")
            
            return {
                "step_id": step.id,
                "success": False,
                "error": str(e),
                "retry_count": step.retry_count
            }
    
    async def _generate_initial_plan(self, user_request: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate initial plan using LLM."""
        
        # Get available tools
        available_tools = await self.tool_registry.list_tools()
        tool_specs = await self._get_tool_specifications(available_tools)
        
        # Create planning prompt
        planning_prompt = self._create_todo_planning_prompt(user_request, tool_specs, context)
        
        try:
            # Use new LLM system for reasoning
            response = await chat(planning_prompt, purpose=Purpose.REASONING)
            
            # Parse JSON response
            json_content = self._extract_json_from_response(response)
            plan_data = json.loads(json_content)
            
            return plan_data
            
        except Exception as e:
            logger.error(f"Failed to generate plan: {e}")
            # Fallback to simple plan
            return {
                "analysis": f"Failed to generate detailed plan: {e}",
                "expected_outcome": "Manual execution required",
                "execution_plan": [
                    {
                        "description": f"Manual completion of: {user_request}",
                        "tool": "thinking",
                        "parameters": {"problem": user_request}
                    }
                ]
            }
    
    def _create_todo_planning_prompt(self, user_request: str, tool_specs: Dict, context: Dict[str, Any]) -> str:
        """Create a planning prompt focused on TODO-style output."""
        
        tools_info = "\n".join([
            f"- {name}: {spec['description']}" 
            for name, spec in tool_specs.items()
        ])
        
        return f"""
You are a workflow planning assistant. Create a TODO-style execution plan for the user's request.

USER REQUEST: {user_request}

AVAILABLE TOOLS:
{tools_info}

CONTEXT: {json.dumps(context, indent=2) if context else "None"}

Create a step-by-step plan using this JSON format:

```json
{{
    "analysis": "Brief analysis of what needs to be done",
    "expected_outcome": "What the user should expect as a result",
    "execution_plan": [
        {{
            "description": "Clear description of what this step does (for TODO list)",
            "tool": "tool_name",
            "parameters": {{"param": "value"}},
            "dependencies": ["step_001", "step_002"]  // Optional: IDs of steps this depends on
        }}
    ]
}}
```

Guidelines:
1. Create clear, actionable step descriptions suitable for a TODO list
2. Break complex tasks into smaller, manageable steps
3. Specify dependencies between steps when needed
4. Use appropriate tools for each step
5. Keep steps focused and testable
6. Plan for potential failures and retries
"""
    
    async def _get_tool_specifications(self, tool_names: List[str]) -> Dict[str, Dict]:
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
                            "required": param.required
                        }
                        for param in capability.parameters
                    ]
                }
        return specs
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON content from LLM response."""
        # Look for JSON code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # Look for JSON objects
        start = response.find("{")
        end = response.rfind("}") + 1
        
        if start != -1 and end > start:
            return response[start:end]
        
        raise ValueError("No valid JSON found in response")
    
    async def _replan(self, plan: TodoPlan, reason: ReplanReason):
        """Re-evaluate and update the plan."""
        logger.info(f"🔄 Re-evaluating plan due to: {reason.value}")
        
        # Add replan to history
        plan.replan_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason.value,
            "version": plan.plan_version
        })
        
        # For now, just mark timestamp - full replanning would be more complex
        plan.last_evaluated = datetime.utcnow()
        plan.plan_version += 1
    
    def _should_retry(self, step: TodoStep, error: Exception) -> bool:
        """Determine if a step should be retried based on the error."""
        # Simple retry logic - could be made more sophisticated
        retry_errors = [
            "timeout",
            "connection",
            "temporary",
            "rate limit"
        ]
        
        error_str = str(error).lower()
        return any(retry_error in error_str for retry_error in retry_errors)
    
    async def _ensure_tools_initialized(self):
        """Ensure tools are properly initialized."""
        if not self._tools_initialized:
            # Tools should be initialized via the registry
            self._tools_initialized = True