"""Coordinator agent that manages worker agents and plans task execution.

The CoordinatorAgent is responsible for:
- Analyzing user requests and creating TodoPlans
- Discovering and selecting appropriate workers
- Delegating tasks to workers via A2A
- Monitoring task progress
- Aggregating results and replanning if needed
"""

import asyncio
from datetime import datetime
import json
import logging
from typing import Any

from aida.agents.base import AgentConfig, BaseAgent
from aida.agents.base.messages import (
    TaskPriority,
    WorkerMessageTypes,
    create_task_assignment_message,
)
from aida.agents.coordination.dispatcher import (
    DispatchStrategy,
    RetryStrategy,
    TaskDispatcher,
    TaskResult,
)
from aida.agents.coordination.plan_models import (
    ReplanReason,
    TodoPlan,
    TodoStatus,
    TodoStep,
)
from aida.agents.coordination.storage import CoordinatorPlanStorage
from aida.agents.coordination.task_reviser import ReflectionAnalyzer, TaskReviser
from aida.core.protocols.a2a import A2AMessage, AgentInfo
from aida.llm import get_llm

logger = logging.getLogger(__name__)


class WorkerCapability:
    """Information about a worker's capability."""

    def __init__(self, worker_id: str, capability: str, last_seen: float):
        self.worker_id = worker_id
        self.capability = capability
        self.last_seen = last_seen
        self.success_count = 0
        self.failure_count = 0
        self.avg_response_time = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def update_stats(self, success: bool, response_time: float) -> None:
        """Update worker statistics."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        # Update moving average of response time
        total = self.success_count + self.failure_count
        self.avg_response_time = (self.avg_response_time * (total - 1) + response_time) / total


class CoordinatorAgent(BaseAgent):
    """Coordinator agent that manages task planning and worker delegation."""

    def __init__(self, config: AgentConfig | None = None):
        """Initialize the coordinator agent.

        Args:
            config: Optional configuration, uses defaults if not provided
        """
        # Default configuration for coordinator
        default_config = AgentConfig(
            agent_id="coordinator_001",
            agent_type="coordinator",
            capabilities=[
                "planning",
                "task_delegation",
                "worker_discovery",
                "result_aggregation",
                "replanning",
            ],
            port=8100,  # Fixed port for coordinator
            max_concurrent_tasks=20,
            task_timeout_seconds=600,
        )

        super().__init__(config or default_config)

        # Worker discovery and tracking
        # Maps worker_id to either WorkerProxy (for task execution) or AgentInfo (for discovery)
        from aida.agents.coordination.worker_proxy import WorkerProxy

        self._known_workers: dict[str, WorkerProxy | AgentInfo] = {}
        self._worker_capabilities: dict[str, list[WorkerCapability]] = {}
        self._capability_index: dict[str, list[WorkerCapability]] = {}

        # Active plan tracking
        self._active_plans: dict[str, TodoPlan] = {}
        self._plan_workers: dict[str, dict[str, str]] = {}  # plan_id -> {step_id: worker_id}
        self._step_tasks: dict[str, asyncio.Task] = {}  # step_id -> monitoring task

        # Initialize dispatcher and revision components
        self._dispatcher: TaskDispatcher | None = None
        self._task_reviser = TaskReviser()
        self._reflection_analyzer = ReflectionAnalyzer()

        # Dispatcher configuration
        self._dispatch_strategy = DispatchStrategy.CAPABILITY_BASED
        self._retry_strategy = RetryStrategy.EXPONENTIAL_BACKOFF
        self._max_retries = 3
        self._task_timeout = 300.0

        # Storage manager
        storage_dir = getattr(config, "storage_dir", ".aida/coordinator/plans")
        self._storage = CoordinatorPlanStorage(storage_dir)

        # LLM for planning
        self._llm_client = None
        self._plan_counter = 0
        self._step_counter = 0

    async def _on_start(self) -> None:
        """Additional initialization after base start."""
        # Initialize LLM client
        try:
            self._llm_client = get_llm()
            logger.info("Initialized LLM client for planning")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            # Continue without LLM - can still coordinate if plans are provided

        # Start worker discovery
        self._tasks.add(asyncio.create_task(self._periodic_worker_discovery()))

        # Start plan monitoring
        self._tasks.add(asyncio.create_task(self._monitor_active_plans()))

        # Initialize dispatcher with discovered workers
        await self._update_dispatcher()

    async def handle_message(self, message: A2AMessage) -> A2AMessage | None:
        """Handle incoming A2A messages.

        Routes messages to appropriate handlers based on type.
        """
        logger.debug(
            f"Coordinator received message type: {message.message_type} from {message.sender_id}"
        )

        handlers = {
            A2AMessage.MessageTypes.TASK_REQUEST: self._handle_task_request,
            A2AMessage.MessageTypes.CAPABILITY_RESPONSE: self._handle_capability_response,
            # New worker message types
            WorkerMessageTypes.WORKER_REGISTRATION: self._handle_worker_registration,
            WorkerMessageTypes.TASK_ACCEPTANCE: self._handle_task_acceptance,
            WorkerMessageTypes.TASK_REJECTION: self._handle_task_rejection,
            WorkerMessageTypes.TASK_PROGRESS: self._handle_task_progress,
            WorkerMessageTypes.TASK_COMPLETION: self._handle_task_completion,
            WorkerMessageTypes.WORKER_STATUS: self._handle_worker_status,
            # Handle task responses from workers
            "task_response": self._handle_task_response,
            # Handle handshake for A2A protocol
            "handshake": self._handle_handshake,
        }

        handler = handlers.get(message.message_type)
        if handler:
            return await handler(message)

        # Unknown message type
        return self.create_error_response(message, f"Unknown message type: {message.message_type}")

    async def create_plan(
        self, user_request: str, context: dict[str, Any] | None = None
    ) -> TodoPlan:
        """Create a new TodoPlan for a user request.

        Args:
            user_request: The user's request to plan for
            context: Optional context information

        Returns:
            Created TodoPlan

        Raises:
            RuntimeError: If planning fails
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized - cannot create plans")

        try:
            # Generate plan using LLM
            plan_data = await self._generate_plan_with_llm(user_request, context)

            # Create TodoStep objects
            steps = []
            step_ids = []  # Track step IDs for dependency mapping

            for i, step_data in enumerate(plan_data.get("execution_plan", [])):
                step_id = f"step_{self._step_counter:04d}"
                self._step_counter += 1
                step_ids.append(step_id)

                # Convert integer dependencies to step IDs
                dependencies = []
                for dep in step_data.get("dependencies", []):
                    if isinstance(dep, int) and 0 <= dep < i:
                        # Convert index to step ID
                        dependencies.append(step_ids[dep])
                    elif isinstance(dep, str):
                        # Already a string ID
                        dependencies.append(dep)

                step = TodoStep(
                    id=step_id,
                    description=step_data["description"],
                    tool_name=step_data.get("capability", "general"),
                    parameters=step_data.get("parameters", {}),
                    dependencies=dependencies,
                    max_retries=3,
                )
                steps.append(step)

            # Create plan
            self._plan_counter += 1
            plan = TodoPlan(
                id=f"plan_{self._plan_counter:04d}_{datetime.utcnow().timestamp():.0f}",
                user_request=user_request,
                analysis=plan_data.get("analysis", ""),
                expected_outcome=plan_data.get("expected_outcome", ""),
                steps=steps,
                context=context or {},
            )

            # Store plan in memory and persist to disk
            self._active_plans[plan.id] = plan
            self._plan_workers[plan.id] = {}
            self._storage.save_plan(plan)

            logger.info(f"Created plan {plan.id} with {len(steps)} steps")
            return plan

        except Exception as e:
            logger.error(f"Failed to create plan: {e}")
            raise RuntimeError(f"Plan creation failed: {e}") from e

    async def execute_plan(self, plan: TodoPlan) -> dict[str, Any]:
        """Execute a TodoPlan by delegating to workers using TaskDispatcher.

        Args:
            plan: The plan to execute

        Returns:
            Execution results
        """
        logger.info(f"Starting execution of plan {plan.id}")

        # Ensure we have current worker information and dispatcher
        await self._discover_workers()
        await self._update_dispatcher()

        if not self._dispatcher:
            logger.error("No dispatcher available - no workers found")
            return {
                "plan_id": plan.id,
                "status": "failed",
                "error": "No workers available",
                "results": [],
            }

        results = []
        step_results = {}  # Track results for each step

        while True:
            # Check if we should replan
            should_replan, reason = plan.should_replan()
            if should_replan and reason != ReplanReason.PERIODIC_CHECK:
                logger.info(f"Replanning needed: {reason.value}")
                await self._replan(plan, reason)

            # Save current state periodically
            self._storage.save_plan(plan)

            # Get next executable step
            next_step = plan.get_next_executable_step()
            if not next_step:
                # Check if we're done or stuck
                progress = plan.get_progress()
                if plan.status == "completed":
                    logger.info(f"Plan {plan.id} completed successfully")
                    # Run reflection analysis
                    await self._run_reflection_analysis(plan, step_results)
                    # Save final state and archive
                    self._storage.save_plan(plan, "archived")
                    break
                elif plan.status in ["failed", "partial_failure"]:
                    logger.error(f"Plan {plan.id} execution failed")
                    # Save final state to failed directory
                    self._storage.save_plan(plan, "failed")
                    break
                else:
                    # Wait for in-progress steps
                    await asyncio.sleep(1)
                    continue

            # Execute step using dispatcher
            try:
                # Mark as in progress
                next_step.status = TodoStatus.IN_PROGRESS
                next_step.started_at = datetime.utcnow()

                # Dispatch with retry logic
                result = await self._dispatcher.dispatch(
                    step=next_step,
                    context={"plan_id": plan.id, "plan_context": plan.context},
                    task_id=f"{plan.id}_{next_step.id}",
                    enable_revision=True,
                )

                # Store result
                step_results[next_step.id] = result

                if result.success:
                    next_step.status = TodoStatus.COMPLETED
                    next_step.result = result.result
                    next_step.completed_at = datetime.utcnow()
                    logger.info(f"Step {next_step.id} completed successfully")
                else:
                    # Check if we should revise and retry
                    # The dispatcher has exhausted its retries, but we can still try revision
                    if result.retriable and self._task_reviser:
                        logger.info(f"Attempting to revise failed step {next_step.id}")
                        # Try to revise the task
                        revision = await self._task_reviser.revise_task(
                            step=next_step,
                            failure_result=result,
                            context=plan.context,
                        )

                        if revision and revision.confidence > 0.6:
                            # Apply revision and retry
                            revised_step = self._task_reviser.apply_revision(next_step, revision)
                            # Replace step in plan
                            for i, step in enumerate(plan.steps):
                                if step.id == next_step.id:
                                    plan.steps[i] = revised_step
                                    break
                            logger.info(
                                f"Applied revision to step {next_step.id}: {revision.approach_changes}"
                            )
                            # Reset status to retry
                            revised_step.status = TodoStatus.PENDING
                            continue

                    # Mark as failed
                    next_step.status = TodoStatus.FAILED
                    next_step.error = result.error
                    next_step.completed_at = datetime.utcnow()
                    logger.error(f"Step {next_step.id} failed: {result.error}")

            except Exception as e:
                logger.error(f"Error executing step {next_step.id}: {e}")
                next_step.status = TodoStatus.FAILED
                next_step.error = str(e)
                next_step.completed_at = datetime.utcnow()

        # Collect final results
        for step in plan.steps:
            if step.result:
                results.append(
                    {
                        "step_id": step.id,
                        "description": step.description,
                        "status": step.status.value,
                        "result": step.result,
                    }
                )

        # Get dispatcher metrics
        metrics = self._dispatcher.get_metrics() if self._dispatcher else {}

        return {
            "plan_id": plan.id,
            "status": plan.status,
            "results": results,
            "summary": plan.get_progress(),
            "step_results": {k: v.to_dict() for k, v in step_results.items()},
            "metrics": metrics,
        }

    async def _update_dispatcher(self) -> None:
        """Update the task dispatcher with current workers."""
        # Get all known workers
        workers = list(self._known_workers.values())

        if workers:
            # Create new dispatcher with current workers
            self._dispatcher = TaskDispatcher(
                agents=workers,
                dispatch_strategy=self._dispatch_strategy,
                retry_strategy=self._retry_strategy,
                max_retries=self._max_retries,
                timeout=self._task_timeout,
            )
            logger.info(f"Updated dispatcher with {len(workers)} workers")
        else:
            logger.warning("No workers available for dispatcher")
            self._dispatcher = None

    async def _run_reflection_analysis(
        self, plan: TodoPlan, step_results: dict[str, TaskResult]
    ) -> None:
        """Run reflection analysis on completed plan."""
        try:
            # Prepare plan summary
            plan_summary = {
                "plan_id": plan.id,
                "user_request": plan.user_request,
                "total_steps": len(plan.steps),
                "completed_steps": sum(1 for s in plan.steps if s.status == TodoStatus.COMPLETED),
                "failed_steps": sum(1 for s in plan.steps if s.status == TodoStatus.FAILED),
                "execution_time": (
                    (plan.last_updated - plan.created_at).total_seconds()
                    if plan.last_updated
                    else 0
                ),
            }

            # Get dispatcher metrics
            metrics = self._dispatcher.get_metrics() if self._dispatcher else {}

            # Run analysis
            analysis = await self._reflection_analyzer.analyze_execution(
                plan_summary=plan_summary,
                metrics=metrics,
            )

            # Store analysis results
            if not hasattr(plan, "reflection_analysis"):
                plan.reflection_analysis = []

            plan.reflection_analysis.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "analysis": analysis,
                }
            )

            logger.info(f"Completed reflection analysis for plan {plan.id}")

        except Exception as e:
            logger.error(f"Failed to run reflection analysis: {e}")

    async def _handle_handshake(self, message: A2AMessage) -> A2AMessage:
        """Handle A2A protocol handshake."""
        # Respond with our capabilities
        return A2AMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type="handshake",
            payload={"capabilities": self.capabilities, "agent_type": self.agent_type},
        )

    async def _handle_task_response(self, message: A2AMessage) -> A2AMessage | None:
        """Handle task response from worker and route to proxy."""
        worker_id = message.sender_id

        # Find the proxy for this worker
        proxy = self._known_workers.get(worker_id)
        if proxy and hasattr(proxy, "handle_worker_response"):
            # Route the response to the proxy
            proxy.handle_worker_response(message)
        else:
            logger.warning(f"Received task response from unknown worker: {worker_id}")

        return None

    async def _handle_task_request(self, message: A2AMessage) -> A2AMessage:
        """Handle task request from user or another agent."""
        try:
            user_request = message.payload.get("request", "")
            context = message.payload.get("context", {})

            # Create plan
            plan = await self.create_plan(user_request, context)

            # Start execution in background
            self._tasks.add(asyncio.create_task(self.execute_plan(plan)))

            return self.create_response(
                message,
                {
                    "status": "accepted",
                    "plan_id": plan.id,
                    "plan_summary": plan.to_markdown(),
                    "step_count": len(plan.steps),
                },
            )

        except Exception as e:
            return self.create_error_response(message, str(e))

    async def _handle_worker_registration(self, message: A2AMessage) -> A2AMessage:
        """Handle worker registration."""
        payload = message.payload
        worker_id = payload["worker_id"]
        worker_type = payload["worker_type"]
        capabilities = payload["capabilities"]

        # Create a proxy for the remote worker
        from aida.agents.coordination.worker_proxy import WorkerProxy

        proxy = WorkerProxy(
            worker_id=worker_id,
            capabilities=capabilities,
            coordinator_agent=self,
        )

        # Store the proxy as the "worker" for the dispatcher
        self._known_workers[worker_id] = proxy

        # Update capability tracking
        self._worker_capabilities[worker_id] = []

        for capability in capabilities:
            worker_cap = WorkerCapability(
                worker_id=worker_id,
                capability=capability,
                last_seen=asyncio.get_event_loop().time(),
            )

            self._worker_capabilities[worker_id].append(worker_cap)

            if capability not in self._capability_index:
                self._capability_index[capability] = []
            self._capability_index[capability].append(worker_cap)

        logger.info(
            f"Registered {worker_type} worker {worker_id} with capabilities: {capabilities}"
        )

        # Send acknowledgment
        return self.create_response(
            message,
            {
                "status": "registered",
                "coordinator_id": self.agent_id,
                "message": f"Worker {worker_id} successfully registered",
            },
        )

    async def _handle_task_acceptance(self, message: A2AMessage) -> A2AMessage | None:
        """Handle task acceptance from worker."""
        payload = message.payload
        task_id = payload["task_id"]
        worker_id = payload["worker_id"]

        logger.info(f"Worker {worker_id} accepted task {task_id}")

        # Find the plan and step
        for plan_id, plan in self._active_plans.items():
            for step in plan.steps:
                if step.id == task_id:
                    # Confirm task assignment
                    step.status = TodoStatus.IN_PROGRESS
                    self._plan_workers[plan_id][task_id] = worker_id
                    return None

        logger.warning(f"Received acceptance for unknown task {task_id}")
        return None

    async def _handle_task_rejection(self, message: A2AMessage) -> A2AMessage | None:
        """Handle task rejection from worker."""
        payload = message.payload
        task_id = payload["task_id"]
        worker_id = payload["worker_id"]
        reason = payload["reason"]

        logger.warning(f"Worker {worker_id} rejected task {task_id}: {reason}")

        # Find the plan and step
        for plan_id, plan in self._active_plans.items():
            for step in plan.steps:
                if step.id == task_id:
                    # Mark step as pending again for reassignment
                    step.status = TodoStatus.PENDING
                    if plan_id in self._plan_workers and task_id in self._plan_workers[plan_id]:
                        del self._plan_workers[plan_id][task_id]

                    # Update worker stats negatively
                    self._update_worker_stats(worker_id, step.tool_name, False, 0)
                    return None

        return None

    async def _handle_task_progress(self, message: A2AMessage) -> A2AMessage | None:
        """Handle task progress update from worker."""
        payload = message.payload
        task_id = payload["task_id"]
        worker_id = payload["worker_id"]
        progress = payload["progress_percentage"]
        status_message = payload["status_message"]

        logger.info(f"Task {task_id} progress: {progress}% - {status_message}")

        # In a full implementation, would update step progress tracking
        # and potentially notify interested parties
        return None

    async def _handle_task_completion(self, message: A2AMessage) -> A2AMessage:
        """Handle task completion from worker."""
        payload = message.payload
        task_id = payload["task_id"]
        worker_id = payload["worker_id"]
        success = payload["success"]
        result = payload.get("result", {})
        error = payload.get("error")
        execution_time = payload["execution_time_seconds"]

        # Find the plan and step
        step_found = False
        for plan_id, plan in self._active_plans.items():
            for step in plan.steps:
                if step.id == task_id:
                    step_found = True

                    if success:
                        step.status = TodoStatus.COMPLETED
                        step.result = result
                        step.completed_at = datetime.utcnow()

                        # Save updated plan state
                        self._storage.save_plan(plan)

                        # Update worker stats
                        self._update_worker_stats(worker_id, step.tool_name, True, execution_time)

                        logger.info(
                            f"Task {task_id} completed successfully by {worker_id} "
                            f"in {execution_time:.1f}s"
                        )
                    else:
                        step.status = TodoStatus.FAILED
                        step.error = error or "Unknown error"
                        step.completed_at = datetime.utcnow()

                        # Update worker stats
                        self._update_worker_stats(worker_id, step.tool_name, False, execution_time)

                        logger.error(f"Task {task_id} failed: {step.error}")

                    # Update plan timestamp
                    plan.last_updated = datetime.utcnow()
                    break

        if not step_found:
            logger.warning(f"Received completion for unknown task {task_id}")

        # Send acknowledgment
        return self.create_response(
            message,
            {"status": "acknowledged", "task_id": task_id, "message": "Task completion recorded"},
        )

    async def _handle_worker_status(self, message: A2AMessage) -> A2AMessage | None:
        """Handle worker status update."""
        payload = message.payload
        worker_id = payload["worker_id"]
        state = payload["state"]

        # Update last seen time
        if worker_id in self._known_workers:
            self._known_workers[worker_id].last_seen = asyncio.get_event_loop().time()
            self._known_workers[worker_id].status = state

        logger.debug(f"Worker {worker_id} status: {state}")
        return None

    async def _handle_task_status(self, message: A2AMessage) -> A2AMessage | None:
        """Handle task status update from worker."""
        # Similar to task response but for progress updates
        status = message.payload.get("status", "")
        progress = message.payload.get("progress", 0)

        logger.info(f"Received status update from {message.sender_id}: {status} ({progress}%)")

        # In a full implementation, would update step progress tracking
        return None

    async def _handle_capability_response(self, message: A2AMessage) -> A2AMessage | None:
        """Handle capability response from discovered worker."""
        worker_id = message.sender_id
        capabilities = message.payload.get("capabilities", [])
        agent_type = message.payload.get("agent_type", "unknown")

        # Check if we already have a WorkerProxy for this worker
        if worker_id in self._known_workers and hasattr(
            self._known_workers[worker_id], "handle_message"
        ):
            # We already have a proxy, just update its capabilities if needed
            logger.debug(f"Updating capabilities for existing worker proxy {worker_id}")
            # Update the proxy's capabilities
            if hasattr(self._known_workers[worker_id], "capabilities"):
                self._known_workers[worker_id].capabilities = capabilities
        else:
            # No proxy exists, this might be from discovery before registration
            # Store as AgentInfo temporarily until proper registration
            logger.debug(
                f"Storing AgentInfo for discovered worker {worker_id} (awaiting registration)"
            )
            self._known_workers[worker_id] = AgentInfo(
                agent_id=worker_id,
                capabilities=capabilities,
                endpoint=f"ws://{message.sender_id}",  # Simplified - would get from protocol
                last_seen=asyncio.get_event_loop().time(),
            )

        # Update capability tracking
        self._worker_capabilities[worker_id] = []

        for capability in capabilities:
            worker_cap = WorkerCapability(
                worker_id=worker_id,
                capability=capability,
                last_seen=asyncio.get_event_loop().time(),
            )

            self._worker_capabilities[worker_id].append(worker_cap)

            # Update capability index
            if capability not in self._capability_index:
                self._capability_index[capability] = []
            self._capability_index[capability].append(worker_cap)

        logger.info(f"Discovered {agent_type} worker {worker_id} with capabilities: {capabilities}")

        return None

    async def _generate_plan_with_llm(
        self, user_request: str, context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Generate a plan using the LLM."""
        # Get available capabilities
        available_capabilities = list(self._capability_index.keys())

        # If no capabilities available yet, use a default based on context
        if not available_capabilities:
            if context and "required_capability" in context:
                available_capabilities = [context["required_capability"]]
            else:
                # Default to common capabilities
                available_capabilities = ["code_analysis", "code_generation"]

        # For simple single-capability tasks, create a deterministic plan
        if context and context.get("required_capability") and context.get("task_type"):
            return self._create_simple_plan(user_request, context, available_capabilities)

        # For complex requests, use LLM with structured prompt
        prompt = self._create_planning_prompt(user_request, context, available_capabilities)

        try:
            # The LLM manager expects a string, not a list of messages
            response = await self._llm_client.chat(prompt)

            logger.debug(f"LLM raw response: {response}")

            # Extract and validate JSON
            plan_data = self._extract_and_validate_json(response, available_capabilities)

            return plan_data

        except Exception as e:
            logger.error(f"LLM planning failed: {e}")
            logger.error(f"Prompt was: {prompt}")
            raise RuntimeError(f"Failed to generate plan with LLM: {e}")

    def _create_simple_plan(
        self, user_request: str, context: dict[str, Any], available_capabilities: list[str]
    ) -> dict[str, Any]:
        """Create a simple single-step plan for straightforward tasks."""
        capability = context["required_capability"]
        task_type = context["task_type"]

        # Extract parameters from context
        parameters = {}
        param_keys = [
            "file_path",
            "specification",
            "language",
            "code",
            "objectives",
            "framework",
            "format",
        ]
        for key in param_keys:
            if key in context:
                parameters[key] = context[key]

        return {
            "analysis": f"User requested {task_type} operation",
            "expected_outcome": f"Complete {task_type} with results",
            "execution_plan": [
                {
                    "description": user_request,
                    "capability": capability,
                    "parameters": parameters,
                    "dependencies": [],
                }
            ],
        }

    def _create_planning_prompt(
        self, user_request: str, context: dict[str, Any] | None, available_capabilities: list[str]
    ) -> str:
        """Create a structured prompt for the LLM."""
        return f"""You are a task planning assistant. Create a workflow plan.

USER REQUEST: {user_request}

AVAILABLE CAPABILITIES: {", ".join(available_capabilities)}

CONTEXT: {json.dumps(context, indent=2) if context else "{}"}

Respond ONLY with valid JSON:

{{
    "analysis": "Brief analysis of what needs to be done",
    "expected_outcome": "What the user should expect as a result",
    "execution_plan": [
        {{
            "description": "Clear description of what this step does",
            "capability": "capability_name_here",
            "parameters": {{"param1": "value1"}},
            "dependencies": []
        }}
    ]
}}

RULES:
1. Use ONLY capabilities from AVAILABLE CAPABILITIES list
2. Create 1-3 steps maximum
3. Include ALL relevant parameters from CONTEXT in step parameters
4. Each step must have: description, capability, parameters, dependencies
5. Respond with ONLY the JSON object, no other text

Example:
{{
    "analysis": "User wants to analyze Python code for quality issues",
    "expected_outcome": "Code analysis report with metrics and suggestions",
    "execution_plan": [
        {{
            "description": "Analyze code structure and quality",
            "capability": "code_analysis",
            "parameters": {{"file_path": "/path/to/file.py", "detailed_analysis": true}},
            "dependencies": []
        }}
    ]
}}"""

    def _extract_and_validate_json(
        self, response: str, available_capabilities: list[str]
    ) -> dict[str, Any]:
        """Extract JSON from LLM response and validate it."""
        if not isinstance(response, str):
            raise ValueError(f"Expected string response from LLM, got {type(response)}")

        # Extract JSON object
        json_str = self._extract_json_object(response)

        # Parse JSON
        try:
            plan_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"JSON string: {json_str[:200]}...")
            raise ValueError(f"Invalid JSON in LLM response: {e}")

        # Validate structure
        self._validate_plan_structure(plan_data, available_capabilities)

        return plan_data

    def _extract_json_object(self, text: str) -> str:
        """Extract the main JSON object from text."""
        if "{" not in text:
            raise ValueError("No JSON object found in response")

        # Find the main JSON object
        brace_count = 0
        start_idx = None
        end_idx = None

        for i, char in enumerate(text):
            if char == "{":
                if start_idx is None:
                    start_idx = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    end_idx = i + 1
                    break

        if start_idx is not None and end_idx is not None:
            return text[start_idx:end_idx]
        else:
            raise ValueError("Could not extract complete JSON object")

    def _validate_plan_structure(
        self, plan_data: dict[str, Any], available_capabilities: list[str]
    ) -> None:
        """Validate the plan data structure."""
        # Check required fields
        required_fields = ["analysis", "expected_outcome", "execution_plan"]
        missing_fields = [field for field in required_fields if field not in plan_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Validate each step
        for i, step in enumerate(plan_data.get("execution_plan", [])):
            step_required = ["description", "capability", "parameters"]
            step_missing = [field for field in step_required if field not in step]
            if step_missing:
                raise ValueError(f"Step {i} missing required fields: {step_missing}")

            # Check capability is valid
            if step["capability"] not in available_capabilities:
                raise ValueError(f"Step {i} has invalid capability: {step['capability']}")

    # Storage management methods

    def load_plan(self, plan_id: str) -> TodoPlan | None:
        """Load a plan from storage.

        Args:
            plan_id: The plan ID to load

        Returns:
            TodoPlan if found, None otherwise
        """
        # Check if already in memory
        if plan_id in self._active_plans:
            return self._active_plans[plan_id]

        # Load from storage
        plan = self._storage.load_plan(plan_id)
        if plan:
            # Add to active plans if it's not completed
            progress = plan.get_progress()
            if plan.status not in ["completed", "failed"]:
                self._active_plans[plan_id] = plan
                self._plan_workers[plan_id] = {}

        return plan

    def list_stored_plans(self, status: str | None = None) -> list[dict[str, Any]]:
        """List all stored plans with metadata.

        Args:
            status: Filter by status (active, archived, failed)

        Returns:
            List of plan metadata
        """
        return self._storage.list_plans(status)

    def get_plan_summary(self, plan_id: str) -> dict[str, Any] | None:
        """Get a summary of a specific plan.

        Args:
            plan_id: The plan ID

        Returns:
            Plan summary or None if not found
        """
        plan = self.load_plan(plan_id)
        if not plan:
            return None

        return plan.get_summary_stats()

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics.

        Returns:
            Storage statistics
        """
        stats = self._storage.get_storage_stats()

        # Add active plan info
        stats["active_in_memory"] = len(self._active_plans)
        stats["active_workers"] = len(self._known_workers)

        return stats

    def archive_completed_plans(self) -> int:
        """Archive all completed plans.

        Returns:
            Number of plans archived
        """
        # First archive from memory
        archived_from_memory = 0
        for plan_id, plan in list(self._active_plans.items()):
            if plan.status == "completed":
                self._storage.save_plan(plan, "archived")
                del self._active_plans[plan_id]
                if plan_id in self._plan_workers:
                    del self._plan_workers[plan_id]
                archived_from_memory += 1

        # Then archive from active directory
        archived_from_disk = self._storage.archive_completed_plans()

        total_archived = archived_from_memory + archived_from_disk
        logger.info(
            f"Archived {total_archived} plans ({archived_from_memory} from memory, {archived_from_disk} from disk)"
        )

        return total_archived

    def cleanup_old_plans(self, days_old: int = 30) -> int:
        """Clean up old plans.

        Args:
            days_old: Age threshold in days

        Returns:
            Number of plans deleted
        """
        return self._storage.cleanup_old_plans(days_old)

    def export_plan_report(self, output_file: str | None = None) -> str:
        """Export a summary report of all plans.

        Args:
            output_file: Output filename

        Returns:
            Path to the exported file
        """
        return self._storage.export_summary_report(output_file)

    async def _select_worker_for_step(self, step: TodoStep) -> str | None:
        """Select the best worker for a given step.

        Selection criteria:
        1. Has the required capability
        2. Is currently available (recently seen)
        3. Has good success rate
        4. Has low average response time
        """
        capability = step.tool_name

        # Get workers with this capability
        candidates = self._capability_index.get(capability, [])
        if not candidates:
            logger.warning(f"No workers found with capability: {capability}")
            return None

        # Filter by recency (seen in last 5 minutes)
        current_time = asyncio.get_event_loop().time()
        active_candidates = [c for c in candidates if current_time - c.last_seen < 300]

        if not active_candidates:
            logger.warning(f"No recently active workers for capability: {capability}")
            return None

        # Score candidates
        scored = []
        for candidate in active_candidates:
            # Prefer high success rate and low response time
            score = candidate.success_rate * 100
            if candidate.avg_response_time > 0:
                score -= candidate.avg_response_time

            scored.append((score, candidate))

        # Sort by score (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)

        selected = scored[0][1]
        logger.info(
            f"Selected worker {selected.worker_id} for capability {capability} "
            f"(success rate: {selected.success_rate:.2%}, "
            f"avg time: {selected.avg_response_time:.1f}s)"
        )

        return selected.worker_id

    async def _delegate_step_to_worker(self, plan_id: str, step: TodoStep, worker_id: str) -> bool:
        """Send task request to worker.

        Returns:
            True if successfully sent
        """
        # Use the new message creation helper
        message_data = create_task_assignment_message(
            coordinator_id=self.agent_id,
            worker_id=worker_id,
            task_id=step.id,
            plan_id=plan_id,
            step_description=step.description,
            capability=step.tool_name,
            parameters=step.parameters,
            dependencies=step.dependencies,
            timeout_seconds=self.config.task_timeout_seconds,
            priority=TaskPriority.HIGH if step.retry_count > 0 else TaskPriority.NORMAL,
            retry_count=step.retry_count,
            max_retries=step.max_retries,
        )

        task_message = A2AMessage(**message_data)
        logger.debug(
            f"Sending task assignment: type={task_message.message_type}, recipient={task_message.recipient_id}"
        )
        success = await self.send_message(task_message)

        if success:
            logger.info(f"Delegated step {step.id} to worker {worker_id}")
        else:
            logger.error(f"Failed to delegate step {step.id} to worker {worker_id}")

        return success

    async def _discover_workers(self) -> None:
        """Discover available workers."""
        discovery_message = A2AMessage(
            sender_id=self.agent_id,
            message_type=A2AMessage.MessageTypes.CAPABILITY_DISCOVERY,
            payload={"seeking": "workers", "coordinator": self.agent_id},
        )

        # Broadcast discovery request
        await self.send_message(discovery_message)

        # Also use A2A protocol's discovery
        discovered = await self.a2a_protocol.discover_agents()

        for agent_info in discovered:
            if agent_info.agent_id != self.agent_id:
                # Request capabilities from discovered agent
                cap_request = A2AMessage(
                    sender_id=self.agent_id,
                    recipient_id=agent_info.agent_id,
                    message_type=A2AMessage.MessageTypes.CAPABILITY_DISCOVERY,
                    payload={"request": "capabilities"},
                )
                await self.send_message(cap_request)

    async def _periodic_worker_discovery(self) -> None:
        """Periodically discover workers."""
        while self._running:
            try:
                await self._discover_workers()
                await asyncio.sleep(60)  # Every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker discovery: {e}")
                await asyncio.sleep(60)

    async def _monitor_active_plans(self) -> None:
        """Monitor active plans for timeout and cleanup."""
        while self._running:
            try:
                current_time = datetime.utcnow()

                for plan_id, plan in list(self._active_plans.items()):
                    # Check for stalled plans
                    last_activity = plan.last_updated
                    if (current_time - last_activity).total_seconds() > 3600:  # 1 hour
                        logger.warning(f"Plan {plan_id} appears stalled, archiving")
                        # Move to failed directory
                        self._storage.move_plan(plan_id, "failed")
                        del self._active_plans[plan_id]
                        del self._plan_workers[plan_id]
                        continue

                    # Check for completed plans
                    progress = plan.get_progress()
                    if plan.status in ["completed", "failed"]:
                        # Keep completed plans for 5 minutes
                        if (current_time - last_activity).total_seconds() > 300:
                            logger.info(f"Archiving completed plan {plan_id}")
                            # Archive completed plans
                            if plan.status == "completed":
                                self._storage.move_plan(plan_id, "archived")
                            else:
                                self._storage.move_plan(plan_id, "failed")
                            del self._active_plans[plan_id]
                            del self._plan_workers[plan_id]

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in plan monitoring: {e}")
                await asyncio.sleep(30)

    def _update_worker_stats(
        self, worker_id: str, capability: str, success: bool, response_time: float
    ) -> None:
        """Update worker performance statistics."""
        if worker_id in self._worker_capabilities:
            for worker_cap in self._worker_capabilities[worker_id]:
                if worker_cap.capability == capability:
                    worker_cap.update_stats(success, response_time)
                    break

    async def _replan(self, plan: TodoPlan, reason: ReplanReason) -> None:
        """Replan based on failures or new information."""
        logger.info(f"Replanning {plan.id} due to: {reason.value}")

        # For now, simple strategy: mark failed steps for retry
        for step in plan.steps:
            if step.status == TodoStatus.FAILED and step.retry_count < step.max_retries:
                step.status = TodoStatus.PENDING
                step.retry_count += 1
                logger.info(f"Retrying step {step.id} (attempt {step.retry_count + 1})")

        # Update plan version
        plan.plan_version += 1
        plan.last_evaluated = datetime.utcnow()

        # Record replan event
        plan.replan_history.append(
            {
                "version": plan.plan_version,
                "reason": reason.value,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
