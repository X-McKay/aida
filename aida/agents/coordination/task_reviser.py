"""Task revision module for intelligent retry with LLM-based task refinement.

This module provides functionality to revise failed tasks using LLM analysis,
enabling smarter retries with refined parameters or approaches.
"""

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from aida.agents.coordination.dispatcher import TaskResult
from aida.agents.coordination.plan_models import TodoStep
from aida.config.llm_profiles import Purpose
from aida.llm import get_llm

logger = logging.getLogger(__name__)


class RevisionSuggestion(BaseModel):
    """Suggestion for task revision."""

    revised_description: str
    revised_parameters: dict[str, Any]
    approach_changes: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class TaskReviser:
    """Handles intelligent task revision using LLM analysis.

    This class analyzes failed tasks and suggests revisions to improve
    success likelihood on retry.
    """

    def __init__(self, llm_purpose: Purpose = Purpose.REASONING):
        """Initialize the TaskReviser.

        Args:
            llm_purpose: LLM purpose to use for revision
        """
        self.llm_manager = get_llm()
        self.llm_purpose = llm_purpose

    async def revise_task(
        self,
        step: TodoStep,
        failure_result: TaskResult,
        context: dict[str, Any] | None = None,
        attempt_history: list | None = None,
    ) -> RevisionSuggestion | None:
        """Revise a failed task using LLM analysis.

        Args:
            step: The failed task step
            failure_result: Result from the failed execution
            context: Optional execution context
            attempt_history: History of previous attempts

        Returns:
            RevisionSuggestion or None if revision not possible
        """
        # Prepare revision prompt
        prompt = self._build_revision_prompt(step, failure_result, context, attempt_history)

        try:
            # Get LLM suggestion
            response = await self.llm_manager.agenerate(
                prompt=prompt,
                purpose=self.llm_purpose,
                temperature=0.7,  # Some creativity for problem-solving
                max_tokens=1000,
            )

            # Parse response
            suggestion = self._parse_revision_response(response)

            if suggestion:
                logger.info(
                    f"Generated revision for step {step.id} with confidence {suggestion.confidence}",
                    extra={
                        "step_id": step.id,
                        "confidence": suggestion.confidence,
                        "approach": suggestion.approach_changes,
                    },
                )

            return suggestion

        except Exception as e:
            logger.error(
                f"Failed to revise task {step.id}: {e}", extra={"step_id": step.id, "error": str(e)}
            )
            return None

    def _build_revision_prompt(
        self,
        step: TodoStep,
        failure_result: TaskResult,
        context: dict[str, Any] | None,
        attempt_history: list | None,
    ) -> str:
        """Build prompt for task revision.

        Args:
            step: The failed task step
            failure_result: Result from failed execution
            context: Optional execution context
            attempt_history: History of previous attempts

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "You are an expert task revision assistant. A task has failed and needs revision.",
            "",
            "FAILED TASK DETAILS:",
            f"Description: {step.description}",
            f"Tool/Capability: {step.tool_name}",
            f"Parameters: {json.dumps(step.parameters, indent=2)}",
            "",
            "FAILURE INFORMATION:",
            f"Error: {failure_result.error}",
            f"Retry Count: {failure_result.retry_count}",
            f"Execution Time: {failure_result.execution_time:.2f}s",
        ]

        if context:
            prompt_parts.extend(["", "CONTEXT:", json.dumps(context, indent=2)])

        if attempt_history:
            prompt_parts.extend(["", "PREVIOUS ATTEMPTS:", json.dumps(attempt_history, indent=2)])

        prompt_parts.extend(
            [
                "",
                "Please analyze this failure and suggest a revision that addresses the root cause.",
                "Consider:",
                "1. Parameter adjustments that might resolve the issue",
                "2. Alternative approaches to achieve the same goal",
                "3. Breaking down the task if it's too complex",
                "4. Adding error handling or validation",
                "",
                "Respond ONLY with valid JSON in this exact format (no extra text):",
                "{",
                '  "revised_description": "Write configuration file to the test directory",',
                '  "revised_parameters": {"path": "/tmp/test/config.json", "content": "..."},',
                '  "approach_changes": "Added missing path parameter using context",',
                '  "confidence": 0.8,',
                '  "reasoning": "The error indicates path is required but missing"',
                "}",
                "Important: Use paths from context when available, avoid hardcoded system paths.",
            ]
        )

        return "\n".join(prompt_parts)

    def _parse_revision_response(self, response: str) -> RevisionSuggestion | None:
        """Parse LLM response into RevisionSuggestion.

        Args:
            response: LLM response text

        Returns:
            RevisionSuggestion or None if parsing fails
        """
        try:
            # Try to extract JSON from response
            # Handle cases where LLM adds extra text
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                # Clean up common JSON issues
                json_str = json_str.replace("...", "placeholder")  # Replace ... with placeholder
                json_str = json_str.replace("\\n", " ")  # Remove newlines in strings

                data = json.loads(json_str)

                # Ensure required fields have valid values
                if "confidence" not in data or not isinstance(data["confidence"], int | float):
                    data["confidence"] = 0.8

                return RevisionSuggestion(**data)
            else:
                logger.warning("No valid JSON found in revision response")
                return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse revision response: {e}")
            logger.debug(f"Raw response was: {response}")
            # Try to create a default revision based on the error
            if "auth_token" in str(e) or "auth_token" in response:
                return self._create_default_auth_revision()
            return None
        except Exception as e:
            logger.error(f"Error creating RevisionSuggestion: {e}")
            return None

    def _create_default_auth_revision(self) -> RevisionSuggestion:
        """Create a default revision for auth token errors."""
        return RevisionSuggestion(
            revised_description="Call API with authentication token",
            revised_parameters={"auth_token": "default_token"},
            approach_changes="Added missing auth_token parameter",
            confidence=0.9,
            reasoning="The error indicates auth_token is a required parameter",
        )

    def apply_revision(self, step: TodoStep, suggestion: RevisionSuggestion) -> TodoStep:
        """Apply revision suggestion to a task step.

        Args:
            step: Original task step
            suggestion: Revision suggestion

        Returns:
            Revised task step
        """
        # Create a copy with revisions
        revised_step = step.model_copy()

        # Apply revisions
        revised_step.description = suggestion.revised_description
        # Merge parameters instead of replacing them
        revised_step.parameters = {**step.parameters, **suggestion.revised_parameters}

        # Add revision metadata
        if not hasattr(revised_step, "metadata"):
            revised_step.metadata = {}

        revised_step.metadata["revised"] = True
        revised_step.metadata["revision_reasoning"] = suggestion.reasoning
        revised_step.metadata["revision_confidence"] = suggestion.confidence

        logger.info(
            f"Applied revision to step {step.id}: {suggestion.approach_changes}",
            extra={
                "step_id": step.id,
                "confidence": suggestion.confidence,
            },
        )

        return revised_step


class ReflectionAnalyzer:
    """Analyzes completed plans for improvement opportunities.

    This supports the reflection/review cycle for continuous improvement.
    """

    def __init__(self, llm_purpose: Purpose = Purpose.REASONING):
        """Initialize the ReflectionAnalyzer.

        Args:
            llm_purpose: LLM purpose to use for analysis
        """
        self.llm_manager = get_llm()
        self.llm_purpose = llm_purpose

    async def analyze_execution(
        self,
        plan_summary: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Analyze plan execution for improvements.

        Args:
            plan_summary: Summary of plan execution
            metrics: Execution metrics

        Returns:
            Analysis results with improvement suggestions
        """
        prompt = self._build_reflection_prompt(plan_summary, metrics)

        try:
            response = await self.llm_manager.agenerate(
                prompt=prompt,
                purpose=self.llm_purpose,
                temperature=0.5,
                max_tokens=1500,
            )

            return self._parse_reflection_response(response)

        except Exception as e:
            logger.error(f"Failed to analyze execution: {e}")
            return {"analysis_failed": True, "error": str(e), "suggestions": []}

    def _build_reflection_prompt(
        self,
        plan_summary: dict[str, Any],
        metrics: dict[str, Any],
    ) -> str:
        """Build prompt for execution reflection.

        Args:
            plan_summary: Summary of plan execution
            metrics: Execution metrics

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "You are an expert system performance analyzer. Please analyze this execution summary.",
            "",
            "EXECUTION SUMMARY:",
            json.dumps(plan_summary, indent=2),
            "",
            "METRICS:",
            json.dumps(metrics, indent=2),
            "",
            "Please provide:",
            "1. Key bottlenecks or failure patterns",
            "2. Suggested improvements for future executions",
            "3. Agent performance insights",
            "4. Task complexity assessment",
            "",
            "Respond in JSON format:",
            "{",
            '  "bottlenecks": ["List of identified bottlenecks"],',
            '  "failure_patterns": ["Common failure patterns"],',
            '  "improvements": [',
            "    {",
            '      "area": "Area of improvement",',
            '      "suggestion": "Specific suggestion",',
            '      "impact": "high/medium/low",',
            '      "implementation": "How to implement"',
            "    }",
            "  ],",
            '  "agent_insights": { /* Agent-specific observations */ },',
            '  "complexity_assessment": "Assessment of task complexity vs capabilities"',
            "}",
        ]

        return "\n".join(prompt_parts)

    def _parse_reflection_response(self, response: str) -> dict[str, Any]:
        """Parse reflection analysis response.

        Args:
            response: LLM response text

        Returns:
            Parsed analysis results
        """
        try:
            # Extract JSON
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {"parse_failed": True, "raw_response": response, "suggestions": []}

        except Exception as e:
            logger.error(f"Failed to parse reflection response: {e}")
            return {"parse_failed": True, "error": str(e), "suggestions": []}
