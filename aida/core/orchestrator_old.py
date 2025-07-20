"""Intelligent workflow orchestration using LLM reasoning."""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from aida.providers.llm.base import LLMMessage, LLMRole
from aida.providers.llm.manager import get_llm_manager
from aida.tools.base import get_tool_registry, ToolResult
from aida.config.llm_defaults import (
    get_reasoning_system_prompt, 
    get_tool_orchestration_prompt
)


logger = logging.getLogger(__name__)


class WorkflowStep:
    """Represents a single step in a workflow execution."""
    
    def __init__(self, tool_name: str, parameters: Dict[str, Any], purpose: str = ""):
        self.tool_name = tool_name
        self.parameters = parameters
        self.purpose = purpose
        self.result: Optional[ToolResult] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
    
    @property
    def status(self) -> str:
        if self.error:
            return "failed"
        elif self.completed_at:
            return "completed"
        elif self.started_at:
            return "running"
        else:
            return "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "purpose": self.purpose,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "result_summary": self._get_result_summary()
        }
    
    def _get_result_summary(self) -> Optional[Dict[str, Any]]:
        if not self.result:
            return None
        
        return {
            "status": self.result.status,
            "execution_time": self.result.duration_seconds,
            "result_type": type(self.result.result).__name__,
            "has_error": bool(self.result.error)
        }


class WorkflowPlan:
    """Represents a planned workflow with analysis and execution steps."""
    
    def __init__(self, user_request: str, analysis: str, steps: List[WorkflowStep], expected_outcome: str):
        self.user_request = user_request
        self.analysis = analysis
        self.steps = steps
        self.expected_outcome = expected_outcome
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.current_step: int = 0
    
    @property
    def status(self) -> str:
        if self.completed_at:
            return "completed"
        elif self.started_at:
            return "running"
        else:
            return "planned"
    
    @property
    def progress(self) -> float:
        if not self.steps:
            return 100.0
        return (self.current_step / len(self.steps)) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_request": self.user_request,
            "analysis": self.analysis,
            "expected_outcome": self.expected_outcome,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "current_step": self.current_step,
            "total_steps": len(self.steps),
            "steps": [step.to_dict() for step in self.steps]
        }


class WorkflowOrchestrator:
    """Orchestrates complex workflows using LLM reasoning and tool execution."""
    
    def __init__(self):
        self.llm_manager = get_llm_manager()
        self.tool_registry = get_tool_registry()
        self.conversation_history: List[LLMMessage] = []
        self.active_workflows: Dict[str, WorkflowPlan] = {}
        self._tools_initialized = False
    
    async def process_request(
        self, 
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowPlan:
        """Process a user request and create a workflow plan."""
        
        # Ensure tools are initialized
        if not self._tools_initialized:
            await self._ensure_tools_initialized()
        
        # Add to conversation history
        self.conversation_history.append(
            LLMMessage(role=LLMRole.USER, content=user_message)
        )
        
        # Get available tools with detailed specifications
        available_tools = await self.tool_registry.list_tools()
        
        # Get detailed tool specifications for the prompt
        tool_specs = {}
        for tool_name in available_tools:
            tool = await self.tool_registry.get_tool(tool_name)
            if tool:
                capability = tool.get_capability()
                tool_specs[tool_name] = {
                    "description": capability.description,
                    "parameters": [
                        {
                            "name": param.name,
                            "type": param.type,
                            "description": param.description,
                            "required": param.required,
                            "default": param.default
                        }
                        for param in capability.parameters
                    ]
                }
        
        # Create orchestration prompt with detailed specifications
        orchestration_prompt = get_tool_orchestration_prompt(user_message, tool_specs)
        
        # Get LLM analysis and planning
        system_prompt = get_reasoning_system_prompt()
        
        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=system_prompt)
        ]
        
        # Add conversation history from context if available
        if context and "conversation_history" in context:
            # Add last few exchanges for context (limit to avoid token overflow)
            recent_history = context["conversation_history"][-6:]  # Last 3 exchanges
            
            # Convert dict format to LLMMessage objects
            for msg in recent_history:
                if isinstance(msg, dict):
                    role = LLMRole.USER if msg["role"] == "user" else LLMRole.ASSISTANT
                    messages.append(LLMMessage(role=role, content=msg["content"]))
                else:
                    # Already an LLMMessage object
                    messages.append(msg)
        
        # Add current user message
        messages.append(LLMMessage(role=LLMRole.USER, content=orchestration_prompt))
        
        try:
            response = await self.llm_manager.chat_completion(
                messages=messages,
                routing_strategy="cost_optimized",
                requirements={"local_only": True}  # Prefer Ollama
            )
            
            # Parse the LLM response
            plan_data = self._parse_llm_planning_response(response.content)
            
            # Create workflow steps
            steps = []
            for step_data in plan_data.get("execution_plan", []):
                step = WorkflowStep(
                    tool_name=step_data["tool"],
                    parameters=step_data["parameters"],
                    purpose=step_data.get("purpose", "")
                )
                steps.append(step)
            
            # Create workflow plan
            workflow = WorkflowPlan(
                user_request=user_message,
                analysis=plan_data.get("analysis", "No analysis provided"),
                steps=steps,
                expected_outcome=plan_data.get("expected_outcome", "")
            )
            
            # Store workflow
            workflow_id = f"workflow_{datetime.utcnow().timestamp()}"
            self.active_workflows[workflow_id] = workflow
            
            return workflow
            
        except Exception as e:
            logger.error(f"Failed to process request with LLM: {e}")
            # For now, return a simple error workflow
            # TODO: Implement proper LLM provider setup instead of fallback
            error_step = WorkflowStep(
                tool_name="thinking",
                parameters={
                    "problem": f"Unable to process request due to LLM unavailability: {user_message}",
                    "reasoning_type": "systematic_analysis"
                },
                purpose="Analyze the request despite LLM limitations"
            )
            
            workflow = WorkflowPlan(
                user_request=user_message,
                analysis=f"LLM processing failed: {e}. Need to setup proper LLM provider.",
                steps=[error_step],
                expected_outcome="Error analysis - please setup LLM provider for full functionality"
            )
            
            # Store workflow
            workflow_id = f"workflow_{datetime.utcnow().timestamp()}"
            self.active_workflows[workflow_id] = workflow
            
            return workflow
    
    async def execute_workflow(
        self, 
        workflow: WorkflowPlan,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Execute a planned workflow step by step."""
        
        workflow.started_at = datetime.utcnow()
        results = []
        
        try:
            for i, step in enumerate(workflow.steps):
                workflow.current_step = i
                
                if progress_callback:
                    await progress_callback(workflow, step)
                
                # Execute step
                step_result = await self._execute_step(step)
                results.append(step_result)
                
                # Check if we should continue based on result
                if step.error and not self._should_continue_on_error(step, workflow):
                    break
            
            workflow.completed_at = datetime.utcnow()
            workflow.current_step = len(workflow.steps)
            
            return {
                "status": "completed",
                "workflow": workflow.to_dict(),
                "results": results,
                "execution_summary": self._create_execution_summary(workflow)
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "workflow": workflow.to_dict(),
                "results": results
            }
    
    async def execute_request(
        self, 
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Process and execute a user request in one go."""
        
        # Plan the workflow
        workflow = await self.process_request(user_message, context)
        
        # Execute the workflow
        return await self.execute_workflow(workflow, progress_callback)
    
    async def _execute_step(self, step: WorkflowStep, retry_count: int = 0) -> Dict[str, Any]:
        """Execute a single workflow step with automatic error correction."""
        step.started_at = datetime.utcnow()
        
        try:
            # Get the tool
            tool = await self.tool_registry.get_tool(step.tool_name)
            if not tool:
                raise ValueError(f"Tool '{step.tool_name}' not found")
            
            # Validate parameters before execution
            capability = tool.get_capability()
            required_params = [p.name for p in capability.parameters if p.required]
            provided_params = list(step.parameters.keys())
            missing_params = set(required_params) - set(provided_params)
            
            if missing_params:
                logger.error(f"Missing required parameters for {step.tool_name}: {missing_params}")
                logger.error(f"Required: {required_params}")
                logger.error(f"Provided: {provided_params}")
                logger.error(f"Step parameters: {step.parameters}")
                raise ValueError(f"Missing required parameters: {missing_params}")
            
            # Execute the tool
            result = await tool.execute_async(**step.parameters)
            step.result = result
            step.completed_at = datetime.utcnow()
            
            # Check for execution errors and attempt auto-correction
            if step.tool_name == "execution" and result and hasattr(result, 'result'):
                execution_result = result.result
                if isinstance(execution_result, dict):
                    exit_code = execution_result.get('exit_code', 0)
                    stderr = execution_result.get('stderr', '')
                    
                    # If execution failed and we haven't retried yet
                    if exit_code != 0 and retry_count < 2:
                        logger.warning(f"🔄 Execution failed (exit {exit_code}), attempting auto-correction...")
                        
                        # Attempt to fix the error
                        corrected_step = await self._attempt_error_correction(step, execution_result, retry_count)
                        if corrected_step:
                            logger.info(f"🛠️ Retrying with corrected code (attempt {retry_count + 1}/2)...")
                            return await self._execute_step(corrected_step, retry_count + 1)
            
            return {
                "step": step.to_dict(),
                "success": True,
                "result": result.dict() if result else None
            }
            
        except Exception as e:
            step.error = str(e)
            step.completed_at = datetime.utcnow()
            logger.error(f"Step execution failed: {e}")
            
            return {
                "step": step.to_dict(),
                "success": False,
                "error": str(e)
            }
    
    async def _attempt_error_correction(self, failed_step: WorkflowStep, execution_result: Dict[str, Any], retry_count: int) -> Optional[WorkflowStep]:
        """Attempt to correct execution errors using LLM analysis."""
        try:
            language = failed_step.parameters.get('language', 'python')
            code = failed_step.parameters.get('code', '')
            exit_code = execution_result.get('exit_code', 0)
            stderr = execution_result.get('stderr', '')
            stdout = execution_result.get('stdout', '')
            
            logger.debug(f"Attempting to fix {language} code error (exit {exit_code})")
            logger.debug(f"Original code: {code}")
            logger.debug(f"Error output: {stderr}")
            
            # Use LLM to analyze and fix the error
            error_analysis_prompt = f"""
            You are a code debugging expert. A {language} script failed with the following details:

            ORIGINAL CODE:
            ```{language}
            {code}
            ```

            ERROR DETAILS:
            - Exit code: {exit_code}
            - Error output: {stderr}
            - Standard output: {stdout}

            Please analyze the error and provide a corrected version. Your response must be in this exact JSON format:

            ```json
            {{
                "analysis": "Brief explanation of what went wrong",
                "corrected_code": "The fixed code with all necessary imports and syntax corrections",
                "required_packages": ["list", "of", "packages", "needed"],
                "changes_made": "Summary of what was fixed"
            }}
            ```

            Requirements:
            1. Fix ALL syntax errors and missing imports
            2. Ensure the code is complete and executable
            3. Add proper output statements (print, echo, console.log) so results are visible
            4. Include ALL required package imports at the top
            5. Make the code robust and handle edge cases
            """
            
            # Get error correction from LLM
            if hasattr(self, 'llm_manager') and self.llm_manager:
                try:
                    correction_response = await self.llm_manager.chat(error_analysis_prompt)
                    correction_data = self._parse_error_correction_response(correction_response.content)
                    
                    if correction_data and correction_data.get('corrected_code'):
                        corrected_code = correction_data['corrected_code']
                        packages = correction_data.get('required_packages', [])
                        
                        # Create corrected step
                        corrected_step = WorkflowStep(
                            tool_name=failed_step.tool_name,
                            parameters={
                                **failed_step.parameters,
                                'code': corrected_code,
                                'packages': packages
                            },
                            purpose=f"{failed_step.purpose} (auto-corrected)"
                        )
                        
                        logger.info(f"🔧 LLM-corrected code: {correction_data.get('changes_made', 'Fixed errors')}")
                        logger.debug(f"Corrected code: {corrected_code}")
                        return corrected_step
                        
                except Exception as llm_error:
                    logger.warning(f"LLM error correction failed: {llm_error}")
            
            # Fallback to simple heuristic fixes if LLM fails
            return self._simple_error_correction(failed_step, execution_result)
            
        except Exception as e:
            logger.error(f"Error correction failed: {e}")
            return None
    
    def _parse_error_correction_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """Parse LLM error correction response."""
        try:
            import re
            import json
            
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                return json.loads(json_content)
            
            # Try to find JSON object directly
            if '{' in response_content and '}' in response_content:
                start = response_content.find('{')
                end = response_content.rfind('}') + 1
                return json.loads(response_content[start:end])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to parse error correction response: {e}")
            return None
    
    def _simple_error_correction(self, failed_step: WorkflowStep, execution_result: Dict[str, Any]) -> Optional[WorkflowStep]:
        """Simple fallback error correction with basic heuristics."""
        try:
            language = failed_step.parameters.get('language', 'python')
            code = failed_step.parameters.get('code', '')
            stderr = execution_result.get('stderr', '')
            
            packages = []
            corrected_code = code
            
            # Basic fixes for common issues
            if language == 'python':
                # Add common packages if mentioned
                if 'pandas' in code or 'pd.' in code:
                    packages.append('pandas')
                if 'numpy' in code or 'np.' in code:
                    packages.append('numpy')
                if 'matplotlib' in code:
                    packages.append('matplotlib')
                    
                # Add basic imports if missing
                if 'pd.' in code and 'import pandas' not in code:
                    corrected_code = 'import pandas as pd; ' + corrected_code
                if 'np.' in code and 'import numpy' not in code:
                    corrected_code = 'import numpy as np; ' + corrected_code
            
            if packages or corrected_code != code:
                return WorkflowStep(
                    tool_name=failed_step.tool_name,
                    parameters={
                        **failed_step.parameters,
                        'code': corrected_code,
                        'packages': packages
                    },
                    purpose=f"{failed_step.purpose} (simple correction)"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Simple error correction failed: {e}")
            return None
    
    def _parse_llm_planning_response(self, response_content: str) -> Dict[str, Any]:
        """Parse LLM response for workflow planning with improved error handling."""
        try:
            import re
            
            # Try to extract JSON from the response
            response_content = response_content.strip()
            logger.debug(f"Parsing LLM response ({len(response_content)} chars)")
            logger.debug(f"Raw response: {response_content[:500]}...")
            
            json_content = None
            
            # Method 1: Look for ```json blocks (preferred)
            json_block_pattern = r'```json\s*(.*?)\s*```'
            json_match = re.search(json_block_pattern, response_content, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                logger.debug("Found JSON in code block")
            
            # Method 2: Look for JSON object boundaries more aggressively
            elif "{" in response_content and "}" in response_content:
                # Find the outermost JSON object
                brace_count = 0
                json_start = None
                json_end = None
                
                for i, char in enumerate(response_content):
                    if char == '{':
                        if json_start is None:
                            json_start = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and json_start is not None:
                            json_end = i + 1
                            break
                
                if json_start is not None and json_end is not None:
                    json_content = response_content[json_start:json_end]
                    logger.debug("Extracted JSON object using brace matching")
            
            # Method 3: Try multiline regex patterns for nested JSON
            if not json_content:
                json_patterns = [
                    r'\{[^{}]*"execution_plan"[^{}]*\[[^\]]*\][^{}]*\}',  # Simple pattern
                    r'\{.*?"execution_plan".*?\[.*?\].*?\}',  # More flexible
                    r'\{(?:[^{}]|{[^{}]*})*"execution_plan"(?:[^{}]|{[^{}]*})*\}',  # Nested objects
                ]
                
                for pattern in json_patterns:
                    match = re.search(pattern, response_content, re.DOTALL)
                    if match:
                        json_content = match.group(0)
                        logger.debug(f"Found JSON using pattern: {pattern[:50]}...")
                        break
            
            if not json_content:
                logger.error("No JSON content found - attempting to create fallback structure")
                # Log the response for debugging
                logger.error(f"Full response: {response_content}")
                raise ValueError("No valid JSON structure found in LLM response")
            
            # Clean up the JSON content
            json_content = json_content.strip()
            
            # Try to parse the JSON
            try:
                parsed_response = json.loads(json_content)
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON decode error: {json_err}")
                logger.error(f"Problematic JSON: {json_content}")
                
                # Try to fix common JSON issues
                fixed_json = json_content
                
                # Fix newlines in JSON strings (major issue causing the error)
                # Replace actual newlines within strings with \n escape sequences
                import re
                
                # Find strings and escape newlines within them
                def fix_newlines_in_strings(match):
                    string_content = match.group(1)
                    # Replace newlines with \n and return the fixed string
                    fixed_content = string_content.replace('\n', '\\n').replace('\r', '\\r')
                    return f'"{fixed_content}"'
                
                # Fix newlines in string values
                fixed_json = re.sub(r'"([^"]*(?:\n|\r)[^"]*)"', fix_newlines_in_strings, fixed_json, flags=re.DOTALL)
                
                # Fix unescaped quotes in strings
                fixed_json = re.sub(r'(?<!\\)"(?=[^,:}\]]*[,:}\]])', r'\"', fixed_json)
                
                # Fix trailing commas before closing braces/brackets
                fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
                
                # Fix trailing commas in arrays more aggressively
                fixed_json = re.sub(r',(\s*\],)', r'\1', fixed_json)
                fixed_json = re.sub(r',(\s*\])', r'\1', fixed_json)
                
                # Try parsing again
                try:
                    parsed_response = json.loads(fixed_json)
                    logger.debug("Fixed JSON parsing issues")
                except json.JSONDecodeError:
                    raise ValueError(f"Could not parse JSON even after fixes: {json_err}")
            
            # Validate the response structure
            if not isinstance(parsed_response, dict):
                raise ValueError("LLM response must be a JSON object")
            
            if "execution_plan" not in parsed_response:
                raise ValueError("LLM response missing 'execution_plan' field")
            
            if not isinstance(parsed_response["execution_plan"], list):
                raise ValueError("'execution_plan' must be a list of steps")
            
            # Validate each step thoroughly
            for i, step in enumerate(parsed_response.get("execution_plan", [])):
                if not isinstance(step, dict):
                    raise ValueError(f"Step {i+1} must be a dictionary")
                
                if "tool" not in step:
                    raise ValueError(f"Step {i+1} missing 'tool' field")
                
                if "parameters" not in step:
                    raise ValueError(f"Step {i+1} missing 'parameters' field")
                
                tool_name = step["tool"]
                parameters = step["parameters"]
                
                if not isinstance(parameters, dict):
                    raise ValueError(f"Step {i+1}: parameters for {tool_name} must be a dictionary")
                
                # Validate tool exists and parameters are reasonable
                valid_tools = ["thinking", "execution", "file_operations", "system", "context", "maintenance", "project", "architecture"]
                if tool_name not in valid_tools:
                    logger.warning(f"Step {i+1}: unknown tool '{tool_name}', valid tools: {valid_tools}")
                    # Don't fail here, just warn - allow for flexibility
            
            # Create a summary of the plan
            plan_summary = []
            for i, step in enumerate(parsed_response['execution_plan'], 1):
                tool = step.get('tool', step.get('tool_name', ''))  # Support both field names
                if tool == "thinking":
                    plan_summary.append(f"{i}. Analyze")
                elif tool == "file_operations":
                    op = step['parameters'].get('operation', '')
                    if op == "write_file":
                        plan_summary.append(f"{i}. Create script")
                    else:
                        plan_summary.append(f"{i}. {op.replace('_', ' ').title()}")
                elif tool == "execution":
                    plan_summary.append(f"{i}. Execute")
                else:
                    plan_summary.append(f"{i}. {tool.title()}")
            
            logger.info(f"📋 Workflow planned: {' → '.join(plan_summary)}")
            return parsed_response
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw LLM response: {response_content}")
            raise ValueError(f"LLM response parsing failed: {e}. Raw response logged for debugging.")
    
    # Removed complex fallback methods - focusing on proper LLM integration
    
    def _should_continue_on_error(self, step: WorkflowStep, workflow: WorkflowPlan) -> bool:
        """Determine if workflow should continue after a step error."""
        # For now, continue on non-critical errors
        critical_tools = ["thinking"]  # These are required for workflow integrity
        
        if step.tool_name in critical_tools:
            return False
        
        # Continue for tool-specific errors that might not affect other steps
        return True
    
    def _create_execution_summary(self, workflow: WorkflowPlan) -> Dict[str, Any]:
        """Create a summary of workflow execution."""
        total_steps = len(workflow.steps)
        completed_steps = sum(1 for step in workflow.steps if step.status == "completed")
        failed_steps = sum(1 for step in workflow.steps if step.status == "failed")
        
        total_time = 0.0
        if workflow.started_at and workflow.completed_at:
            total_time = (workflow.completed_at - workflow.started_at).total_seconds()
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "success_rate": (completed_steps / max(1, total_steps)) * 100,
            "total_execution_time": total_time,
            "workflow_status": workflow.status
        }
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific workflow."""
        workflow = self.active_workflows.get(workflow_id)
        return workflow.to_dict() if workflow else None
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows."""
        return [workflow.to_dict() for workflow in self.active_workflows.values()]
    
    async def _ensure_tools_initialized(self):
        """Ensure tools are initialized in the registry."""
        try:
            from aida.tools.base import initialize_default_tools
            await initialize_default_tools()
            self._tools_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            # Continue without tools - fallback will handle this


# Global orchestrator instance
_global_orchestrator: Optional[WorkflowOrchestrator] = None


def get_orchestrator() -> WorkflowOrchestrator:
    """Get the global workflow orchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = WorkflowOrchestrator()
    return _global_orchestrator