"""Main thinking tool implementation."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

from aida.tools.base import ToolResult, ToolCapability, ToolParameter, ToolStatus
from aida.tools.base_tool import BaseModularTool
from aida.llm import chat

from .models import (
    ThinkingRequest, 
    ThinkingResponse,
    ThinkingSection,
    ReasoningType,
    Perspective,
    OutputFormat
)
from .config import ThinkingConfig
from .prompt_builder import ThinkingPromptBuilder
from .response_parser import ThinkingResponseParser

logger = logging.getLogger(__name__)


class ThinkingTool(BaseModularTool[ThinkingRequest, ThinkingResponse, ThinkingConfig]):
    """Tool for complex reasoning, analysis, and strategic planning using LLM."""
    
    def __init__(self):
        super().__init__()
        self._response_cache = {}
        
        # Initialize components
        self.prompt_builder = ThinkingPromptBuilder()
        self.response_parser = ThinkingResponseParser()
    
    def _get_tool_name(self) -> str:
        return "thinking"
    
    def _get_tool_version(self) -> str:
        return "2.0.0"
    
    def _get_tool_description(self) -> str:
        return "Enables complex reasoning, chain-of-thought analysis, and strategic planning"
    
    def _get_default_config(self):
        return ThinkingConfig
    
    def get_capability(self) -> ToolCapability:
        """Get tool capability descriptor."""
        return ToolCapability(
            name=self.name,
            version=self.version,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="problem",
                    type="str",
                    description="The problem or question to analyze",
                    required=True
                ),
                ToolParameter(
                    name="context",
                    type="str",
                    description="Additional context for the problem",
                    required=False,
                    default=""
                ),
                ToolParameter(
                    name="reasoning_type",
                    type="str", 
                    description="Type of reasoning to apply",
                    required=False,
                    default="systematic_analysis",
                    choices=[t.value for t in ReasoningType]
                ),
                ToolParameter(
                    name="depth",
                    type="int",
                    description="Depth of analysis (1-5)",
                    required=False,
                    default=3,
                    min_value=1,
                    max_value=5
                ),
                ToolParameter(
                    name="perspective",
                    type="str",
                    description="Analysis perspective to take",
                    required=False,
                    default="balanced",
                    choices=[p.value for p in Perspective]
                ),
                ToolParameter(
                    name="output_format",
                    type="str",
                    description="Format for the analysis output",
                    required=False,
                    default="structured",
                    choices=[f.value for f in OutputFormat]
                )
            ]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute thinking and analysis using LLM."""
        start_time = datetime.utcnow()
        
        try:
            # Create request model
            request = ThinkingRequest(**kwargs)
            
            # Check cache if enabled
            cache_key = self._get_cache_key(request)
            if ThinkingConfig.ENABLE_RESPONSE_CACHE and cache_key in self._response_cache:
                cached_response = self._response_cache[cache_key]
                logger.debug(f"Returning cached response for {cache_key}")
                return self._create_tool_result(cached_response, start_time)
            
            # Build prompt
            prompt = self.prompt_builder.build(request)
            
            # Get LLM response
            llm_response = await chat(prompt, purpose=ThinkingConfig.LLM_PURPOSE)
            
            # Parse response
            response = self.response_parser.parse(
                llm_response,
                request
            )
            
            # Update processing time
            response.processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Cache response if enabled
            if ThinkingConfig.ENABLE_RESPONSE_CACHE:
                self._response_cache[cache_key] = response
                # TODO: Implement cache expiration
            
            return self._create_tool_result(response, start_time)
            
        except Exception as e:
            logger.error(f"Thinking tool failed: {e}")
            return ToolResult(
                tool_name=self.name,
                execution_id="",
                status=ToolStatus.FAILED,
                error=str(e),
                started_at=start_time,
                completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - start_time).total_seconds()
            )
    
    def _get_cache_key(self, request: ThinkingRequest) -> str:
        """Generate cache key for request."""
        return f"{request.problem[:50]}_{request.reasoning_type}_{request.depth}_{request.perspective}"
    
    def _create_tool_result(
        self, 
        response: ThinkingResponse, 
        start_time: datetime
    ) -> ToolResult:
        """Create ToolResult from ThinkingResponse."""
        # Convert response to dict for result
        result_data = {
            "analysis": response.analysis,
            "reasoning_type": response.reasoning_type.value,
            "format": response.perspective.value
        }
        
        # Add optional fields if present
        if response.summary:
            result_data["summary"] = response.summary
        if response.recommendations:
            result_data["recommendations"] = response.recommendations
        if response.key_insights:
            result_data["key_insights"] = response.key_insights
        if response.sections:
            result_data["sections"] = {
                k: v.dict() for k, v in response.sections.items()
            }
        
        return ToolResult(
            tool_name=self.name,
            execution_id=response.request_id,
            status=ToolStatus.COMPLETED,
            result=result_data,
            started_at=start_time,
            completed_at=datetime.utcnow(),
            duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
            metadata={
                "reasoning_type": response.reasoning_type.value,
                "depth": response.depth,
                "perspective": response.perspective.value,
                "cached": False  # Will be True when implementing proper caching
            }
        )
    
    # ============================================================================
    # HYBRID ARCHITECTURE METHODS
    # ============================================================================
    
    def _create_pydantic_tools(self) -> Dict[str, Callable]:
        """Convert to PydanticAI-compatible tool functions."""
        
        async def analyze_problem(
            problem: str,
            context: str = "",
            reasoning_type: str = "systematic_analysis",
            depth: int = 3,
            perspective: str = "balanced"
        ) -> Dict[str, Any]:
            """Analyze a problem using structured reasoning."""
            result = await self.execute(
                problem=problem,
                context=context,
                reasoning_type=reasoning_type,
                depth=depth,
                perspective=perspective
            )
            return result.result
        
        async def brainstorm_solutions(problem: str, context: str = "") -> Dict[str, Any]:
            """Brainstorm creative solutions for a problem."""
            result = await self.execute(
                problem=problem,
                context=context,
                reasoning_type="brainstorming",
                depth=4
            )
            return result.result
        
        async def strategic_planning(goal: str, context: str = "") -> Dict[str, Any]:
            """Create a strategic plan for achieving a goal."""
            result = await self.execute(
                problem=goal,
                context=context,
                reasoning_type="strategic_planning",
                depth=4,
                perspective="business"
            )
            return result.result
        
        async def analyze_decision(
            decision: str,
            options: List[str],
            context: str = ""
        ) -> Dict[str, Any]:
            """Analyze a decision with given options."""
            options_context = "Options:\n" + "\n".join(f"- {opt}" for opt in options)
            full_context = f"{context}\n\n{options_context}" if context else options_context
            
            result = await self.execute(
                problem=decision,
                context=full_context,
                reasoning_type="decision_analysis",
                depth=4
            )
            return result.result
        
        return {
            "analyze_problem": analyze_problem,
            "brainstorm_solutions": brainstorm_solutions,
            "strategic_planning": strategic_planning,
            "analyze_decision": analyze_decision
        }
    
    def _create_mcp_server(self):
        """Create MCP server instance."""
        from .mcp_server import ThinkingMCPServer
        return ThinkingMCPServer(self)
    
    def _create_observability(self, config: Dict[str, Any]):
        """Create observability instance."""
        from .observability import ThinkingObservability
        return ThinkingObservability(self, config)