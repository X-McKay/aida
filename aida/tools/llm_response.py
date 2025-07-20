"""Direct LLM response tool for answering questions."""

import asyncio
import json
from typing import Any, Dict, Optional
from datetime import datetime
import logging

from aida.tools.base import Tool, ToolResult, ToolCapability, ToolParameter
from aida.llm import chat

logger = logging.getLogger(__name__)


class LLMResponseTool(Tool):
    """Tool for getting direct LLM responses to questions."""
    
    def __init__(self):
        super().__init__(
            name="llm_response",
            description="Get a direct response from the LLM for questions, explanations, and general knowledge",
            version="1.0.0"
        )
    
    def get_capability(self) -> ToolCapability:
        """Get tool capability descriptor."""
        return ToolCapability(
            name=self.name,
            version=self.version,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="question",
                    type="str",
                    description="The question or request to answer",
                    required=True
                ),
                ToolParameter(
                    name="context",
                    type="str",
                    description="Additional context for the question",
                    required=False,
                    default=""
                ),
                ToolParameter(
                    name="max_length",
                    type="int",
                    description="Maximum response length in characters",
                    required=False,
                    default=2000
                )
            ]
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Get LLM response to the question."""
        question = kwargs["question"]
        context = kwargs.get("context", "")
        max_length = kwargs.get("max_length", 2000)
        
        try:
            # Build prompt
            prompt = question
            if context:
                # Handle both string context and dict/list context
                if isinstance(context, (dict, list)):
                    context_str = json.dumps(context, indent=2)
                else:
                    context_str = str(context)
                prompt = f"Previous conversation context:\n{context_str}\n\nCurrent question: {question}"
            
            # Add instruction for length
            prompt += f"\n\nPlease provide a helpful, informative response (max {max_length} characters)."
            
            # Get LLM response
            response = await chat(prompt)
            
            # Truncate if needed
            if len(response) > max_length:
                response = response[:max_length-3] + "..."
            
            return ToolResult(
                tool_name=self.name,
                execution_id="",
                status="completed",
                result=response,
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                duration_seconds=0.1,
                metadata={
                    "question_length": len(question),
                    "response_length": len(response)
                }
            )
            
        except Exception as e:
            logger.error(f"LLM response failed: {e}")
            return ToolResult(
                tool_name=self.name,
                execution_id="",
                status="failed",
                error=str(e),
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                duration_seconds=0.1
            )
    
    async def execute_async(self, **kwargs) -> ToolResult:
        """Async execution (same as execute for this tool)."""
        return await self.execute(**kwargs)