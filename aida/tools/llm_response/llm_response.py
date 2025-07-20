"""Main LLM response tool implementation."""

import logging
from typing import Any, Dict, Optional, Callable
from datetime import datetime

from aida.tools.base import ToolResult, ToolCapability, ToolParameter, ToolStatus
from aida.tools.base_tool import SimpleToolBase
from aida.llm import chat

from .models import LLMResponseRequest, LLMResponseResult
from .config import LLMResponseConfig

logger = logging.getLogger(__name__)


class LLMResponseTool(SimpleToolBase):
    """Tool for getting direct LLM responses to questions."""
    
    def __init__(self):
        super().__init__()
        self._pydantic_tools_cache = {}
    
    def _get_tool_name(self) -> str:
        return "llm_response"
    
    def _get_tool_version(self) -> str:
        return "1.0.2"
    
    def _get_tool_description(self) -> str:
        return "Get a direct response from the LLM for questions, explanations, and general knowledge"
    
    def _get_default_config(self):
        return LLMResponseConfig
    
    def _get_default_operation(self) -> str:
        return "answer"
    
    def _create_processors(self) -> Dict[str, Callable]:
        return {
            "answer": self._answer_question
        }
    
    async def _answer_question(self, **kwargs) -> str:
        """Process the answer operation."""
        # Create request model
        request = LLMResponseRequest(**kwargs)
        
        # Build prompt
        prompt = LLMResponseConfig.build_prompt(
            request.question,
            request.context,
            request.max_length
        )
        
        # Get LLM response
        response = await chat(prompt, purpose=LLMResponseConfig.LLM_PURPOSE)
        
        # Truncate if needed
        if len(response) > request.max_length:
            truncate_at = request.max_length - LLMResponseConfig.TRUNCATION_BUFFER
            response = response[:truncate_at] + LLMResponseConfig.TRUNCATION_SUFFIX
        
        return response
    
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
                    default=LLMResponseConfig.DEFAULT_MAX_LENGTH,
                    min_value=LLMResponseConfig.MIN_LENGTH,
                    max_value=LLMResponseConfig.ABSOLUTE_MAX_LENGTH
                )
            ]
        )
    
    # Override execute to add custom metadata
    async def execute(self, **kwargs) -> ToolResult:
        """Get LLM response to the question."""
        # Default operation if not specified
        if 'operation' not in kwargs:
            kwargs['operation'] = 'answer'
        
        # Call parent execute
        result = await super().execute(**kwargs)
        
        # Add custom metadata if successful
        if result.status == ToolStatus.COMPLETED and 'question' in kwargs:
            result.metadata.update({
                "question_length": len(kwargs.get('question', '')),
                "context_provided": bool(kwargs.get('context', '')),
                "response_length": len(result.result) if isinstance(result.result, str) else 0
            })
        
        return result
    
    # ============================================================================
    # HYBRID ARCHITECTURE METHODS
    # ============================================================================
    
    def _create_pydantic_tools(self) -> Dict[str, Callable]:
        """Convert to PydanticAI-compatible tool functions."""
        async def ask_llm(
            question: str,
            context: str = "",
            max_length: int = LLMResponseConfig.DEFAULT_MAX_LENGTH
        ) -> str:
            """Get a direct LLM response to a question."""
            result = await self.execute(
                question=question,
                context=context,
                max_length=max_length
            )
            return result.result
        
        async def explain_concept(concept: str, context: str = "") -> str:
            """Get an explanation of a concept from the LLM."""
            question = f"Please explain {concept}"
            result = await self.execute(
                question=question,
                context=context,
                max_length=3000
            )
            return result.result
        
        async def answer_question(question: str, context: str = "") -> str:
            """Answer a question using the LLM."""
            result = await self.execute(
                question=question,
                context=context
            )
            return result.result
        
        return {
            "ask_llm": ask_llm,
            "explain_concept": explain_concept,
            "answer_question": answer_question
        }
    
    def _create_mcp_server(self):
        """Create MCP server instance."""
        from .mcp_server import LLMResponseMCPServer
        return LLMResponseMCPServer(self)
    
    def _create_observability(self, config: Dict[str, Any]):
        """Create observability instance."""
        from .observability import LLMResponseObservability
        return LLMResponseObservability(self, config)