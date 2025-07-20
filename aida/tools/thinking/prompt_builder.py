"""Prompt building for thinking tool."""

import logging
from typing import Optional

from .models import ThinkingRequest
from .config import ThinkingConfig

logger = logging.getLogger(__name__)


class ThinkingPromptBuilder:
    """Builds prompts for thinking operations."""
    
    def build(self, request: ThinkingRequest) -> str:
        """Build the prompt for the LLM based on request."""
        base_prompt = self._build_base_prompt(request)
        reasoning_prompt = self._build_reasoning_prompt(request)
        format_instruction = self._build_format_instruction(request)
        
        return base_prompt + reasoning_prompt + format_instruction
    
    def _build_base_prompt(self, request: ThinkingRequest) -> str:
        """Build the base problem and context prompt."""
        prompt = f"Problem: {request.problem}"
        
        if request.context:
            prompt += f"\n\nContext: {request.context}"
            
        return prompt
    
    def _build_reasoning_prompt(self, request: ThinkingRequest) -> str:
        """Build reasoning-specific prompt section."""
        template = ThinkingConfig.get_prompt_template(request.reasoning_type)
        
        # Format the template with request parameters
        return template.format(
            perspective=request.perspective.value,
            depth=request.depth
        )
    
    def _build_format_instruction(self, request: ThinkingRequest) -> str:
        """Build output format instructions."""
        return ThinkingConfig.get_output_instruction(request.output_format.value)
    
    def build_refinement_prompt(
        self, 
        original_response: str,
        refinement_request: str
    ) -> str:
        """Build a prompt for refining an existing analysis."""
        return f"""
Original Analysis:
{original_response}

Refinement Request:
{refinement_request}

Please refine the analysis based on the refinement request, maintaining the same structure and depth.
"""