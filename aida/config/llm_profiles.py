"""LLM profiles with purpose-based configurations."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel
from .models import ModelSpec, Provider


class Purpose(str, Enum):
    """LLM usage purposes."""
    DEFAULT = "default"
    CODING = "coding"
    REASONING = "reasoning" 
    MULTIMODAL = "multimodal"
    QUICK = "quick"


class LLMProfile(BaseModel):
    """LLM profile for a specific purpose."""
    purpose: Purpose
    model: ModelSpec
    prompt: str


# Default profiles with inline model definitions
DEFAULT_PROFILES = {
    Purpose.DEFAULT: LLMProfile(
        purpose=Purpose.DEFAULT,
        model=ModelSpec(
            provider=Provider.OLLAMA,
            model_id="llama3.2:latest",
            base_url="http://localhost:11434",
            max_tokens=16384,
            temperature=0,
        ),
        prompt="You are AIDA, a helpful AI assistant. Provide clear, accurate responses."
    ),
    
    Purpose.CODING: LLMProfile(
        purpose=Purpose.CODING,
        model=ModelSpec(
            provider=Provider.OLLAMA,
            model_id="codellama:latest",
            base_url="http://localhost:11434"
        ),
        prompt="""You are AIDA's coding specialist. Write clean, well-documented code.

Guidelines:
- Use clear variable and function names
- Include proper error handling
- Add comments for complex logic
- Follow best practices for the language
- Provide complete, runnable examples"""
    ),
    
    Purpose.REASONING: LLMProfile(
        purpose=Purpose.REASONING,
        model=ModelSpec(
            provider=Provider.OLLAMA,
            model_id="deepseek-r1:8b",
            base_url="http://localhost:11434",
            max_tokens=16384,
            temperature=0,
        ),
        prompt="""You are AIDA's reasoning specialist for complex analysis.

Approach:
- Think step-by-step
- Break down complex problems
- Explain your reasoning clearly
- Consider multiple perspectives
- Provide structured conclusions"""
    ),
    
    Purpose.MULTIMODAL: LLMProfile(
        purpose=Purpose.MULTIMODAL,
        model=ModelSpec(
            provider=Provider.OLLAMA,
            model_id="llama3.2:latest",
            base_url="http://localhost:11434",
            temperature=0,
        ),
        prompt="""You are AIDA's multimodal specialist. Analyze images and multimedia content.

Tasks:
- Describe visual content accurately
- Extract relevant text from images
- Identify patterns and relationships
- Provide structured analysis"""
    ),
    
    Purpose.QUICK: LLMProfile(
        purpose=Purpose.QUICK,
        model=ModelSpec(
            provider=Provider.OLLAMA,
            model_id="tinyllama:latest",
            base_url="http://localhost:11434",
            max_tokens=4096,
            temperature=0,
        ),
        prompt="You are AIDA's quick response specialist. Be brief but complete."
    )
}


def get_profile(purpose: Purpose) -> LLMProfile:
    """Get profile for purpose."""
    return DEFAULT_PROFILES[purpose]


def get_available_purposes() -> list[Purpose]:
    """Get purposes with available models."""
    available = []
    for purpose, profile in DEFAULT_PROFILES.items():
        if profile.model.is_available:
            available.append(purpose)
    return available