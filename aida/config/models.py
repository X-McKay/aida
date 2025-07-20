"""Model specifications and provider definitions."""

import os
from enum import Enum
from typing import Optional
from pydantic import BaseModel


class Provider(str, Enum):
    """LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    VLLM = "vllm"


class ModelSpec(BaseModel):
    """Model specification."""
    provider: Provider
    model_id: str
    temperature: float = 0.1
    max_tokens: int = 4000
    base_url: Optional[str] = None
    
    @property
    def api_key(self) -> Optional[str]:
        """Get API key from environment."""
        if self.provider == Provider.OPENAI:
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == Provider.ANTHROPIC:
            return os.getenv("ANTHROPIC_API_KEY")
        return None
    
    @property
    def is_available(self) -> bool:
        """Check if model is available."""
        if self.provider in [Provider.OPENAI, Provider.ANTHROPIC]:
            return bool(self.api_key)
        return True  # Local models assumed available