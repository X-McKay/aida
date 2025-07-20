"""Base LLM provider interface."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
from enum import Enum
import uuid
import logging

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class LLMRole(str, Enum):
    """LLM message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class LLMMessage(BaseModel):
    """LLM message format."""
    
    role: LLMRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMUsage(BaseModel):
    """LLM usage statistics."""
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: Optional[float] = None


class LLMResponse(BaseModel):
    """LLM response format."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    role: LLMRole = LLMRole.ASSISTANT
    model: str
    provider: str
    usage: Optional[LLMUsage] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class LLMError(Exception):
    """Base LLM provider error."""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "LLM_ERROR",
        provider: str = None,
        model: str = None,
        details: Dict[str, Any] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.provider = provider
        self.model = model
        self.details = details or {}


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    
    provider_name: str
    model: str
    api_key: Optional[str] = None
    api_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_rpm: Optional[int] = None
    rate_limit_tpm: Optional[int] = None
    custom_headers: Dict[str, str] = Field(default_factory=dict)
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider_name = config.provider_name
        self.model = config.model
        
        # Rate limiting
        self._request_times: List[float] = []
        self._token_usage: List[int] = []
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            "requests_sent": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_response_time": 0.0
        }
    
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[LLMMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[LLMResponse, None]]:
        """Generate chat completion."""
        pass
    
    @abstractmethod
    async def text_completion(
        self, 
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[LLMResponse, None]]:
        """Generate text completion."""
        pass
    
    @abstractmethod
    async def embedding(
        self, 
        text: Union[str, List[str]],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings."""
        pass
    
    @abstractmethod
    async def check_health(self) -> bool:
        """Check provider health."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass
    
    async def chat(
        self,
        message: str,
        system_message: Optional[str] = None,
        conversation_history: Optional[List[LLMMessage]] = None,
        **kwargs
    ) -> LLMResponse:
        """Simplified chat interface."""
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append(LLMMessage(
                role=LLMRole.SYSTEM,
                content=system_message
            ))
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add user message
        messages.append(LLMMessage(
            role=LLMRole.USER,
            content=message
        ))
        
        return await self.chat_completion(messages, **kwargs)
    
    async def rate_limit_check(self, estimated_tokens: int = 0) -> None:
        """Check and enforce rate limits."""
        async with self._lock:
            now = datetime.utcnow().timestamp()
            
            # Clean old entries (older than 1 minute)
            cutoff = now - 60
            self._request_times = [t for t in self._request_times if t > cutoff]
            self._token_usage = self._token_usage[-len(self._request_times):]
            
            # Check RPM limit
            if self.config.rate_limit_rpm:
                if len(self._request_times) >= self.config.rate_limit_rpm:
                    sleep_time = 60 - (now - self._request_times[0])
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
            
            # Check TPM limit
            if self.config.rate_limit_tpm and estimated_tokens > 0:
                total_tokens = sum(self._token_usage) + estimated_tokens
                if total_tokens >= self.config.rate_limit_tpm:
                    sleep_time = 60 - (now - self._request_times[0])
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
            
            # Record this request
            self._request_times.append(now)
            if estimated_tokens > 0:
                self._token_usage.append(estimated_tokens)
    
    def update_stats(self, response: LLMResponse, response_time: float):
        """Update provider statistics."""
        self._stats["requests_sent"] += 1
        self._stats["requests_successful"] += 1
        
        if response.usage:
            self._stats["total_tokens"] += response.usage.total_tokens
            if response.usage.estimated_cost:
                self._stats["total_cost"] += response.usage.estimated_cost
        
        # Update average response time
        total_requests = self._stats["requests_successful"]
        current_avg = self._stats["average_response_time"]
        self._stats["average_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    def update_error_stats(self):
        """Update error statistics."""
        self._stats["requests_sent"] += 1
        self._stats["requests_failed"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return {
            **self._stats,
            "provider": self.provider_name,
            "model": self.model,
            "success_rate": (
                self._stats["requests_successful"] / 
                max(1, self._stats["requests_sent"])
            ) * 100
        }
    
    async def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        # Simple estimation: ~4 characters per token
        return max(1, len(text) // 4)
    
    async def estimate_cost(self, usage: LLMUsage) -> float:
        """Estimate cost for usage."""
        # Override in provider implementations with actual pricing
        return 0.0
    
    def create_message(
        self, 
        role: LLMRole, 
        content: str, 
        **kwargs
    ) -> LLMMessage:
        """Create an LLM message."""
        return LLMMessage(
            role=role,
            content=content,
            **kwargs
        )
    
    async def validate_config(self) -> bool:
        """Validate provider configuration."""
        try:
            return await self.check_health()
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup if needed
        pass


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register(cls, provider_name: str, provider_class: type):
        """Register a provider class."""
        cls._providers[provider_name] = provider_class
    
    @classmethod
    def create(cls, config: LLMConfig) -> LLMProvider:
        """Create a provider instance."""
        provider_class = cls._providers.get(config.provider_name)
        if not provider_class:
            raise LLMError(
                f"Unknown provider: {config.provider_name}",
                "UNKNOWN_PROVIDER"
            )
        
        return provider_class(config)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List available providers."""
        return list(cls._providers.keys())