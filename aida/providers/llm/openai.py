"""OpenAI LLM provider implementation."""

import asyncio
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import time
import logging

try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from aida.providers.llm.base import (
    LLMProvider, LLMMessage, LLMResponse, LLMError, LLMConfig, 
    LLMRole, LLMUsage, LLMProviderFactory
)


logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self, config: LLMConfig):
        if not OPENAI_AVAILABLE:
            raise LLMError(
                "OpenAI package not available. Install with: pip install openai",
                "MISSING_DEPENDENCY",
                provider="openai"
            )
        
        super().__init__(config)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_url,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        
        # Model pricing (tokens per dollar)
        self.pricing = self._get_model_pricing()
    
    async def chat_completion(
        self, 
        messages: List[LLMMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[LLMResponse, None]]:
        """Generate chat completion using OpenAI."""
        start_time = time.time()
        
        try:
            # Rate limiting
            estimated_tokens = sum([
                await self.estimate_tokens(msg.content) for msg in messages
            ])
            await self.rate_limit_check(estimated_tokens)
            
            # Convert messages to OpenAI format
            openai_messages = [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    **({"name": msg.name} if msg.name else {}),
                    **({"function_call": msg.function_call} if msg.function_call else {}),
                    **({"tool_calls": msg.tool_calls} if msg.tool_calls else {})
                }
                for msg in messages
            ]
            
            # Prepare parameters
            params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "stream": stream,
                **self._get_optional_params(kwargs)
            }
            
            if stream:
                return self._stream_chat_completion(params, start_time)
            else:
                return await self._complete_chat_completion(params, start_time)
                
        except Exception as e:
            self.update_error_stats()
            raise self._handle_error(e)
    
    async def text_completion(
        self, 
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[LLMResponse, None]]:
        """Generate text completion using OpenAI."""
        # Convert to chat format for modern OpenAI models
        messages = [LLMMessage(role=LLMRole.USER, content=prompt)]
        return await self.chat_completion(messages, stream=stream, **kwargs)
    
    async def embedding(
        self, 
        text: Union[str, List[str]],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings using OpenAI."""
        try:
            # Rate limiting
            if isinstance(text, str):
                text = [text]
            
            estimated_tokens = sum([await self.estimate_tokens(t) for t in text])
            await self.rate_limit_check(estimated_tokens)
            
            # Get embeddings
            response = await self.client.embeddings.create(
                model=kwargs.get("embedding_model", "text-embedding-ada-002"),
                input=text
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            self.update_error_stats()
            raise self._handle_error(e)
    
    async def check_health(self) -> bool:
        """Check OpenAI provider health."""
        try:
            # Try a simple completion
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information."""
        model_info = {
            "provider": "openai",
            "model": self.model,
            "type": "chat",
            "max_context": self._get_model_context_length(),
            "supports_streaming": True,
            "supports_functions": True,
            "supports_tools": True
        }
        
        # Add pricing info if available
        if self.model in self.pricing:
            model_info["pricing"] = self.pricing[self.model]
        
        return model_info
    
    async def _complete_chat_completion(
        self, 
        params: Dict[str, Any], 
        start_time: float
    ) -> LLMResponse:
        """Complete non-streaming chat completion."""
        response = await self.client.chat.completions.create(**params)
        response_time = time.time() - start_time
        
        choice = response.choices[0]
        message = choice.message
        
        # Calculate usage and cost
        usage = None
        if response.usage:
            usage = LLMUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                estimated_cost=await self._calculate_cost(response.usage)
            )
        
        # Create response
        llm_response = LLMResponse(
            content=message.content or "",
            model=response.model,
            provider=self.provider_name,
            usage=usage,
            function_call=message.function_call.dict() if message.function_call else None,
            tool_calls=[call.dict() for call in message.tool_calls] if message.tool_calls else None,
            finish_reason=choice.finish_reason,
            metadata={"response_id": response.id}
        )
        
        self.update_stats(llm_response, response_time)
        return llm_response
    
    async def _stream_chat_completion(
        self, 
        params: Dict[str, Any], 
        start_time: float
    ) -> AsyncGenerator[LLMResponse, None]:
        """Stream chat completion responses."""
        full_content = ""
        response_id = None
        model = None
        
        async for chunk in await self.client.chat.completions.create(**params):
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                
                if not response_id:
                    response_id = chunk.id
                    model = chunk.model
                
                if delta.content:
                    full_content += delta.content
                    
                    yield LLMResponse(
                        content=delta.content,
                        model=model or self.model,
                        provider=self.provider_name,
                        finish_reason=chunk.choices[0].finish_reason,
                        metadata={
                            "response_id": response_id,
                            "is_stream_chunk": True,
                            "full_content": full_content
                        }
                    )
        
        # Final response with complete content and usage
        response_time = time.time() - start_time
        final_response = LLMResponse(
            content=full_content,
            model=model or self.model,
            provider=self.provider_name,
            metadata={
                "response_id": response_id,
                "is_final": True,
                "response_time": response_time
            }
        )
        
        self.update_stats(final_response, response_time)
    
    def _get_optional_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get optional parameters for OpenAI API."""
        params = {}
        
        if "max_tokens" in kwargs or self.config.max_tokens:
            params["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)
        
        if "top_p" in kwargs or self.config.top_p:
            params["top_p"] = kwargs.get("top_p", self.config.top_p)
        
        if "frequency_penalty" in kwargs or self.config.frequency_penalty:
            params["frequency_penalty"] = kwargs.get(
                "frequency_penalty", self.config.frequency_penalty
            )
        
        if "presence_penalty" in kwargs or self.config.presence_penalty:
            params["presence_penalty"] = kwargs.get(
                "presence_penalty", self.config.presence_penalty
            )
        
        if "stop" in kwargs or self.config.stop:
            params["stop"] = kwargs.get("stop", self.config.stop)
        
        # Add function calling parameters
        if "functions" in kwargs:
            params["functions"] = kwargs["functions"]
        
        if "function_call" in kwargs:
            params["function_call"] = kwargs["function_call"]
        
        if "tools" in kwargs:
            params["tools"] = kwargs["tools"]
        
        if "tool_choice" in kwargs:
            params["tool_choice"] = kwargs["tool_choice"]
        
        return params
    
    def _get_model_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get OpenAI model pricing information."""
        return {
            "gpt-4": {
                "prompt_tokens_per_dollar": 1000000 / 30,  # $30 per 1M tokens
                "completion_tokens_per_dollar": 1000000 / 60  # $60 per 1M tokens
            },
            "gpt-4-32k": {
                "prompt_tokens_per_dollar": 1000000 / 60,
                "completion_tokens_per_dollar": 1000000 / 120
            },
            "gpt-3.5-turbo": {
                "prompt_tokens_per_dollar": 1000000 / 0.5,  # $0.50 per 1M tokens
                "completion_tokens_per_dollar": 1000000 / 1.5  # $1.50 per 1M tokens
            },
            "gpt-3.5-turbo-16k": {
                "prompt_tokens_per_dollar": 1000000 / 3,
                "completion_tokens_per_dollar": 1000000 / 4
            }
        }
    
    def _get_model_context_length(self) -> int:
        """Get model context length."""
        context_lengths = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000
        }
        return context_lengths.get(self.model, 4096)
    
    async def _calculate_cost(self, usage) -> float:
        """Calculate cost for OpenAI usage."""
        if self.model not in self.pricing:
            return 0.0
        
        pricing = self.pricing[self.model]
        
        prompt_cost = usage.prompt_tokens / pricing["prompt_tokens_per_dollar"]
        completion_cost = usage.completion_tokens / pricing["completion_tokens_per_dollar"]
        
        return prompt_cost + completion_cost
    
    def _handle_error(self, error: Exception) -> LLMError:
        """Handle OpenAI-specific errors."""
        if isinstance(error, openai.RateLimitError):
            return LLMError(
                "OpenAI rate limit exceeded",
                "RATE_LIMIT_EXCEEDED",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )
        elif isinstance(error, openai.AuthenticationError):
            return LLMError(
                "OpenAI authentication failed",
                "AUTHENTICATION_FAILED",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )
        elif isinstance(error, openai.APIError):
            return LLMError(
                f"OpenAI API error: {error}",
                "API_ERROR",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )
        else:
            return LLMError(
                f"OpenAI provider error: {error}",
                "PROVIDER_ERROR",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )


# Register the provider
if OPENAI_AVAILABLE:
    LLMProviderFactory.register("openai", OpenAIProvider)