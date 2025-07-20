"""Anthropic LLM provider implementation."""

import asyncio
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import time
import logging

try:
    import anthropic
    from anthropic import AsyncAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from aida.providers.llm.base import (
    LLMProvider, LLMMessage, LLMResponse, LLMError, LLMConfig, 
    LLMRole, LLMUsage, LLMProviderFactory
)


logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider implementation."""
    
    def __init__(self, config: LLMConfig):
        if not ANTHROPIC_AVAILABLE:
            raise LLMError(
                "Anthropic package not available. Install with: pip install anthropic",
                "MISSING_DEPENDENCY",
                provider="anthropic"
            )
        
        super().__init__(config)
        
        # Initialize Anthropic client
        self.client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.api_url,
            timeout=config.timeout,
            max_retries=config.max_retries
        )
        
        # Model pricing
        self.pricing = self._get_model_pricing()
    
    async def chat_completion(
        self, 
        messages: List[LLMMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[LLMResponse, None]]:
        """Generate chat completion using Anthropic."""
        start_time = time.time()
        
        try:
            # Rate limiting
            estimated_tokens = sum([
                await self.estimate_tokens(msg.content) for msg in messages
            ])
            await self.rate_limit_check(estimated_tokens)
            
            # Convert messages to Anthropic format
            anthropic_messages, system_message = self._convert_messages(messages)
            
            # Prepare parameters
            params = {
                "model": self.model,
                "messages": anthropic_messages,
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens or 1000),
                "stream": stream,
                **self._get_optional_params(kwargs)
            }
            
            # Add system message if present
            if system_message:
                params["system"] = system_message
            
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
        """Generate text completion using Anthropic."""
        # Convert to chat format
        messages = [LLMMessage(role=LLMRole.USER, content=prompt)]
        return await self.chat_completion(messages, stream=stream, **kwargs)
    
    async def embedding(
        self, 
        text: Union[str, List[str]],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings using Anthropic."""
        # Anthropic doesn't have a native embedding API
        # This would need to be implemented with a third-party service
        # or using Claude to generate embeddings through prompting
        raise LLMError(
            "Anthropic does not provide native embedding API",
            "FEATURE_NOT_SUPPORTED",
            provider=self.provider_name
        )
    
    async def check_health(self) -> bool:
        """Check Anthropic provider health."""
        try:
            # Try a simple completion
            response = await self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Anthropic model information."""
        model_info = {
            "provider": "anthropic",
            "model": self.model,
            "type": "chat",
            "max_context": self._get_model_context_length(),
            "supports_streaming": True,
            "supports_functions": False,  # Anthropic uses tool calling differently
            "supports_tools": True
        }
        
        # Add pricing info if available
        if self.model in self.pricing:
            model_info["pricing"] = self.pricing[self.model]
        
        return model_info
    
    def _convert_messages(self, messages: List[LLMMessage]) -> tuple:
        """Convert AIDA messages to Anthropic format."""
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if msg.role == LLMRole.SYSTEM:
                # Anthropic handles system messages separately
                system_message = msg.content
            elif msg.role in [LLMRole.USER, LLMRole.ASSISTANT]:
                anthropic_messages.append({
                    "role": msg.role.value,
                    "content": msg.content
                })
            # Skip function/tool messages for now - would need special handling
        
        return anthropic_messages, system_message
    
    async def _complete_chat_completion(
        self, 
        params: Dict[str, Any], 
        start_time: float
    ) -> LLMResponse:
        """Complete non-streaming chat completion."""
        response = await self.client.messages.create(**params)
        response_time = time.time() - start_time
        
        # Extract content from response
        content = ""
        if response.content:
            content = "".join([
                block.text for block in response.content 
                if hasattr(block, 'text')
            ])
        
        # Calculate usage and cost
        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = LLMUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                estimated_cost=await self._calculate_cost(response.usage)
            )
        
        # Create response
        llm_response = LLMResponse(
            content=content,
            model=response.model,
            provider=self.provider_name,
            usage=usage,
            finish_reason=response.stop_reason,
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
        
        async with self.client.messages.stream(**params) as stream:
            async for event in stream:
                if event.type == "message_start":
                    response_id = event.message.id
                    model = event.message.model
                elif event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        delta_text = event.delta.text
                        full_content += delta_text
                        
                        yield LLMResponse(
                            content=delta_text,
                            model=model or self.model,
                            provider=self.provider_name,
                            metadata={
                                "response_id": response_id,
                                "is_stream_chunk": True,
                                "full_content": full_content
                            }
                        )
        
        # Final response
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
        """Get optional parameters for Anthropic API."""
        params = {}
        
        if "temperature" in kwargs or self.config.temperature:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        
        if "top_p" in kwargs or self.config.top_p:
            params["top_p"] = kwargs.get("top_p", self.config.top_p)
        
        if "stop" in kwargs or self.config.stop:
            stop_sequences = kwargs.get("stop", self.config.stop)
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            params["stop_sequences"] = stop_sequences
        
        # Add tool calling parameters
        if "tools" in kwargs:
            params["tools"] = kwargs["tools"]
        
        if "tool_choice" in kwargs:
            params["tool_choice"] = kwargs["tool_choice"]
        
        return params
    
    def _get_model_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get Anthropic model pricing information."""
        return {
            "claude-3-opus-20240229": {
                "prompt_tokens_per_dollar": 1000000 / 15,  # $15 per 1M tokens
                "completion_tokens_per_dollar": 1000000 / 75  # $75 per 1M tokens
            },
            "claude-3-sonnet-20240229": {
                "prompt_tokens_per_dollar": 1000000 / 3,  # $3 per 1M tokens
                "completion_tokens_per_dollar": 1000000 / 15  # $15 per 1M tokens
            },
            "claude-3-haiku-20240307": {
                "prompt_tokens_per_dollar": 1000000 / 0.25,  # $0.25 per 1M tokens
                "completion_tokens_per_dollar": 1000000 / 1.25  # $1.25 per 1M tokens
            }
        }
    
    def _get_model_context_length(self) -> int:
        """Get model context length."""
        context_lengths = {
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-2.1": 200000,
            "claude-2.0": 100000,
            "claude-instant-1.2": 100000
        }
        return context_lengths.get(self.model, 100000)
    
    async def _calculate_cost(self, usage) -> float:
        """Calculate cost for Anthropic usage."""
        if self.model not in self.pricing:
            return 0.0
        
        pricing = self.pricing[self.model]
        
        prompt_cost = usage.input_tokens / pricing["prompt_tokens_per_dollar"]
        completion_cost = usage.output_tokens / pricing["completion_tokens_per_dollar"]
        
        return prompt_cost + completion_cost
    
    def _handle_error(self, error: Exception) -> LLMError:
        """Handle Anthropic-specific errors."""
        if isinstance(error, anthropic.RateLimitError):
            return LLMError(
                "Anthropic rate limit exceeded",
                "RATE_LIMIT_EXCEEDED",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )
        elif isinstance(error, anthropic.AuthenticationError):
            return LLMError(
                "Anthropic authentication failed",
                "AUTHENTICATION_FAILED",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )
        elif isinstance(error, anthropic.APIError):
            return LLMError(
                f"Anthropic API error: {error}",
                "API_ERROR",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )
        else:
            return LLMError(
                f"Anthropic provider error: {error}",
                "PROVIDER_ERROR",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )


# Register the provider
if ANTHROPIC_AVAILABLE:
    LLMProviderFactory.register("anthropic", AnthropicProvider)