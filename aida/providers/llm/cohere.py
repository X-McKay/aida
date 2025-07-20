"""Cohere LLM provider implementation."""

import asyncio
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import time
import logging

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

from aida.providers.llm.base import (
    LLMProvider, LLMMessage, LLMResponse, LLMError, LLMConfig, 
    LLMRole, LLMUsage, LLMProviderFactory
)


logger = logging.getLogger(__name__)


class CohereProvider(LLMProvider):
    """Cohere LLM provider implementation."""
    
    def __init__(self, config: LLMConfig):
        if not COHERE_AVAILABLE:
            raise LLMError(
                "Cohere package not available. Install with: pip install cohere",
                "MISSING_DEPENDENCY",
                provider="cohere"
            )
        
        super().__init__(config)
        
        # Initialize Cohere client
        self.client = cohere.AsyncClient(
            api_key=config.api_key,
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
        """Generate chat completion using Cohere."""
        start_time = time.time()
        
        try:
            # Rate limiting
            estimated_tokens = sum([
                await self.estimate_tokens(msg.content) for msg in messages
            ])
            await self.rate_limit_check(estimated_tokens)
            
            # Convert messages to Cohere format
            cohere_messages, system_message = self._convert_messages(messages)
            
            # Prepare parameters
            params = {
                "model": self.model,
                "message": cohere_messages[-1]["content"] if cohere_messages else "",
                "chat_history": cohere_messages[:-1] if len(cohere_messages) > 1 else [],
                "stream": stream,
                **self._get_optional_params(kwargs)
            }
            
            # Add system message (preamble in Cohere)
            if system_message:
                params["preamble"] = system_message
            
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
        """Generate text completion using Cohere."""
        start_time = time.time()
        
        try:
            estimated_tokens = await self.estimate_tokens(prompt)
            await self.rate_limit_check(estimated_tokens)
            
            params = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                **self._get_optional_params(kwargs)
            }
            
            if stream:
                return self._stream_text_completion(params, start_time)
            else:
                return await self._complete_text_completion(params, start_time)
                
        except Exception as e:
            self.update_error_stats()
            raise self._handle_error(e)
    
    async def embedding(
        self, 
        text: Union[str, List[str]],
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings using Cohere."""
        try:
            if isinstance(text, str):
                text = [text]
            
            estimated_tokens = sum([await self.estimate_tokens(t) for t in text])
            await self.rate_limit_check(estimated_tokens)
            
            response = await self.client.embed(
                model=kwargs.get("embedding_model", "embed-english-v3.0"),
                texts=text,
                input_type=kwargs.get("input_type", "search_document")
            )
            
            return response.embeddings
            
        except Exception as e:
            self.update_error_stats()
            raise self._handle_error(e)
    
    async def check_health(self) -> bool:
        """Check Cohere provider health."""
        try:
            # Try a simple chat completion
            response = await self.client.chat(
                model=self.model,
                message="Hello",
                max_tokens=1
            )
            return True
        except Exception as e:
            logger.error(f"Cohere health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Cohere model information."""
        model_info = {
            "provider": "cohere",
            "model": self.model,
            "type": "chat",
            "max_context": self._get_model_context_length(),
            "supports_streaming": True,
            "supports_functions": False,
            "supports_tools": True,
            "supports_embeddings": True
        }
        
        # Add pricing info if available
        if self.model in self.pricing:
            model_info["pricing"] = self.pricing[self.model]
        
        return model_info
    
    def _convert_messages(self, messages: List[LLMMessage]) -> tuple:
        """Convert AIDA messages to Cohere format."""
        cohere_messages = []
        system_message = None
        
        for msg in messages:
            if msg.role == LLMRole.SYSTEM:
                # Cohere uses preamble for system messages
                system_message = msg.content
            elif msg.role == LLMRole.USER:
                cohere_messages.append({
                    "role": "USER",
                    "content": msg.content
                })
            elif msg.role == LLMRole.ASSISTANT:
                cohere_messages.append({
                    "role": "CHATBOT",
                    "content": msg.content
                })
        
        return cohere_messages, system_message
    
    async def _complete_chat_completion(
        self, 
        params: Dict[str, Any], 
        start_time: float
    ) -> LLMResponse:
        """Complete non-streaming chat completion."""
        response = await self.client.chat(**params)
        response_time = time.time() - start_time
        
        # Calculate usage and cost
        usage = None
        if hasattr(response, 'meta') and response.meta:
            billed_units = response.meta.billed_units
            if billed_units:
                usage = LLMUsage(
                    prompt_tokens=billed_units.input_tokens or 0,
                    completion_tokens=billed_units.output_tokens or 0,
                    total_tokens=(billed_units.input_tokens or 0) + (billed_units.output_tokens or 0),
                    estimated_cost=await self._calculate_cost(billed_units)
                )
        
        # Create response
        llm_response = LLMResponse(
            content=response.text,
            model=self.model,
            provider=self.provider_name,
            usage=usage,
            finish_reason=response.finish_reason if hasattr(response, 'finish_reason') else None,
            metadata={
                "generation_id": response.generation_id if hasattr(response, 'generation_id') else None,
                "response_id": response.response_id if hasattr(response, 'response_id') else None
            }
        )
        
        self.update_stats(llm_response, response_time)
        return llm_response
    
    async def _complete_text_completion(
        self, 
        params: Dict[str, Any], 
        start_time: float
    ) -> LLMResponse:
        """Complete non-streaming text completion."""
        response = await self.client.generate(**params)
        response_time = time.time() - start_time
        
        # Get the first generation
        generation = response.generations[0] if response.generations else None
        content = generation.text if generation else ""
        
        # Calculate usage
        usage = None
        if hasattr(response, 'meta') and response.meta:
            billed_units = response.meta.billed_units
            if billed_units:
                usage = LLMUsage(
                    prompt_tokens=billed_units.input_tokens or 0,
                    completion_tokens=billed_units.output_tokens or 0,
                    total_tokens=(billed_units.input_tokens or 0) + (billed_units.output_tokens or 0),
                    estimated_cost=await self._calculate_cost(billed_units)
                )
        
        # Create response
        llm_response = LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            usage=usage,
            finish_reason=generation.finish_reason if generation else None,
            metadata={
                "generation_id": generation.id if generation else None
            }
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
        
        stream = self.client.chat_stream(**params)
        
        async for event in stream:
            if event.event_type == "text-generation":
                delta_text = event.text
                full_content += delta_text
                
                yield LLMResponse(
                    content=delta_text,
                    model=self.model,
                    provider=self.provider_name,
                    metadata={
                        "is_stream_chunk": True,
                        "full_content": full_content,
                        "event_type": event.event_type
                    }
                )
            elif event.event_type == "stream-end":
                # Final response with usage info
                response_time = time.time() - start_time
                
                usage = None
                if hasattr(event, 'response') and event.response and hasattr(event.response, 'meta'):
                    billed_units = event.response.meta.billed_units
                    if billed_units:
                        usage = LLMUsage(
                            prompt_tokens=billed_units.input_tokens or 0,
                            completion_tokens=billed_units.output_tokens or 0,
                            total_tokens=(billed_units.input_tokens or 0) + (billed_units.output_tokens or 0),
                            estimated_cost=await self._calculate_cost(billed_units)
                        )
                
                final_response = LLMResponse(
                    content=full_content,
                    model=self.model,
                    provider=self.provider_name,
                    usage=usage,
                    metadata={
                        "is_final": True,
                        "response_time": response_time
                    }
                )
                
                self.update_stats(final_response, response_time)
                break
    
    async def _stream_text_completion(
        self, 
        params: Dict[str, Any], 
        start_time: float
    ) -> AsyncGenerator[LLMResponse, None]:
        """Stream text completion responses."""
        full_content = ""
        
        stream = self.client.generate_stream(**params)
        
        async for event in stream:
            if event.event_type == "text-generation":
                delta_text = event.text
                full_content += delta_text
                
                yield LLMResponse(
                    content=delta_text,
                    model=self.model,
                    provider=self.provider_name,
                    metadata={
                        "is_stream_chunk": True,
                        "full_content": full_content,
                        "event_type": event.event_type
                    }
                )
            elif event.event_type == "stream-end":
                # Final response
                response_time = time.time() - start_time
                final_response = LLMResponse(
                    content=full_content,
                    model=self.model,
                    provider=self.provider_name,
                    metadata={
                        "is_final": True,
                        "response_time": response_time
                    }
                )
                
                self.update_stats(final_response, response_time)
                break
    
    def _get_optional_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get optional parameters for Cohere API."""
        params = {}
        
        if "temperature" in kwargs or self.config.temperature:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        
        if "max_tokens" in kwargs or self.config.max_tokens:
            params["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)
        
        if "top_p" in kwargs or self.config.top_p:
            params["p"] = kwargs.get("top_p", self.config.top_p)
        
        if "frequency_penalty" in kwargs or self.config.frequency_penalty:
            params["frequency_penalty"] = kwargs.get(
                "frequency_penalty", self.config.frequency_penalty
            )
        
        if "presence_penalty" in kwargs or self.config.presence_penalty:
            params["presence_penalty"] = kwargs.get(
                "presence_penalty", self.config.presence_penalty
            )
        
        if "stop" in kwargs or self.config.stop:
            stop_sequences = kwargs.get("stop", self.config.stop)
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            params["stop_sequences"] = stop_sequences
        
        # Cohere-specific parameters
        if "k" in kwargs:  # top-k sampling
            params["k"] = kwargs["k"]
        
        if "return_likelihoods" in kwargs:
            params["return_likelihoods"] = kwargs["return_likelihoods"]
        
        return params
    
    def _get_model_pricing(self) -> Dict[str, Dict[str, float]]:
        """Get Cohere model pricing information."""
        return {
            "command": {
                "prompt_tokens_per_dollar": 1000000 / 1.0,  # $1 per 1M tokens
                "completion_tokens_per_dollar": 1000000 / 2.0  # $2 per 1M tokens
            },
            "command-light": {
                "prompt_tokens_per_dollar": 1000000 / 0.3,  # $0.30 per 1M tokens
                "completion_tokens_per_dollar": 1000000 / 0.6  # $0.60 per 1M tokens
            },
            "command-nightly": {
                "prompt_tokens_per_dollar": 1000000 / 1.0,
                "completion_tokens_per_dollar": 1000000 / 2.0
            }
        }
    
    def _get_model_context_length(self) -> int:
        """Get model context length."""
        context_lengths = {
            "command": 4096,
            "command-light": 4096,
            "command-nightly": 4096,
            "command-r": 128000,
            "command-r-plus": 128000
        }
        return context_lengths.get(self.model, 4096)
    
    async def _calculate_cost(self, billed_units) -> float:
        """Calculate cost for Cohere usage."""
        if self.model not in self.pricing:
            return 0.0
        
        pricing = self.pricing[self.model]
        
        input_tokens = billed_units.input_tokens or 0
        output_tokens = billed_units.output_tokens or 0
        
        prompt_cost = input_tokens / pricing["prompt_tokens_per_dollar"]
        completion_cost = output_tokens / pricing["completion_tokens_per_dollar"]
        
        return prompt_cost + completion_cost
    
    def _handle_error(self, error: Exception) -> LLMError:
        """Handle Cohere-specific errors."""
        if isinstance(error, cohere.errors.TooManyRequestsError):
            return LLMError(
                "Cohere rate limit exceeded",
                "RATE_LIMIT_EXCEEDED",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )
        elif isinstance(error, cohere.errors.UnauthorizedError):
            return LLMError(
                "Cohere authentication failed",
                "AUTHENTICATION_FAILED",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )
        elif isinstance(error, cohere.errors.CohereAPIError):
            return LLMError(
                f"Cohere API error: {error}",
                "API_ERROR",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )
        else:
            return LLMError(
                f"Cohere provider error: {error}",
                "PROVIDER_ERROR",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )


# Register the provider
if COHERE_AVAILABLE:
    LLMProviderFactory.register("cohere", CohereProvider)