"""vLLM LLM provider implementation."""

import asyncio
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import time
import logging

import aiohttp

from aida.providers.llm.base import (
    LLMProvider, LLMMessage, LLMResponse, LLMError, LLMConfig, 
    LLMRole, LLMUsage, LLMProviderFactory
)


logger = logging.getLogger(__name__)


class VLLMProvider(LLMProvider):
    """vLLM LLM provider implementation."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        # Set default vLLM URL if not provided
        self.base_url = config.api_url or "http://localhost:8000"
        
        # Initialize HTTP session
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.config.custom_headers
            )
        return self.session
    
    async def chat_completion(
        self, 
        messages: List[LLMMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[LLMResponse, None]]:
        """Generate chat completion using vLLM."""
        start_time = time.time()
        
        try:
            # Rate limiting
            estimated_tokens = sum([
                await self.estimate_tokens(msg.content) for msg in messages
            ])
            await self.rate_limit_check(estimated_tokens)
            
            # Convert messages to OpenAI-compatible format (vLLM uses OpenAI API)
            openai_messages = [
                {
                    "role": msg.role.value,
                    "content": msg.content
                }
                for msg in messages
            ]
            
            # Prepare parameters
            params = {
                "model": self.model,
                "messages": openai_messages,
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
        """Generate text completion using vLLM."""
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
        """Generate embeddings using vLLM."""
        try:
            session = await self._get_session()
            
            if isinstance(text, str):
                text = [text]
            
            params = {
                "model": kwargs.get("embedding_model", self.model),
                "input": text
            }
            
            async with session.post(
                f"{self.base_url}/v1/embeddings",
                json=params,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [item["embedding"] for item in data["data"]]
                else:
                    error_text = await response.text()
                    raise LLMError(
                        f"vLLM embedding failed: {error_text}",
                        "API_ERROR",
                        provider=self.provider_name
                    )
            
        except Exception as e:
            self.update_error_stats()
            raise self._handle_error(e)
    
    async def check_health(self) -> bool:
        """Check vLLM provider health."""
        try:
            session = await self._get_session()
            
            # Check if vLLM is running
            async with session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model["id"] for model in data.get("data", [])]
                    return self.model in models
                return False
                
        except Exception as e:
            logger.error(f"vLLM health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get vLLM model information."""
        return {
            "provider": "vllm",
            "model": self.model,
            "type": "chat",
            "max_context": 4096,  # Default, varies by model
            "supports_streaming": True,
            "supports_functions": False,
            "supports_tools": False,
            "local_model": True,
            "base_url": self.base_url
        }
    
    async def _complete_chat_completion(
        self, 
        params: Dict[str, Any], 
        start_time: float
    ) -> LLMResponse:
        """Complete non-streaming chat completion."""
        session = await self._get_session()
        
        async with session.post(
            f"{self.base_url}/v1/chat/completions",
            json=params,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                data = await response.json()
                response_time = time.time() - start_time
                
                choice = data["choices"][0]
                message = choice["message"]
                
                # Calculate usage
                usage = None
                if "usage" in data:
                    usage_data = data["usage"]
                    usage = LLMUsage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0),
                        estimated_cost=0.0  # Local model
                    )
                
                # Create response
                llm_response = LLMResponse(
                    content=message.get("content", ""),
                    model=data.get("model", self.model),
                    provider=self.provider_name,
                    usage=usage,
                    finish_reason=choice.get("finish_reason"),
                    metadata={"response_id": data.get("id")}
                )
                
                self.update_stats(llm_response, response_time)
                return llm_response
            else:
                error_text = await response.text()
                raise LLMError(
                    f"vLLM chat completion failed: {error_text}",
                    "API_ERROR",
                    provider=self.provider_name
                )
    
    async def _complete_text_completion(
        self, 
        params: Dict[str, Any], 
        start_time: float
    ) -> LLMResponse:
        """Complete non-streaming text completion."""
        session = await self._get_session()
        
        async with session.post(
            f"{self.base_url}/v1/completions",
            json=params,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                data = await response.json()
                response_time = time.time() - start_time
                
                choice = data["choices"][0]
                
                # Calculate usage
                usage = None
                if "usage" in data:
                    usage_data = data["usage"]
                    usage = LLMUsage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0),
                        estimated_cost=0.0  # Local model
                    )
                
                # Create response
                llm_response = LLMResponse(
                    content=choice.get("text", ""),
                    model=data.get("model", self.model),
                    provider=self.provider_name,
                    usage=usage,
                    finish_reason=choice.get("finish_reason"),
                    metadata={"response_id": data.get("id")}
                )
                
                self.update_stats(llm_response, response_time)
                return llm_response
            else:
                error_text = await response.text()
                raise LLMError(
                    f"vLLM text completion failed: {error_text}",
                    "API_ERROR",
                    provider=self.provider_name
                )
    
    async def _stream_chat_completion(
        self, 
        params: Dict[str, Any], 
        start_time: float
    ) -> AsyncGenerator[LLMResponse, None]:
        """Stream chat completion responses."""
        session = await self._get_session()
        full_content = ""
        
        async with session.post(
            f"{self.base_url}/v1/chat/completions",
            json=params,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                async for line in response.content:
                    if line:
                        line_str = line.decode().strip()
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]  # Remove "data: " prefix
                            
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                import json
                                data = json.loads(data_str)
                                
                                if "choices" in data and data["choices"]:
                                    choice = data["choices"][0]
                                    delta = choice.get("delta", {})
                                    
                                    if "content" in delta:
                                        delta_content = delta["content"]
                                        full_content += delta_content
                                        
                                        yield LLMResponse(
                                            content=delta_content,
                                            model=data.get("model", self.model),
                                            provider=self.provider_name,
                                            finish_reason=choice.get("finish_reason"),
                                            metadata={
                                                "is_stream_chunk": True,
                                                "full_content": full_content,
                                                "response_id": data.get("id")
                                            }
                                        )
                            except json.JSONDecodeError:
                                continue
            else:
                error_text = await response.text()
                raise LLMError(
                    f"vLLM streaming failed: {error_text}",
                    "API_ERROR",
                    provider=self.provider_name
                )
        
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
    
    async def _stream_text_completion(
        self, 
        params: Dict[str, Any], 
        start_time: float
    ) -> AsyncGenerator[LLMResponse, None]:
        """Stream text completion responses."""
        session = await self._get_session()
        full_content = ""
        
        async with session.post(
            f"{self.base_url}/v1/completions",
            json=params,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                async for line in response.content:
                    if line:
                        line_str = line.decode().strip()
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]  # Remove "data: " prefix
                            
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                import json
                                data = json.loads(data_str)
                                
                                if "choices" in data and data["choices"]:
                                    choice = data["choices"][0]
                                    
                                    if "text" in choice:
                                        delta_content = choice["text"]
                                        full_content += delta_content
                                        
                                        yield LLMResponse(
                                            content=delta_content,
                                            model=data.get("model", self.model),
                                            provider=self.provider_name,
                                            finish_reason=choice.get("finish_reason"),
                                            metadata={
                                                "is_stream_chunk": True,
                                                "full_content": full_content,
                                                "response_id": data.get("id")
                                            }
                                        )
                            except json.JSONDecodeError:
                                continue
            else:
                error_text = await response.text()
                raise LLMError(
                    f"vLLM streaming failed: {error_text}",
                    "API_ERROR",
                    provider=self.provider_name
                )
        
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
    
    def _get_optional_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get optional parameters for vLLM API."""
        params = {}
        
        if "temperature" in kwargs or self.config.temperature:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        
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
        
        # vLLM-specific parameters
        if "top_k" in kwargs:
            params["top_k"] = kwargs["top_k"]
        
        if "repetition_penalty" in kwargs:
            params["repetition_penalty"] = kwargs["repetition_penalty"]
        
        if "length_penalty" in kwargs:
            params["length_penalty"] = kwargs["length_penalty"]
        
        return params
    
    def _handle_error(self, error: Exception) -> LLMError:
        """Handle vLLM-specific errors."""
        if isinstance(error, aiohttp.ClientError):
            return LLMError(
                f"vLLM connection error: {error}",
                "CONNECTION_ERROR",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )
        else:
            return LLMError(
                f"vLLM provider error: {error}",
                "PROVIDER_ERROR",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )
    
    async def list_models(self) -> List[str]:
        """List available models in vLLM."""
        try:
            session = await self._get_session()
            
            async with session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    return [model["id"] for model in data.get("data", [])]
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to list vLLM models: {e}")
            return []
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()


# Register the provider
LLMProviderFactory.register("vllm", VLLMProvider)