"""Ollama LLM provider implementation."""

import asyncio
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import time
import logging

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

import aiohttp

from aida.providers.llm.base import (
    LLMProvider, LLMMessage, LLMResponse, LLMError, LLMConfig, 
    LLMRole, LLMUsage, LLMProviderFactory
)


logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation."""
    
    def __init__(self, config: LLMConfig):
        if not OLLAMA_AVAILABLE:
            raise LLMError(
                "Ollama package not available. Install with: pip install ollama",
                "MISSING_DEPENDENCY",
                provider="ollama"
            )
        
        super().__init__(config)
        
        # Set default Ollama URL if not provided
        self.base_url = config.api_url or "http://localhost:11434"
        
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
        """Generate chat completion using Ollama."""
        start_time = time.time()
        
        try:
            # Rate limiting (basic for local models)
            estimated_tokens = sum([
                await self.estimate_tokens(msg.content) for msg in messages
            ])
            await self.rate_limit_check(estimated_tokens)
            
            # Convert messages to Ollama format
            ollama_messages = [
                {
                    "role": msg.role.value,
                    "content": msg.content
                }
                for msg in messages
            ]
            
            # Prepare parameters
            params = {
                "model": self.model,
                "messages": ollama_messages,
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
        """Generate text completion using Ollama."""
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
        """Generate embeddings using Ollama."""
        try:
            session = await self._get_session()
            
            if isinstance(text, str):
                text = [text]
            
            embeddings = []
            
            for txt in text:
                params = {
                    "model": kwargs.get("embedding_model", self.model),
                    "prompt": txt
                }
                
                async with session.post(
                    f"{self.base_url}/api/embeddings",
                    json=params
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        embeddings.append(data.get("embedding", []))
                    else:
                        error_text = await response.text()
                        raise LLMError(
                            f"Ollama embedding failed: {error_text}",
                            "API_ERROR",
                            provider=self.provider_name
                        )
            
            return embeddings
            
        except Exception as e:
            self.update_error_stats()
            raise self._handle_error(e)
    
    async def check_health(self) -> bool:
        """Check Ollama provider health."""
        try:
            session = await self._get_session()
            
            # Check if Ollama is running
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    # Check if model is available
                    data = await response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    return self.model in models
                return False
                
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information."""
        return {
            "provider": "ollama",
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
            f"{self.base_url}/api/chat",
            json=params
        ) as response:
            if response.status == 200:
                data = await response.json()
                response_time = time.time() - start_time
                
                # Create response
                llm_response = LLMResponse(
                    content=data.get("message", {}).get("content", ""),
                    model=self.model,
                    provider=self.provider_name,
                    usage=self._extract_usage(data),
                    metadata={
                        "created_at": data.get("created_at"),
                        "done": data.get("done", True)
                    }
                )
                
                self.update_stats(llm_response, response_time)
                return llm_response
            else:
                error_text = await response.text()
                raise LLMError(
                    f"Ollama chat completion failed: {error_text}",
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
            f"{self.base_url}/api/generate",
            json=params
        ) as response:
            if response.status == 200:
                data = await response.json()
                response_time = time.time() - start_time
                
                # Create response
                llm_response = LLMResponse(
                    content=data.get("response", ""),
                    model=self.model,
                    provider=self.provider_name,
                    usage=self._extract_usage(data),
                    metadata={
                        "created_at": data.get("created_at"),
                        "done": data.get("done", True)
                    }
                )
                
                self.update_stats(llm_response, response_time)
                return llm_response
            else:
                error_text = await response.text()
                raise LLMError(
                    f"Ollama text completion failed: {error_text}",
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
            f"{self.base_url}/api/chat",
            json=params
        ) as response:
            if response.status == 200:
                async for line in response.content:
                    if line:
                        try:
                            import json
                            data = json.loads(line.decode().strip())
                            
                            if "message" in data and "content" in data["message"]:
                                delta_content = data["message"]["content"]
                                full_content += delta_content
                                
                                yield LLMResponse(
                                    content=delta_content,
                                    model=self.model,
                                    provider=self.provider_name,
                                    metadata={
                                        "is_stream_chunk": True,
                                        "full_content": full_content,
                                        "done": data.get("done", False)
                                    }
                                )
                                
                                if data.get("done", False):
                                    break
                        except json.JSONDecodeError:
                            continue
            else:
                error_text = await response.text()
                raise LLMError(
                    f"Ollama streaming failed: {error_text}",
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
            f"{self.base_url}/api/generate",
            json=params
        ) as response:
            if response.status == 200:
                async for line in response.content:
                    if line:
                        try:
                            import json
                            data = json.loads(line.decode().strip())
                            
                            if "response" in data:
                                delta_content = data["response"]
                                full_content += delta_content
                                
                                yield LLMResponse(
                                    content=delta_content,
                                    model=self.model,
                                    provider=self.provider_name,
                                    metadata={
                                        "is_stream_chunk": True,
                                        "full_content": full_content,
                                        "done": data.get("done", False)
                                    }
                                )
                                
                                if data.get("done", False):
                                    break
                        except json.JSONDecodeError:
                            continue
            else:
                error_text = await response.text()
                raise LLMError(
                    f"Ollama streaming failed: {error_text}",
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
        """Get optional parameters for Ollama API."""
        params = {}
        
        # Ollama-specific parameters
        if "temperature" in kwargs or self.config.temperature:
            params["temperature"] = kwargs.get("temperature", self.config.temperature)
        
        if "top_p" in kwargs or self.config.top_p:
            params["top_p"] = kwargs.get("top_p", self.config.top_p)
        
        if "stop" in kwargs or self.config.stop:
            stop_sequences = kwargs.get("stop", self.config.stop)
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]
            params["stop"] = stop_sequences
        
        # Ollama-specific options
        options = {}
        
        if "num_predict" in kwargs:
            options["num_predict"] = kwargs["num_predict"]
        elif self.config.max_tokens:
            options["num_predict"] = self.config.max_tokens
        
        if "repeat_penalty" in kwargs:
            options["repeat_penalty"] = kwargs["repeat_penalty"]
        
        if "seed" in kwargs:
            options["seed"] = kwargs["seed"]
        
        if options:
            params["options"] = options
        
        return params
    
    def _extract_usage(self, data: Dict[str, Any]) -> Optional[LLMUsage]:
        """Extract usage information from Ollama response."""
        # Ollama doesn't provide detailed token usage by default
        # We can estimate based on response length
        if "response" in data or "message" in data:
            content = data.get("response", "") or data.get("message", {}).get("content", "")
            estimated_tokens = max(1, len(content) // 4)
            
            return LLMUsage(
                prompt_tokens=0,  # Not available
                completion_tokens=estimated_tokens,
                total_tokens=estimated_tokens,
                estimated_cost=0.0  # Local model
            )
        
        return None
    
    def _handle_error(self, error: Exception) -> LLMError:
        """Handle Ollama-specific errors."""
        if isinstance(error, aiohttp.ClientError):
            return LLMError(
                f"Ollama connection error: {error}",
                "CONNECTION_ERROR",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )
        else:
            return LLMError(
                f"Ollama provider error: {error}",
                "PROVIDER_ERROR",
                provider=self.provider_name,
                model=self.model,
                details={"original_error": str(error)}
            )
    
    async def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            session = await self._get_session()
            
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return [model["name"] for model in data.get("models", [])]
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama repository."""
        try:
            session = await self._get_session()
            
            params = {"name": model_name}
            
            async with session.post(
                f"{self.base_url}/api/pull",
                json=params
            ) as response:
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Failed to pull Ollama model {model_name}: {e}")
            return False
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()


# Register the provider
if OLLAMA_AVAILABLE:
    LLMProviderFactory.register("ollama", OllamaProvider)