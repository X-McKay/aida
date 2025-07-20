"""LLM manager and routing system."""

import asyncio
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
import logging
import random

from aida.providers.llm.base import (
    LLMProvider, LLMMessage, LLMResponse, LLMError, LLMConfig, 
    LLMProviderFactory
)


logger = logging.getLogger(__name__)


class LLMRouter:
    """Intelligent routing for LLM requests."""
    
    def __init__(self):
        self.provider_health: Dict[str, bool] = {}
        self.provider_latency: Dict[str, float] = {}
        self.provider_costs: Dict[str, float] = {}
        self.last_health_check: Dict[str, datetime] = {}
        self.health_check_interval = timedelta(minutes=5)
    
    async def select_provider(
        self,
        providers: List[LLMProvider],
        routing_strategy: str = "cost_optimized",
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[LLMProvider]:
        """Select the best provider based on strategy."""
        if not providers:
            return None
        
        # Filter healthy providers
        healthy_providers = []
        for provider in providers:
            if await self._is_provider_healthy(provider):
                healthy_providers.append(provider)
        
        if not healthy_providers:
            # Fallback to any provider if all appear unhealthy
            healthy_providers = providers
        
        # Apply requirements filter
        if requirements:
            healthy_providers = self._filter_by_requirements(healthy_providers, requirements)
        
        if not healthy_providers:
            return None
        
        # Route based on strategy
        if routing_strategy == "cost_optimized":
            return self._select_by_cost(healthy_providers)
        elif routing_strategy == "performance":
            return self._select_by_performance(healthy_providers)
        elif routing_strategy == "round_robin":
            return self._select_round_robin(healthy_providers)
        elif routing_strategy == "random":
            return random.choice(healthy_providers)
        else:
            # Default to first healthy provider
            return healthy_providers[0]
    
    async def _is_provider_healthy(self, provider: LLMProvider) -> bool:
        """Check if provider is healthy with caching."""
        provider_key = f"{provider.provider_name}:{provider.model}"
        now = datetime.utcnow()
        
        # Check if we need to update health status
        last_check = self.last_health_check.get(provider_key)
        if not last_check or now - last_check > self.health_check_interval:
            try:
                health = await provider.check_health()
                self.provider_health[provider_key] = health
                self.last_health_check[provider_key] = now
                
                # Also update latency if healthy
                if health:
                    start_time = datetime.utcnow()
                    await provider.check_health()
                    latency = (datetime.utcnow() - start_time).total_seconds()
                    self.provider_latency[provider_key] = latency
                    
            except Exception as e:
                logger.warning(f"Health check failed for {provider_key}: {e}")
                self.provider_health[provider_key] = False
                self.last_health_check[provider_key] = now
        
        return self.provider_health.get(provider_key, True)  # Default to healthy
    
    def _filter_by_requirements(
        self, 
        providers: List[LLMProvider], 
        requirements: Dict[str, Any]
    ) -> List[LLMProvider]:
        """Filter providers by requirements."""
        filtered = []
        
        for provider in providers:
            model_info = provider.get_model_info()
            
            # Check streaming requirement
            if requirements.get("streaming") and not model_info.get("supports_streaming"):
                continue
            
            # Check function calling requirement
            if requirements.get("functions") and not model_info.get("supports_functions"):
                continue
            
            # Check context length requirement
            min_context = requirements.get("min_context_length")
            if min_context and model_info.get("max_context", 0) < min_context:
                continue
            
            # Check provider type
            required_provider = requirements.get("provider")
            if required_provider and provider.provider_name != required_provider:
                continue
            
            # Check if local model is required
            if requirements.get("local_only") and not model_info.get("local_model"):
                continue
            
            filtered.append(provider)
        
        return filtered
    
    def _select_by_cost(self, providers: List[LLMProvider]) -> LLMProvider:
        """Select provider with lowest cost."""
        # For local models, cost is 0
        local_providers = [
            p for p in providers 
            if p.get_model_info().get("local_model", False)
        ]
        
        if local_providers:
            return local_providers[0]
        
        # For cloud providers, estimate cost based on pricing
        best_provider = providers[0]
        best_cost = float('inf')
        
        for provider in providers:
            model_info = provider.get_model_info()
            pricing = model_info.get("pricing", {})
            
            if pricing:
                # Estimate cost for 1000 tokens
                estimated_cost = (
                    1000 / pricing.get("prompt_tokens_per_dollar", 1000000) +
                    1000 / pricing.get("completion_tokens_per_dollar", 1000000)
                )
                
                if estimated_cost < best_cost:
                    best_cost = estimated_cost
                    best_provider = provider
        
        return best_provider
    
    def _select_by_performance(self, providers: List[LLMProvider]) -> LLMProvider:
        """Select provider with best performance."""
        best_provider = providers[0]
        best_latency = float('inf')
        
        for provider in providers:
            provider_key = f"{provider.provider_name}:{provider.model}"
            latency = self.provider_latency.get(provider_key, 1.0)  # Default 1s
            
            if latency < best_latency:
                best_latency = latency
                best_provider = provider
        
        return best_provider
    
    def _select_round_robin(self, providers: List[LLMProvider]) -> LLMProvider:
        """Select provider using round-robin."""
        # Simple round-robin based on current time
        index = int(datetime.utcnow().timestamp()) % len(providers)
        return providers[index]


class LLMManager:
    """Manages multiple LLM providers with intelligent routing."""
    
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.router = LLMRouter()
        self.fallback_chain: List[str] = []
        self.default_provider: Optional[str] = None
        
        # Statistics
        self._stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_used": 0,
            "provider_usage": {}
        }
    
    def add_provider(self, provider: LLMProvider, is_default: bool = False) -> None:
        """Add an LLM provider."""
        provider_key = f"{provider.provider_name}:{provider.model}"
        self.providers[provider_key] = provider
        self._stats["provider_usage"][provider_key] = 0
        
        if is_default:
            self.default_provider = provider_key
        
        logger.debug(f"Added LLM provider: {provider_key}")
    
    def add_provider_config(self, config: LLMConfig, is_default: bool = False) -> None:
        """Add provider from configuration."""
        try:
            provider = LLMProviderFactory.create(config)
            self.add_provider(provider, is_default)
        except Exception as e:
            logger.error(f"Failed to create provider from config: {e}")
    
    def set_fallback_chain(self, provider_keys: List[str]) -> None:
        """Set the fallback chain for providers."""
        self.fallback_chain = provider_keys
        logger.debug(f"Set fallback chain: {provider_keys}")
    
    async def chat_completion(
        self,
        messages: List[LLMMessage],
        stream: bool = False,
        routing_strategy: str = "cost_optimized",
        requirements: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[LLMResponse, None]]:
        """Generate chat completion with intelligent routing."""
        self._stats["total_requests"] += 1
        
        # Get available providers
        available_providers = list(self.providers.values())
        
        # Select provider
        provider = await self.router.select_provider(
            available_providers,
            routing_strategy,
            requirements
        )
        
        if not provider:
            self._stats["failed_requests"] += 1
            raise LLMError(
                "No suitable provider available",
                "NO_PROVIDER_AVAILABLE"
            )
        
        # Try primary provider
        try:
            response = await provider.chat_completion(messages, stream=stream, **kwargs)
            self._stats["successful_requests"] += 1
            self._update_provider_usage(provider)
            return response
            
        except Exception as e:
            logger.warning(f"Primary provider {provider.provider_name} failed: {e}")
            
            # Try fallback chain
            for fallback_key in self.fallback_chain:
                if fallback_key in self.providers:
                    fallback_provider = self.providers[fallback_key]
                    
                    try:
                        response = await fallback_provider.chat_completion(
                            messages, stream=stream, **kwargs
                        )
                        self._stats["successful_requests"] += 1
                        self._stats["fallback_used"] += 1
                        self._update_provider_usage(fallback_provider)
                        
                        logger.info(f"Fallback to {fallback_key} successful")
                        return response
                        
                    except Exception as fallback_error:
                        logger.warning(f"Fallback provider {fallback_key} failed: {fallback_error}")
                        continue
            
            # All providers failed
            self._stats["failed_requests"] += 1
            raise LLMError(
                f"All providers failed. Primary error: {e}",
                "ALL_PROVIDERS_FAILED",
                details={"primary_error": str(e)}
            )
    
    async def text_completion(
        self,
        prompt: str,
        stream: bool = False,
        routing_strategy: str = "cost_optimized",
        requirements: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[LLMResponse, None]]:
        """Generate text completion with intelligent routing."""
        # Convert to chat format
        messages = [LLMMessage(role="user", content=prompt)]
        return await self.chat_completion(
            messages, stream=stream, routing_strategy=routing_strategy,
            requirements=requirements, **kwargs
        )
    
    async def embedding(
        self,
        text: Union[str, List[str]],
        provider_preference: Optional[str] = None,
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings with provider selection."""
        # Filter providers that support embeddings
        embedding_providers = [
            provider for provider in self.providers.values()
            if provider.get_model_info().get("supports_embeddings", False)
        ]
        
        if not embedding_providers:
            raise LLMError(
                "No providers support embeddings",
                "FEATURE_NOT_SUPPORTED"
            )
        
        # Use preferred provider if specified
        if provider_preference:
            for provider in embedding_providers:
                if provider.provider_name == provider_preference:
                    return await provider.embedding(text, **kwargs)
        
        # Use first available provider
        provider = embedding_providers[0]
        return await provider.embedding(text, **kwargs)
    
    async def chat(
        self,
        message: str,
        system_message: Optional[str] = None,
        conversation_history: Optional[List[LLMMessage]] = None,
        **kwargs
    ) -> LLMResponse:
        """Simplified chat interface."""
        messages = []
        
        if system_message:
            messages.append(LLMMessage(role="system", content=system_message))
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append(LLMMessage(role="user", content=message))
        
        return await self.chat_completion(messages, **kwargs)
    
    def get_provider(self, provider_key: str) -> Optional[LLMProvider]:
        """Get a specific provider."""
        return self.providers.get(provider_key)
    
    def list_providers(self) -> List[str]:
        """List all provider keys."""
        return list(self.providers.keys())
    
    def get_provider_info(self, provider_key: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get provider information."""
        if provider_key:
            provider = self.providers.get(provider_key)
            if provider:
                return provider.get_model_info()
            return {}
        else:
            return [
                {**provider.get_model_info(), "key": key}
                for key, provider in self.providers.items()
            ]
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers."""
        health_status = {}
        
        tasks = [
            (key, provider.check_health())
            for key, provider in self.providers.items()
        ]
        
        results = await asyncio.gather(
            *[task[1] for task in tasks],
            return_exceptions=True
        )
        
        for (key, _), result in zip(tasks, results):
            if isinstance(result, Exception):
                health_status[key] = False
            else:
                health_status[key] = result
        
        return health_status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            **self._stats,
            "total_providers": len(self.providers),
            "success_rate": (
                self._stats["successful_requests"] / 
                max(1, self._stats["total_requests"])
            ) * 100,
            "fallback_rate": (
                self._stats["fallback_used"] / 
                max(1, self._stats["total_requests"])
            ) * 100
        }
    
    def _update_provider_usage(self, provider: LLMProvider):
        """Update provider usage statistics."""
        provider_key = f"{provider.provider_name}:{provider.model}"
        self._stats["provider_usage"][provider_key] += 1
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cleanup all providers
        for provider in self.providers.values():
            if hasattr(provider, '__aexit__'):
                await provider.__aexit__(exc_type, exc_val, exc_tb)


# Global LLM manager instance
_global_llm_manager: Optional[LLMManager] = None


def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance."""
    global _global_llm_manager
    if _global_llm_manager is None:
        _global_llm_manager = LLMManager()
    return _global_llm_manager


async def setup_llm_providers(configs: List[Dict[str, Any]]) -> LLMManager:
    """Setup LLM providers from configuration."""
    manager = get_llm_manager()
    
    for config_dict in configs:
        try:
            config = LLMConfig(**config_dict)
            is_default = config_dict.get("is_default", False)
            manager.add_provider_config(config, is_default)
        except Exception as e:
            logger.error(f"Failed to setup provider from config {config_dict}: {e}")
    
    return manager