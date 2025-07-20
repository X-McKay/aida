"""Simple LLM manager with purpose-based routing."""

from typing import Dict, AsyncGenerator, Union
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.anthropic import AnthropicModel  
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.anthropic import AnthropicProvider

from ..config.models import ModelSpec, Provider
from ..config.llm_profiles import Purpose, LLMProfile, get_profile, get_available_purposes


class LLMManager:
    """Simple LLM manager with purpose-based routing."""
    
    def __init__(self):
        self._agents: Dict[Purpose, Agent] = {}
        self._setup_models()
    
    def _setup_models(self):
        """Setup models and agents for available purposes."""
        for purpose in get_available_purposes():
            profile = get_profile(purpose)
            model = self._create_model_for_profile(profile)
            
            self._agents[purpose] = Agent(
                model=model,
                system_prompt=profile.prompt
            )
    
    def _create_model_for_profile(self, profile: LLMProfile) -> Model:
        """Create model for profile."""
        if profile.model.is_available:
            return self._create_pydantic_model(profile.model)
        
        raise ValueError(f"No available model for purpose: {profile.purpose}")
    
    def _create_pydantic_model(self, spec: ModelSpec) -> Model:
        """Create PydanticAI model from spec."""
        settings = {
            'temperature': spec.temperature,
            'max_tokens': spec.max_tokens
        }
        
        if spec.provider == Provider.OPENAI:
            return OpenAIModel(
                spec.model_id,
                provider=OpenAIProvider(api_key=spec.api_key),
                settings=settings
            )
        elif spec.provider == Provider.ANTHROPIC:
            return AnthropicModel(
                spec.model_id,
                provider=AnthropicProvider(api_key=spec.api_key),
                settings=settings
            )
        elif spec.provider in [Provider.OLLAMA, Provider.VLLM]:
            # Both Ollama and vLLM use OpenAI-compatible interface
            base_url = spec.base_url + "/v1" if spec.provider == Provider.OLLAMA else spec.base_url
            return OpenAIModel(
                spec.model_id,
                provider=OpenAIProvider(base_url=base_url),
                settings=settings
            )
        else:
            raise ValueError(f"Unsupported provider: {spec.provider}")
    
    async def chat(
        self, 
        message: str,
        purpose: Purpose = Purpose.DEFAULT,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Chat with LLM."""
        agent = self._agents.get(purpose)
        if not agent:
            raise ValueError(f"No agent for purpose: {purpose}")
        
        if stream:
            return self._stream_chat(agent, message)
        else:
            result = await agent.run(message)
            return result.data
    
    async def _stream_chat(self, agent: Agent, message: str) -> AsyncGenerator[str, None]:
        """Stream chat response."""
        async with agent.run_stream(message) as result:
            async for text in result.stream_text(delta=True):
                yield text
    
    def list_purposes(self) -> list[Purpose]:
        """List available purposes."""
        return list(self._agents.keys())
    
    async def health_check(self) -> Dict[Purpose, bool]:
        """Check health of all agents."""
        health = {}
        for purpose, agent in self._agents.items():
            try:
                await agent.run("test")
                health[purpose] = True
            except Exception:
                health[purpose] = False
        return health