"""LLM provider setup utilities."""

import asyncio
import subprocess
import sys
import os
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LLMSetupGuide:
    """Guide users through LLM provider setup."""
    
    async def check_ollama_availability(self) -> bool:
        """Check if Ollama is installed and running."""
        try:
            # Check if ollama command exists
            result = subprocess.run(["which", "ollama"], capture_output=True)
            if result.returncode != 0:
                return False
            
            # Check if Ollama service is running
            result = subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
    
    async def check_available_models(self) -> List[str]:
        """Check what models are available in Ollama."""
        try:
            result = subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to check Ollama models: {e}")
        return []
    
    async def suggest_model_download(self) -> Optional[str]:
        """Suggest a good model to download."""
        models = await self.check_available_models()
        
        # Preferred models in order
        preferred = [
            "llama3.2:latest",
            "llama3.2:3b", 
            "mistral:latest",
            "codellama:latest",
            "tinyllama:latest"
        ]
        
        # Check if any preferred model is available
        for model in preferred:
            if model in models:
                return model
        
        # If no models, suggest downloading one
        if not models:
            return "llama3.2:3b"  # Smaller, good model
        
        # Return first available model
        return models[0] if models else None
    
    def generate_setup_instructions(self) -> str:
        """Generate setup instructions based on current state."""
        return """
🤖 AIDA LLM Provider Setup Guide

To enable full AIDA functionality, you need an LLM provider:

OPTION 1: Local LLM with Ollama (Recommended)
-------------------------------------------
1. Install Ollama:
   curl -fsSL https://ollama.ai/install.sh | sh

2. Start Ollama service:
   ollama serve

3. Download a model (in another terminal):
   ollama pull llama3.2:3b        # Small, fast model
   # OR
   ollama pull llama3.2:latest    # Larger, more capable

4. Test AIDA:
   uv run aida interactive start

OPTION 2: Cloud LLM Providers
-----------------------------
Set environment variables:

For OpenAI:
   export OPENAI_API_KEY="your-api-key-here"

For Anthropic:
   export ANTHROPIC_API_KEY="your-api-key-here"

OPTION 3: Quick Test (No Setup)
------------------------------
AIDA will work with basic functionality even without LLM providers,
but responses will be limited.

🎯 Once setup, AIDA will provide intelligent tool orchestration
   and natural language responses!
"""

    async def interactive_setup(self):
        """Interactive setup process."""
        print("🔧 AIDA LLM Setup Assistant")
        print("=" * 40)
        
        # Check current state
        ollama_available = await self.check_ollama_availability()
        
        if ollama_available:
            models = await self.check_available_models()
            print(f"✅ Ollama is running with {len(models)} models")
            if models:
                print(f"   Available models: {', '.join(models[:3])}")
                suggested_model = await self.suggest_model_download()
                print(f"✅ AIDA is ready to use with model: {suggested_model}")
                return True
            else:
                print("⚠️  Ollama is running but no models are downloaded")
                print("\nTo download a model, run:")
                print("   ollama pull llama3.2:3b")
                return False
        else:
            print("❌ Ollama not found or not running")
            
            # Check for API keys
            has_openai = bool(os.getenv("OPENAI_API_KEY"))
            has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
            
            if has_openai:
                print("✅ OpenAI API key found")
                return True
            elif has_anthropic:
                print("✅ Anthropic API key found")  
                return True
            else:
                print("❌ No API keys found")
                print("\n" + self.generate_setup_instructions())
                return False


async def setup_llm_providers_interactive():
    """Interactive LLM provider setup."""
    guide = LLMSetupGuide()
    success = await guide.interactive_setup()
    
    if success:
        print("\n🚀 LLM providers are configured!")
        print("   You can now use: uv run aida interactive start")
    else:
        print("\n⚠️  Please complete LLM setup for full functionality")
        
    return success


async def test_llm_integration():
    """Test LLM integration and provide feedback."""
    from aida.providers.llm.manager import get_llm_manager
    from aida.config.llm_defaults import auto_configure_llm_providers
    
    print("🧪 Testing LLM Integration")
    print("-" * 30)
    
    try:
        # Try to configure providers
        manager = await auto_configure_llm_providers()
        providers = manager.list_providers()
        
        if providers:
            print(f"✅ Found {len(providers)} LLM providers:")
            for provider in providers:
                print(f"   - {provider}")
            
            # Test health
            health = await manager.health_check()
            healthy_providers = [k for k, v in health.items() if v]
            
            if healthy_providers:
                print(f"✅ {len(healthy_providers)} providers are healthy")
                return True
            else:
                print("⚠️  No providers are healthy")
                return False
        else:
            print("❌ No providers configured")
            return False
            
    except Exception as e:
        print(f"❌ LLM integration test failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(setup_llm_providers_interactive())