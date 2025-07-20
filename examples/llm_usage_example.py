#!/usr/bin/env python3
"""
Example usage of the simplified LLM system.

This example shows how to use AIDA's purpose-based LLM interface
with automatic provider fallbacks.

Prerequisites:
    1. Install PydanticAI: pip install pydantic-ai
    2. For Ollama models: Start Ollama server and pull models
    3. For cloud models: Set API keys in environment variables
"""

import asyncio
import sys
from pathlib import Path

# Add AIDA to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aida.llm import chat, get_llm
from aida.config.llm_profiles import Purpose


async def basic_usage_examples():
    """Show basic usage examples for different purposes."""
    print("🚀 AIDA Simplified LLM System Examples")
    print("=" * 50)
    
    # Default purpose - general chat
    print("\n1. Default Purpose (General Chat)")
    response = await chat("What is the capital of France?")
    print(f"Response: {response}")
    
    # Coding purpose - specialized for code generation
    print("\n2. Coding Purpose")
    code_request = "Write a Python function to calculate fibonacci numbers"
    response = await chat(code_request, purpose=Purpose.CODING)
    print(f"Code Response: {response}")
    
    # Reasoning purpose - for complex analysis
    print("\n3. Reasoning Purpose")
    reasoning_request = "Analyze the pros and cons of remote work vs office work"
    response = await chat(reasoning_request, purpose=Purpose.REASONING)
    print(f"Analysis: {response}")
    
    # Quick purpose - for fast responses
    print("\n4. Quick Purpose")
    quick_request = "What's 2+2?"
    response = await chat(quick_request, purpose=Purpose.QUICK)
    print(f"Quick Response: {response}")


async def streaming_example():
    """Show streaming response example."""
    print("\n" + "=" * 50)
    print("🌊 Streaming Response Example")
    print("=" * 50)
    
    request = "Write a short story about a robot learning to paint"
    print(f"Request: {request}")
    print("Streaming response:")
    
    async for chunk in await chat(request, purpose=Purpose.DEFAULT, stream=True):
        print(chunk, end="", flush=True)
    print("\n")


async def health_check_example():
    """Show health check functionality."""
    print("\n" + "=" * 50)
    print("🏥 Health Check Example")
    print("=" * 50)
    
    llm = get_llm()
    
    # Show available purposes
    purposes = llm.list_purposes()
    print(f"Available purposes: {[p.value for p in purposes]}")
    
    # Check health of all agents
    health_status = await llm.health_check()
    print("Health status:")
    for purpose, is_healthy in health_status.items():
        status = "✅ Healthy" if is_healthy else "❌ Unhealthy"
        print(f"  {purpose.value}: {status}")


async def error_handling_example():
    """Show error handling with fallbacks."""
    print("\n" + "=" * 50)
    print("🛡️ Error Handling Example")
    print("=" * 50)
    
    try:
        # This will demonstrate fallback behavior
        # Primary model (Ollama) might not be available, will fallback to cloud
        response = await chat("Hello, test the fallback system", purpose=Purpose.DEFAULT)
        print(f"Response (with potential fallback): {response}")
        
        print("✅ Fallback system working correctly")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("This might happen if no models are available for the purpose")


async def multimodal_example():
    """Show multimodal capabilities (requires OpenAI API key)."""
    print("\n" + "=" * 50)
    print("🖼️ Multimodal Example (requires OpenAI API key)")
    print("=" * 50)
    
    try:
        # Note: This would require actual image input in real usage
        response = await chat(
            "Describe what you would look for when analyzing an image of a sunset",
            purpose=Purpose.MULTIMODAL
        )
        print(f"Multimodal guidance: {response}")
        
    except Exception as e:
        print(f"Multimodal not available: {e}")
        print("Make sure OPENAI_API_KEY is set for multimodal features")


async def main():
    """Run all examples."""
    try:
        await basic_usage_examples()
        await streaming_example()
        await health_check_example()
        await error_handling_example()
        await multimodal_example()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed!")
        print("\nUsage Summary:")
        print("- Use Purpose.DEFAULT for general questions")
        print("- Use Purpose.CODING for code generation")
        print("- Use Purpose.REASONING for complex analysis")
        print("- Use Purpose.QUICK for fast, simple responses")
        print("- Use Purpose.MULTIMODAL for image/document analysis")
        print("- Add stream=True for streaming responses")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: 'ollama serve'")
        print("2. Pull required models: 'ollama pull llama3.2'")
        print("3. Set API keys: export OPENAI_API_KEY=your_key")
        print("4. Install dependencies: 'pip install pydantic-ai'")


if __name__ == "__main__":
    asyncio.run(main())