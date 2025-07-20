"""Default LLM configuration with Ollama as primary provider."""

from typing import Dict, List, Any
from aida.providers.llm.base import LLMConfig


# Default Ollama models to try in order of preference
DEFAULT_OLLAMA_MODELS = [
    "llama3.2:latest",  # General purpose model
    "mistral:latest",  # General purpose model
    "codellama:latest",  # For code generation
    "deepseek-r1:8b",  # For Reasoning
]

# Lightweight models for testing/fallback
LIGHTWEIGHT_OLLAMA_MODELS = [
    "tinyllama:latest",
    "orca-mini:latest", 
    "phi3:mini",
]


def get_default_llm_configs() -> List[Dict[str, Any]]:
    """Get default LLM provider configurations with Ollama as primary."""
    configs = []
    
    # Primary Ollama provider with best available model
    configs.append({
        "provider_name": "ollama",
        "model": "llama3.2:latest",  # Default to Llama 3
        "api_url": "http://localhost:11434",
        "temperature": 0.1,  # Conservative for agent tasks
        "max_tokens": 16384,
        "timeout": 30,
        "is_default": True,
        "custom_headers": {},
        "rate_limit": {
            "requests_per_minute": 60,
            "tokens_per_minute": 50000
        }
    })
    
    # Fallback Ollama providers with different models
    for i, model in enumerate(DEFAULT_OLLAMA_MODELS[1:3], 1):  # Add 2 more fallbacks
        configs.append({
            "provider_name": "ollama", 
            "model": model,
            "api_url": "http://localhost:11434",
            "temperature": 0.1,
            "max_tokens": 16384,
            "timeout": 30,
            "is_default": False,
            "fallback_priority": i
        })
    
    # Add OpenAI as fallback if API key is available
    configs.append({
        "provider_name": "openai",
        "model": "gpt-4o-mini",  # Cost-effective fallback
        "api_key": None,  # Will be loaded from environment
        "temperature": 0.1,
        "max_tokens": 16384,
        "timeout": 30,
        "is_default": False,
        "fallback_priority": 10,
        "enabled": False  # Only enable if API key is provided
    })
    
    return configs


def get_reasoning_system_prompt() -> str:
    """Get system prompt for LLM reasoning layer."""
    return """You are AIDA (Advanced Intelligent Distributed Agent System), a tool orchestrator that MUST respond only in valid JSON format.

CRITICAL: You MUST ALWAYS respond with the exact JSON structure requested. NO conversational text. NO explanations. NO acknowledgments.

Available tools:
- thinking: Complex reasoning and analysis (requires: problem)
- execution: Run code/commands in safe containers (requires: command)  
- file_operations: File management (requires: operation, path)
- system: Monitor system resources (various parameters)
- context: Manage conversation memory (various parameters)
- maintenance: System cleanup (various parameters)
- project: Project management (various parameters)
- architecture: System design (various parameters)

WORKFLOW PATTERNS:
1. General Questions: thinking tool only for analysis and reasoning
2. Script/Code Creation: thinking → file_operations(write_file) → execution(test) → provide result
3. File Operations: thinking → file_operations(appropriate operation) → analysis
4. System Tasks: thinking → execution(commands) → verification
5. Information Requests: thinking tool for research and analysis

TASK CLASSIFICATION:
- Pure information/advice questions: Use thinking tool only
- Code/script requests: Use thinking → file_operations → execution workflow
- File management: Use thinking → file_operations workflow
- System operations: Use thinking → execution workflow
- Research/analysis: Use thinking tool primarily

MANDATORY: Follow the exact JSON format. Include required parameters. Use correct tool names."""


def get_tool_orchestration_prompt(user_message: str, tool_specs: Dict[str, Dict[str, Any]]) -> str:
    """Generate a detailed prompt for tool orchestration with complete tool specifications."""
    
    # Build detailed tool descriptions
    tools_description = []
    for tool_name, spec in tool_specs.items():
        params_desc = []
        for param in spec.get("parameters", []):
            required_marker = " (REQUIRED)" if param.get("required", False) else " (optional)"
            default_info = f" [default: {param.get('default')}]" if param.get("default") is not None else ""
            params_desc.append(f"    - {param['name']}: {param['type']}{required_marker} - {param.get('description', '')}{default_info}")
        
        params_text = "\n".join(params_desc) if params_desc else "    - No parameters"
        
        tools_description.append(f"""
**{tool_name}**: {spec.get('description', 'No description')}
  Parameters:
{params_text}""")
    
    tools_list = "\n".join(tools_description)
    
    return f"""SYSTEM: You are an agent orchestrator. You MUST respond ONLY with valid JSON in the exact format specified below. NO conversational text, NO explanations, NO additional content.

USER REQUEST: {user_message}

AVAILABLE TOOLS:
{tools_list}

RESPONSE FORMAT REQUIREMENT:
You MUST respond with ONLY the following JSON structure. NO other text is allowed:

```json
{{
    "analysis": "Technical planning note: what tool(s) to use and why (NOT the answer to user's question)",
    "execution_plan": [
        {{
            "tool": "exact_tool_name",
            "parameters": {{
                "exact_param_name": "appropriate_value"
            }},
            "purpose": "Clear purpose of this step"
        }}
    ],
    "expected_outcome": "Brief description of what will be provided to the user"
}}
```

MANDATORY RULES:
1. RESPOND ONLY WITH THE JSON BLOCK ABOVE
2. NO conversational text before or after
3. NO acknowledgments or confirmations
4. Use exact tool names from the list above
5. Use exact parameter names as specified
6. Include ALL required parameters
7. Start with ```json and end with ```
8. CRITICAL: For multi-line content, use \\n escape sequences instead of literal newlines in JSON strings
9. CRITICAL: The "analysis" field is for workflow planning only - DO NOT put the user's answer there. Let the tools generate the actual response.

PARAMETER REQUIREMENTS:
- thinking tool: requires 'problem' parameter (for analysis and planning)
- file_operations tool: requires 'operation' and 'path' parameters
  Valid operations: list_files, read_file, write_file, edit_file, search_files, create_directory, delete_file, copy_file, move_file, get_file_info, find_files
- execution tool: requires 'language' and 'code' parameters (runs code in safe containers)
  Valid languages: python, javascript, bash, node, go, rust, java
- For home directory operations, use '$HOME' or '~' for the user's home directory

IMPORTANT TASK ANALYSIS:
FIRST: Consider the conversation context and determine what type of request this is:

CONTEXT-AWARE CLASSIFICATION:
- If this is a follow-up question (asking "why?", "how?", "explain more", etc.) about a previous response, treat it as an INFORMATION question that only needs the thinking tool
- If this is asking for clarification or elaboration on something already discussed, use thinking tool only
- If this is a completely new request, classify based on the content below

1. INFORMATION/ADVICE QUESTIONS (travel, recommendations, explanations, follow-ups, etc.):
   - Use ONLY the thinking tool for comprehensive analysis and response
   - No file operations or execution needed
   - Example: "Where should I vacation?", "Why those places?", "How does X work?", "What is the best approach for Y?"

2. CODE/SCRIPT CREATION REQUESTS:
   - Use thinking tool to analyze requirements, dependencies, and plan complete solution
   - Use file_operations to write complete, executable script with ALL necessary imports
   - Use execution tool to test script with CORRECT language and required packages
   - FILENAME RULES: Create descriptive names (e.g., "dataframe_generator.py", "file_counter.sh")
   - DEPENDENCY ANALYSIS: Identify and include ALL required packages/imports
   - CODE COMPLETENESS: Ensure scripts are complete and include output statements

3. FILE MANAGEMENT REQUESTS:
   - Use thinking tool to plan approach
   - Use file_operations for list/read/search/create operations
   - Example: "Find files containing X", "List directory contents"

4. SYSTEM OPERATION REQUESTS:
   - Use thinking tool to plan approach
   - Use execution tool for system commands
   - Example: "Check system resources", "Install package"

5. ANALYSIS/RESEARCH REQUESTS:
   - Use thinking tool for deep analysis and reasoning
   - May combine with file_operations if analyzing existing files

EXAMPLES FOR DIFFERENT REQUEST TYPES:

1. INFORMATION/ADVICE QUESTION (including follow-ups):
{{
  "analysis": "User asking for travel/advice information, need thinking tool to research and provide detailed recommendations",
  "execution_plan": [
    {{
      "tool": "thinking",
      "parameters": {{
        "problem": "Based on the conversation context, analyze the user's question about [topic] and provide comprehensive, detailed advice. Include specific recommendations, explanations, and relevant details. If this is a follow-up question, reference and build upon previous discussion. Use \\n for line breaks to organize detailed information clearly."
      }},
      "purpose": "Research and provide detailed response considering conversation history"
    }}
  ],
  "expected_outcome": "Detailed recommendations and advice about [topic]"
}}

2. CODE/SCRIPT CREATION:
{{
  "execution_plan": [
    {{
      "tool": "thinking",
      "parameters": {{
        "problem": "Analyze requirements: What language is best? What libraries are needed? What should the complete code structure look like?"
      }},
      "purpose": "Plan complete solution with dependencies"
    }},
    {{
      "tool": "file_operations", 
      "parameters": {{
        "operation": "write_file",
        "path": "./temp/[descriptive_name].[extension]",
        "content": "[complete executable script with imports and logic]"
      }},
      "purpose": "Create executable script file"
    }},
    {{
      "tool": "execution",
      "parameters": {{
        "language": "[language]",
        "code": "[same content]",
        "packages": ["[packages]"]
      }},
      "purpose": "Test script execution"
    }}
  ]
}}

3. FILE OPERATION:
{{
  "execution_plan": [
    {{
      "tool": "thinking",
      "parameters": {{
        "problem": "Plan how to [find/analyze/manage] files based on user requirements"
      }},
      "purpose": "Analyze file operation approach"
    }},
    {{
      "tool": "file_operations",
      "parameters": {{
        "operation": "[list_files/read_file/search_files]",
        "path": "[appropriate_path]"
      }},
      "purpose": "Perform file operation"
    }}
  ]
}}

RESPOND NOW WITH ONLY THE JSON BLOCK:"""


async def setup_default_ollama_provider():
    """Setup Ollama with automatic model detection and fallback."""
    from aida.providers.llm.manager import get_llm_manager
    from aida.providers.llm.base import LLMConfig
    
    manager = get_llm_manager()
    
    # Try to find the best available Ollama model
    ollama_config = LLMConfig(
        provider_name="ollama",
        model="llama3.2:latest",  # Default
        api_url="http://localhost:11434",
        temperature=0.1,
        max_tokens=4096,
        timeout=30
    )
    
    try:
        from aida.providers.llm.ollama import OllamaProvider
        ollama_provider = OllamaProvider(ollama_config)
        
        # Check which models are available
        available_models = await ollama_provider.list_models()
        
        if available_models:
            # Find the best available model from our preferred list
            best_model = None
            for preferred_model in DEFAULT_OLLAMA_MODELS:
                if preferred_model in available_models:
                    best_model = preferred_model
                    break
            
            if not best_model:
                # Fallback to first available model
                best_model = available_models[0]
            
            # Update config with best available model
            ollama_config.model = best_model
            ollama_provider = OllamaProvider(ollama_config)
            
            # Add as default provider
            manager.add_provider(ollama_provider, is_default=True)
            
            return ollama_provider
        else:
            # No models available, try to pull a lightweight one
            await ollama_provider.pull_model("tinyllama:latest")
            ollama_config.model = "tinyllama:latest"
            ollama_provider = OllamaProvider(ollama_config)
            manager.add_provider(ollama_provider, is_default=True)
            return ollama_provider
            
    except Exception as e:
        # Ollama not available, will need to use other providers
        raise RuntimeError(f"Failed to setup Ollama provider: {e}")


async def auto_configure_llm_providers():
    """Automatically configure LLM providers with intelligent fallbacks."""
    from aida.providers.llm.manager import get_llm_manager, setup_llm_providers
    import os
    
    # Start with default configs
    configs = get_default_llm_configs()
    
    # Enable OpenAI if API key is available
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        for config in configs:
            if config.get("provider_name") == "openai":
                config["api_key"] = openai_key
                config["enabled"] = True
                break
    
    # Enable Anthropic if API key is available
    anthropic_key = os.getenv("ANTHROPIC_API_KEY") 
    if anthropic_key:
        configs.append({
            "provider_name": "anthropic",
            "model": "claude-3-haiku-20240307",
            "api_key": anthropic_key,
            "temperature": 0.1,
            "max_tokens": 4096,
            "timeout": 30,
            "is_default": False,
            "fallback_priority": 5,
            "enabled": True
        })
    
    # Setup providers
    manager = await setup_llm_providers(configs)
    
    # Set fallback chain prioritizing local models
    fallback_keys = []
    for config in sorted(configs, key=lambda x: x.get("fallback_priority", 99)):
        if config.get("enabled", True):
            provider_key = f"{config['provider_name']}:{config['model']}"
            fallback_keys.append(provider_key)
    
    if fallback_keys:
        manager.set_fallback_chain(fallback_keys)
    
    return manager