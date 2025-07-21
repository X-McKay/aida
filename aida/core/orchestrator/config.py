"""Configuration for the TODO orchestrator."""

from typing import Any

from aida.config.llm_profiles import Purpose


class OrchestratorConfig:
    """Configuration class for the TODO orchestrator."""

    # LLM Configuration
    LLM_PURPOSE = Purpose.DEFAULT  # Use default model instead of reasoning model

    # Plan Generation Configuration
    MAX_STEPS_PER_PLAN = 10
    DEFAULT_TOOL_NAME = "thinking"
    DEFAULT_STORAGE_DIR = ".aida/plans"

    # Retry Configuration
    DEFAULT_MAX_RETRIES = 2
    RETRY_ERRORS = ["timeout", "connection", "temporary", "rate limit"]

    # Replan Configuration
    REPLAN_AFTER_STEPS = 5  # Replan after this many completed steps
    REPLAN_AFTER_SECONDS = 600  # Replan after 10 minutes

    # JSON Validation
    REQUIRED_PLAN_FIELDS = ["analysis", "expected_outcome", "execution_plan"]
    REQUIRED_STEP_FIELDS = ["description"]

    @classmethod
    def get_planning_prompt_template(cls) -> str:
        """Get the planning prompt template."""
        return """
You are a workflow planning assistant. Analyze the user's request and determine the best approach.

USER REQUEST: {user_request}

AVAILABLE TOOLS:
{tools_info}

CONTEXT: {context}

IMPORTANT: First determine if this request needs tools or can be answered directly:
- General knowledge questions -> Use "llm_response" tool to provide an answer
- Questions about information -> Use "llm_response" tool
- Questions like "What are good places to...", "How do I...", "Explain..." -> Use "llm_response" tool
- File operations -> Use "file_operations" tool ONLY when user explicitly asks about files
- Code execution -> Use "execution" tool ONLY when user wants to run code
- System commands -> Use "system" tool ONLY when user wants system info

CRITICAL: When the user's question refers to previous conversation (uses words like "those", "it", "them", "the above", etc.):
1. Look at the CONTEXT for conversation_history
2. Extract relevant previous messages
3. Pass them as the "context" parameter to llm_response tool

Respond ONLY with valid JSON:

{{
    "analysis": "Brief analysis of what needs to be done",
    "expected_outcome": "What the user should expect as a result",
    "execution_plan": [
        {{
            "description": "Clear description of what this step does",
            "tool_name": "tool_name_here",
            "parameters": {{"param1": "value1", "param2": "value2"}},
            "dependencies": []
        }}
    ]
}}

Guidelines:
- For general questions -> Create ONE step using llm_response tool
- IMPORTANT: If the CONTEXT includes conversation_history, you MUST pass it to llm_response as context
- Only use file_operations when the user explicitly asks about files
- Only use execution when the user wants to run code
- Create 1-3 steps maximum to keep plans simple
- Respond with ONLY the JSON object, no other text

Example for a general question with context:
{{
    "analysis": "User wants information about which snowboarding location is best in March, referring to previously mentioned locations",
    "expected_outcome": "Information about the best snowboarding location in March from the previously listed options",
    "execution_plan": [
        {{
            "description": "Provide information about the best snowboarding location in March",
            "tool_name": "llm_response",
            "parameters": {{
                "question": "Which of those is typically the best or most popular in March?",
                "context": "Previous conversation discussed these East Coast snowboarding locations: Stowe Vermont, Killington Vermont, Hunter Mountain New York, Sunday River Maine, and Bretton Woods New Hampshire"
            }},
            "dependencies": []
        }}
    ]
}}
"""

    @classmethod
    def get_step_validation_rules(cls) -> dict[str, Any]:
        """Get validation rules for plan steps."""
        return {
            "max_description_length": 200,
            "allowed_tools": ["thinking", "execution", "file_operations"],
            "max_dependencies": 5,
            "max_parameters": 10,
        }

    @classmethod
    def get_storage_settings(cls) -> dict[str, Any]:
        """Get storage configuration settings."""
        return {
            "auto_save": True,
            "backup_count": 5,
            "compress_old_plans": True,
            "archive_after_days": 30,
        }
