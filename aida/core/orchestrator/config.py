"""Configuration for the TODO orchestrator."""

from aida.config.llm_profiles import Purpose
from typing import Dict, Any


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
    RETRY_ERRORS = [
        "timeout",
        "connection", 
        "temporary",
        "rate limit"
    ]
    
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
You are a workflow planning assistant. Create a TODO-style execution plan for the user's request.

USER REQUEST: {user_request}

AVAILABLE TOOLS:
{tools_info}

CONTEXT: {context}

IMPORTANT: Respond ONLY with valid JSON. Do not include any explanatory text before or after the JSON.

Respond with exactly this JSON structure:

{{
    "analysis": "Brief analysis of what needs to be done",
    "expected_outcome": "What the user should expect as a result",
    "execution_plan": [
        {{
            "description": "Clear description of what this step does (for TODO list)",
            "tool": "thinking",
            "parameters": {{"problem": "description of what to think about"}},
            "dependencies": []
        }}
    ]
}}

Requirements:
- Keep analysis and expected_outcome to 1-2 sentences each
- Use "thinking" as the tool for most steps since specific tools may not be available
- Create 1-3 steps maximum to keep plans simple
- Respond with ONLY the JSON object, no other text
"""
    
    @classmethod
    def get_step_validation_rules(cls) -> Dict[str, Any]:
        """Get validation rules for plan steps."""
        return {
            "max_description_length": 200,
            "allowed_tools": ["thinking", "execution", "file_operations"],
            "max_dependencies": 5,
            "max_parameters": 10
        }
    
    @classmethod
    def get_storage_settings(cls) -> Dict[str, Any]:
        """Get storage configuration settings."""
        return {
            "auto_save": True,
            "backup_count": 5,
            "compress_old_plans": True,
            "archive_after_days": 30
        }