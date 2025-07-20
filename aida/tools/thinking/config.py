"""Configuration for the thinking tool."""

from typing import Dict, Any
from aida.config.llm_profiles import Purpose
from .models import ReasoningType


class ThinkingConfig:
    """Configuration for thinking tool operations."""
    
    # LLM Configuration
    LLM_PURPOSE = Purpose.DEFAULT
    
    # Processing Configuration
    DEFAULT_DEPTH = 3
    MAX_RESPONSE_LENGTH = 4000
    SECTION_EXTRACTION_ENABLED = True
    
    # Caching Configuration
    ENABLE_RESPONSE_CACHE = True
    CACHE_TTL_SECONDS = 3600  # 1 hour
    
    # Prompt Templates
    REASONING_PROMPTS = {
        ReasoningType.SYSTEMATIC_ANALYSIS: """
Perform a systematic analysis of this problem from a {perspective} perspective.
Depth level: {depth}/5 (where 5 is most thorough).

Please analyze:
1. Problem breakdown and key components
2. Key factors and variables
3. Constraints and limitations
4. Opportunities and possibilities
5. Risks and challenges
6. Recommendations and next steps
""",
        ReasoningType.CHAIN_OF_THOUGHT: """
Use chain-of-thought reasoning to work through this problem step by step.
Show your thinking process clearly at depth level {depth}/5.

Walk through:
1. Initial understanding
2. Step-by-step reasoning
3. Intermediate conclusions
4. Final insights
""",
        ReasoningType.PROBLEM_DECOMPOSITION: """
Decompose this problem into smaller, manageable sub-problems.
Analysis depth: {depth}/5.

Break down into:
1. Core problem components
2. Sub-problems and their relationships
3. Dependencies between components
4. Solution approach for each sub-problem
5. Integration strategy
""",
        ReasoningType.STRATEGIC_PLANNING: """
Create a strategic plan to address this problem from a {perspective} perspective.
Planning depth: {depth}/5.

Include:
1. Current state analysis
2. Desired future state
3. Gap analysis
4. Strategic objectives
5. Action plan with milestones
6. Success metrics
""",
        ReasoningType.BRAINSTORMING: """
Brainstorm creative solutions and ideas for this problem.
Generate ideas at depth level {depth}/5.

Provide:
1. Multiple solution approaches
2. Creative alternatives
3. Unconventional ideas
4. Pros and cons of each
5. Recommended approaches
""",
        ReasoningType.ROOT_CAUSE_ANALYSIS: """
Perform a root cause analysis of this problem.
Analysis depth: {depth}/5.

Investigate:
1. Symptoms vs root causes
2. Causal chain analysis
3. Contributing factors
4. Why-why analysis
5. Preventive measures
""",
        ReasoningType.DECISION_ANALYSIS: """
Analyze the decision options for this problem from a {perspective} perspective.
Analysis depth: {depth}/5.

Evaluate:
1. Available options
2. Criteria for evaluation
3. Pros and cons of each option
4. Risk assessment
5. Recommended decision with rationale
"""
    }
    
    # Output Format Instructions
    OUTPUT_FORMAT_INSTRUCTIONS = {
        "structured": "\n\nProvide a well-structured response with clear sections.",
        "narrative": "\n\nProvide a flowing narrative response.",
        "bullet_points": "\n\nProvide response in clear bullet points.",
        "detailed": "\n\nProvide a comprehensive, detailed response."
    }
    
    # Section Keywords for Extraction
    SECTION_KEYWORDS = [
        "problem", "analysis", "factors", "constraints", "opportunities",
        "risks", "recommendations", "conclusion", "summary", "next steps",
        "insights", "approach", "solution", "strategy", "objectives",
        "metrics", "dependencies", "components", "causes", "options"
    ]
    
    # Validation Rules
    MIN_PROBLEM_LENGTH = 10
    MAX_PROBLEM_LENGTH = 5000
    MAX_CONTEXT_LENGTH = 10000
    
    @classmethod
    def get_prompt_template(cls, reasoning_type: ReasoningType) -> str:
        """Get the prompt template for a reasoning type."""
        return cls.REASONING_PROMPTS.get(
            reasoning_type, 
            cls.REASONING_PROMPTS[ReasoningType.SYSTEMATIC_ANALYSIS]
        )
    
    @classmethod
    def get_output_instruction(cls, format_type: str) -> str:
        """Get output format instruction."""
        return cls.OUTPUT_FORMAT_INSTRUCTIONS.get(format_type, "")
    
    @classmethod
    def get_mcp_config(cls) -> Dict[str, Any]:
        """Get MCP server configuration."""
        return {
            "server_name": "aida-thinking",
            "max_concurrent_requests": 10,
            "timeout_seconds": 60
        }
    
    @classmethod
    def get_observability_config(cls) -> Dict[str, Any]:
        """Get OpenTelemetry configuration."""
        return {
            "service_name": "aida-thinking-tool",
            "trace_enabled": True,
            "metrics_enabled": True,
            "export_endpoint": "http://localhost:4317"
        }