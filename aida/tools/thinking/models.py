"""Data models for the thinking tool."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator


class ReasoningType(str, Enum):
    """Types of reasoning approaches."""

    SYSTEMATIC_ANALYSIS = "systematic_analysis"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    PROBLEM_DECOMPOSITION = "problem_decomposition"
    STRATEGIC_PLANNING = "strategic_planning"
    BRAINSTORMING = "brainstorming"
    ROOT_CAUSE_ANALYSIS = "root_cause_analysis"
    DECISION_ANALYSIS = "decision_analysis"


class Perspective(str, Enum):
    """Analysis perspectives."""

    TECHNICAL = "technical"
    BUSINESS = "business"
    USER = "user"
    SECURITY = "security"
    BALANCED = "balanced"


class OutputFormat(str, Enum):
    """Output formatting options."""

    STRUCTURED = "structured"
    NARRATIVE = "narrative"
    BULLET_POINTS = "bullet_points"
    DETAILED = "detailed"


class ThinkingRequest(BaseModel):
    """Request model for thinking operations."""

    problem: str = Field(..., description="The problem or question to analyze")
    context: str | None = Field("", description="Additional context for the problem")
    reasoning_type: ReasoningType = Field(
        ReasoningType.SYSTEMATIC_ANALYSIS, description="Type of reasoning to apply"
    )
    depth: int = Field(
        3, ge=1, le=5, description="Depth of analysis (1-5, where 5 is most thorough)"
    )
    perspective: Perspective = Field(
        Perspective.BALANCED, description="Analysis perspective to take"
    )
    output_format: OutputFormat = Field(
        OutputFormat.STRUCTURED, description="Format for the analysis output"
    )

    @validator("problem")
    def problem_not_empty(cls, v):
        """Validate that problem statement is not empty.

        Args:
            v: Problem value to validate

        Returns:
            Stripped problem string

        Raises:
            ValueError: If problem statement is empty
        """
        if not v or not v.strip():
            raise ValueError("Problem statement cannot be empty")
        return v.strip()


class ThinkingSection(BaseModel):
    """A section of the thinking analysis."""

    title: str
    content: str
    subsections: dict[str, str] | None = None


class ThinkingResponse(BaseModel):
    """Response model for thinking operations."""

    request_id: str = Field(default_factory=lambda: f"think_{datetime.utcnow().timestamp()}")
    problem: str
    reasoning_type: ReasoningType
    perspective: Perspective
    depth: int

    # Analysis results
    analysis: str = Field(..., description="Full analysis text")
    sections: dict[str, ThinkingSection] | None = Field(
        None, description="Structured sections (for structured output)"
    )
    summary: str | None = Field(None, description="Executive summary")

    # Specific reasoning outputs
    recommendations: list[str] | None = Field(None, description="List of recommendations")
    key_insights: list[str] | None = Field(None, description="Key insights from analysis")
    action_items: list[str] | None = Field(None, description="Actionable next steps")
    risks: list[str] | None = Field(None, description="Identified risks")
    opportunities: list[str] | None = Field(None, description="Identified opportunities")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time: float | None = None
    model_used: str | None = None

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class ThinkingPromptTemplate(BaseModel):
    """Template for generating prompts."""

    reasoning_type: ReasoningType
    template: str
    sections: list[str]


class ThinkingMetrics(BaseModel):
    """Metrics for thinking operations."""

    total_analyses: int = 0
    analyses_by_type: dict[str, int] = Field(default_factory=dict)
    average_processing_time: float = 0.0
    average_depth: float = 0.0
    most_common_perspective: str | None = None
