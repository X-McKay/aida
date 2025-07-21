"""Data models for LLM response tool."""

from datetime import datetime

from pydantic import BaseModel, Field, validator


class LLMResponseRequest(BaseModel):
    """Request model for LLM response operations."""

    question: str = Field(..., description="The question or request to answer")
    context: str | None = Field("", description="Additional context for the question")
    max_length: int = Field(
        2000, ge=100, le=10000, description="Maximum response length in characters"
    )

    @validator("question")
    def question_not_empty(cls, v):
        """Validate that question is not empty.

        Args:
            v: Question value to validate

        Returns:
            Stripped question string

        Raises:
            ValueError: If question is empty
        """
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

    @validator("context")
    def clean_context(cls, v):
        """Clean and format context input.

        Args:
            v: Context value to clean (can be string, dict, or list)

        Returns:
            Formatted context string
        """
        # Handle dict/list context by converting to string
        if isinstance(v, dict | list):
            import json

            return json.dumps(v, indent=2)
        return str(v) if v else ""


class LLMResponseResult(BaseModel):
    """Result model for LLM response operations."""

    request_id: str = Field(default_factory=lambda: f"llm_resp_{datetime.utcnow().timestamp()}")
    question: str
    response: str
    context_provided: bool = False
    response_length: int = 0
    truncated: bool = False

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processing_time: float | None = None
    model_used: str | None = None

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class LLMResponseMetrics(BaseModel):
    """Metrics for LLM response operations."""

    total_responses: int = 0
    average_response_length: float = 0.0
    average_processing_time: float = 0.0
    questions_with_context: int = 0
    truncated_responses: int = 0
