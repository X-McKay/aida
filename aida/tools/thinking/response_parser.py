"""Response parsing for thinking tool."""

import logging
import re

from .config import ThinkingConfig
from .models import ReasoningType, ThinkingRequest, ThinkingResponse, ThinkingSection

logger = logging.getLogger(__name__)


class ThinkingResponseParser:
    """Parses LLM responses into structured thinking responses."""

    def parse(self, llm_response: str, request: ThinkingRequest) -> ThinkingResponse:
        """Parse LLM response into ThinkingResponse."""
        response = ThinkingResponse(
            problem=request.problem,
            reasoning_type=request.reasoning_type,
            perspective=request.perspective,
            depth=request.depth,
            analysis=llm_response,
        )

        # Extract structured components based on reasoning type
        if request.output_format.value == "structured":
            self._extract_structured_components(response, llm_response, request)

        # Extract summary
        response.summary = self._extract_summary(llm_response)

        return response

    def _extract_structured_components(
        self, response: ThinkingResponse, llm_response: str, request: ThinkingRequest
    ) -> None:
        """Extract structured components from response."""
        # Extract sections
        if ThinkingConfig.SECTION_EXTRACTION_ENABLED:
            response.sections = self._extract_sections(llm_response)

        # Extract specific components based on reasoning type
        if request.reasoning_type == ReasoningType.SYSTEMATIC_ANALYSIS:
            response.recommendations = self._extract_list_items(
                llm_response, ["recommendations", "recommend", "suggest"]
            )
            response.risks = self._extract_list_items(
                llm_response, ["risks", "risk", "challenges", "concerns"]
            )
            response.opportunities = self._extract_list_items(
                llm_response, ["opportunities", "opportunity", "possibilities"]
            )

        elif request.reasoning_type == ReasoningType.BRAINSTORMING:
            response.key_insights = self._extract_list_items(
                llm_response, ["ideas", "solutions", "approaches", "alternatives"]
            )

        elif request.reasoning_type == ReasoningType.STRATEGIC_PLANNING:
            response.action_items = self._extract_list_items(
                llm_response, ["action", "steps", "milestones", "tasks"]
            )
            response.key_insights = self._extract_list_items(
                llm_response, ["objectives", "goals", "metrics"]
            )

    def _extract_sections(self, response: str) -> dict[str, ThinkingSection]:
        """Extract sections from structured response."""
        sections = {}
        current_section = None
        current_content = []

        lines = response.split("\n")

        for line in lines:
            # Check if line is a section header
            if self._is_section_header(line):
                # Save previous section
                if current_section:
                    sections[current_section] = ThinkingSection(
                        title=current_section, content="\n".join(current_content).strip()
                    )

                current_section = self._clean_section_header(line)
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = ThinkingSection(
                title=current_section, content="\n".join(current_content).strip()
            )

        return sections

    def _is_section_header(self, line: str) -> bool:
        """Check if a line is a section header."""
        line = line.strip()

        # Check for numbered headers (1., 2., etc.)
        if re.match(r"^\d+\.", line):
            return True

        # Check for markdown headers
        if line.startswith("#"):
            return True

        # Check for keyword-based headers
        lower_line = line.lower()
        for keyword in ThinkingConfig.SECTION_KEYWORDS:
            if keyword in lower_line and line.endswith(":"):
                return True

        return False

    def _clean_section_header(self, line: str) -> str:
        """Clean a section header."""
        line = line.strip()

        # Remove number prefix
        line = re.sub(r"^\d+\.\s*", "", line)

        # Remove markdown prefix
        line = line.lstrip("#").strip()

        # Remove trailing colon
        line = line.rstrip(":")

        return line

    def _extract_list_items(self, response: str, keywords: list[str]) -> list[str] | None:
        """Extract list items following keywords."""
        items = []

        lines = response.split("\n")
        in_list = False

        for _i, line in enumerate(lines):
            lower_line = line.lower()

            # Check if we're starting a relevant list
            if any(keyword in lower_line for keyword in keywords):
                in_list = True
                continue

            # Extract list items
            if in_list:
                # Check if line is a list item
                if re.match(r"^\s*[-*•]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
                    # Clean and add item
                    item = re.sub(r"^\s*[-*•]\s+", "", line)
                    item = re.sub(r"^\s*\d+\.\s+", "", item)
                    items.append(item.strip())
                # Stop if we hit a new section or empty lines
                elif self._is_section_header(line) or (not line.strip() and items):
                    break

        return items if items else None

    def _extract_summary(self, response: str) -> str | None:
        """Extract summary from response."""
        # Look for explicit summary sections
        summary_keywords = [
            "summary",
            "conclusion",
            "in conclusion",
            "recommendations",
            "next steps",
            "key takeaway",
        ]

        lines = response.split("\n")

        for i, line in enumerate(lines):
            lower_line = line.lower()

            for keyword in summary_keywords:
                if keyword in lower_line:
                    # Extract next few lines as summary
                    summary_lines = []

                    for j in range(i + 1, min(i + 5, len(lines))):
                        if lines[j].strip():
                            summary_lines.append(lines[j].strip())
                        elif summary_lines:  # Stop at empty line after content
                            break

                    if summary_lines:
                        return " ".join(summary_lines)

        # If no explicit summary found in text, continue to fallback

        # Fallback: Use first substantial paragraph
        for line in lines:
            if line.strip() and len(line.strip()) > 50:
                return line.strip()

        return None
