"""MCP server implementation for thinking tool."""

import logging
from typing import Dict, Any

from aida.tools.base_mcp import SimpleMCPServer
from .models import ReasoningType, Perspective

logger = logging.getLogger(__name__)


class ThinkingMCPServer(SimpleMCPServer):
    """MCP server wrapper for ThinkingTool."""
    
    def __init__(self, thinking_tool):
        operations = {
            "analyze": {
                "description": "Analyze problems using various reasoning methods",
                "parameters": {
                    "problem": {
                        "type": "string",
                        "description": "The problem to analyze"
                    },
                    "context": {
                        "type": "string", 
                        "description": "Additional context"
                    },
                    "reasoning_type": {
                        "type": "string",
                        "enum": [t.value for t in ReasoningType],
                        "description": "Type of reasoning to apply"
                    },
                    "depth": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "description": "Depth of analysis (1-5)"
                    },
                    "perspective": {
                        "type": "string",
                        "enum": [p.value for p in Perspective],
                        "description": "Analysis perspective"
                    }
                },
                "required": ["problem"]
            },
            "brainstorm": {
                "description": "Brainstorm creative solutions",
                "parameters": {
                    "problem": {
                        "type": "string",
                        "description": "Problem to brainstorm solutions for"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context"
                    }
                },
                "required": ["problem"],
                "handler": self._handle_brainstorm
            },
            "decide": {
                "description": "Analyze a decision with options",
                "parameters": {
                    "decision": {
                        "type": "string",
                        "description": "Decision to analyze"
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Available options"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context"
                    }
                },
                "required": ["decision", "options"],
                "handler": self._handle_decide
            }
        }
        super().__init__(thinking_tool, operations)
    
    async def _handle_brainstorm(self, arguments: Dict[str, Any]):
        """Handle brainstorm operation."""
        result = await self.tool.execute(
            problem=arguments["problem"],
            context=arguments.get("context", ""),
            reasoning_type="brainstorming",
            depth=4
        )
        return result.result
    
    async def _handle_decide(self, arguments: Dict[str, Any]):
        """Handle decision analysis."""
        options = arguments["options"]
        options_context = "Options:\n" + "\n".join(f"- {opt}" for opt in options)
        context = arguments.get("context", "")
        full_context = f"{context}\n\n{options_context}" if context else options_context
        
        result = await self.tool.execute(
            problem=arguments["decision"],
            context=full_context,
            reasoning_type="decision_analysis",
            depth=4
        )
        return result.result