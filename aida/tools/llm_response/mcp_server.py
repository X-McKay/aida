"""MCP server implementation for LLM response tool."""

import logging
from typing import Dict, Any

from aida.tools.base_mcp import SimpleMCPServer

logger = logging.getLogger(__name__)


class LLMResponseMCPServer(SimpleMCPServer):
    """MCP server wrapper for LLMResponseTool."""
    
    def __init__(self, llm_response_tool):
        operations = {
            "answer": {
                "description": "Get a direct response from the LLM",
                "parameters": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask"
                    },
                    "context": {
                        "type": "string", 
                        "description": "Additional context for the question"
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum response length",
                        "minimum": 100,
                        "maximum": 10000
                    }
                },
                "required": ["question"]
            },
            "explain": {
                "description": "Get an explanation of a concept",
                "parameters": {
                    "concept": {
                        "type": "string",
                        "description": "The concept to explain"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context"
                    }
                },
                "required": ["concept"],
                "handler": self._handle_explain
            }
        }
        super().__init__(llm_response_tool, operations)
    
    async def _handle_explain(self, arguments: Dict[str, Any]):
        """Handle explain operation by converting to question format."""
        # Convert concept to question format
        question = f"Please explain {arguments['concept']}"
        result = await self.tool.execute(
            operation="answer",
            question=question,
            context=arguments.get("context", ""),
            max_length=3000
        )
        return result.result