# Thinking Tool

## Overview
The Thinking Tool provides structured reasoning and analysis capabilities, enabling systematic problem-solving, brainstorming, and decision-making through various cognitive frameworks.

## Features
- Multiple reasoning types for different cognitive tasks
- Configurable depth of analysis (1-5 levels)
- Structured output with clear reasoning chains
- Support for complex problem decomposition
- Full hybrid architecture support (AIDA, PydanticAI, MCP)

## Configuration
The tool uses the following configuration parameters:

- `LLM_PURPOSE`: Set to `Purpose.REASONING` for analytical tasks
- `DEFAULT_REASONING_TYPE`: Default "systematic_analysis"
- `DEFAULT_DEPTH`: Default depth level 3
- `MAX_DEPTH`: Maximum depth level 5

## Reasoning Types

1. **systematic_analysis**: Step-by-step breakdown of complex problems
2. **chain_of_thought**: Sequential reasoning with explicit thought progression
3. **pros_and_cons**: Balanced evaluation of options
4. **brainstorming**: Creative idea generation
5. **root_cause_analysis**: Identifying underlying causes
6. **decision_tree**: Mapping out decision paths and outcomes
7. **swot_analysis**: Strengths, Weaknesses, Opportunities, Threats
8. **scenario_planning**: Exploring different future scenarios
9. **risk_assessment**: Evaluating potential risks and mitigations
10. **creative_problem_solving**: Innovative approaches to challenges

## Usage Examples

### Basic Usage
```python
from aida.tools.thinking import ThinkingTool

tool = ThinkingTool()
result = await tool.execute(
    problem="How can we improve code review processes?",
    reasoning_type="systematic_analysis",
    depth=3
)
print(result.result['analysis'])
```

### Brainstorming Session
```python
result = await tool.execute(
    problem="Generate ideas for making our API more developer-friendly",
    reasoning_type="brainstorming",
    depth=4
)
for idea in result.result['ideas']:
    print(f"- {idea}")
```

### Decision Making
```python
result = await tool.execute(
    problem="Should we migrate from monolith to microservices?",
    reasoning_type="pros_and_cons",
    depth=5
)
print("Pros:", result.result['pros'])
print("Cons:", result.result['cons'])
print("Recommendation:", result.result['recommendation'])
```

### PydanticAI Integration
```python
from pydantic_ai import Agent
from aida.tools.thinking import ThinkingTool

tool = ThinkingTool()
agent = Agent(
    "You are a strategic advisor",
    tools=tool.to_pydantic_tools()
)

response = await agent.run(
    "Analyze the risks of adopting a new technology stack"
)
```

### MCP Server Usage
```python
mcp_server = tool.get_mcp_server()

result = await mcp_server.call_tool(
    "thinking_analyze",
    {
        "problem": "How to scale our infrastructure?",
        "reasoning_type": "scenario_planning",
        "depth": 4
    }
)
```

## Operations

### analyze (default)
**Description**: Apply structured reasoning to analyze a problem or question

**Parameters**:
- `problem` (str, required): The problem or question to analyze
- `reasoning_type` (str, optional): Type of reasoning to apply (default: "systematic_analysis")
- `depth` (int, optional): Depth of analysis from 1-5 (default: 3)

**Returns**: Dictionary containing the analysis results, structure varies by reasoning type

**Example**:
```python
result = await tool.execute(
    problem="How can we reduce technical debt?",
    reasoning_type="root_cause_analysis",
    depth=4
)
```

## Output Formats by Reasoning Type

### systematic_analysis
```json
{
    "analysis": "Detailed step-by-step analysis",
    "steps": ["Step 1", "Step 2", ...],
    "conclusion": "Final conclusion"
}
```

### brainstorming
```json
{
    "ideas": ["Idea 1", "Idea 2", ...],
    "categories": {"Category": ["Ideas"], ...},
    "top_recommendations": ["Best ideas"]
}
```

### pros_and_cons
```json
{
    "pros": ["Advantage 1", "Advantage 2", ...],
    "cons": ["Disadvantage 1", "Disadvantage 2", ...],
    "recommendation": "Balanced recommendation"
}
```

## Error Handling
Common errors and solutions:

- **Invalid reasoning type**: Check available types in ReasoningType enum
- **Depth out of range**: Use depth between 1 and 5
- **LLM timeout**: Complex problems with high depth may take longer; reduce depth or simplify problem
- **Incomplete analysis**: Ensure problem statement is clear and specific

## Performance Considerations
- Higher depth levels increase processing time exponentially
- Complex reasoning types (scenario_planning, decision_tree) take longer
- Break down very complex problems into smaller sub-problems
- Cache results for repeated analyses of the same problem

## Dependencies
- AIDA LLM system with reasoning capabilities
- No external services required
- Works with any LLM backend that supports analytical prompts

## Changelog
- **1.0.0**: Initial implementation with 10 reasoning types
- **1.0.1**: Added structured prompt templates for each reasoning type
- **1.0.2**: Improved output parsing and error handling
- **1.0.3**: Added depth configuration for nuanced analysis
