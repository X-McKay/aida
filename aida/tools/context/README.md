# Context Tool

## Overview
The Context Tool provides advanced context management capabilities for AI systems, including compression, summarization, key point extraction, and intelligent search within large text contexts. It's designed to help manage token limits and extract relevant information from extensive conversations or documents.

## Features
- Intelligent context compression with priority-based selection
- Multi-format summarization (structured, narrative, bullet points)
- Key point extraction with categorization
- Semantic search within context
- Token optimization for LLM constraints
- Context snapshots for state management
- Export functionality in multiple formats
- Full hybrid architecture support (AIDA, PydanticAI, MCP)

## Configuration
The tool uses the following configuration parameters:

- `LLM_PURPOSE`: Set to `Purpose.DEFAULT` for general context processing
- `DEFAULT_COMPRESSION_LEVEL`: 0.5 (keep 50% of content)
- `DEFAULT_MAX_TOKENS`: 2000 tokens
- `DEFAULT_SEARCH_RESULTS`: 10 results
- `DEFAULT_MAX_KEY_POINTS`: 10 points
- `SNAPSHOT_DIR`: Default snapshot directory path

## Usage Examples

### Basic Compression
```python
from aida.tools.context import ContextTool

tool = ContextTool()
result = await tool.execute(
    operation="compress",
    content=long_conversation,
    compression_level=0.3,  # Keep 30% of content
    priority="recency"      # Prioritize recent content
)
print(f"Compressed from {len(long_conversation)} to {len(result.result)} chars")
```

### Intelligent Summarization
```python
result = await tool.execute(
    operation="summarize",
    content=document_text,
    max_tokens=500,
    output_format="structured"
)
summary = result.result
print(f"Overview: {summary['overview']}")
print(f"Key Topics: {summary['key_topics']}")
```

### Key Point Extraction
```python
result = await tool.execute(
    operation="extract_key_points",
    content=meeting_transcript,
    max_results=5
)
for point in result.result:
    print(f"- {point}")
```

### Context Search
```python
result = await tool.execute(
    operation="search",
    content=knowledge_base,
    query="machine learning optimization techniques",
    max_results=10
)
for match in result.result:
    print(f"Found: {match['text']} (relevance: {match['score']})")
```

### PydanticAI Integration
```python
from pydantic_ai import Agent
from aida.tools.context import ContextTool

tool = ContextTool()
agent = Agent(
    "You are a document analyst",
    tools=tool.to_pydantic_tools()
)

# Agent can now use context compression, summarization, etc.
```

### MCP Server Usage
```python
mcp_server = tool.get_mcp_server()

# Compress via MCP
result = await mcp_server.call_tool(
    "context_compress",
    {
        "content": large_text,
        "compression_level": 0.4,
        "priority": "importance"
    }
)
```

## Operations

### compress
**Description**: Compress context while preserving important information

**Parameters**:
- `content` (str, required): Content to compress
- `compression_level` (float, optional): Amount to keep (0.1-0.9, default: 0.5)
- `priority` (str, optional): Priority strategy - "recency", "relevance", "importance", "balanced"

**Returns**: Compressed text maintaining key information

### summarize
**Description**: Create a summary of context content

**Parameters**:
- `content` (str, required): Content to summarize
- `max_tokens` (int, optional): Maximum tokens in summary (default: 2000)
- `output_format` (str, optional): Format - "structured", "narrative", "bullet_points"

**Returns**: Summary in requested format

### extract_key_points
**Description**: Extract key points from context

**Parameters**:
- `content` (str, required): Content to analyze
- `max_results` (int, optional): Maximum points to extract (default: 10)

**Returns**: List of key points with categories

### search
**Description**: Search for information within context

**Parameters**:
- `content` (str, required): Content to search in
- `query` (str, required): Search query
- `max_results` (int, optional): Maximum results (default: 10)

**Returns**: List of matches with relevance scores

### optimize
**Description**: Optimize context to fit within token limits

**Parameters**:
- `content` (str, required): Content to optimize
- `max_tokens` (int, required): Target token limit

**Returns**: Optimized content fitting within limit

### snapshot
**Description**: Create a snapshot of current context

**Parameters**:
- `content` (str, required): Content to snapshot
- `file_path` (str, optional): Custom save location

**Returns**: Snapshot metadata including ID and location

### export
**Description**: Export context to a file

**Parameters**:
- `content` (str, required): Content to export
- `file_path` (str, required): Output file path
- `format_type` (str, optional): Format - "json", "markdown", "yaml", "text"

**Returns**: Export confirmation with file path

## Priority Strategies

- **recency**: Prioritizes recent information (end of context)
- **relevance**: Prioritizes semantically relevant content
- **importance**: Prioritizes critical information (decisions, actions)
- **balanced**: Balanced approach combining all strategies

## Output Formats

### Structured Summary
```json
{
    "overview": "High-level summary",
    "key_topics": ["Topic 1", "Topic 2"],
    "main_points": ["Point 1", "Point 2"],
    "action_items": ["Action 1", "Action 2"],
    "decisions": ["Decision 1", "Decision 2"]
}
```

### Search Results
```json
[
    {
        "text": "Matched text segment",
        "score": 0.95,
        "position": 1234,
        "context": "Surrounding context"
    }
]
```

## Error Handling
Common errors and solutions:

- **Compression too aggressive**: Increase compression_level parameter
- **Summary too short**: Increase max_tokens parameter
- **Search no results**: Try broader search terms or check content
- **Token limit exceeded**: Use optimize operation first

## Performance Considerations
- Large contexts (>10K chars) may take longer to process
- Compression operations are memory-intensive
- Search performance depends on content size
- Consider chunking very large documents
- Cache results for repeated operations

## Dependencies
- AIDA LLM system for intelligent processing
- No external APIs required
- Optional: OpenTelemetry for observability

## Changelog
- **1.0.0**: Initial implementation with core operations
- **1.0.1**: Added priority strategies for compression
- **1.0.2**: Improved search with semantic matching
- **1.0.3**: Added snapshot and export functionality
- **1.0.4**: Enhanced summarization formats