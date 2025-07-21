# Hybrid ContextTool Guide

## Overview

The ContextTool provides advanced context management capabilities including compression, summarization, token optimization, and intelligent search. It's designed to help manage large conversation contexts efficiently while preserving important information.

**Version:** 2.0.0

## Key Features

- **Context Compression**: Reduce context size while preserving key information
- **Intelligent Summarization**: Create structured summaries in various formats
- **Key Point Extraction**: Identify and rank the most important points
- **Token Optimization**: Optimize content for LLM token limits
- **Context Search**: Search within context with multiple matching strategies
- **Snapshot Management**: Save and restore context states
- **Context Splitting**: Split large contexts into manageable chunks
- **Import/Export**: Exchange context data in multiple formats
- **Hybrid Architecture**: Compatible with AIDA, PydanticAI, MCP, and OpenTelemetry

## Architecture

The ContextTool implements a hybrid architecture that supports:

1. **Core Tool Interface** - Full feature set with all operations
2. **PydanticAI Tools** - Streamlined functions for common operations
3. **MCP Server** - Universal AI compatibility via Model Context Protocol
4. **OpenTelemetry** - Production-ready observability with compression metrics

## Usage Examples

### 1. Core Interface

```python
from aida.tools.context import ContextTool

tool = ContextTool()

# Compress context
result = await tool.execute(
    operation="compress",
    content=long_conversation,
    compression_ratio=0.5,  # Target 50% compression
    preserve_priority="important"  # Focus on important content
)

print(f"Original size: {result.result['compression_stats']['original_size']}")
print(f"Compressed size: {result.result['compression_stats']['compressed_size']}")
print(f"Efficiency: {result.result['compression_stats']['efficiency']}%")

# Summarize context
result = await tool.execute(
    operation="summarize",
    content=conversation_text,
    max_tokens=1000,
    format_type="structured"
)

summary = result.result["summary"]
print(f"Key topics: {summary['key_topics']}")
print(f"Action items: {summary['action_items']}")
print(f"Decisions: {summary['decisions']}")

# Extract key points
result = await tool.execute(
    operation="extract_key_points",
    content=document_text,
    max_tokens=10  # Extract top 10 points
)

for point in result.result["key_points"]:
    print(f"{point['category']}: {point['content']} (score: {point['importance_score']})")
```

### 2. PydanticAI Interface

```python
from aida.tools.context import ContextTool
from pydantic_ai import Agent

# Get PydanticAI-compatible tools
tool = ContextTool()
tools = tool.to_pydantic_tools()

# Use with PydanticAI agent
agent = Agent("gpt-4")
tool.register_with_pydantic_agent(agent)

# Or use functions directly
# Compress context
compressed = await tools["compress_context"](
    ctx,  # RunContext from PydanticAI
    content=large_text,
    compression_ratio=0.3,
    preserve_priority="recent"
)

# Summarize with different formats
summary = await tools["summarize_context"](
    ctx,
    content=meeting_notes,
    max_tokens=500,
    format_type="bullet_points"
)

# Extract key information
key_points = await tools["extract_key_points"](
    ctx,
    content=research_document,
    max_points=15
)

# Optimize for token limits
optimized = await tools["optimize_tokens"](
    ctx,
    content=verbose_text,
    max_tokens=2000
)

# Search within context
results = await tools["search_context"](
    ctx,
    content=conversation_history,
    query="decision about architecture"
)
```

### 3. MCP Server Interface

```python
from aida.tools.context import ContextTool

tool = ContextTool()
mcp_server = tool.get_mcp_server()

# Compress via MCP
result = await mcp_server.call_tool("context_compress", {
    "content": long_text,
    "compression_ratio": 0.4,
    "preserve_priority": "balanced"
})

# Summarize via MCP
result = await mcp_server.call_tool("context_summarize", {
    "content": document,
    "max_tokens": 1500,
    "format_type": "narrative"
})

# Search via MCP
result = await mcp_server.call_tool("context_search", {
    "content": knowledge_base,
    "query": "implementation details"
})

# Optimize tokens via MCP
result = await mcp_server.call_tool("context_optimize_tokens", {
    "content": verbose_content,
    "max_tokens": 3000
})
```

### 4. Advanced Operations

```python
# Merge multiple contexts
result = await tool.execute(
    operation="merge_contexts",
    context_data={
        "conversation_1": context1,
        "conversation_2": context2,
        "notes": context3
    },
    preserve_priority="balanced"
)

merged = result.result["merged_content"]
conflicts = result.result["conflict_resolution"]

# Split large context
result = await tool.execute(
    operation="split_context",
    content=very_long_document,
    max_tokens=2000  # Each chunk max 2000 tokens
)

chunks = result.result["chunks"]
for i, chunk in enumerate(chunks):
    metadata = result.result["chunk_metadata"][i]
    print(f"Chunk {metadata['chunk_id']}: {metadata['size']} words")

# Analyze relevance
result = await tool.execute(
    operation="analyze_relevance",
    content=document,
    search_query="machine learning applications"
)

relevance = result.result["overall_relevance"]
insights = result.result["insights"]
print(f"Relevance level: {result.result['relevance_level']}")

# Create and restore snapshots
# Create snapshot
snapshot_result = await tool.execute(
    operation="create_snapshot",
    context_data={
        "conversation": messages,
        "metadata": session_info,
        "state": current_state
    },
    file_path="/path/to/snapshot.json"
)

# Later, restore snapshot
restore_result = await tool.execute(
    operation="restore_snapshot",
    file_path="/path/to/snapshot.json"
)

restored = restore_result.result["restored_context"]
```

### 5. Import/Export Operations

```python
# Export context in different formats
# JSON format
await tool.execute(
    operation="export_context",
    context_data=conversation_data,
    file_path="/exports/conversation.json",
    format_type="json"
)

# Markdown format
await tool.execute(
    operation="export_context",
    context_data=conversation_data,
    file_path="/exports/conversation.md",
    format_type="markdown"
)

# Import context
import_result = await tool.execute(
    operation="import_context",
    file_path="/imports/previous_conversation.json"
)

imported_data = import_result.result["context_data"]
```

### 6. With Observability

```python
# Enable tracing
observability = tool.enable_observability({
    "enabled": True,
    "service_name": "context-management",
    "endpoint": "http://localhost:4317"
})

# Traced compression
with observability.trace_operation("compress", len(content)):
    result = await tool.execute(
        operation="compress",
        content=content,
        compression_ratio=0.5
    )

    # Record metrics
    observability.record_operation(
        "compress",
        result.duration_seconds,
        result.status == ToolStatus.COMPLETED
    )

    # Record compression ratio
    actual_ratio = result.result["compression_stats"]["actual_ratio"]
    observability.record_compression(actual_ratio)

    # Record token reduction
    efficiency = result.result["compression_stats"]["efficiency"]
    observability.record_token_reduction(efficiency)
```

## Operations

### Content Processing

| Operation | Description | Key Parameters |
|-----------|-------------|----------------|
| `compress` | Compress content while preserving important information | `content`, `compression_ratio`, `preserve_priority` |
| `summarize` | Create intelligent summary | `content`, `max_tokens`, `format_type` |
| `extract_key_points` | Extract and rank key points | `content`, `max_tokens` (as max points) |
| `optimize_tokens` | Optimize content for token limits | `content`, `max_tokens` |

### Context Management

| Operation | Description | Key Parameters |
|-----------|-------------|----------------|
| `merge_contexts` | Merge multiple contexts intelligently | `context_data`, `preserve_priority` |
| `split_context` | Split large context into chunks | `content`, `max_tokens` |
| `analyze_relevance` | Analyze content relevance to query | `content`, `search_query` |
| `search_context` | Search within context | `content`, `search_query` |

### Persistence Operations

| Operation | Description | Key Parameters |
|-----------|-------------|----------------|
| `create_snapshot` | Save context state | `context_data`, `file_path` |
| `restore_snapshot` | Restore context from snapshot | `file_path` |
| `export_context` | Export context to file | `context_data`, `file_path`, `format_type` |
| `import_context` | Import context from file | `file_path` |

## Parameters

### Common Parameters

| Parameter | Type | Description | Values |
|-----------|------|-------------|--------|
| `content` | str | Content to process | Any text |
| `compression_ratio` | float | Target compression ratio | 0.1-0.9 |
| `preserve_priority` | str | Content preservation strategy | recent, important, balanced, comprehensive |
| `max_tokens` | int | Maximum tokens for output | 100-32000 |
| `format_type` | str | Output format | structured, narrative, bullet_points, json, markdown |
| `search_query` | str | Query for search/relevance | Any search term |
| `file_path` | str | Path for import/export | Valid file path |

### Preservation Priorities

- **recent**: Prioritize recent content
- **important**: Focus on content with importance markers
- **balanced**: Balance between recency and importance
- **comprehensive**: Try to preserve as much as possible

### Format Types

- **structured**: Organized with categories and sections
- **narrative**: Flowing text summary
- **bullet_points**: Concise bullet list
- **json**: JSON structure
- **markdown**: Markdown formatted

## Return Values

### Compression Result

```python
{
    "original_content": "...",
    "compressed_content": "...",
    "compression_stats": {
        "original_size": 5000,
        "compressed_size": 2500,
        "target_ratio": 0.5,
        "actual_ratio": 0.5,
        "space_saved": 2500,
        "efficiency": 50.0
    },
    "preserve_priority": "balanced",
    "key_elements_preserved": ["key_concepts", "main_decisions", "critical_facts"]
}
```

### Summary Result

```python
{
    "original_length": 1000,
    "summary": {
        "overview": "Overview of content...",
        "key_topics": ["topic1", "topic2"],
        "important_facts": ["fact1", "fact2"],
        "action_items": ["action1", "action2"],
        "decisions": ["decision1"],
        "context_type": "technical_discussion"
    },
    "format_type": "structured",
    "compression_achieved": 0.8,
    "key_elements": {
        "topics_count": 3,
        "facts_count": 5,
        "actions_count": 2,
        "decisions_count": 1
    }
}
```

### Key Points Result

```python
{
    "key_points": [
        {
            "category": "main_concepts",
            "content": "Hybrid architecture design",
            "importance_score": 0.9,
            "relevance_score": 0.85
        },
        {
            "category": "critical_information",
            "content": "Backward compatibility requirement",
            "importance_score": 0.95,
            "relevance_score": 0.8
        }
    ],
    "total_points_found": 25,
    "points_selected": 10,
    "categories_represented": ["main_concepts", "critical_information", "requirements"],
    "average_importance": 0.87
}
```

## Best Practices

### Compression Strategy

1. **Choose appropriate preservation priority**:
   - Use "recent" for ongoing conversations
   - Use "important" for technical documentation
   - Use "balanced" for general content
   - Use "comprehensive" when unsure

2. **Set realistic compression ratios**:
   - 0.7-0.9: Light compression, minimal loss
   - 0.5-0.7: Moderate compression, good balance
   - 0.3-0.5: Heavy compression, focus on essentials
   - 0.1-0.3: Extreme compression, key points only

### Summarization Tips

1. **Select appropriate format**:
   - "structured" for comprehensive analysis
   - "narrative" for readable summaries
   - "bullet_points" for quick overview
   - "json" for programmatic processing

2. **Set token limits based on use case**:
   - 500-1000: Brief summaries
   - 1000-2000: Detailed summaries
   - 2000-4000: Comprehensive summaries

### Token Optimization

1. **Progressive optimization**:
   ```python
   # Start with light optimization
   result = await tool.execute(
       operation="optimize_tokens",
       content=content,
       max_tokens=original_tokens * 0.8
   )

   # If needed, optimize further
   if still_too_large:
       result = await tool.execute(
           operation="optimize_tokens",
           content=result.result["optimized_content"],
           max_tokens=target_tokens
       )
   ```

2. **Monitor quality preservation**:
   - Check quality_preservation score
   - Review optimized content for coherence
   - Ensure key information retained

### Context Splitting

1. **Choose appropriate chunk sizes**:
   - 1000-2000 tokens: Standard chunks
   - 2000-4000 tokens: Large chunks
   - 500-1000 tokens: Small, focused chunks

2. **Consider overlap strategy**:
   - Semantic boundaries preserve meaning
   - Some overlap helps maintain context
   - Review chunk metadata for continuity

## Error Handling

```python
try:
    result = await tool.execute(
        operation="compress",
        content=content,
        compression_ratio=0.5
    )

    if result.status == ToolStatus.COMPLETED:
        compressed = result.result["compressed_content"]
        stats = result.result["compression_stats"]
    else:
        print(f"Operation failed: {result.error}")

except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Invalid parameter: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

### Large Context Handling

```python
# For very large contexts, process in stages
if len(content) > 100000:  # Very large
    # First, split into chunks
    split_result = await tool.execute(
        operation="split_context",
        content=content,
        max_tokens=5000
    )

    # Process each chunk
    processed_chunks = []
    for chunk in split_result.result["chunks"]:
        chunk_result = await tool.execute(
            operation="compress",
            content=chunk,
            compression_ratio=0.5
        )
        processed_chunks.append(chunk_result.result["compressed_content"])

    # Merge results
    merge_result = await tool.execute(
        operation="merge_contexts",
        context_data={f"chunk_{i}": c for i, c in enumerate(processed_chunks)},
        preserve_priority="balanced"
    )
```

### Caching Strategies

- Cache compression results for repeated content
- Store summaries for frequently accessed contexts
- Use snapshots for checkpoint management

## MCP Tools

The MCP server provides these tools:

- `context_compress` - Compress content with preservation
- `context_summarize` - Create intelligent summaries
- `context_search` - Search within context
- `context_optimize_tokens` - Optimize for token limits

## PydanticAI Functions

Available typed functions:

- `compress_context(ctx, content, compression_ratio, preserve_priority)`
- `summarize_context(ctx, content, max_tokens, format_type)`
- `extract_key_points(ctx, content, max_points)`
- `optimize_tokens(ctx, content, max_tokens)`
- `search_context(ctx, content, query)`

## Integration Examples

### With LLM Conversations

```python
from aida.tools.context import ContextTool
from aida.llm import chat

context_tool = ContextTool()

# Manage conversation context
conversation_history = []

async def manage_conversation(new_message: str):
    conversation_history.append(new_message)

    # Check if context is getting large
    full_context = "\n".join(conversation_history)

    if len(full_context) > 10000:  # Getting large
        # Compress older parts
        compress_result = await context_tool.execute(
            operation="compress",
            content=full_context,
            compression_ratio=0.5,
            preserve_priority="recent"
        )

        # Use compressed context for LLM
        response = await chat(
            compress_result.result["compressed_content"],
            purpose=Purpose.DEFAULT
        )
    else:
        response = await chat(full_context, purpose=Purpose.DEFAULT)

    return response
```

### Context Window Management

```python
async def manage_context_window(
    messages: List[str],
    max_tokens: int = 4000
) -> str:
    tool = ContextTool()

    # Join messages
    full_context = "\n".join(messages)

    # Check token count
    token_result = await tool.execute(
        operation="optimize_tokens",
        content=full_context,
        max_tokens=max_tokens
    )

    optimized = token_result.result["optimized_content"]

    # If still too large, use compression
    if token_result.result["token_analysis"]["final_tokens"] > max_tokens:
        compress_result = await tool.execute(
            operation="compress",
            content=optimized,
            compression_ratio=0.5,
            preserve_priority="important"
        )
        optimized = compress_result.result["compressed_content"]

    return optimized
```

### Knowledge Base Search

```python
async def search_knowledge_base(
    knowledge_base: Dict[str, str],
    query: str
) -> List[Dict[str, Any]]:
    tool = ContextTool()
    results = []

    for doc_id, content in knowledge_base.items():
        # Analyze relevance
        relevance_result = await tool.execute(
            operation="analyze_relevance",
            content=content,
            search_query=query
        )

        if relevance_result.result["relevance_level"] in ["high", "medium"]:
            # Extract relevant sections
            search_result = await tool.execute(
                operation="search_context",
                content=content,
                search_query=query
            )

            results.append({
                "doc_id": doc_id,
                "relevance": relevance_result.result["overall_relevance"],
                "matches": search_result.result["top_matches"],
                "insights": relevance_result.result["insights"]
            })

    # Sort by relevance
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results
```

## Troubleshooting

### Compression Issues
- If compression ratio not achieved, try different preservation priorities
- For technical content, use "important" priority
- Check key_elements_preserved to ensure critical info retained

### Summary Quality
- Adjust max_tokens based on content complexity
- Try different format_types for better results
- For structured data, use "structured" format

### Token Optimization
- If quality drops, increase max_tokens slightly
- Check quality_preservation score
- Consider manual review for critical content

### Search Accuracy
- Use specific search queries
- Check all match types (exact, partial, semantic)
- Review search_coverage metric

## Related Documentation

- [Hybrid Architecture Overview](./hybrid_architecture.md)
- [FileOperationsTool Guide](./hybrid_file_operations_tool.md)
- [SystemTool Guide](./hybrid_system_tool.md)
- [ExecutionTool Guide](./hybrid_execution_tool.md)
