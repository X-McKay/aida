# Test Coverage Improvement Plan

## Current Status
- Overall coverage: ~26%
- Target coverage: 60%
- Many test failures due to API changes

## Completed Tasks ✅
1. **Fixed memory module tests** - Updated parameter names and method signatures
   - `max_size` → `max_entries`
   - `recall_memory` → `retrieve_memory`
   - `forget_memory` → `delete_memory`

2. **Fixed thinking tool tests** - Updated enum values to match implementation
   - `ReasoningType.ANALYTICAL` → `ReasoningType.SYSTEMATIC_ANALYSIS`
   - Updated Perspective and OutputFormat enums

3. **Fixed context tool tests** - Corrected mock return types
   - Changed string/list returns to proper dict structures

4. **Fixed execution tool tests** - Updated to mock correct methods
   - Mock `_execute_in_container` instead of non-existent methods

5. **Added config module tests** - Easy wins for coverage
   - Tests for Provider, ModelSpec, Purpose enums
   - Tests for LLM profiles and defaults

## In Progress 🔄
1. **Fix state module tests** - API changes in progress
   - Need to complete updating all test methods to use correct API

## Pending Tasks 📋

### High Priority
1. **Complete state module fixes**
   - Finish updating test methods to use `update_agent_state` instead of specific methods
   - Fix callback tests to use subscription model

2. **Fix base tool tests**
   - Address missing imports/attributes
   - Update JSON formatting expectations

### Medium Priority
1. **Add tests for agent module** (219 lines, 25% coverage)
   - Core agent functionality
   - Event handling
   - State management

2. **Add tests for LLM manager** (57 lines, 32% coverage)
   - Provider initialization
   - Model selection logic

3. **Add tests for orchestrator modules**
   - TodoOrchestrator functionality
   - Plan execution
   - Storage operations

4. **Add tests for protocol modules**
   - A2A protocol (22% coverage)
   - MCP protocol (35% coverage)

5. **Add tests for tool modules** (0% coverage for many)
   - File operations tool
   - System tool
   - Thinking tool
   - Context tool

### Low Priority (Nice to Have)
1. **Add tests for utility modules**
   - Event system
   - State persistence
   - Observability helpers

## Excluded from Testing
- CLI modules (pending refactor)
- UI modules (console, progress, status, tables)
- Provider implementations (may change)

## Testing Strategy
1. Fix all failing tests first
2. Focus on high-impact modules with low coverage
3. Prioritize core functionality over utilities
4. Aim for 60% coverage threshold
5. Write focused unit tests, not integration tests

## Notes
- Many tests were using outdated APIs
- Focus on testing public interfaces, not implementation details
- Mock external dependencies consistently
- Use pytest-asyncio for async tests
