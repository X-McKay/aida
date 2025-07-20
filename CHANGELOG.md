# Changelog

All notable changes to AIDA (Advanced Intelligent Distributed Agent System) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New simplified LLM system with PydanticAI integration
- Purpose-based LLM routing (DEFAULT, CODING, REASONING, MULTIMODAL, QUICK)
- Unified model interface supporting Ollama, OpenAI, Anthropic, and vLLM providers
- Comprehensive integration test framework with real LLM calls
- CLI integration for test runner (`aida test` command)
- Streaming support for LLM responses with proper async iteration
- Automatic provider selection and configuration
- Mock tool system for testing orchestrator execution

### Changed
- **BREAKING**: Replaced complex LLM provider system with simplified PydanticAI-based approach
- **BREAKING**: Removed fallback mechanism and hardcoded model lists for simplified configuration
- Migrated Todo Orchestrator to use new LLM system
- Updated all LLM imports to use `aida.llm` module
- Improved error handling in Todo Orchestrator with proper defaults for missing fields
- Enhanced test coverage with real functionality testing (no mocks for core features)

### Fixed
- EOF error when piping input to todo orchestrator standalone example
- LLM streaming test failure due to incorrect async iterator usage
- Todo Orchestrator plan execution failure due to missing 'tool' field in step data
- Pattern matching issues in plan complexity tests
- Import errors related to removed FallbackModel functionality

### Removed
- Legacy LLM provider system (`aida.providers.llm`)
- FallbackModel and complex fallback chain mechanisms
- Hardcoded model availability checks
- Mocked LLM responses in core functionality tests

### Technical Details

#### LLM System Refactoring
- Implemented PydanticAI-based LLM manager with automatic provider detection
- Added purpose-based model routing for optimized responses per use case
- Simplified configuration using environment variables for API keys
- Ollama integration using OpenAI-compatible interface
- Support for streaming responses with proper delta handling

#### Testing Infrastructure
- Created `BaseTestSuite` and `TestRegistry` for structured testing
- Implemented integration tests for LLM and Todo Orchestrator systems
- Added CLI command `aida test` with suite-specific and verbose options
- Real LLM calls in tests to ensure actual functionality works
- Mock tools for orchestrator execution testing

#### Todo Orchestrator Improvements
- Migrated to new LLM system maintaining all existing functionality
- Added robust error handling for malformed LLM responses
- Improved plan creation with proper field defaults
- Enhanced progress tracking and execution reliability

## [Previous Versions]

*Previous version history to be documented as needed.*