"""AIDA integration tests for refactored components."""

# Import all test suites to register them
from . import (
    test_chat_cli,  # noqa: F401
    test_hybrid_context,  # noqa: F401
    test_hybrid_execution,  # noqa: F401
    test_hybrid_file_operations,  # noqa: F401
    test_hybrid_system,  # noqa: F401
    test_llm,  # noqa: F401
    test_orchestrator,  # noqa: F401
)
