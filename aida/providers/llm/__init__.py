"""LLM provider integrations.

DEPRECATED: This module is deprecated. Use aida.llm instead.

The new simplified LLM system is available at:
- aida.llm.chat() - Simple chat interface
- aida.config.llm_profiles - Purpose-based configuration
- aida.config.models - Model specifications
"""

import warnings

warnings.warn(
    "aida.providers.llm is deprecated. Use aida.llm instead.", DeprecationWarning, stacklevel=2
)

# Keep these for backward compatibility until migration is complete
__all__ = []
