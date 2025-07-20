"""AIDA configuration module."""

from aida.config.models import (
    Provider,
    ModelSpec
)

from aida.config.llm_defaults import (
    auto_configure_llm_providers,
    get_default_llm_config
)

__all__ = [
    "Provider",
    "ModelSpec",
    "auto_configure_llm_providers",
    "get_default_llm_config"
]