"""AIDA configuration module."""

from aida.config.llm_defaults import auto_configure_llm_providers, get_default_llm_config
from aida.config.models import ModelSpec, Provider

__all__ = ["Provider", "ModelSpec", "auto_configure_llm_providers", "get_default_llm_config"]
