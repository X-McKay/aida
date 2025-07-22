"""Tests for configuration modules."""

import os
from unittest.mock import Mock, patch

import pytest

from aida.config.llm_defaults import auto_configure_llm_providers, get_default_llm_config
from aida.config.llm_profiles import DEFAULT_PROFILES, Purpose, get_available_purposes, get_profile
from aida.config.models import ModelSpec, Provider


class TestLLMDefaults:
    """Test LLM defaults module."""

    @pytest.mark.asyncio
    async def test_auto_configure_llm_providers_success(self):
        """Test successful auto-configuration of LLM providers."""
        # Mock the LLM manager
        mock_manager = Mock()
        mock_manager.list_purposes.return_value = ["default", "coding", "reasoning"]

        with patch("aida.llm.get_llm", return_value=mock_manager):
            result = await auto_configure_llm_providers()
            assert result == 3
            mock_manager.list_purposes.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_configure_llm_providers_no_purposes(self):
        """Test auto-configuration with no purposes configured."""
        # Mock the LLM manager with no purposes
        mock_manager = Mock()
        mock_manager.list_purposes.return_value = []

        with (
            patch("aida.llm.get_llm", return_value=mock_manager),
            pytest.raises(Exception, match="No LLM purposes configured"),
        ):
            await auto_configure_llm_providers()

    def test_get_default_llm_config(self):
        """Test getting default LLM configuration."""
        config = get_default_llm_config()

        assert config["default_provider"] == "ollama"
        assert "providers" in config
        assert "ollama" in config["providers"]

        ollama_config = config["providers"]["ollama"]
        assert ollama_config["type"] == "ollama"
        assert ollama_config["api_url"] == "http://localhost:11434"
        assert ollama_config["model"] == "llama3.2:latest"
        assert ollama_config["timeout"] == 120


class TestLLMProfiles:
    """Test LLM profiles module."""

    def test_get_profile_default(self):
        """Test getting default profile."""
        profile = get_profile(Purpose.DEFAULT)
        assert profile.purpose == Purpose.DEFAULT
        assert profile.model.provider == Provider.OLLAMA
        assert profile.model.model_id == "llama3.2:latest"
        assert "helpful AI assistant" in profile.prompt

    def test_get_profile_coding(self):
        """Test getting coding profile."""
        profile = get_profile(Purpose.CODING)
        assert profile.purpose == Purpose.CODING
        assert profile.model.provider == Provider.OLLAMA
        assert profile.model.model_id == "codellama:latest"
        assert "coding specialist" in profile.prompt

    def test_get_profile_reasoning(self):
        """Test getting reasoning profile."""
        profile = get_profile(Purpose.REASONING)
        assert profile.purpose == Purpose.REASONING
        assert profile.model.provider == Provider.OLLAMA
        assert profile.model.model_id == "deepseek-r1:8b"
        assert profile.model.temperature == 0
        assert "reasoning specialist" in profile.prompt

    def test_get_profile_multimodal(self):
        """Test getting multimodal profile."""
        profile = get_profile(Purpose.MULTIMODAL)
        assert profile.purpose == Purpose.MULTIMODAL
        assert profile.model.provider == Provider.OLLAMA
        assert "multimodal specialist" in profile.prompt

    def test_get_profile_quick(self):
        """Test getting quick profile."""
        profile = get_profile(Purpose.QUICK)
        assert profile.purpose == Purpose.QUICK
        assert profile.model.provider == Provider.OLLAMA
        assert profile.model.model_id == "tinyllama:latest"
        assert profile.model.max_tokens == 4096

    def test_get_available_purposes_all_available(self):
        """Test getting available purposes when all models are available."""
        # All Ollama models should be available by default
        available = get_available_purposes()
        # Should have all purposes since they all use Ollama
        assert len(available) == len(Purpose)

    def test_get_available_purposes_with_api_key_models(self):
        """Test getting available purposes with API key models."""
        # Create a profile that requires API key
        get_profile(Purpose.DEFAULT)

        # All default profiles use Ollama which doesn't need API key
        available = get_available_purposes()
        assert len(available) > 0

        # Verify all available purposes use Ollama (no API key needed)
        for purpose in available:
            profile = DEFAULT_PROFILES[purpose]
            assert profile.model.provider == Provider.OLLAMA

    def test_all_profiles_have_required_fields(self):
        """Test that all profiles have required fields."""
        for purpose, profile in DEFAULT_PROFILES.items():
            assert profile.purpose == purpose
            assert profile.model is not None
            assert profile.prompt is not None
            assert isinstance(profile.model, ModelSpec)
            assert len(profile.prompt) > 0


class TestModelSpec:
    """Test ModelSpec model."""

    def test_model_spec_creation(self):
        """Test creating ModelSpec."""
        spec = ModelSpec(
            provider=Provider.OLLAMA,
            model_id="llama3.2:latest",
            temperature=0.5,
            max_tokens=2000,
            base_url="http://localhost:11434",
        )

        assert spec.provider == Provider.OLLAMA
        assert spec.model_id == "llama3.2:latest"
        assert spec.temperature == 0.5
        assert spec.max_tokens == 2000
        assert spec.base_url == "http://localhost:11434"

    def test_model_spec_defaults(self):
        """Test ModelSpec default values."""
        spec = ModelSpec(
            provider=Provider.OLLAMA,
            model_id="test_model",
        )

        assert spec.temperature == 0.1
        assert spec.max_tokens == 4000
        assert spec.base_url is None

    def test_api_key_openai(self):
        """Test getting OpenAI API key from environment."""
        spec = ModelSpec(
            provider=Provider.OPENAI,
            model_id="gpt-4",
        )

        # Without API key
        with patch.dict(os.environ, {}, clear=True):
            assert spec.api_key is None

        # With API key
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test_value"},  # pragma: allowlist secret
        ):
            assert spec.api_key == "test_value"  # pragma: allowlist secret

    def test_api_key_anthropic(self):
        """Test getting Anthropic API key from environment."""
        spec = ModelSpec(
            provider=Provider.ANTHROPIC,
            model_id="claude-3",
        )

        # Without API key
        with patch.dict(os.environ, {}, clear=True):
            assert spec.api_key is None

        # With API key
        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "test_key"},  # pragma: allowlist secret
        ):
            assert spec.api_key == "test_key"  # pragma: allowlist secret

    def test_api_key_ollama(self):
        """Test Ollama doesn't need API key."""
        spec = ModelSpec(
            provider=Provider.OLLAMA,
            model_id="llama3.2:latest",
        )

        assert spec.api_key is None

    def test_is_available_openai(self):
        """Test OpenAI availability based on API key."""
        spec = ModelSpec(
            provider=Provider.OPENAI,
            model_id="gpt-4",
        )

        # Without API key
        with patch.dict(os.environ, {}, clear=True):
            assert spec.is_available is False

        # With API key
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):  # pragma: allowlist secret
            assert spec.is_available is True

    def test_is_available_anthropic(self):
        """Test Anthropic availability based on API key."""
        spec = ModelSpec(
            provider=Provider.ANTHROPIC,
            model_id="claude-3",
        )

        # Without API key
        with patch.dict(os.environ, {}, clear=True):
            assert spec.is_available is False

        # With API key
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"}):  # pragma: allowlist secret
            assert spec.is_available is True

    def test_is_available_ollama(self):
        """Test Ollama is always available."""
        spec = ModelSpec(
            provider=Provider.OLLAMA,
            model_id="llama3.2:latest",
        )

        # Always available for local models
        assert spec.is_available is True

    def test_is_available_vllm(self):
        """Test VLLM is always available."""
        spec = ModelSpec(
            provider=Provider.VLLM,
            model_id="test_model",
        )

        # Always available for local models
        assert spec.is_available is True


class TestProvider:
    """Test Provider enum."""

    def test_provider_values(self):
        """Test provider enum values."""
        assert Provider.OPENAI == "openai"
        assert Provider.ANTHROPIC == "anthropic"
        assert Provider.OLLAMA == "ollama"
        assert Provider.VLLM == "vllm"

    def test_provider_from_string(self):
        """Test creating provider from string."""
        assert Provider("openai") == Provider.OPENAI
        assert Provider("anthropic") == Provider.ANTHROPIC
        assert Provider("ollama") == Provider.OLLAMA
        assert Provider("vllm") == Provider.VLLM

    def test_provider_invalid_string(self):
        """Test creating provider from invalid string."""
        with pytest.raises(ValueError):
            Provider("invalid_provider")


class TestPurpose:
    """Test Purpose enum."""

    def test_purpose_values(self):
        """Test purpose enum values."""
        assert Purpose.DEFAULT == "default"
        assert Purpose.CODING == "coding"
        assert Purpose.REASONING == "reasoning"
        assert Purpose.MULTIMODAL == "multimodal"
        assert Purpose.QUICK == "quick"

    def test_purpose_from_string(self):
        """Test creating purpose from string."""
        assert Purpose("default") == Purpose.DEFAULT
        assert Purpose("coding") == Purpose.CODING
        assert Purpose("reasoning") == Purpose.REASONING

    def test_purpose_invalid_string(self):
        """Test creating purpose from invalid string."""
        with pytest.raises(ValueError):
            Purpose("invalid_purpose")


if __name__ == "__main__":
    pytest.main([__file__])
