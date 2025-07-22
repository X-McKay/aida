"""Tests for LLM manager module."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from aida.config.llm_profiles import Purpose
from aida.config.models import ModelSpec, Provider
from aida.llm import chat, get_llm
from aida.llm.manager import LLMManager


class TestLLMManager:
    """Test LLMManager class."""

    @pytest.fixture
    def manager(self):
        """Create a fresh LLM manager with mocked setup."""
        with patch.object(LLMManager, "_setup_models"):
            return LLMManager()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = Mock()
        agent.run = AsyncMock()
        agent.run_stream = AsyncMock()
        return agent

    def test_llm_manager_initialization(self):
        """Test LLM manager initialization."""
        with patch.object(LLMManager, "_setup_models") as mock_setup:
            manager = LLMManager()
            assert manager._agents == {}
            mock_setup.assert_called_once()

    def test_setup_models(self):
        """Test setting up models for all purposes."""
        mock_purposes = [Purpose.DEFAULT, Purpose.CODING]
        mock_profile = Mock()
        mock_model = Mock()

        with (
            patch("aida.llm.manager.get_available_purposes", return_value=mock_purposes),
            patch("aida.llm.manager.get_profile", return_value=mock_profile),
            patch.object(LLMManager, "_create_model_for_profile", return_value=mock_model),
            patch("aida.llm.manager.Agent") as mock_agent_class,
        ):
            manager = LLMManager()

            # Should create agents for each purpose
            assert len(manager._agents) == len(mock_purposes)
            assert mock_agent_class.call_count == len(mock_purposes)

    def test_create_model_for_profile_ollama(self):
        """Test creating Ollama model from profile."""
        profile = Mock()
        profile.purpose = Purpose.DEFAULT
        profile.model = ModelSpec(
            provider=Provider.OLLAMA,
            model_id="llama3.2:latest",
            base_url="http://localhost:11434",
            temperature=0.1,
            max_tokens=4096,
        )

        with patch("aida.llm.manager.OpenAIModel") as mock_model_class:
            manager = LLMManager.__new__(LLMManager)
            manager._create_model_for_profile(profile)

            mock_model_class.assert_called_once()
            call_args = mock_model_class.call_args
            assert call_args[0][0] == "llama3.2:latest"

    def test_create_model_for_profile_openai(self):
        """Test creating OpenAI model from profile."""
        profile = Mock()
        profile.purpose = Purpose.DEFAULT
        profile.model = Mock()
        profile.model.provider = Provider.OPENAI
        profile.model.model_id = "gpt-4"
        profile.model.api_key = "test_key"  # pragma: allowlist secret
        profile.model.temperature = 0.7
        profile.model.max_tokens = 4096
        profile.model.is_available = True

        with patch("aida.llm.manager.OpenAIModel") as mock_model_class:
            manager = LLMManager.__new__(LLMManager)
            manager._create_model_for_profile(profile)

            mock_model_class.assert_called_once()
            call_args = mock_model_class.call_args
            assert call_args[0][0] == "gpt-4"

    def test_create_model_for_profile_anthropic(self):
        """Test creating Anthropic model from profile."""
        profile = Mock()
        profile.purpose = Purpose.DEFAULT
        profile.model = Mock()
        profile.model.provider = Provider.ANTHROPIC
        profile.model.model_id = "claude-3"
        profile.model.api_key = "test_key"  # pragma: allowlist secret
        profile.model.temperature = 0.7
        profile.model.max_tokens = 4096
        profile.model.is_available = True

        with patch("aida.llm.manager.AnthropicModel") as mock_model_class:
            manager = LLMManager.__new__(LLMManager)
            manager._create_model_for_profile(profile)

            mock_model_class.assert_called_once()
            call_args = mock_model_class.call_args
            assert call_args[0][0] == "claude-3"

    def test_create_model_unavailable(self):
        """Test error when model is not available."""
        profile = Mock()
        profile.purpose = Purpose.DEFAULT
        profile.model = Mock()
        profile.model.is_available = False

        manager = LLMManager.__new__(LLMManager)
        with pytest.raises(ValueError, match="No available model"):
            manager._create_model_for_profile(profile)

    def test_list_purposes(self, manager):
        """Test listing available purposes."""
        # Add some test agents
        manager._agents[Purpose.DEFAULT] = Mock()
        manager._agents[Purpose.CODING] = Mock()

        purposes = manager.list_purposes()
        assert Purpose.DEFAULT in purposes
        assert Purpose.CODING in purposes
        assert len(purposes) == 2

    def test_create_model_for_profile_ollama_no_base_url(self, manager):
        """Test creating Ollama model without base_url raises error."""
        from aida.config.llm_profiles import LLMProfile, Purpose
        from aida.config.models import ModelSpec, Provider

        profile = LLMProfile(
            purpose=Purpose.DEFAULT,
            prompt="Test prompt",
            model=ModelSpec(
                provider=Provider.OLLAMA,
                model_id="test_model",
                # base_url is None - should trigger error
                base_url=None,
            ),
        )

        manager = LLMManager.__new__(LLMManager)
        with pytest.raises(ValueError, match="base_url is required for Provider.OLLAMA"):
            manager._create_model_for_profile(profile)

    @pytest.mark.skip(reason="Provider enum prevents unsupported providers at validation time")
    def test_create_model_for_profile_unsupported_provider(self, manager):
        """Test creating model with unsupported provider raises error."""
        # This test is not feasible because the Pydantic enum validation
        # prevents unsupported providers from being created in the first place
        pass

    @pytest.mark.asyncio
    async def test_chat_non_streaming(self, manager, mock_agent):
        """Test non-streaming chat."""
        # Setup
        manager._agents[Purpose.DEFAULT] = mock_agent
        mock_result = Mock()
        mock_result.data = "Test response"
        mock_agent.run.return_value = mock_result

        # Test
        response = await manager.chat("Hello", purpose=Purpose.DEFAULT, stream=False)

        assert response == "Test response"
        mock_agent.run.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_chat_streaming(self, manager, mock_agent):
        """Test streaming chat."""

        # Setup mock streaming with async context manager
        class MockStreamContext:
            def __init__(self):
                self.entered = False

            async def __aenter__(self):
                self.entered = True
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

            async def stream_text(self, delta=True):
                yield "Hello"
                yield " world"

        mock_stream_context = MockStreamContext()
        mock_agent.run_stream = Mock(return_value=mock_stream_context)
        manager._agents[Purpose.DEFAULT] = mock_agent

        # Test
        stream = await manager.chat("Hello", purpose=Purpose.DEFAULT, stream=True)
        assert hasattr(stream, "__anext__")  # Check it's an async generator

        # Collect stream results
        results = []
        async for chunk in stream:
            results.append(chunk)

        assert results == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_chat_invalid_purpose(self, manager):
        """Test chat with invalid purpose."""
        with pytest.raises(ValueError, match="No agent for purpose"):
            await manager.chat("Hello", purpose="invalid_purpose")

    @pytest.mark.asyncio
    async def test_health_check(self, manager, mock_agent):
        """Test health check for all agents."""
        # Setup agents
        manager._agents[Purpose.DEFAULT] = mock_agent
        manager._agents[Purpose.CODING] = Mock(run=AsyncMock())

        # One healthy, one failing
        mock_agent.run.return_value = Mock()
        manager._agents[Purpose.CODING].run.side_effect = Exception("Failed")

        # Test
        health = await manager.health_check()

        assert health[Purpose.DEFAULT] is True
        assert health[Purpose.CODING] is False


class TestGlobalLLMManager:
    """Test global LLM manager functions."""

    def test_get_llm_singleton(self):
        """Test that get_llm returns singleton."""
        manager1 = get_llm()
        manager2 = get_llm()
        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_chat_shortcut(self):
        """Test the chat shortcut function."""
        # Mock the manager
        mock_manager = Mock()
        mock_manager.chat = AsyncMock(return_value="Test response")

        with patch("aida.llm.get_llm", return_value=mock_manager):
            response = await chat("Test message", purpose=Purpose.QUICK)

            assert response == "Test response"
            mock_manager.chat.assert_called_once_with("Test message", Purpose.QUICK, False)

    @pytest.mark.asyncio
    async def test_chat_shortcut_streaming(self):
        """Test the chat shortcut function with streaming."""
        # Mock the manager
        mock_manager = Mock()

        async def mock_stream():
            yield "chunk1"
            yield "chunk2"

        mock_manager.chat = AsyncMock(return_value=mock_stream())

        with patch("aida.llm.get_llm", return_value=mock_manager):
            stream = await chat("Test message", stream=True)

            # Collect results
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)

            assert chunks == ["chunk1", "chunk2"]


if __name__ == "__main__":
    pytest.main([__file__])
