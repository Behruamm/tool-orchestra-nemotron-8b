"""
Pytest configuration and shared fixtures.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_lm_studio_client():
    """Mock LM Studio client for testing without local models."""
    with patch("src.models.lm_studio.LMStudioClient") as mock:
        client = MagicMock()

        # Default orchestrator response
        client.chat.return_value = {"choices": [{"message": {"content": "phi4_query"}}]}

        mock.return_value = client
        yield client


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini API client for testing without API calls."""
    with patch("src.models.gemini.GeminiClient") as mock:
        client = MagicMock()

        response = MagicMock()
        response.text = "Mocked Gemini response"
        response.usage_metadata = {"total_token_count": 100}

        client.generate_content.return_value = response
        mock.return_value = client
        yield client


@pytest.fixture
def sample_state():
    """Sample agent state for testing."""
    return {
        "query": "What is 2 + 2?",
        "messages": [],
        "tool_results": [],
        "iteration": 0,
        "final_response": "",
        "total_cost": 0.0,
    }
