"""Tests for the Brave Search-backed web search tool."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.tools.web_search import WebSearchTool


@pytest.fixture
def mock_settings():
    """Mock settings with Brave API key configured."""
    with patch("src.tools.web_search.get_settings") as mock:
        settings = MagicMock()
        settings.brave_search.api_key = "test-brave-key"
        settings.brave_search.base_url = "https://api.search.brave.com/res/v1/web/search"
        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_settings_no_key():
    """Mock settings without Brave API key."""
    with patch("src.tools.web_search.get_settings") as mock:
        settings = MagicMock()
        settings.brave_search.api_key = ""
        settings.brave_search.base_url = "https://api.search.brave.com/res/v1/web/search"
        mock.return_value = settings
        yield settings


@pytest.fixture
def brave_response():
    """Sample Brave Search API response."""
    return {
        "web": {
            "results": [
                {
                    "title": "Python Tutorial",
                    "description": "Learn Python programming from scratch.",
                    "url": "https://python.org/tutorial",
                },
                {
                    "title": "Python Docs",
                    "description": "Official Python documentation.",
                    "url": "https://docs.python.org",
                },
                {
                    "title": "Real Python",
                    "description": "Python tutorials and articles.",
                    "url": "https://realpython.com",
                },
            ]
        }
    }


class TestWebSearchTool:
    def test_config(self, mock_settings):
        tool = WebSearchTool()
        assert tool.name == "web_search"
        assert tool.is_local is False
        assert tool.estimated_cost > 0

    def test_missing_api_key(self, mock_settings_no_key):
        tool = WebSearchTool()
        result = tool.run(query="python tutorial")
        assert result.error is not None
        assert "BRAVE_API_KEY" in result.error
        assert not result.success

    def test_successful_search(self, mock_settings, brave_response):
        tool = WebSearchTool()
        mock_response = MagicMock()
        mock_response.json.return_value = brave_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = tool.run(query="python tutorial", num_results=3)

        assert result.success
        assert result.output["query"] == "python tutorial"
        assert len(result.output["results"]) == 3
        assert result.output["results"][0]["title"] == "Python Tutorial"
        assert result.metadata.get("source") == "brave_search"

    def test_num_results_limit(self, mock_settings, brave_response):
        tool = WebSearchTool()
        mock_response = MagicMock()
        mock_response.json.return_value = brave_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            result = tool.run(query="test", num_results=2)

        assert result.success
        assert len(result.output["results"]) == 2

    def test_http_error(self, mock_settings):
        tool = WebSearchTool()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)

            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_client.get.return_value.raise_for_status.side_effect = (
                httpx.HTTPStatusError("rate limited", request=MagicMock(), response=mock_resp)
            )
            mock_client_cls.return_value = mock_client

            result = tool.run(query="test")

        assert result.error is not None
        assert "429" in result.error

    def test_request_error(self, mock_settings):
        tool = WebSearchTool()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.side_effect = httpx.RequestError("connection failed")
            mock_client_cls.return_value = mock_client

            result = tool.run(query="test")

        assert result.error is not None
        assert "connection failed" in result.error

    def test_api_headers(self, mock_settings, brave_response):
        tool = WebSearchTool()
        mock_response = MagicMock()
        mock_response.json.return_value = brave_response
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_cls.return_value = mock_client

            tool.run(query="test", num_results=5)

        call_kwargs = mock_client.get.call_args
        headers = call_kwargs.kwargs["headers"]
        assert headers["X-Subscription-Token"] == "test-brave-key"
        assert headers["Accept"] == "application/json"
