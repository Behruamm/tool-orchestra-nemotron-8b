"""
Web Search Tool - Search the public internet via Brave Search API.
"""

import logging
import time
from typing import Any

import httpx

from src.config import get_settings
from src.tools.base import BaseTool, ToolConfig, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """
    Web Search tool using Brave Search API for real-time internet information.

    Capabilities:
    - Search public web for current information
    - Find news, articles, documentation
    - Access real-time data

    Note: Disabled when privacy=True in preferences.
    """

    def __init__(self, config: ToolConfig | None = None) -> None:
        super().__init__(config)
        settings = get_settings()
        self._api_key = settings.brave_search.api_key
        self._base_url = settings.brave_search.base_url

    def default_config(self) -> ToolConfig:
        return ToolConfig(
            name="web_search",
            description=(
                "Searches the public internet for real-time information. "
                "Use for current events, documentation, or information not in local knowledge. "
                "Disabled if privacy=True."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                    },
                },
                "required": ["query"],
            },
            estimated_cost=0.001,
            estimated_latency_ms=1500.0,
            is_local=False,
        )

    def run(
        self,
        query: str,
        num_results: int = 5,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute web search via Brave Search API.

        Args:
            query: The search query
            num_results: Number of results to return

        Returns:
            ToolResult with search results
        """
        start_time = time.perf_counter()

        if not self._api_key:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                output=None,
                cost=0.0,
                latency_ms=latency_ms,
                error="BRAVE_API_KEY not configured. Set it in your .env file.",
            )

        try:
            results = self._search(query, num_results)
            latency_ms = (time.perf_counter() - start_time) * 1000

            return ToolResult(
                output={
                    "query": query,
                    "results": results,
                    "total_results": len(results),
                },
                cost=0.0,  # Brave Search free tier: 2000 queries/month
                latency_ms=latency_ms,
                metadata={"source": "brave_search"},
            )
        except httpx.HTTPStatusError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error("Brave Search API error: %s", e)
            return ToolResult(
                output=None,
                cost=0.0,
                latency_ms=latency_ms,
                error=f"Brave Search API error: {e.response.status_code}",
            )
        except httpx.RequestError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.error("Brave Search request failed: %s", e)
            return ToolResult(
                output=None,
                cost=0.0,
                latency_ms=latency_ms,
                error=f"Brave Search request failed: {e}",
            )

    def _search(self, query: str, num_results: int) -> list[dict[str, Any]]:
        """Call Brave Search API and parse results."""
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self._api_key,
        }
        params: dict[str, str | int] = {
            "q": query,
            "count": min(num_results, 20),  # Brave max is 20
        }

        with httpx.Client(timeout=10.0) as client:
            response = client.get(self._base_url, headers=headers, params=params)
            response.raise_for_status()

        data = response.json()
        results: list[dict[str, Any]] = []

        for item in data.get("web", {}).get("results", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("description", ""),
                "url": item.get("url", ""),
            })

        return results
