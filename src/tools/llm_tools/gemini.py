"""
Gemini LLM Tool - Cloud language model exposed as a callable tool.

Wraps the Google Gemini client to provide a unified tool interface.
Cost: Paid API (pricing varies by model)
"""

from typing import Any

from src.models.gemini import GeminiClient, get_gemini_client
from src.tools.base import BaseTool, ToolConfig, ToolResult


class GeminiTool(BaseTool):
    """
    Google Gemini model as a tool.

    Capabilities:
    - Complex multi-step reasoning
    - Advanced analysis and synthesis
    - Nuanced understanding of complex queries
    - High-quality responses for difficult tasks

    This is the high-quality option for complex tasks.
    Use when local models are insufficient.
    """

    def __init__(self) -> None:
        super().__init__()
        self._client: GeminiClient | None = None  # Lazy initialization

    @property
    def client(self) -> GeminiClient:
        """Lazy load the Gemini client."""
        if self._client is None:
            self._client = get_gemini_client()
        return self._client

    def default_config(self) -> ToolConfig:
        return ToolConfig(
            name="gemini",
            description=(
                "Google Gemini cloud model for complex reasoning, advanced analysis, "
                "and high-quality responses. Paid API - use when local models are "
                "insufficient. Best for: multi-step reasoning, complex analysis, "
                "nuanced understanding, synthesis of complex information."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt/instruction for the model",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional system prompt to set context",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate (default: 2048)",
                    },
                },
                "required": ["prompt"],
            },
            estimated_cost=0.001,  # ~$0.001 per call estimate
            estimated_latency_ms=3000.0,  # ~3 seconds typical
            is_local=False,
        )

    def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute Gemini with the given prompt.

        Args:
            prompt: The user prompt/instruction
            system_prompt: Optional system context
            max_tokens: Maximum tokens to generate

        Returns:
            ToolResult with model output and cost
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat(messages, max_tokens=max_tokens)

            return ToolResult(
                output=response.content,
                cost=response.cost,  # Actual API cost
                latency_ms=response.latency_ms,
                metadata={
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        except Exception as e:
            return ToolResult(
                output=None,
                error=str(e),
                cost=0.0,
            )

    async def arun(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> ToolResult:
        """Async execution of Gemini."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.achat(messages, max_tokens=max_tokens)

            return ToolResult(
                output=response.content,
                cost=response.cost,
                latency_ms=response.latency_ms,
                metadata={
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
            )
        except Exception as e:
            return ToolResult(
                output=None,
                error=str(e),
                cost=0.0,
            )
