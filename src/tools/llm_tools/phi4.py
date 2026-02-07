"""
Phi-4 LLM Tool - Local language model exposed as a callable tool.

Wraps the LM Studio Phi-4 client to provide a unified tool interface.
Cost: Free (local execution)
"""

from typing import Any

from src.models.lm_studio import LMStudioClient, get_phi4_client
from src.tools.base import BaseTool, ToolConfig, ToolResult


class Phi4Tool(BaseTool):
    """
    Local Phi-4 model as a tool.

    Capabilities:
    - Code generation and execution planning
    - Query formulation and reformulation
    - Summarization and synthesis
    - General reasoning (simpler tasks)

    This is the cost-efficient option for most tasks.
    """

    def __init__(self) -> None:
        super().__init__()
        self._client: LMStudioClient | None = None  # Lazy initialization

    @property
    def client(self) -> LMStudioClient:
        """Lazy load the LM Studio client."""
        if self._client is None:
            self._client = get_phi4_client()
        return self._client

    def default_config(self) -> ToolConfig:
        return ToolConfig(
            name="phi4",
            description=(
                "Local Phi-4 language model for code generation, summarization, "
                "query formulation, and general reasoning. Fast and free. "
                "Use for: writing code, drafting queries, summarizing results, "
                "simple reasoning tasks."
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
            estimated_cost=0.0,  # Free - local model
            estimated_latency_ms=2000.0,  # ~2 seconds typical
            is_local=True,
        )

    def run(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Execute Phi-4 with the given prompt.

        Args:
            prompt: The user prompt/instruction
            system_prompt: Optional system context
            max_tokens: Maximum tokens to generate

        Returns:
            ToolResult with model output
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat(messages, max_tokens=max_tokens)

            return ToolResult(
                output=response.content,
                cost=0.0,  # Local is free
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
        """Async execution of Phi-4."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.achat(messages, max_tokens=max_tokens)

            return ToolResult(
                output=response.content,
                cost=0.0,
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
