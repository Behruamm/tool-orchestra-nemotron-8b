"""
Base interface for model clients.

All model clients (LM Studio, Gemini) implement this interface
for consistent usage across the orchestrator.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    """Standardized response from any model."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0
    latency_ms: float = 0.0
    raw_response: Any = None

    @property
    def input_tokens(self) -> int:
        return self.usage.get("input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        return self.usage.get("output_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class ModelConfig:
    """Configuration for a model client."""

    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    stop: list[str] | None = None


class BaseModelClient(ABC):
    """Abstract base class for model clients."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @property
    @abstractmethod
    def is_local(self) -> bool:
        """Return True if this is a local model (no API cost)."""
        pass

    @property
    @abstractmethod
    def cost_per_1k_input(self) -> float:
        """Cost per 1000 input tokens in USD."""
        pass

    @property
    @abstractmethod
    def cost_per_1k_output(self) -> float:
        """Cost per 1000 output tokens in USD."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional model-specific parameters

        Returns:
            ModelResponse with generated content
        """
        pass

    @abstractmethod
    async def achat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Async version of chat."""
        pass

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost for a request."""
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        return input_cost + output_cost

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, local={self.is_local})"
