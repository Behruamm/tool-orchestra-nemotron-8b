"""
Base classes for the unified tool interface.

All tools (including LLMs) implement this interface for consistent
usage across the orchestrator, following the ToolOrchestra paper pattern.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolConfig:
    """Configuration for a tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    # Cost and latency metadata for routing decisions
    estimated_cost: float = 0.0  # Estimated cost per call in USD
    estimated_latency_ms: float = 0.0  # Estimated latency in milliseconds
    is_local: bool = True  # Whether tool runs locally (no external API)


@dataclass
class ToolResult:
    """
    Standardized result from any tool execution.

    Matches the paper's observation format with cost/latency tracking.
    """

    output: Any  # The tool's output (string, dict, etc.)
    cost: float = 0.0  # Actual cost incurred in USD
    latency_ms: float = 0.0  # Actual latency in milliseconds
    is_terminal: bool = False  # True if this ends the workflow (finish tool)
    error: str | None = None  # Error message if execution failed
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional info

    @property
    def success(self) -> bool:
        """Check if the tool executed successfully."""
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "output": self.output,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "is_terminal": self.is_terminal,
            "error": self.error,
            "success": self.success,
            "metadata": self.metadata,
        }


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Tools include:
    - Basic tools: python_sandbox, web_search, local_search
    - LLM tools: phi4, gemini (LLMs exposed as callable tools)
    - Special tools: finish (terminates workflow)
    """

    def __init__(self, config: ToolConfig | None = None):
        if config:
            self.config = config
        else:
            self.config = self.default_config()

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def description(self) -> str:
        return self.config.description

    @property
    def estimated_cost(self) -> float:
        return self.config.estimated_cost

    @property
    def estimated_latency_ms(self) -> float:
        return self.config.estimated_latency_ms

    @property
    def is_local(self) -> bool:
        return self.config.is_local

    @abstractmethod
    def default_config(self) -> ToolConfig:
        """Return default configuration for this tool."""
        pass

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool synchronously.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with output, cost, and latency
        """
        pass

    async def arun(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool asynchronously.

        Default implementation wraps sync run().
        Override for true async execution.
        """
        return self.run(**kwargs)

    def to_schema(self) -> dict[str, Any]:
        """
        Return JSON schema for the tool.

        This is passed to the orchestrator for tool selection.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.config.parameters,
            "estimated_cost": self.estimated_cost,
            "estimated_latency_ms": self.estimated_latency_ms,
            "is_local": self.is_local,
        }

    def _measure_execution(self, func: Any, **kwargs: Any) -> ToolResult:
        """
        Helper to measure execution time and wrap result.

        Use this in run() implementations:
            return self._measure_execution(self._execute, **kwargs)
        """
        start_time = time.perf_counter()
        try:
            result = func(**kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # If result is already a ToolResult, update latency
            if isinstance(result, ToolResult):
                result.latency_ms = latency_ms
                return result

            # Otherwise wrap in ToolResult
            return ToolResult(
                output=result,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                output=None,
                latency_ms=latency_ms,
                error=str(e),
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, local={self.is_local})"
