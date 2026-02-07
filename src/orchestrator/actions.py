"""
Orchestrator Action Schema - Structured output format for the orchestrator.

Following the ToolOrchestra paper pattern, the orchestrator outputs
structured JSON with reasoning, tool selection, and parameters.
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrchestratorAction:
    """
    Structured action output from the orchestrator.

    The orchestrator produces chain-of-thought reasoning followed by
    a structured tool call with parameters.
    """

    reasoning: str  # Chain-of-thought explanation
    tool: str  # Tool name from registry
    parameters: dict[str, Any]  # Tool-specific parameters
    confidence: float = 1.0  # 0-1 confidence score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reasoning": self.reasoning,
            "tool": self.tool,
            "parameters": self.parameters,
            "confidence": self.confidence,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrchestratorAction":
        """Create from dictionary."""
        return cls(
            reasoning=data.get("reasoning", ""),
            tool=data.get("tool", ""),
            parameters=data.get("parameters", {}),
            confidence=data.get("confidence", 1.0),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "OrchestratorAction":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def is_terminal(self) -> bool:
        """Check if this action terminates the workflow."""
        return self.tool == "finish"

    def __repr__(self) -> str:
        return f"OrchestratorAction(tool={self.tool}, confidence={self.confidence})"


@dataclass
class TrajectoryStep:
    """
    A single step in the execution trajectory.

    Matches the paper's history format: h_k = (query, obs_0, action_0, obs_1, ...)
    """

    step_type: str  # "action" or "observation"
    content: dict[str, Any]  # Action details or tool result
    tool_name: str | None = None  # Tool that was called (for observations)
    cost: float = 0.0  # Cost incurred
    latency_ms: float = 0.0  # Latency incurred

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "step_type": self.step_type,
            "content": self.content,
            "tool_name": self.tool_name,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
        }


@dataclass
class Trajectory:
    """
    Full execution trajectory for a query.

    Accumulates all actions and observations during workflow execution.
    """

    query: str  # Original user query
    steps: list[TrajectoryStep] = field(default_factory=list)

    def add_action(self, action: OrchestratorAction) -> None:
        """Add an action to the trajectory."""
        self.steps.append(
            TrajectoryStep(
                step_type="action",
                content=action.to_dict(),
                tool_name=action.tool,
            )
        )

    def add_observation(
        self,
        tool_name: str,
        result: dict[str, Any],
        cost: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        """Add an observation (tool result) to the trajectory."""
        self.steps.append(
            TrajectoryStep(
                step_type="observation",
                content=result,
                tool_name=tool_name,
                cost=cost,
                latency_ms=latency_ms,
            )
        )

    def get_history_for_prompt(self) -> list[dict[str, Any]]:
        """
        Get trajectory formatted for the orchestrator prompt.

        Returns a list of steps suitable for including in the prompt context.
        """
        return [step.to_dict() for step in self.steps]

    def total_cost(self) -> float:
        """Calculate total cost across all steps."""
        return sum(step.cost for step in self.steps)

    def total_latency_ms(self) -> float:
        """Calculate total latency across all steps."""
        return sum(step.latency_ms for step in self.steps)

    def __len__(self) -> int:
        return len(self.steps)
