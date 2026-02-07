"""
Finish Tool - Special tool to terminate the workflow.

Following the ToolOrchestra paper pattern, termination is part of
the action space as a callable tool with structured output.
"""

from typing import Any

from src.tools.base import BaseTool, ToolConfig, ToolResult


class FinishTool(BaseTool):
    """
    Special tool to terminate the workflow with a final answer.

    The orchestrator calls this when it has gathered enough information
    to provide a complete response to the user's query.
    """

    def default_config(self) -> ToolConfig:
        return ToolConfig(
            name="finish",
            description=(
                "Call this tool when you have the final answer to return to the user. "
                "Use this to complete the task and end the workflow. "
                "Provide the complete, well-formatted answer."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to return to the user",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score from 0.0 to 1.0",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of sources/tools used to derive the answer",
                    },
                },
                "required": ["answer"],
            },
            estimated_cost=0.0,
            estimated_latency_ms=0.0,
            is_local=True,
        )

    def run(
        self,
        answer: str,
        confidence: float = 1.0,
        sources: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """
        Finish the workflow with the final answer.

        Args:
            answer: The final answer to return to the user
            confidence: Confidence score (0.0 to 1.0)
            sources: List of sources used

        Returns:
            ToolResult with is_terminal=True to signal workflow end
        """
        return ToolResult(
            output=answer,
            cost=0.0,
            latency_ms=0.0,
            is_terminal=True,  # Signals the workflow to end
            metadata={
                "confidence": confidence,
                "sources": sources or [],
            },
        )

    async def arun(
        self,
        answer: str,
        confidence: float = 1.0,
        sources: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Async version (same as sync for finish tool)."""
        return self.run(answer=answer, confidence=confidence, sources=sources)
