"""
Tool Registry - Central registry for all available tools.

Manages both basic tools and LLM tools through a unified interface.
"""

from typing import Any

from src.tools.base import BaseTool, ToolConfig, ToolResult


class ToolRegistry:
    """
    Registry to manage available tools.

    All tools (basic tools, LLM tools, finish tool) are registered here
    and accessible through the same interface.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        """List all tool schemas (for orchestrator prompt)."""
        return [tool.to_schema() for tool in self._tools.values()]

    def list_tool_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_local_tools(self) -> list[BaseTool]:
        """Get all tools that run locally (no external API)."""
        return [tool for tool in self._tools.values() if tool.is_local]

    def get_external_tools(self) -> list[BaseTool]:
        """Get all tools that require external APIs."""
        return [tool for tool in self._tools.values() if not tool.is_local]

    def get_tools_by_cost(self, max_cost: float) -> list[BaseTool]:
        """Get tools with estimated cost below threshold."""
        return [
            tool for tool in self._tools.values() if tool.estimated_cost <= max_cost
        ]

    async def execute(self, tool_name: str, parameters: dict) -> ToolResult:
        """
        Execute a tool by name with given parameters.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool-specific parameters

        Returns:
            ToolResult from the tool execution
        """
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                output=None,
                error=f"Tool '{tool_name}' not found",
            )
        return await tool.arun(**parameters)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Global registry instance
registry = ToolRegistry()


# Re-export base classes for backward compatibility
__all__ = ["ToolRegistry", "registry", "BaseTool", "ToolConfig", "ToolResult"]
