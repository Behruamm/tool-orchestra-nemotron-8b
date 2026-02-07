"""
Tool Orchestra - Unified Tool Registry

All tools (basic tools, LLM tools, finish tool) are registered here
and accessible through the same interface following the ToolOrchestra paper pattern.
"""

from src.tools.base import BaseTool, ToolConfig, ToolResult
from src.tools.finish import FinishTool

# LLM tools
from src.tools.llm_tools import GeminiTool, Phi4Tool
from src.tools.local_search import LocalSearchTool

# Basic tools
from src.tools.python_sandbox import PythonSandboxTool
from src.tools.registry import ToolRegistry, registry
from src.tools.web_search import WebSearchTool


def register_default_tools() -> None:
    """
    Register all default tools in the global registry.

    This includes:
    - Basic tools: python_sandbox, local_search, web_search
    - LLM tools: phi4, gemini
    - Special tools: finish
    """
    # Basic tools
    registry.register(PythonSandboxTool())
    registry.register(LocalSearchTool())
    registry.register(WebSearchTool())

    # LLM tools (following paper pattern - LLMs as tools)
    registry.register(Phi4Tool())
    registry.register(GeminiTool())

    # Special tools
    registry.register(FinishTool())


# Auto-register on import
register_default_tools()


__all__ = [
    # Base classes
    "BaseTool",
    "ToolConfig",
    "ToolResult",
    # Registry
    "registry",
    "ToolRegistry",
    # Basic tools
    "PythonSandboxTool",
    "LocalSearchTool",
    "WebSearchTool",
    # LLM tools
    "Phi4Tool",
    "GeminiTool",
    # Special tools
    "FinishTool",
    # Registration
    "register_default_tools",
]
