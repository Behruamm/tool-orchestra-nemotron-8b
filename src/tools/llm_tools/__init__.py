"""
LLM Tools - Language models exposed as callable tools.

Following the ToolOrchestra paper pattern, LLMs are treated uniformly
as tools through the same interface as basic tools.
"""

from src.tools.llm_tools.gemini import GeminiTool
from src.tools.llm_tools.phi4 import Phi4Tool

__all__ = ["Phi4Tool", "GeminiTool"]
