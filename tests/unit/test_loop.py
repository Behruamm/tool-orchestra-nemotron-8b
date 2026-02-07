"""
Unit tests for the execution loop.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.orchestrator.loop import run, execute_tool, _build_system_prompt
from src.orchestrator.actions import OrchestratorAction
from src.tools.base import ToolResult


class TestBuildSystemPrompt:
    """Tests for system prompt building."""

    def test_build_system_prompt_includes_tools(self):
        """System prompt should include tool definitions."""
        prompt = _build_system_prompt()

        # Should contain tool names
        assert "finish" in prompt
        # Should contain instructions
        assert "JSON" in prompt
        assert "reasoning" in prompt


class TestExecuteTool:
    """Tests for tool execution."""

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self):
        """Should return error for unknown tool."""
        action = OrchestratorAction(
            reasoning="test",
            tool="nonexistent_tool",
            parameters={},
        )

        result = await execute_tool(action)

        assert result.error is not None
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_finish_tool(self):
        """Should execute finish tool successfully."""
        action = OrchestratorAction(
            reasoning="Done",
            tool="finish",
            parameters={"answer": "The answer is 42"},
        )

        result = await execute_tool(action)

        assert result.success
        assert result.is_terminal
        assert result.output == "The answer is 42"


class TestRunLoop:
    """Tests for the main run loop."""

    @pytest.mark.asyncio
    async def test_run_returns_on_finish(self):
        """Loop should return when finish tool is called."""
        # Mock orchestrator to return finish action immediately
        mock_response = MagicMock()
        mock_response.content = '{"reasoning": "Done", "tool": "finish", "parameters": {"answer": "42"}, "confidence": 1.0}'
        mock_response.cost = 0.0

        with patch("src.orchestrator.loop.get_orchestrator_client") as mock_client:
            mock_client.return_value.achat = AsyncMock(return_value=mock_response)

            result = await run("What is the answer?")

            assert result["answer"] == "42"
            assert result["turns"] == 1

    @pytest.mark.asyncio
    async def test_run_respects_max_turns(self):
        """Loop should stop after max turns."""
        # Mock orchestrator to call phi4 (correct tool name) repeatedly
        mock_response = MagicMock()
        mock_response.content = '{"reasoning": "Thinking", "tool": "phi4", "parameters": {"prompt": "test"}, "confidence": 0.8}'
        mock_response.cost = 0.0

        # Mock phi4 tool
        mock_tool = MagicMock()
        mock_tool.arun = AsyncMock(
            return_value=ToolResult(output="response", cost=0.0, latency_ms=10)
        )

        # Mock settings
        mock_settings = MagicMock()
        mock_settings.max_iterations = 3

        with (
            patch("src.orchestrator.loop.get_orchestrator_client") as mock_client,
            patch("src.orchestrator.loop.registry") as mock_registry,
            patch("src.orchestrator.loop.get_settings", return_value=mock_settings),
        ):

            mock_client.return_value.achat = AsyncMock(return_value=mock_response)
            mock_registry.get.return_value = mock_tool
            mock_registry.list_tools.return_value = []

            result = await run("Endless query")

            assert result["turns"] == 3
            assert "unable to complete" in result["answer"].lower()
