"""
Orchestrator Execution Loop - Core of the Orchestrator-8B system.

Simple loop that:
1. Sends query + tool definitions to Orchestrator-8B
2. Parses structured JSON response
3. Executes tool calls
4. Feeds results back until finish tool is called

No LangGraph, no routing logic — the model decides everything.
"""

import logging
from typing import Any

from src.config import get_settings
from src.models.lm_studio import get_orchestrator_client
from src.orchestrator.actions import OrchestratorAction, Trajectory
from src.orchestrator.parser import parse_orchestrator_response, ParseError
from src.tools import registry
from src.tools.base import ToolResult

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an intelligent research assistant that answers questions accurately using available tools.

You have access to the following tools:
{tools}

For each turn, analyze the query and available information, then output a JSON object with:
{{
    "reasoning": "Your step-by-step thinking about what to do next",
    "tool": "tool_name",
    "parameters": {{"param1": "value1", ...}},
    "confidence": 0.0-1.0
}}

Important rules:
1. Always output valid JSON - no markdown code blocks, just raw JSON
2. Use web_search to find current information from the internet
3. Use python_sandbox to perform calculations or data analysis
4. Use phi4 for quick local reasoning tasks
5. Use gemini for complex reasoning requiring a powerful model
6. Call finish when you have the complete answer with sources

Example response:
{{"reasoning": "I need to search for current information", "tool": "web_search", "parameters": {{"query": "latest news on topic"}}, "confidence": 0.9}}
"""


def _build_system_prompt() -> str:
    """Build the system prompt with tool definitions."""
    tools = registry.list_tools()
    tool_descriptions = []
    for tool in tools:
        params = tool.get("parameters", {}).get("properties", {})
        param_str = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in params.items())
        tool_descriptions.append(
            f"- {tool['name']}: {tool['description']} ({param_str})"
        )

    return SYSTEM_PROMPT.format(tools="\n".join(tool_descriptions))


def _format_observation(tool_name: str, result: ToolResult) -> str:
    """Format a tool result as an observation message."""
    if result.error:
        return f"[{tool_name}] Error: {result.error}"

    output = result.output
    if isinstance(output, dict):
        import json

        output = json.dumps(output, indent=2)

    return f"[{tool_name}] Result:\n{output}"


async def execute_tool(action: OrchestratorAction) -> ToolResult:
    """Execute a tool based on the orchestrator's action."""
    tool = registry.get(action.tool)
    if not tool:
        return ToolResult(
            output=None,
            error=f"Tool '{action.tool}' not found in registry",
        )

    try:
        result = await tool.arun(**action.parameters)
        return result
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return ToolResult(
            output=None,
            error=str(e),
        )


async def run(query: str, verbose: bool = False) -> dict[str, Any]:
    """
    Run the orchestrator loop for a query.

    Args:
        query: The user's query
        verbose: If True, log detailed execution trace

    Returns:
        Dict with 'answer', 'sources', 'cost', 'turns', 'trajectory'
    """
    settings = get_settings()
    max_turns = settings.max_iterations  # Will rename to max_turns later

    # Initialize
    client = get_orchestrator_client()
    trajectory = Trajectory(query=query)
    system_prompt = _build_system_prompt()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    total_cost = 0.0

    for turn in range(max_turns):
        if verbose:
            logger.info(f"Turn {turn + 1}/{max_turns}")

        # Call orchestrator
        try:
            response = await client.achat(messages)
            total_cost += response.cost
        except Exception as e:
            logger.error(f"Orchestrator call failed: {e}")
            return {
                "answer": f"Error: Failed to get response from orchestrator: {e}",
                "sources": [],
                "cost": total_cost,
                "turns": turn + 1,
                "trajectory": trajectory,
            }

        # Parse response
        try:
            action = parse_orchestrator_response(response.content)
            trajectory.add_action(action)

            if verbose:
                logger.info(f"Action: {action.tool} - {action.reasoning[:100]}...")
        except ParseError as e:
            logger.warning(f"Parse error on turn {turn + 1}: {e}")
            # Add error observation and continue
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": f"Error: Could not parse your response. Please output valid JSON. Error: {e}",
                }
            )
            continue

        # Check for finish
        if action.tool == "finish":
            answer = action.parameters.get("answer", "No answer provided")
            sources = action.parameters.get("sources", [])

            return {
                "answer": answer,
                "sources": sources,
                "cost": total_cost,
                "turns": turn + 1,
                "trajectory": trajectory,
            }

        # Execute tool
        result = await execute_tool(action)
        total_cost += result.cost

        trajectory.add_observation(
            tool_name=action.tool,
            result=result.to_dict(),
            cost=result.cost,
            latency_ms=result.latency_ms,
        )

        if verbose:
            logger.info(f"Result: {str(result.output)[:200]}...")

        # Add to messages
        messages.append({"role": "assistant", "content": action.to_json()})
        messages.append(
            {"role": "user", "content": _format_observation(action.tool, result)}
        )

    # Max turns reached
    return {
        "answer": "I was unable to complete the task within the allowed number of steps. Here's what I found so far:\n\n"
        + "\n".join(
            f"- Used {s.tool_name}"
            for s in trajectory.steps
            if s.step_type == "observation"
        ),
        "sources": [],
        "cost": total_cost,
        "turns": max_turns,
        "trajectory": trajectory,
    }


def run_sync(query: str, verbose: bool = False) -> dict[str, Any]:
    """Synchronous wrapper for run()."""
    import asyncio

    return asyncio.run(run(query, verbose=verbose))


__all__ = ["run", "run_sync", "execute_tool"]
