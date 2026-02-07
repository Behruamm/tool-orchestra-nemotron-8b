"""
Response Parser - Parse structured JSON output from the orchestrator.

Handles extraction and validation of orchestrator responses,
with graceful fallback for malformed outputs.
"""

import json
import logging
import re

from src.orchestrator.actions import OrchestratorAction
from src.tools.registry import registry

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Raised when response parsing fails."""

    pass


def extract_json_from_response(response: str) -> str | None:
    """
    Extract the first valid JSON object from a response using bracket counting.

    This is more robust than regex for nested structures and "extra data" issues.
    """
    text = response.strip()

    # Strip model reasoning tags if present
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    text = text.replace("</think>", "").strip()

    # Find first '{'
    start = text.find("{")
    if start == -1:
        return None

    # Count brackets to find the matching closing '}'
    # This simple counter handles nested braces but ignores strings/escapes for speed.
    # For a stricter parser, we'd need a full tokenizer, but this is usually sufficient for LLM output.
    count = 0
    in_string = False
    escape = False

    for i, char in enumerate(text):
        if i < start:
            continue

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if not in_string:
            if char == "{":
                count += 1
            elif char == "}":
                count -= 1
                if count == 0:
                    return text[start : i + 1]

    return None


def extract_code_from_text(text: str) -> str | None:
    """
    Extract Python code from text that may contain code blocks.

    Useful when phi4 outputs code that needs to go to python_sandbox.
    """
    # Try markdown code block
    code_block_pattern = r"```(?:python)?\s*([\s\S]*?)\s*```"
    match = re.search(code_block_pattern, text)
    if match:
        return match.group(1).strip()

    # If the text looks like code (contains def, import, print, etc.)
    code_indicators = ["def ", "import ", "print(", "for ", "while ", "if ", "class "]
    if any(indicator in text for indicator in code_indicators):
        return text.strip()

    return None


def validate_action(action: OrchestratorAction) -> tuple[bool, str | None]:
    """
    Validate that an action is well-formed and references a valid tool.

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check tool exists
    if action.tool not in registry:
        available = registry.list_tool_names()
        return False, f"Tool '{action.tool}' not found. Available: {available}"

    # Check required parameters
    tool = registry.get(action.tool)
    if tool:
        schema = tool.config.parameters
        required = schema.get("required", [])
        for param in required:
            if param not in action.parameters:
                return (
                    False,
                    f"Missing required parameter '{param}' for tool '{action.tool}'",
                )

    # Validate confidence range
    if not 0.0 <= action.confidence <= 1.0:
        action.confidence = max(0.0, min(1.0, action.confidence))

    return True, None


def parse_orchestrator_response(response: str) -> OrchestratorAction:
    """
    Parse the orchestrator's response into a structured action.

    Args:
        response: Raw response string from the orchestrator model

    Returns:
        OrchestratorAction with parsed content

    Raises:
        ParseError: If response cannot be parsed into a valid action
    """
    # Extract JSON from response
    json_str = extract_json_from_response(response)

    if not json_str:
        logger.warning(f"Could not extract JSON from response: {response[:200]}...")
        raise ParseError(f"No valid JSON found in response: {response[:100]}...")

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        raise ParseError(f"Invalid JSON: {e}") from e

    # Validate required fields
    if "tool" not in data:
        raise ParseError("Missing 'tool' field in response")

    # Create action
    action = OrchestratorAction(
        reasoning=data.get("reasoning", ""),
        tool=data.get("tool", ""),
        parameters=data.get("parameters", {}),
        confidence=float(data.get("confidence", 1.0)),
    )

    # Validate action
    is_valid, error = validate_action(action)
    if not is_valid:
        raise ParseError(error)

    return action


def create_fallback_action(
    query: str,
    error_message: str,
    default_tool: str = "phi4",
) -> OrchestratorAction:
    """
    Create a fallback action when parsing fails.

    Uses phi4 as the default fallback since it's local and free.
    """
    logger.info(f"Creating fallback action with {default_tool}: {error_message}")

    return OrchestratorAction(
        reasoning=f"Fallback due to parse error: {error_message}",
        tool=default_tool,
        parameters={
            "prompt": f"Please help with this query: {query}",
        },
        confidence=0.5,
    )


def safe_parse_response(
    response: str,
    query: str,
    default_tool: str = "phi4",
) -> OrchestratorAction:
    """
    Safely parse response with fallback on failure.

    This is the main entry point for parsing orchestrator responses.
    """
    try:
        return parse_orchestrator_response(response)
    except ParseError as e:
        logger.warning(f"Parse failed, using fallback: {e}")
        return create_fallback_action(query, str(e), default_tool)
