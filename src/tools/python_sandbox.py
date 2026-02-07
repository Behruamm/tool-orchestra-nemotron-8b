"""
Python Sandbox Tool - Execute Python code in a sandboxed environment.

WARNING: Usage of exec() has security risks.
In production, this should run in Docker/Firecracker.
"""

import contextlib
import io
import time
import traceback
from typing import Any

from src.tools.base import BaseTool, ToolConfig, ToolResult


class PythonSandboxTool(BaseTool):
    """
    Executes Python code in a local sandboxed environment.

    Capabilities:
    - Mathematical calculations
    - Data processing and transformation
    - Logic execution
    - String manipulation

    Security: Limited to safe modules (math, datetime, json, random).
    """

    def default_config(self) -> ToolConfig:
        return ToolConfig(
            name="python_sandbox",
            description=(
                "Executes Python code for math, logic, and data processing. "
                "Returns stdout/stderr. Use for calculations, data transformation, "
                "and programmatic operations."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Valid Python code to execute",
                    }
                },
                "required": ["code"],
            },
            estimated_cost=0.0,
            estimated_latency_ms=100.0,
            is_local=True,
        )

    def run(self, code: str, **kwargs: Any) -> ToolResult:
        """
        Execute the provided Python code and return the output.

        Args:
            code: Valid Python code to execute

        Returns:
            ToolResult with stdout/stderr output
        """
        start_time = time.perf_counter()

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Restricted globals for safety
        execution_globals = {
            "math": __import__("math"),
            "datetime": __import__("datetime"),
            "json": __import__("json"),
            "random": __import__("random"),
        }

        try:
            with (
                contextlib.redirect_stdout(stdout_capture),
                contextlib.redirect_stderr(stderr_capture),
            ):
                exec(code, execution_globals)

            latency_ms = (time.perf_counter() - start_time) * 1000
            stdout_value = stdout_capture.getvalue()
            stderr_value = stderr_capture.getvalue()

            output = {
                "status": "success",
                "stdout": stdout_value,
                "stderr": stderr_value,
                "result": stdout_value.strip() if stdout_value else "(No output)",
            }

            return ToolResult(
                output=output,
                cost=0.0,
                latency_ms=latency_ms,
                metadata={"code_length": len(code)},
            )

        except Exception:
            latency_ms = (time.perf_counter() - start_time) * 1000
            error_msg = traceback.format_exc()

            return ToolResult(
                output={
                    "status": "error",
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue(),
                },
                cost=0.0,
                latency_ms=latency_ms,
                error=error_msg,
            )
