"""Internal tool used to verify model tool-calling support."""
from __future__ import annotations

from .base_tool import BaseTool
from ..core import ToolMetadata


class ModelTestTool(BaseTool):
    """Report a requested value during a model connectivity test."""

    metadata = ToolMetadata(
        name="report_model_test",
        description="Report the requested value to complete an SRHarness model test.",
        parameters={
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The exact value requested by the model-test prompt.",
                },
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    )

    def execute(self, answer: str) -> dict[str, str]:
        """Return the value supplied by the model-test request.

        Args:
            answer: Exact value requested by the test prompt.

        Returns:
            The reported answer.
        """
        return {"answer": answer}
