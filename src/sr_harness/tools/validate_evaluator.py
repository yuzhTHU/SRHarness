"""Validate a workspace evaluator against the current dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core import AgentContext, ToolMetadata
from ..evaluator import load_custom_evaluator
from .base_tool import BaseTool
from .evaluate_formula import EvaluateTool


@BaseTool.register("validate_evaluator")
class ValidateEvaluatorTool(BaseTool):
    """Load an evaluator file in isolation and use it to evaluate one formula."""

    metadata = ToolMetadata(name="validate_evaluator")

    def execute(self, f: str, evaluator_file: str | None = None, y: str | None = None, fit: bool = False, show_diagnostics: bool = False) -> dict[str, Any]:
        """Validate a custom evaluator without changing the live context.

        Args:
            evaluator_file: Optional workspace-relative Python file under context.evaluator/. When omitted, validate the evaluator already attached to the context.
            f: Formula to evaluate.
            y: Optional target expression; defaults to context.target.
            fit: Whether to fit formula parameters first.
            show_diagnostics: Whether to include residual diagnostics.
        """
        relative = None
        evaluator = self.context.evaluator
        if evaluator_file is not None:
            relative = Path(evaluator_file)
            if relative.is_absolute() or not relative.parts or relative.parts[0] != "context.evaluator":
                raise ValueError("evaluator_file must be under context.evaluator/")
            path = (self.context.workspace / relative).resolve()
            evaluator_root = (self.context.workspace / "context.evaluator").resolve()
            if evaluator_root not in path.parents or path.suffix != ".py" or not path.is_file():
                raise ValueError("evaluator_file must identify an existing Python file under context.evaluator/")
            evaluator = load_custom_evaluator(file=path)
        test_context = AgentContext(
            args=self.context.args, data=self.context.data, target=self.context.target,
            variable_descriptions=self.context.variable_descriptions,
            variable_axes=self.context.variable_axes,
            variable_structures=self.context.variable_structures,
            num_nodes=self.context.num_nodes, evaluator=evaluator, workspace=self.context.workspace,
        )
        result = EvaluateTool(context=test_context).execute(
            f=f, y=y, fit=fit, show_diagnostics=show_diagnostics,
        )
        return {
            "evaluator_file": str(relative) if relative is not None else None,
            "evaluator": type(evaluator).__name__,
            **result,
        }

    @classmethod
    def format_result_dict(cls, result: dict[str, Any]) -> str:
        reference = result["evaluator_file"] or result["evaluator"]
        return f"Evaluator: {reference}\n" + EvaluateTool.format_result_dict(result)
