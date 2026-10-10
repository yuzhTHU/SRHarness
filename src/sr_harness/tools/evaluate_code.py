# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Evaluate Python-defined models inside a data-only OS sandbox."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import sr_harness_engine as engine

from ..runtime.sandbox import _run_sandbox_protocol
from .base_tool import BaseTool, ToolMetadata
from .code_executor import CodeExecutorTool


@BaseTool.register("evaluate_code")
class EvaluateCodeTool(BaseTool):
    """Implementation of the evaluate code tool."""

    metadata = ToolMetadata(name="evaluate_code")
    DEFAULT_TIMEOUT_SECONDS = CodeExecutorTool.DEFAULT_TIMEOUT_SECONDS
    DEFAULT_MEMORY_LIMIT_MB = CodeExecutorTool.DEFAULT_MEMORY_LIMIT_MB
    DEFAULT_OUTPUT_LIMIT_BYTES = CodeExecutorTool.DEFAULT_OUTPUT_LIMIT_BYTES
    MAX_TIMEOUT_SECONDS = CodeExecutorTool.MAX_TIMEOUT_SECONDS
    MAX_MEMORY_LIMIT_MB = CodeExecutorTool.MAX_MEMORY_LIMIT_MB
    MAX_OUTPUT_LIMIT_BYTES = CodeExecutorTool.MAX_OUTPUT_LIMIT_BYTES

    def execute(
        self,
        model_code: str,
        predict_code: str,
        y: str = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
        output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
        show_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate a Python-defined candidate model on the current dataset.

        If possible, first use ``read_skill`` to review ``sr-harness-engine-syntax`` and any
        relevant companion documentation, such as ``sr-harness-engine-graph-syntax`` for graph
        or hypergraph models. Prefer a structured SRHarness Engine formula when it can express
        the candidate, and use this tool when it cannot do so conveniently. The code runs in a
        restricted sandbox, then the tool computes metrics against the target and returns the
        formatted model under the ``formula`` key.

        This tool can return candidate formulas for submission when `y` is the target
        variable and `predict_code` does not depend on the target variable.

        Args:
            model_code: Code containing exactly one function with signature
                `def func(data)` plus optional top-level imports. `data` is a dictionary
                mapping variable names to numeric arrays, including the target variable.
                This function should return a Python dict as the fitted `model`, which
                will be passed to `predict_code` and `format_code`. The returned model
                should contain a `description` field that identifies the model with a
                concise mathematical formula (e.g., y = aₖx² + bₖx + cₖ,
                yᵢ = MLP1(xᵢ) + Σ Aᵢⱼ MLP2(xᵢ, xⱼ)).
            predict_code: Code containing exactly one function with signature
                `def func(data, model)` plus optional top-level imports.
                The function should return the predicted value for the target `y`,
                which must be array-like and compatible with the target shape.
            y: Target variable name. Use target variable by default.
                Expressions are also supported, e.g., "log(y)", "y - x1"
            timeout_seconds: Wall-clock timeout in seconds. The effective value is capped.
            memory_limit_mb: Address-space memory limit in MB. The effective value is capped.
            output_limit_bytes: Limit on the amount of output (in bytes) that can be produced.
            show_diagnostics: Whether metrics should include compact residual diagnostics.
        """
        train_data = self.context.train_split.data
        validation_data = self.context.validation_split.data
        y = (y or self.context.target).strip().strip('"').strip("'")
        eq_y = self.parse_formula(y)
        timeout_seconds = CodeExecutorTool.bounded_int(
            timeout_seconds, self.DEFAULT_TIMEOUT_SECONDS, 1, self.MAX_TIMEOUT_SECONDS
        )
        memory_limit_mb = CodeExecutorTool.bounded_int(
            memory_limit_mb, self.DEFAULT_MEMORY_LIMIT_MB, 64, self.MAX_MEMORY_LIMIT_MB
        )
        output_limit_bytes = CodeExecutorTool.bounded_int(
            output_limit_bytes, self.DEFAULT_OUTPUT_LIMIT_BYTES, 1024, self.MAX_OUTPUT_LIMIT_BYTES
        )
        groups = {"train": train_data}
        if validation_data:
            groups["validation"] = validation_data
        result = _run_sandbox_protocol(
            operation="evaluate-code",
            request={
                "model_code": CodeExecutorTool.extract_code(model_code),
                "predict_code": CodeExecutorTool.extract_code(predict_code),
                "target": self.context.target,
            },
            arrays=groups,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
            output_limit_bytes=output_limit_bytes,
            interruption_event=getattr(self, "cancel_event", None),
        )
        predictions = result.arrays.get("predictions", {})
        opaque_f = engine.parse("__code_model_prediction__")
        data_split_results: dict[str, dict[str, Any]] = {}
        for split_name, split_values in groups.items():
            if split_name not in predictions:
                raise ValueError(f"Sandbox did not return {split_name} predictions")
            y_pred = np.asarray(predictions[split_name])
            y_true = np.asarray(eq_y.eval(split_values))
            y_pred, y_true = np.broadcast_arrays(y_pred, y_true)
            metrics = self.calculate_metrics(opaque_f, y_true, y_pred)
            metrics["complexity"] = len(model_code) + len(predict_code)
            metrics.pop("aic", None)
            metrics.pop("bic", None)
            split_result: dict[str, Any] = {"metrics": metrics}
            if show_diagnostics:
                split_result["diagnostics"] = self.residual_diagnostics(
                    y_true=y_true,
                    y_pred=y_pred,
                    data=split_values,
                    target_expression=eq_y.to_str(),
                )
            data_split_results[split_name] = split_result
        return {
            "formula": str(result.payload["model_str"]),
            "is_candidate": bool(result.payload["is_candidate"])
            and eq_y.to_str() == self.context.target,
            "data_split_results": data_split_results,
        }

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        """Format a tool result for the language model.

        Args:
            result: Result mapping to format or update.

        Returns:
            str: The operation result.
        """
        text = cls.format_evaluation_result(result, title="Evaluated code-defined model")
        marker = "Formula Complexity="
        if text.count(marker) > 1:
            raise ValueError(
                f"Expected at most one '{marker}' field in formatted evaluation output, "
                f"but found {text.count(marker)}."
            )
        text = text.replace(marker, "Code Complexity=")
        return (
            text + "\n"
            "(Note: The Code Complexity is measured as source-code character count, so it is not "
            "directly comparable to Formula Complexity defined as the symbolic formula node count.)"
        )
