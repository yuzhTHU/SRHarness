"""Proxy custom evaluator calls into the data-only OS sandbox."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import sr_harness_engine as engine

from ..core import AgentContext
from ..runtime.sandbox import _run_sandbox_protocol
from .default_evaluator import DefaultEvaluator, MetricDict


def _json_value(value: Any) -> Any:
    """Return a JSON-safe value or omit unsupported runtime objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        converted = [_json_value(item) for item in value]
        return converted if all(item is not _UNSUPPORTED for item in converted) else _UNSUPPORTED
    if isinstance(value, dict):
        converted = {str(key): _json_value(item) for key, item in value.items()}
        return (
            converted
            if all(item is not _UNSUPPORTED for item in converted.values())
            else _UNSUPPORTED
        )
    return _UNSUPPORTED


_UNSUPPORTED = object()


def _context_payload(context: AgentContext) -> dict[str, Any]:
    args = {
        name: converted
        for name, value in vars(context.args).items()
        if (converted := _json_value(value)) is not _UNSUPPORTED
    }
    return {
        "args": args,
        "target": context.target,
        "variable_descriptions": context.variable_descriptions,
        "variable_axes": {name: list(axes) for name, axes in context.variable_axes.items()},
        "variable_structures": context.variable_structures,
        "relation_names": sorted(context.relation_names),
        "num_nodes": context.num_nodes,
    }


class SandboxedCustomEvaluator(DefaultEvaluator):
    """DefaultEvaluator-compatible proxy whose implementation runs out of process."""

    CUSTOM_EVALUATOR: dict[str, Any]
    SANDBOX_SOURCE: str
    SANDBOX_FILENAME: str

    @classmethod
    def _request(
        cls,
        action: str,
        context: AgentContext | None = None,
        *,
        f: engine.Expression | None = None,
        y: engine.Expression | None = None,
        candidate: bool = False,
    ):
        request: dict[str, Any] = {
            "action": action,
            "source": cls.SANDBOX_SOURCE,
            "filename": cls.SANDBOX_FILENAME,
        }
        arrays = None
        if context is not None:
            request["context"] = _context_payload(context)
            arrays = {"context": context.data}
        if f is not None:
            request["f"] = f.to_str(number_format=".17g")
        if y is not None:
            request["y"] = y.to_str(number_format=".17g")
        request["candidate"] = candidate
        return _run_sandbox_protocol(
            operation="evaluator",
            request=request,
            arrays=arrays,
            timeout_seconds=120,
            memory_limit_mb=4096,
            output_limit_bytes=64 * 1024,
        )

    @classmethod
    def split(cls, context: AgentContext):
        """Split context data in the isolated evaluator process."""
        result = cls._request("split", context)
        if set(result.arrays) != {"train", "validation"}:
            raise ValueError("Custom evaluator split() must return train and validation data")
        return {name: context.with_data(data) for name, data in result.arrays.items()}

    @classmethod
    def fit(
        cls, f: engine.Expression, y: engine.Expression, context: AgentContext
    ) -> engine.Expression:
        """Fit a general equality in the isolated evaluator process."""
        result = cls._request("fit", context, f=f, y=y)
        return engine.parse(str(result.payload["expression"]))

    @classmethod
    def evaluate(
        cls, f: engine.Expression, y: engine.Expression, context: AgentContext
    ) -> MetricDict:
        """Evaluate a general equality in the isolated evaluator process."""
        result = cls._request("evaluate", context, f=f, y=y)
        return dict(result.payload["metrics"])

    @classmethod
    def fit_candidate(cls, f: engine.Expression, context: AgentContext) -> engine.Expression:
        """Fit a candidate in the isolated evaluator process."""
        result = cls._request("fit", context, f=f, candidate=True)
        return engine.parse(str(result.payload["expression"]))

    @classmethod
    def evaluate_candidate(cls, f: engine.Expression, context: AgentContext) -> MetricDict:
        """Evaluate a candidate in the isolated evaluator process."""
        result = cls._request("evaluate", context, f=f, candidate=True)
        return dict(result.payload["metrics"])

    @classmethod
    def _evaluate_formula_isolated(
        cls,
        f: engine.Expression,
        y: engine.Expression,
        context: AgentContext,
        *,
        fit: bool,
        candidate: bool,
    ) -> tuple[engine.Expression, dict[str, MetricDict], dict[str, AgentContext]]:
        """Run split, optional fit, and both evaluations in one sandbox call."""
        request = {
            "action": "evaluate-formula",
            "source": cls.SANDBOX_SOURCE,
            "filename": cls.SANDBOX_FILENAME,
            "context": _context_payload(context),
            "f": f.to_str(number_format=".17g"),
            "y": y.to_str(number_format=".17g"),
            "fit": fit,
            "candidate": candidate,
        }
        result = _run_sandbox_protocol(
            operation="evaluator",
            request=request,
            arrays={"context": context.data},
            timeout_seconds=120,
            memory_limit_mb=4096,
            output_limit_bytes=64 * 1024,
        )
        split_contexts = {name: context.with_data(data) for name, data in result.arrays.items()}
        return (
            engine.parse(str(result.payload["expression"])),
            {name: dict(values) for name, values in result.payload["metrics"].items()},
            split_contexts,
        )


@lru_cache(maxsize=128)
def _inspect_source(source: str, filename: str) -> tuple[str, tuple[tuple[str, Any], ...]]:
    """Inspect immutable evaluator source once per process."""
    result = _run_sandbox_protocol(
        operation="evaluator",
        request={"action": "inspect", "source": source, "filename": filename},
        timeout_seconds=30,
        memory_limit_mb=2048,
        output_limit_bytes=64 * 1024,
    )
    return (
        str(result.payload["class_name"]),
        tuple(sorted(dict(result.payload.get("class_attributes", {})).items())),
    )


def create_sandboxed_evaluator(source: str, file: Path | None = None) -> DefaultEvaluator:
    """Validate source in the sandbox and return a named proxy instance."""
    filename = file.name if file is not None else "custom_evaluator.py"
    class_name, class_attributes = _inspect_source(source, filename)
    module_name = f"sr_harness.evaluator.{Path(filename).stem}"
    proxy_class = type(
        class_name,
        (SandboxedCustomEvaluator,),
        {
            **dict(class_attributes),
            "__module__": module_name,
            "CUSTOM_EVALUATOR": {"source": source, "file": file},
            "SANDBOX_SOURCE": source,
            "SANDBOX_FILENAME": filename,
        },
    )
    return proxy_class()
