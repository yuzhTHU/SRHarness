"""Evaluator catalog and guarded custom-evaluator loading for the Web UI."""
from __future__ import annotations

import ast
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import sr_harness_engine as engine

from ..evaluator import (
    BaseEvaluator,
    DefaultEvaluator,
    GraphEvaluator,
    TemplateCustomEvaluator,
)


EVALUATOR_DIRECTORY = Path(__file__).resolve().parents[1] / "evaluator"

BUILTIN_EVALUATORS = OrderedDict((
    ("base", {
        "label": "BaseEvaluator（抽象基类）",
        "label_en": "BaseEvaluator (abstract)",
        "class": BaseEvaluator,
        "filename": "base_evaluator.py",
        "abstract": True,
    }),
    ("default", {
        "label": "DefaultEvaluator（默认评测）",
        "label_en": "DefaultEvaluator (default)",
        "class": DefaultEvaluator,
        "filename": "default_evaluator.py",
    }),
    ("graph", {
        "label": "GraphEvaluator（图与超图）",
        "label_en": "GraphEvaluator (graph/hypergraph)",
        "class": GraphEvaluator,
        "filename": "graph_evaluator.py",
    }),
))


def _source(filename: str) -> str:
    return (EVALUATOR_DIRECTORY / filename).read_text(encoding="utf-8")


CUSTOM_TEMPLATE = _source("template_custom_evaluator.py")

_ALLOWED_IMPORT_ROOTS = {
    "abc", "collections", "math", "numpy", "scipy", "sr_harness",
    "sr_harness_engine", "typing",
}
_ALLOWED_RELATIVE_MODULES = {
    "base_evaluator", "default_evaluator", "graph_evaluator", "core",
}
_FORBIDDEN_NAMES = {
    "breakpoint", "compile", "delattr", "eval", "exec", "getattr", "globals",
    "input", "locals", "open", "setattr", "vars", "__import__",
}


def evaluator_source(evaluator_id: str) -> str:
    """Read the complete source file displayed by the evaluator editor."""
    if evaluator_id == "custom":
        return _source("template_custom_evaluator.py")
    try:
        return _source(BUILTIN_EVALUATORS[evaluator_id]["filename"])
    except KeyError as exc:
        raise ValueError(f"Unknown evaluator: {evaluator_id}") from exc


def evaluator_catalog() -> list[dict[str, str]]:
    """Return user-facing metadata and complete source for built-in evaluators."""
    return [
        {
            "id": evaluator_id,
            "label": item["label"],
            "label_en": item["label_en"],
            "source": evaluator_source(evaluator_id),
            "abstract": bool(item.get("abstract", False)),
        }
        for evaluator_id, item in BUILTIN_EVALUATORS.items()
    ]


def create_builtin_evaluator(evaluator_id: str) -> BaseEvaluator:
    """Instantiate a selectable built-in evaluator."""
    try:
        item = BUILTIN_EVALUATORS[evaluator_id]
        if item.get("abstract"):
            raise ValueError("BaseEvaluator is an abstract reference and cannot be selected")
        return item["class"]()
    except KeyError as exc:
        raise ValueError(f"Unknown evaluator: {evaluator_id}") from exc


def compile_custom_evaluator(source: str) -> BaseEvaluator:
    """Validate and instantiate one custom BaseEvaluator subclass."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Custom evaluator source must not be empty")
    if len(source) > 50_000:
        raise ValueError("Custom evaluator source exceeds 50000 characters")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"Invalid evaluator syntax: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            allowed = all(
                name.split(".", 1)[0] in _ALLOWED_IMPORT_ROOTS
                or (
                    isinstance(node, ast.ImportFrom)
                    and bool(node.level)
                    and name in _ALLOWED_RELATIVE_MODULES
                )
                for name in names
            )
            if not allowed:
                raise ValueError("Custom evaluators may only import scientific SRHarness modules")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"Custom evaluators may not use {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Custom evaluators may not access dunder attributes")
    namespace: dict[str, Any] = {
        "__name__": "sr_harness.evaluator.custom_evaluator",
        "__package__": "sr_harness.evaluator",
        "BaseEvaluator": BaseEvaluator,
        "DefaultEvaluator": DefaultEvaluator,
        "GraphEvaluator": GraphEvaluator,
        "engine": engine,
        "np": np,
        "numpy": np,
    }
    exec(compile(tree, "<custom-evaluator>", "exec"), namespace)
    builtins = {BaseEvaluator, DefaultEvaluator, GraphEvaluator, TemplateCustomEvaluator}
    classes = [
        value for value in namespace.values()
        if isinstance(value, type)
        and value not in builtins
        and issubclass(value, BaseEvaluator)
        and value.__module__ == "sr_harness.evaluator.custom_evaluator"
    ]
    if len(classes) != 1:
        raise ValueError("Define exactly one BaseEvaluator subclass")
    try:
        return classes[0]()
    except Exception as exc:
        raise ValueError(f"Could not instantiate custom evaluator: {exc}") from exc
