"""Load built-in and user-authored evaluator implementations."""
from __future__ import annotations

import ast
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import sr_harness_engine as engine

from .default_evaluator import DefaultEvaluator
from .graph_evaluator import GraphEvaluator

EVALUATOR_DIRECTORY = Path(__file__).resolve().parent
CUSTOM_TEMPLATE = (
    '"""Custom evaluator."""\n'
    'from .default_evaluator import DefaultEvaluator\n\n\n'
    'class CustomEvaluator(DefaultEvaluator):\n'
    '    pass\n'
)
BUILTIN_EVALUATORS = OrderedDict((
    ("default", {"label": "DefaultEvaluator（默认评测）", "label_en": "DefaultEvaluator (default)", "class": DefaultEvaluator, "filename": "default_evaluator.py"}),
    ("graph", {"label": "GraphEvaluator（图与超图）", "label_en": "GraphEvaluator (graph/hypergraph)", "class": GraphEvaluator, "filename": "graph_evaluator.py"}),
))
BUILTIN_EVALUATOR_CLASS_NAMES = {DefaultEvaluator.__name__, GraphEvaluator.__name__}
_ALLOWED_IMPORT_ROOTS = {
    "__future__", "abc", "collections", "math", "numbers", "numpy", "scipy",
    "sr_harness", "sr_harness_engine", "typing", "warnings",
}
_ALLOWED_RELATIVE_MODULES = {"default_evaluator", "graph_evaluator", "utils", "core"}
_FORBIDDEN_NAMES = {
    "breakpoint", "compile", "delattr", "eval", "exec", "getattr", "globals",
    "input", "locals", "open", "setattr", "vars", "__import__",
}


def evaluator_source(evaluator_id: str) -> str:
    try:
        filename = BUILTIN_EVALUATORS[evaluator_id]["filename"]
    except KeyError as exc:
        raise ValueError(f"Unknown evaluator: {evaluator_id}") from exc
    return (EVALUATOR_DIRECTORY / filename).read_text(encoding="utf-8")


def evaluator_catalog() -> list[dict[str, str]]:
    return [{
        "id": evaluator_id, "label": item["label"], "label_en": item["label_en"],
        "source": evaluator_source(evaluator_id), "abstract": False,
    } for evaluator_id, item in BUILTIN_EVALUATORS.items()]


def create_builtin_evaluator(evaluator_id: str) -> DefaultEvaluator:
    try:
        return BUILTIN_EVALUATORS[evaluator_id]["class"]()
    except KeyError as exc:
        raise ValueError(f"Unknown evaluator: {evaluator_id}") from exc


def evaluator_filename(class_name: str) -> str:
    stem = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", class_name)
    stem = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", stem).lower()
    return f"{stem}.py"


def load_custom_evaluator(source: str | None = None, file: Path | None = None) -> DefaultEvaluator:
    """Load exactly one ``DefaultEvaluator`` subclass as an evaluator package module."""
    origin = Path(file).expanduser().resolve() if file is not None else None
    if source is None:
        if origin is None:
            raise ValueError("source or file must be provided")
        try:
            source = origin.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Could not read custom evaluator: {exc}") from exc
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
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            relative_allowed = (
                isinstance(node, ast.ImportFrom)
                and bool(node.level)
                and (
                    (node.module or "") in _ALLOWED_RELATIVE_MODULES
                    or node.module is None and all(alias.name == "utils" for alias in node.names)
                )
            )
            if not relative_allowed and not all(name.split(".", 1)[0] in _ALLOWED_IMPORT_ROOTS for name in names):
                raise ValueError("Custom evaluators may only import scientific SRHarness modules")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"Custom evaluators may not use {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Custom evaluators may not access dunder attributes")

    virtual_path = EVALUATOR_DIRECTORY / (origin.name if origin is not None else "custom_evaluator.py")
    module_name = f"sr_harness.evaluator.{virtual_path.stem}"
    namespace: dict[str, Any] = {
        "__name__": module_name, "__package__": "sr_harness.evaluator", "__file__": str(virtual_path),
        "DefaultEvaluator": DefaultEvaluator, "GraphEvaluator": GraphEvaluator,
        "engine": engine, "np": np, "numpy": np,
    }
    try:
        exec(compile(tree, str(virtual_path), "exec", dont_inherit=True), namespace)
    except Exception as exc:
        raise ValueError(f"Could not load custom evaluator: {exc}") from exc
    classes = [
        value for value in namespace.values()
        if isinstance(value, type) and value not in {DefaultEvaluator, GraphEvaluator}
        and issubclass(value, DefaultEvaluator) and value.__module__ == module_name
    ]
    if len(classes) != 1:
        raise ValueError("Define exactly one DefaultEvaluator subclass")
    evaluator_class = classes[0]
    evaluator_class.CUSTOM_EVALUATOR = {"source": source, "file": origin}
    try:
        return evaluator_class()
    except Exception as exc:
        raise ValueError(f"Could not instantiate custom evaluator: {exc}") from exc
