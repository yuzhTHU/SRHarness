"""Load built-in and user-authored evaluator implementations."""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

from .default_evaluator import DefaultEvaluator
from .graph_evaluator import GraphEvaluator
from .sandboxed_evaluator import create_sandboxed_evaluator

EVALUATOR_DIRECTORY = Path(__file__).resolve().parent
CUSTOM_TEMPLATE = (
    '"""Custom evaluator."""\n'
    "from .default_evaluator import DefaultEvaluator\n\n\n"
    "class CustomEvaluator(DefaultEvaluator):\n"
    "    pass\n"
)
BUILTIN_EVALUATORS = OrderedDict(
    (
        (
            "default",
            {
                "label": "DefaultEvaluator（默认评测）",
                "label_en": "DefaultEvaluator (default)",
                "class": DefaultEvaluator,
                "filename": "default_evaluator.py",
            },
        ),
        (
            "graph",
            {
                "label": "GraphEvaluator（图与超图）",
                "label_en": "GraphEvaluator (graph/hypergraph)",
                "class": GraphEvaluator,
                "filename": "graph_evaluator.py",
            },
        ),
    )
)
BUILTIN_EVALUATOR_CLASS_NAMES = {DefaultEvaluator.__name__, GraphEvaluator.__name__}


def evaluator_source(evaluator_id: str) -> str:
    """Read the source of a built-in evaluator.

    Args:
        evaluator_id: Built-in evaluator identifier.

    Returns:
        UTF-8 Python source for the evaluator.

    Raises:
        ValueError: If ``evaluator_id`` is unknown.
    """
    try:
        filename = BUILTIN_EVALUATORS[evaluator_id]["filename"]
    except KeyError as exc:
        raise ValueError(f"Unknown evaluator: {evaluator_id}") from exc
    return (EVALUATOR_DIRECTORY / filename).read_text(encoding="utf-8")


def evaluator_catalog() -> list[dict[str, str]]:
    """Return serializable metadata and source for built-in evaluators."""
    return [
        {
            "id": evaluator_id,
            "label": item["label"],
            "label_en": item["label_en"],
            "source": evaluator_source(evaluator_id),
            "abstract": False,
        }
        for evaluator_id, item in BUILTIN_EVALUATORS.items()
    ]


def create_builtin_evaluator(evaluator_id: str) -> DefaultEvaluator:
    """Instantiate a built-in evaluator.

    Args:
        evaluator_id: Built-in evaluator identifier.

    Returns:
        A new evaluator instance.

    Raises:
        ValueError: If ``evaluator_id`` is unknown.
    """
    try:
        return BUILTIN_EVALUATORS[evaluator_id]["class"]()
    except KeyError as exc:
        raise ValueError(f"Unknown evaluator: {evaluator_id}") from exc


def evaluator_filename(class_name: str) -> str:
    """Convert an evaluator class name to a snake-case Python filename.

    Args:
        class_name: Evaluator class name.

    Returns:
        Filename ending in ``.py``.
    """
    stem = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", class_name)
    stem = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", stem).lower()
    return f"{stem}.py"


def load_custom_evaluator(source: str | None = None, file: Path | None = None) -> DefaultEvaluator:
    """Load exactly one evaluator subclass as a virtual package module.

    Args:
        source: Python source defining one ``DefaultEvaluator`` subclass.
        file: Optional source file. Its contents are used when ``source`` is
            omitted and its path is retained for provenance.

    Returns:
        An instantiated custom evaluator with ``CUSTOM_EVALUATOR`` provenance.

    Raises:
        ValueError: If the source is unsafe, invalid, or does not define exactly
            one loadable evaluator subclass.
    """
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
        return create_sandboxed_evaluator(source, origin)
    except Exception as exc:
        if "SyntaxError:" in str(exc):
            raise ValueError(f"Invalid evaluator syntax: {exc}") from exc
        raise ValueError(f"Could not load custom evaluator: {exc}") from exc
