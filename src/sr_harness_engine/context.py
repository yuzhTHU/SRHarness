"""Duck-typed data-context helpers for expression evaluation and fitting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_context(
    context: Mapping[str, Any] | Any | None,
    target: Any = None,
    num_nodes: int | None = None,
) -> tuple[Mapping[str, Any], Any, int | None]:
    """Resolve values, target observations, and node metadata from a context.

    Ordinary value mappings continue to work. Context-like objects may expose
    a mapping-valued ``data`` field plus ``target`` and ``num_nodes`` metadata.
    """
    if context is None:
        values: Mapping[str, Any] = {}
    elif isinstance(context, Mapping):
        candidate = context.get("data")
        values = candidate if isinstance(candidate, Mapping) else context
    else:
        candidate = getattr(context, "data", None)
        if not isinstance(candidate, Mapping):
            raise TypeError("context must expose mapping-valued data")
        values = candidate

    if num_nodes is None:
        num_nodes = getattr(values, "num_nodes", None)
    if num_nodes is None:
        num_nodes = getattr(context, "num_nodes", None)
    if num_nodes is None and isinstance(context, Mapping):
        num_nodes = context.get("num_nodes")

    if target is None:
        target = getattr(context, "target", None)
        if target is None and isinstance(context, Mapping):
            target = context.get("target")
    if isinstance(target, str) and target in values:
        target = values[target]
    return values, target, num_nodes
