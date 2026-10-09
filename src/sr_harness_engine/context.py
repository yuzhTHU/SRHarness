"""Resolve expression inputs from value mappings or context objects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_context(
    context: Mapping[str, Any] | Any | None,
    target: Any = None,
    num_nodes: int | None = None,
) -> tuple[Mapping[str, Any], Any, int | None]:
    """Resolve values, target observations, and node metadata from a context.

    A mapping is always interpreted directly as symbol values. Context objects
    must expose a mapping-valued ``data`` field and may expose ``target`` and
    ``num_nodes`` metadata.
    """
    if context is None:
        values: Mapping[str, Any] = {}
    elif isinstance(context, Mapping):
        values = context
    else:
        candidate = getattr(context, "data", None)
        if not isinstance(candidate, Mapping):
            raise TypeError("context must expose mapping-valued data")
        values = candidate

    if num_nodes is None:
        num_nodes = getattr(context, "num_nodes", None)

    if target is None:
        target = getattr(context, "target", None)
    if isinstance(target, str) and target in values:
        target = values[target]
    return values, target, num_nodes
