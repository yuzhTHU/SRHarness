"""Expression simplification and structural analysis."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .evaluation import grouped_parameter_key
from .expression import Binary, Expression, Function, GroupedParameter, Number, Parameter, Unary
from .tree import iter_preorder, transform


def fold_constants(expression: Expression) -> Expression:
    """Evaluate closed numerical subexpressions without reordering terms.

    Args:
        expression: Symbolic expression to process.

    Returns:
        A simplified expression with constant-only branches evaluated.
    """

    def fold(node: Expression) -> Expression:
        if isinstance(node, Unary) and isinstance(node.operand, Number):
            return Number(-node.operand.value)
        if (
            isinstance(node, Binary)
            and isinstance(node.left, Number)
            and isinstance(node.right, Number)
        ):
            operations = {
                "+": lambda left, right: left + right,
                "-": lambda left, right: left - right,
                "*": lambda left, right: left * right,
            }
            if node.operator not in operations:
                return node
            try:
                value = operations[node.operator](node.left.value, node.right.value)
            except (ArithmeticError, OverflowError, ValueError):
                return node
            return Number(value) if _finite_scalar(value) else node
        return node

    return transform(expression, fold)


def count_parameters(
    expression: Expression,
    values: Mapping[str, Any] | None = None,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> int:
    """Count independent fitted values represented by an expression.

    Repeated named parameters count once. A grouped parameter counts once per
    known category. When categories are unavailable, it counts as one
    unresolved parameter family.

    Args:
        expression: Symbolic expression to process.
        values: Values keyed by symbol name.
        parameters: Fitted parameter values keyed by parameter name.

    Returns:
        The number of independent scalar parameter values.
    """
    values = dict(values or {})
    parameters = dict(parameters or {})
    named: set[str] = set()
    grouped: dict[str, GroupedParameter] = {}
    for node in iter_preorder(expression):
        if isinstance(node, Parameter):
            named.add(node.name)
        elif isinstance(node, GroupedParameter):
            grouped.setdefault(grouped_parameter_key(node), node)

    count = len(named)
    for key, node in grouped.items():
        if isinstance(parameters.get(key), Mapping):
            count += len(parameters[key])
            continue
        try:
            labels = np.asarray(node.by.evaluate(values), dtype=object).reshape(-1)
        except (KeyError, ValueError):
            count += len(node.value) if isinstance(node.value, Mapping) else 1
        else:
            count += len({_hashable_label(label) for label in labels})
    return count


def _finite_scalar(value: Any) -> bool:
    array = np.asarray(value)
    return array.ndim == 0 and bool(np.isfinite(array))


def _hashable_label(value: Any) -> Any:
    value = value.item() if isinstance(value, np.generic) else value
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value
