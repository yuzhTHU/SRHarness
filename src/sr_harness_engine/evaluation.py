"""NumPy evaluator for symbolic expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .expression import (
    Aggregate,
    Binary,
    Expression,
    Function,
    GroupedParameter,
    Indexed,
    Number,
    Parameter,
    Reduction,
    RelationLift,
    Symbol,
    Unary,
)
from .tree import children, iter_preorder


@dataclass(slots=True)
class _RelationScope:
    name: str
    table: np.ndarray
    indices: tuple[str, ...]
    legacy: bool = False

    @property
    def bindings(self) -> dict[str, np.ndarray]:
        return {name: self.table[:, column] for column, name in enumerate(self.indices)}


class Evaluator:
    """NumPy expression-tree evaluator."""
    def __init__(
        self,
        values: Mapping[str, Any] | None,
        parameters: Mapping[str, Any] | None,
        time: Any,
        delay_resolver: Callable[..., Any] | None,
        expression: Expression,
    ):
        self.values = dict(values or {})
        self.parameters = dict(parameters or {})
        self.time = None if time is None else np.asarray(time, dtype=float)
        self.delay_resolver = delay_resolver
        self.parameter_defaults = self._parameter_defaults(expression)

    def __call__(self, expression: Expression) -> Any:
        return self._eval(expression, None)

    def _eval(self, node: Expression, scope: _RelationScope | None) -> Any:
        if isinstance(node, Number):
            return np.asarray(node.value)
        if isinstance(node, Symbol):
            if node.name in self.values:
                return np.asarray(self.values[node.name])
            if node.value is not None:
                return np.asarray(node.value)
            raise KeyError(f"No value was provided for symbol {node.name!r}.")
        if isinstance(node, Parameter):
            value = self.parameters.get(node.name, self.parameter_defaults.get(node.name))
            if value is None:
                raise ValueError(
                    f"Parameter {node.name!r} has no value. Fit it or provide "
                    f"parameters[{node.name!r}]."
                )
            return np.asarray(value, dtype=float)
        if isinstance(node, GroupedParameter):
            labels = np.asarray(self._eval(node.by, scope), dtype=object)
            key = grouped_parameter_key(node)
            fitted = self.parameters.get(key, node.value)
            fitted = {} if fitted is None else dict(fitted)
            default = node.default
            missing = [label for label in _unique(labels) if label not in fitted]
            if missing and default is None:
                raise ValueError(
                    f"Grouped parameter {key!r} has no values for categories "
                    f"{missing!r}. Fit it first."
                )
            result = np.asarray(
                [fitted.get(label, default) for label in labels.reshape(-1)], dtype=float
            )
            return result.reshape(labels.shape)
        if isinstance(node, Unary):
            value = self._eval(node.operand, scope)
            return -value
        if isinstance(node, Binary):
            left, right = _align(self._eval(node.left, scope), self._eval(node.right, scope))
            with np.errstate(all="ignore"):
                return {
                    "+": np.add,
                    "-": np.subtract,
                    "*": np.multiply,
                    "/": np.divide,
                    "**": np.power,
                }[node.operator](left, right)
        if isinstance(node, Function):
            return self._function(node, scope)
        if isinstance(node, Indexed):
            return self._indexed(node, scope)
        if isinstance(node, Reduction):
            return self._reduction(node)
        if isinstance(node, Aggregate):
            return self._aggregate(node)
        if isinstance(node, RelationLift):
            return self._relation_lift(node, scope)
        raise TypeError(f"Cannot evaluate {type(node).__name__}.")

    def _function(self, node: Function, scope: _RelationScope | None) -> Any:
        arguments = [self._eval(argument, scope) for argument in node.arguments]
        if node.name == "delay":
            return self._delay(arguments[0], arguments[1])
        functions = {
            "abs": np.abs,
            "arccos": np.arccos,
            "arcsin": np.arcsin,
            "arctan": np.arctan,
            "cos": np.cos,
            "cosh": np.cosh,
            "cot": lambda value: 1.0 / np.tan(value),
            "csc": lambda value: 1.0 / np.sin(value),
            "exp": np.exp,
            "inv": lambda value: 1.0 / value,
            "log": np.log,
            "log10": np.log10,
            "max": np.maximum,
            "min": np.minimum,
            "pow2": lambda value: value**2,
            "pow3": lambda value: value**3,
            "sec": lambda value: 1.0 / np.cos(value),
            "sech": lambda value: 1.0 / np.cosh(value),
            "sin": np.sin,
            "sinh": np.sinh,
            "sign": np.sign,
            "sqrt": np.sqrt,
            "tan": np.tan,
            "tanh": np.tanh,
            "sigmoid": lambda value: 1.0 / (1.0 + np.exp(-value)),
        }
        with np.errstate(all="ignore"):
            return functions[node.name](*arguments)

    def _indexed(self, node: Indexed, scope: _RelationScope | None) -> Any:
        value = np.asarray(self._eval(node.base, None))
        names = tuple(index.name for index in node.indices)
        if scope is not None and len(names) > 1:
            if names != scope.indices:
                raise ValueError(
                    f"Relation fields must use bound indices {scope.indices}, not {names}."
                )
            if isinstance(node.base, Symbol) and node.base.name == scope.name:
                return np.ones(len(scope.table), dtype=float)
            edge_count = len(scope.table)
            if value.ndim == 1 and value.shape[0] == edge_count:
                return value
            if value.ndim > 1 and value.shape[-1] == edge_count:
                return np.moveaxis(value, -1, 0)
            raise ValueError(
                f"Relation field {node.base!s} must have an edge axis of length "
                f"{edge_count} in its last dimension."
            )
        if scope is not None and len(names) == 1 and names[0] in scope.bindings:
            return value[scope.bindings[names[0]]]
        if len(names) > value.ndim:
            raise ValueError(
                f"{len(names)} symbolic indices were applied to a {value.ndim}-dimensional value."
            )
        # A free index names an existing leading axis; it does not slice it.
        return value

    def _reduction(self, node: Reduction) -> Any:
        scope = self._relation_scope(node.relation) if node.relation is not None else None
        value = np.asarray(self._eval(node.operand, scope))
        reduced = {index.name for index in node.indices}
        if scope is None:
            if self._find_relation_scope(node.operand) is not None:
                raise ValueError(
                    "Pass the relation separately, for example "
                    "sum[j](A[i, j], expression)."
                )
            return np.sum(value, axis=tuple(range(min(len(reduced), value.ndim))))

        unknown = reduced - set(scope.indices)
        if unknown:
            raise ValueError(
                f"Indices {sorted(unknown)!r} are not bound by relation {scope.name!r}."
            )
        free = [name for name in scope.indices if name not in reduced]
        if not free:
            return np.sum(value, axis=0)
        coordinates = tuple(scope.bindings[name] for name in free)
        sizes = tuple(
            self._index_size(node.operand, name, coordinate)
            for name, coordinate in zip(free, coordinates)
        )
        result = np.zeros(sizes + value.shape[1:], dtype=np.result_type(value, float))
        np.add.at(result, coordinates[0] if len(coordinates) == 1 else coordinates, value)
        return result

    def _relation_scope(self, relation: Expression) -> _RelationScope:
        if not isinstance(relation, Indexed) or not isinstance(relation.base, Symbol):
            raise ValueError("A reduction relation must be indexed, such as A[i, j].")
        name, table = self._relation(relation.base)
        indices = tuple(index.name for index in relation.indices)
        if table.shape[1] != len(indices):
            raise ValueError(
                f"Relation {name!r} has {table.shape[1]} columns but "
                f"{len(indices)} indices were supplied."
            )
        return _RelationScope(name, table, indices)

    def _aggregate(self, node: Aggregate) -> Any:
        relation_name, table = self._relation(node.relation)
        if table.shape[1] != 2:
            raise ValueError("aggr currently expects a two-column edge list.")
        scope = _RelationScope(relation_name, table, ("source", "target"), legacy=True)
        value = np.asarray(self._eval(node.operand, scope))
        target = table[:, 1]
        fallback = int(target.max()) + 1 if target.size else 0
        size = self._leading_size(node.operand, fallback=fallback)
        result = np.zeros((size,) + value.shape[1:], dtype=np.result_type(value, float))
        np.add.at(result, target, value)
        return result

    def _relation_lift(self, node: RelationLift, scope: _RelationScope | None) -> Any:
        if node.relation is not None:
            name, table = self._relation(node.relation)
            scope = _RelationScope(name, table, ("source", "target"), legacy=True)
        if scope is None:
            raise ValueError(f"{node.role} requires a relation argument or an enclosing aggr(...).")
        column = 1 if node.role == "target" else 0
        return np.asarray(self._eval(node.operand, None))[scope.table[:, column]]

    def _delay(self, value: Any, lag: Any) -> Any:
        value = np.asarray(value, dtype=float)
        lag = np.asarray(lag, dtype=float)
        if self.delay_resolver is not None:
            return np.asarray(self.delay_resolver(value, lag, self.time))
        if value.ndim == 0:
            return value
        time = np.arange(value.shape[0], dtype=float) if self.time is None else self.time
        if time.shape != (value.shape[0],):
            raise ValueError("time must be one-dimensional and match the leading data axis.")
        query = time - lag
        try:
            query = np.broadcast_to(query, (value.shape[0],))
        except ValueError as error:
            raise ValueError(
                "delay delta must be scalar or match the leading data axis."
            ) from error
        flat = value.reshape(value.shape[0], -1)
        delayed = np.column_stack([
            np.interp(query, time, column, left=np.nan, right=np.nan) for column in flat.T
        ])
        return delayed.reshape(value.shape)

    def _find_relation_scope(self, node: Expression) -> _RelationScope | None:
        relations: dict[str, tuple[np.ndarray, tuple[str, ...]]] = {}
        for item in walk(node):
            if not isinstance(item, Indexed) or not isinstance(item.base, Symbol):
                continue
            if item.base.name not in self.values:
                continue
            value = np.asarray(self.values[item.base.name])
            if value.ndim != 2 or value.shape[1] != len(item.indices) or len(item.indices) < 2:
                continue
            if value.dtype.kind not in "iu":
                continue
            relations[item.base.name] = (
                value.astype(int, copy=False),
                tuple(index.name for index in item.indices),
            )
        if not relations:
            return None
        if len(relations) > 1:
            raise ValueError(
                "One reduction cannot currently combine multiple relations: "
                f"{sorted(relations)}."
            )
        name, (table, indices) = next(iter(relations.items()))
        if np.any(table < 0):
            raise ValueError("Relation indices must be non-negative integers.")
        return _RelationScope(name, table, indices)

    def _relation(self, node: Expression) -> tuple[str, np.ndarray]:
        value = np.asarray(self._eval(node, None))
        if value.ndim != 2 or value.dtype.kind not in "iu" or value.shape[1] < 2:
            raise ValueError("A relation must be an integer array shaped (relations, arity).")
        name = node.name if isinstance(node, Symbol) else "<relation>"
        return name, value.astype(int, copy=False)

    def _index_size(self, node: Expression, index: str, coordinate: np.ndarray) -> int:
        sizes = []
        for item in walk(node):
            if not isinstance(item, Indexed) or len(item.indices) != 1:
                continue
            if not isinstance(item.base, Symbol):
                continue
            if item.base.name in self.values:
                value = np.asarray(self.values[item.base.name])
                # Every singly-indexed node field belongs to the same node domain
                # inside one relation contraction, even when this particular free
                # index does not occur on that field (for example A[i,j] with x[j]).
                if value.ndim:
                    sizes.append(value.shape[0])
        fallback = int(coordinate.max()) + 1 if coordinate.size else 0
        return max(sizes, default=fallback)

    def _leading_size(self, node: Expression, fallback: int) -> int:
        sizes = []
        for item in walk(node):
            if isinstance(item, Symbol) and item.name in self.values:
                value = np.asarray(self.values[item.name])
                is_edge_list = (
                    value.ndim == 2 and value.dtype.kind in "iu" and value.shape[1] == 2
                )
                if value.ndim and not is_edge_list:
                    sizes.append(value.shape[0])
        return max(sizes, default=fallback)

    @staticmethod
    def _parameter_defaults(expression: Expression) -> dict[str, float | None]:
        defaults: dict[str, float | None] = {}
        for node in walk(expression):
            if not isinstance(node, Parameter):
                continue
            previous = defaults.get(node.name)
            if previous is not None and node.value is not None and previous != node.value:
                raise ValueError(f"Conflicting initial values for parameter {node.name!r}.")
            if node.value is not None or node.name not in defaults:
                defaults[node.name] = node.value
        return defaults


def _align(left: Any, right: Any) -> tuple[np.ndarray, np.ndarray]:
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim == 1 and right.ndim > 1 and left.shape[0] == right.shape[0]:
        left = left.reshape((left.shape[0],) + (1,) * (right.ndim - 1))
    if right.ndim == 1 and left.ndim > 1 and right.shape[0] == left.shape[0]:
        right = right.reshape((right.shape[0],) + (1,) * (left.ndim - 1))
    return left, right


def _unique(values: np.ndarray) -> list[Any]:
    result = []
    for value in values.reshape(-1):
        value = value.item() if isinstance(value, np.generic) else value
        if value not in result:
            result.append(value)
    return result


def grouped_parameter_key(node: GroupedParameter) -> str:
    """Return the storage key for a grouped parameter.

    Args:
        node: Grouped parameter to identify.

    Returns:
        Its explicit name or a stable name derived from the grouping symbol.
    """
    if node.name:
        return node.name
    if isinstance(node.by, Symbol):
        return f"grouped:{node.by.name}"
    return f"grouped:{node.by}"


def walk(node: Expression):
    """Iterate over an expression tree in preorder.

    Args:
        node: Expression or syntax-tree node.

    Yields:
        Expression nodes in parent-before-children order.
    """
    yield from iter_preorder(node)


def evaluate(
    expression: Expression,
    values: Mapping[str, Any] | None = None,
    *,
    parameters: Mapping[str, Any] | None = None,
    time: Any = None,
    delay_resolver: Callable[..., Any] | None = None,
) -> Any:
    """Evaluate an expression without executing arbitrary Python code.

    Args:
        expression: Symbolic expression to process.
        values: Values keyed by symbol name.
        parameters: Fitted parameter values keyed by parameter name.
        time: Optional sample times.
        delay_resolver: Optional callback that resolves delayed values.

    Returns:
        The evaluated scalar or NumPy array.
    """
    return Evaluator(values, parameters, time, delay_resolver, expression)(expression)
