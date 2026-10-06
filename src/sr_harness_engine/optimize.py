"""Parameter optimization for symbolic expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize

from .evaluation import grouped_parameter_key, walk
from .expression import Expression, GroupedParameter, Parameter


@dataclass(frozen=True, slots=True)
class FitResult:
    """Result of fitting an expression to target observations."""
    expression: Expression
    parameters: dict[str, Any]
    loss: float
    success: bool
    message: str
    n_iter: int

    def evaluate(
        self, values: Mapping[str, Any], *, time: Any = None,
        delay_resolver: Any = None, num_nodes: int | None = None,
    ):
        """Evaluate the supplied model or expression.

        Args:
            values: Values keyed by symbol name.
            time: Optional sample times.
            delay_resolver: Optional callback that resolves delayed values.
            num_nodes: Explicit node count for indexed expressions.

        Returns:
            Predictions from the fitted expression.
        """
        return self.expression.evaluate(
            values, parameters=self.parameters, time=time,
            delay_resolver=delay_resolver, num_nodes=num_nodes,
        )

    predict = evaluate


def fit(
    expression: Expression,
    values: Mapping[str, Any],
    target: Any,
    *,
    initial: Mapping[str, Any] | None = None,
    method: str = "BFGS",
    options: Mapping[str, Any] | None = None,
    num_nodes: int | None = None,
) -> FitResult:
    """Minimize mean squared error and return fitted parameter values.

    Args:
        expression: Symbolic expression to process.
        values: Values keyed by symbol name.
        target: Target name or target values.
        initial: Optional initial parameter values.
        method: Optimization method name.
        options: Optional optimizer settings.
        num_nodes: Explicit node count for indexed expressions.

    Returns:
        Fitted parameters, expression, predictions, and loss.
    """
    initial = dict(initial or {})
    target = np.asarray(target, dtype=float)
    specs: list[tuple[str, Any | None]] = []
    named: dict[str, float | None] = {}
    grouped: dict[str, tuple[GroupedParameter, list[Any]]] = {}

    for node in walk(expression):
        if isinstance(node, Parameter):
            previous = named.get(node.name)
            if previous is not None and node.value is not None and previous != node.value:
                raise ValueError(f"Conflicting initial values for parameter {node.name!r}.")
            if node.name not in named or node.value is not None:
                named[node.name] = node.value
        elif isinstance(node, GroupedParameter):
            key = grouped_parameter_key(node)
            labels = np.asarray(node.by.evaluate(values), dtype=object).reshape(-1)
            unique = []
            for label in labels:
                label = label.item() if isinstance(label, np.generic) else label
                if label not in unique:
                    unique.append(label)
            grouped.setdefault(key, (node, unique))

    vector = []
    for name, default in named.items():
        value = initial.get(name, default if default is not None else 1.0)
        vector.append(float(value))
        specs.append((name, None))
    for key, (node, labels) in grouped.items():
        supplied = initial.get(key, node.value or {})
        if not isinstance(supplied, Mapping):
            raise TypeError(f"Initial value for grouped parameter {key!r} must be a mapping.")
        for label in labels:
            default = node.default if node.default is not None else 1.0
            vector.append(float(supplied.get(label, default)))
            specs.append((key, label))

    if not specs:
        prediction = np.asarray(expression.evaluate(values, num_nodes=num_nodes), dtype=float)
        loss = float(np.mean((prediction - target) ** 2))
        return FitResult(expression, {}, loss, True, "No parameters to optimize.", 0)

    def unpack(raw: np.ndarray) -> dict[str, Any]:
        parameters: dict[str, Any] = {}
        for value, (name, label) in zip(raw, specs):
            if label is None:
                parameters[name] = float(value)
            else:
                parameters.setdefault(name, {})[label] = float(value)
        return parameters

    def objective(raw: np.ndarray) -> float:
        try:
            prediction = np.asarray(
                expression.evaluate(values, parameters=unpack(raw), num_nodes=num_nodes),
                dtype=float,
            )
            if prediction.shape != target.shape:
                prediction = np.broadcast_to(prediction, target.shape)
            loss = np.mean((prediction - target) ** 2)
            return float(loss) if np.isfinite(loss) else 1e100
        except (FloatingPointError, OverflowError, ValueError):
            return 1e100

    result = minimize(
        objective,
        np.asarray(vector, dtype=float),
        method=method,
        options=dict(options or {}),
    )
    return FitResult(
        expression=expression,
        parameters=unpack(result.x),
        loss=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
        n_iter=int(getattr(result, "nit", 0)),
    )
