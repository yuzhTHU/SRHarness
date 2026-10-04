# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Numerical optimization for parameters and opted-in numeric literals."""
from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from typing import Any, Dict
import warnings

import numpy as np
import sr_harness_engine as engine
from scipy.optimize import differential_evolution, least_squares
from sr_harness_engine.evaluation import grouped_parameter_key
from sr_harness_engine.tree import children, with_children, transform


@dataclass(frozen=True)
class ConstantOptimizerConfig:
    """Numerical budget and reproducibility controls for ``fit_constants``."""

    random_state: int = 0
    n_restarts: int = 3
    max_nfev: int = 700
    differential_evolution_maxiter: int = 15
    differential_evolution_popsize: int = 5
    use_global_search: bool = True
    snap_exponents: bool = True
    use_eps: float = 1e-8


@dataclass(frozen=True)
class _ParameterSpec:
    name: str
    label: Any | None
    initial: float
    lower: float
    upper: float
    literal: bool = False


def _parameterize_literals(
    expression: engine.Expression,
    data: Dict[str, np.ndarray],
) -> tuple[engine.Expression, list[_ParameterSpec]]:
    specs: list[_ParameterSpec] = []
    named: dict[str, int] = {}
    grouped: dict[str, engine.GroupedParameter] = {}
    literal_index = 0

    def visit(
        node: engine.Expression,
        parent: engine.Expression | None = None,
        child_index: int | None = None,
    ) -> engine.Expression:
        nonlocal literal_index
        if isinstance(node, engine.Number):
            name = f"__literal_{literal_index}"
            literal_index += 1
            value = float(node.value)
            limit = max(100.0, 10.0 * abs(value) + 1.0)
            lower, upper = -limit, limit
            if isinstance(parent, engine.Binary) and parent.operator == "**" and child_index == 1:
                lower, upper = -10.0, 10.0
            if (
                isinstance(parent, engine.Binary)
                and parent.operator == "-"
                and child_index == 1
                and isinstance(parent.left, engine.Symbol)
                and parent.left.name in data
            ):
                feature = np.asarray(data[parent.left.name], dtype=float).reshape(-1)
                feature = feature[np.isfinite(feature)]
                if feature.size:
                    width = float(np.max(feature) - np.min(feature))
                    margin = max(0.2 * width, 10.0)
                    lower = float(np.min(feature) - margin)
                    upper = float(np.max(feature) + margin)
            specs.append(_ParameterSpec(name, None, value, lower, upper, literal=True))
            return engine.Parameter(name, value)
        if isinstance(node, engine.Parameter):
            if node.name not in named:
                initial = 1.0 if node.value is None else float(node.value)
                limit = max(100.0, 10.0 * abs(initial) + 1.0)
                named[node.name] = len(specs)
                specs.append(_ParameterSpec(node.name, None, initial, -limit, limit))
            return node
        if isinstance(node, engine.GroupedParameter):
            key = grouped_parameter_key(node)
            grouped.setdefault(key, node)
            return dataclass_replace(node, by=visit(node.by, node, 0))
        transformed_children = tuple(
            visit(child, node, index) for index, child in enumerate(children(node))
        )
        return with_children(node, transformed_children)

    parameterized = visit(expression)
    for key, node in grouped.items():
        labels = np.asarray(node.by.evaluate(data), dtype=object).reshape(-1)
        unique = []
        for label in labels:
            label = label.item() if isinstance(label, np.generic) else label
            if label not in unique:
                unique.append(label)
        supplied = dict(node.value or {})
        for label in unique:
            initial = float(supplied.get(label, node.default if node.default is not None else 1.0))
            limit = max(100.0, 10.0 * abs(initial) + 1.0)
            specs.append(_ParameterSpec(key, label, initial, -limit, limit))
    return parameterized, specs


def _unpack(raw: np.ndarray, specs: list[_ParameterSpec]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value, spec in zip(raw, specs):
        if spec.label is None:
            result[spec.name] = float(value)
        else:
            result.setdefault(spec.name, {})[spec.label] = float(value)
    return result


def _bind(
    expression: engine.Expression,
    parameters: dict[str, Any],
    literal_names: set[str],
) -> engine.Expression:
    def bind(node: engine.Expression) -> engine.Expression:
        if isinstance(node, engine.Parameter) and node.name in parameters:
            value = float(parameters[node.name])
            if node.name in literal_names:
                return engine.Number(value)
            return engine.Parameter(node.name, value)
        if isinstance(node, engine.GroupedParameter):
            key = grouped_parameter_key(node)
            if key in parameters:
                return dataclass_replace(node, value=dict(parameters[key]))
        return node

    return transform(expression, bind)


def fit_constants(
    f: engine.Expression,
    data: Dict[str, np.ndarray],
    y: np.ndarray,
    *,
    config: ConstantOptimizerConfig | None = None,
) -> engine.Expression:
    """Return a fitted expression without modifying ``f``.

    This function is an explicit opt-in optimization operation, so numeric
    literals are temporarily promoted to independent parameters. Outside this
    function, literals remain fixed and only ``param(...)`` nodes are fitable.
    """
    if not isinstance(f, engine.Expression):
        raise TypeError("f must be an sr_harness_engine.Expression")
    config = config or ConstantOptimizerConfig()
    target = np.asarray(y, dtype=float).reshape(-1)
    if target.size == 0 or not np.all(np.isfinite(target)):
        raise ValueError("y must contain at least one finite-only sample")
    arrays = {name: np.asarray(value) for name, value in data.items()}
    if any(np.asarray(value).reshape(-1).size != target.size for value in arrays.values()):
        raise ValueError("all data arrays must have the same number of samples as y")

    expression, specs = _parameterize_literals(f.copy(), arrays)
    if not specs:
        return expression.copy()
    initial = np.asarray([spec.initial for spec in specs], dtype=float)
    lower = np.asarray([spec.lower for spec in specs], dtype=float)
    upper = np.asarray([spec.upper for spec in specs], dtype=float)
    scale = max(float(np.std(target)), float(np.max(np.abs(target))) * 1e-12, 1e-12)
    invalid = np.full(target.size, 1e6, dtype=float)

    def prediction(raw: np.ndarray) -> np.ndarray | None:
        try:
            with np.errstate(all="ignore"):
                value = np.asarray(
                    expression.evaluate(arrays, parameters=_unpack(raw, specs)), dtype=float
                ).reshape(-1)
            if value.size == 1 and target.size != 1:
                value = np.full(target.size, float(value[0]))
            if value.size != target.size or not np.all(np.isfinite(value)):
                return None
            return value
        except (ArithmeticError, FloatingPointError, KeyError, TypeError, ValueError):
            return None

    def residual(raw: np.ndarray) -> np.ndarray:
        value = prediction(raw)
        return invalid if value is None else (value - target) / scale

    def loss(raw: np.ndarray) -> float:
        value = prediction(raw)
        return 1e100 if value is None else float(np.mean(np.square(value - target)))

    starts = [np.clip(initial, lower, upper)]
    rng = np.random.default_rng(config.random_state)
    for _ in range(max(0, config.n_restarts - 1)):
        starts.append(rng.uniform(lower, upper))
    if config.use_global_search and len(specs) <= 12:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            global_result = differential_evolution(
                loss,
                list(zip(lower, upper)),
                seed=config.random_state,
                maxiter=config.differential_evolution_maxiter,
                popsize=config.differential_evolution_popsize,
                polish=False,
                updating="immediate",
            )
        starts.append(global_result.x)

    best = starts[0]
    best_loss = loss(best)
    for start in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = least_squares(
                residual,
                start,
                bounds=(lower, upper),
                max_nfev=config.max_nfev,
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
            )
        current_loss = loss(result.x)
        if current_loss < best_loss:
            best, best_loss = result.x, current_loss

    parameters = _unpack(best, specs)
    literal_names = {spec.name for spec in specs if spec.literal}
    return _bind(expression, parameters, literal_names)
