"""Evaluator for broadcast-compatible graph and hypergraph arrays."""
from __future__ import annotations

import numpy as np

import sr_harness_engine as engine

from ..core import AgentContext
from . import utils
from .default_evaluator import ContextSplits, DefaultEvaluator, MetricDict


class GraphEvaluator(DefaultEvaluator):
    """Evaluate variables with ``(..., N/E/H)`` graph-aligned dimensions."""

    @classmethod
    def fit(cls, f: engine.Expression, y: engine.Expression, context: AgentContext) -> engine.Expression:
        target = engine.evaluate(y, context.data, num_nodes=context.num_nodes)
        return engine.fit(f, context.data, target, num_nodes=context.num_nodes).expression

    @classmethod
    def evaluate(cls, f: engine.Expression, y: engine.Expression, context: AgentContext) -> MetricDict:
        target = engine.evaluate(y, context.data, num_nodes=context.num_nodes)
        prediction = engine.evaluate(f, context.data, num_nodes=context.num_nodes)
        return utils.regression_metrics(f, target, prediction)

    @classmethod
    def split(cls, context: AgentContext) -> ContextSplits:
        data = context.data
        target = np.asarray(data[context.target])
        if target.ndim < 2:
            raise ValueError(f"target variable {context.target!r} must have at least 2 dimensions for graph evaluation")
        n_samples = target.shape[0]
        n_evaluation = int(round(n_samples * context.args.validation_fraction))
        n_train = n_samples - n_evaluation
        if n_train <= 0 or n_evaluation <= 0:
            return {"train": context.with_data(data), "validation": context.with_data(data)}
        if context.args.split_by == "random":
            train_indices, validation_indices = utils.select_random_indices(
                seed=context.args.split_random_state, n_samples=n_samples, n_train=n_train
            )
        elif context.args.split_by == "ood":
            split_name = context.args.split_ood_variable
            if not split_name or split_name not in data:
                raise ValueError(f"split_ood_variable {split_name!r} not found in context.data")
            train_indices, validation_indices = utils.select_OOD_indices(
                seed=context.args.split_random_state, n_samples=n_samples, n_train=n_train, value=data[split_name]
            )
        else:
            raise ValueError(f"invalid split_by value: {context.args.split_by}")

        relation_names = context.relation_names
        target_axes = context.variable_axes.get(context.target, ())
        sample_axis = target_axes[0] if target_axes else None

        def select(indices: np.ndarray) -> dict[str, np.ndarray]:
            selected = {}
            for variable, value in data.items():
                axes = context.variable_axes.get(variable, ())
                sample_aligned = variable not in relation_names and (
                    variable == sample_axis
                    or bool(axes and axes[0] == sample_axis)
                    or (not axes and value.ndim > 1 and value.shape[0] == n_samples)
                )
                selected[variable] = value[indices] if sample_aligned else value
            return selected

        return {
            "train": context.with_data(select(train_indices)),
            "validation": context.with_data(select(validation_indices)),
        }
