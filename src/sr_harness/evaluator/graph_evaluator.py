"""Evaluator for broadcast-compatible graph and hypergraph arrays."""
from __future__ import annotations
from typing import Any
import numpy as np
import sr_harness_engine as engine
from ..core import AgentContext
from .default_evaluator import DefaultEvaluator


class GraphEvaluator(DefaultEvaluator):
    """
    Evaluator for broadcast-compatible graph and hypergraph arrays.
    
    Each value of context.data should has shape:
    - (E, 2) for graph topology (in edge list format, [target, source])
    - (H, 3) for hypergraph topology (in hyperedge list format, [target, source1, source2])
    - (..., 1) for graph-level variables
    - (..., V) for node-level variables, where V is the number of nodes
    - (..., E) for edge-level variables, where E is the number of edges
    - (..., H) for hyperedge-level variables, where H is the number of hyperedges
    The first dimensions (``...``) must be broadcast-compatible across all values of context.data.
    """

    @staticmethod
    def fit(expression: engine.Expression, context: AgentContext, target: np.ndarray) -> engine.Expression:
        return engine.fit(expression, context.data, target, num_nodes=context.num_nodes).expression

    @staticmethod
    def evaluate(expression: engine.Expression, context: AgentContext, target: np.ndarray) -> dict[str, Any]:
        prediction = engine.evaluate(expression, context.data, num_nodes=context.num_nodes)
        num_parameters = engine.count_parameters(expression)
        return {
            "mse": DefaultEvaluator.calc_mse(target, prediction),
            "rmse": DefaultEvaluator.calc_rmse(target, prediction),
            "mae": DefaultEvaluator.calc_mae(target, prediction),
            "mape": DefaultEvaluator.calc_mape(target, prediction),
            "r2": DefaultEvaluator.calc_r2(target, prediction),
            "aic": DefaultEvaluator.calc_aic(target, prediction, num_parameters),
            "bic": DefaultEvaluator.calc_bic(target, prediction, num_parameters),
            "pearson_r": DefaultEvaluator.calc_pearson_r(target, prediction),
            "spearman_r": DefaultEvaluator.calc_spearman_r(target, prediction),
            "complexity": DefaultEvaluator.calc_complexity(expression),
        }

    @staticmethod
    def split_data(context: AgentContext) -> dict[str, dict[str, np.ndarray]]:
        data = context.data
        target = np.asarray(data[context.target])
        if target.ndim < 2:
            raise ValueError(f"target variable {context.target!r} must have at least 2 dimensions for graph evaluation")
        n_samples = target.shape[0]
        n_evaluation = int(round(n_samples * context.args.validation_fraction))
        n_train = n_samples - n_evaluation
        if n_train <= 0 or n_evaluation <= 0:
            return {"train": data, "evaluation": data}
        if context.args.split_by == "random":
            train_indices, evaluation_indices = DefaultEvaluator.select_random_indices(
                seed=context.args.split_random_state,
                n_samples=n_samples,
                n_train=n_train,
            )
        elif context.args.split_by == "ood":
            split_name = context.args.split_ood_variable
            assert split_name in data, f"split_ood_variable {split_name!r} not found in context.data"
            train_indices, evaluation_indices = DefaultEvaluator.select_ood_indices(
                seed=context.args.split_random_state,
                n_samples=n_samples,
                n_train=n_train,
                value=data[split_name],
            )
        else:
            raise ValueError(f"invalid split_by value: {context.args.split_by}")

        splited_data = {"train": {}, "evaluation": {}}
        for variable, value in data.items():
            splited_data["train"][variable] = value[train_indices]
            splited_data["evaluation"][variable] = value[evaluation_indices]
        return splited_data
