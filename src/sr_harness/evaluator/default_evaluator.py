"""Default evaluator for equally shaped one-dimensional variables."""
from __future__ import annotations
import numpy as np
import sr_harness_engine as engine
from typing import Any
from ..core import AgentContext
from .base_evaluator import BaseEvaluator


class DefaultEvaluator(BaseEvaluator):
    """
    Default evaluator for equally shaped one-dimensional variables.
    """
    @staticmethod
    def fit(expression: engine.Expression, context: AgentContext, target: np.ndarray) -> engine.Expression:
        """Fit expression parameters against a target (that need not be context.target)."""
        return engine.fit(expression, context.data, target).expression

    @staticmethod
    def evaluate(expression: engine.Expression, context: AgentContext, target: np.ndarray) -> dict[str, Any]:
        """Predict and calculate metrics against a target (that need not be context.target)."""
        prediction = engine.evaluate(expression, context.data)
        num_parameters = engine.count_parameters(expression)
        return {
            "mse": BaseEvaluator.calc_mse(target, prediction),
            "rmse": BaseEvaluator.calc_rmse(target, prediction),
            "mae": BaseEvaluator.calc_mae(target, prediction),
            "mape": BaseEvaluator.calc_mape(target, prediction),
            "r2": BaseEvaluator.calc_r2(target, prediction),
            "aic": BaseEvaluator.calc_aic(target, prediction, num_parameters),
            "bic": BaseEvaluator.calc_bic(target, prediction, num_parameters),
            "pearson_r": BaseEvaluator.calc_pearson_r(target, prediction),
            "spearman_r": BaseEvaluator.calc_spearman_r(target, prediction),
            "complexity": BaseEvaluator.calc_complexity(expression),
        }

    @staticmethod
    def split_data(context: AgentContext) -> dict[str, dict[str, np.ndarray]]:
        """Split equally shaped ``(N,)`` context arrays into train and evaluation data."""
        data = context.data
        n_samples = len(data[context.target])
        n_evaluation = int(round(n_samples * context.args.validation_fraction))
        n_train = n_samples - n_evaluation
        if n_train <= 0 or n_evaluation <= 0:
            return {"train": data, "evaluation": data}
        if context.args.split_by == "random":
            train_indices, evaluation_indices = BaseEvaluator.select_random_indices(
                seed=context.args.split_random_state,
                n_samples=n_samples,
                n_train=n_train,
            )
        elif context.args.split_by == "ood":
            split_ood_variable = context.args.split_ood_variable
            assert split_ood_variable in data, f"split_ood_variable {split_ood_variable!r} not found in context.data"
            train_indices, evaluation_indices = BaseEvaluator.select_ood_indices(
                seed=context.args.split_random_state,
                n_samples=n_samples,
                n_train=n_train,
                value=data[split_ood_variable],
            )
        else:
            raise ValueError(f"invalid split_by value: {context.args.split_by}")
        split_data = {"train": {}, "evaluation": {}}
        for variable, value in data.items():
            split_data["train"][variable] = value[train_indices]
            split_data["evaluation"][variable] = value[evaluation_indices]
        return split_data
