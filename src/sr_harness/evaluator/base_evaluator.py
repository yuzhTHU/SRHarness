"""Abstract evaluator contract and reusable regression primitives."""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

import sr_harness_engine as engine

if TYPE_CHECKING:
    from ..core import AgentContext


def _regression_arrays(target: Any, prediction: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        prediction, target = np.broadcast_arrays(np.asarray(prediction), np.asarray(target))
    except ValueError as exc:
        raise ValueError(
            f"Prediction shape {np.shape(prediction)} and target shape {np.shape(target)} "
            "cannot be broadcast to a common shape."
        ) from exc
    return (
        target.astype(float, copy=False).reshape(-1),
        prediction.astype(float, copy=False).reshape(-1),
    )


class BaseEvaluator(ABC):
    """Abstract fitting, evaluation, and splitting contract."""

    @staticmethod
    @abstractmethod
    def fit(expression: engine.Expression, context: AgentContext, target: np.ndarray) -> engine.Expression:
        """Fit every expression parameter against an explicit target array."""

    @staticmethod
    @abstractmethod
    def evaluate(expression: engine.Expression, context: AgentContext, target: np.ndarray) -> dict[str, Any]:
        """Evaluate one fitted expression against an explicit target array."""

    @staticmethod
    @abstractmethod
    def split_data(context: AgentContext) -> dict[str, dict[str, np.ndarray]]:
        """Split an unsplit context into train and evaluation mappings."""

    @staticmethod
    def select_random_indices(*, seed: int, n_samples: int, n_train: int) -> tuple[np.ndarray, np.ndarray]:
        indices = np.random.RandomState(seed).permutation(n_samples)
        return indices[:n_train], indices[n_train:]

    @staticmethod
    def select_ood_indices(*, seed: int, n_samples: int, n_train: int, value: Any) -> tuple[np.ndarray, np.ndarray]:
        value = np.asarray(value)
        if value.ndim != 1 or len(value) != n_samples:
            raise ValueError("split_ood_variable must have shape (N,)")
        shuffled = np.random.RandomState(seed).permutation(n_samples)
        indices = shuffled[np.argsort(value[shuffled], kind="stable")]
        return indices[:n_train], indices[n_train:]

    @staticmethod
    def calc_mse(y_true: Any, y_pred: Any) -> float:
        y_true, y_pred = _regression_arrays(y_true, y_pred)
        return float(np.mean((y_pred - y_true) ** 2))

    @staticmethod
    def calc_rmse(y_true: Any, y_pred: Any) -> float:
        return float(np.sqrt(BaseEvaluator.calc_mse(y_true, y_pred)))

    @staticmethod
    def calc_mae(y_true: Any, y_pred: Any) -> float:
        y_true, y_pred = _regression_arrays(y_true, y_pred)
        return float(np.mean(np.abs(y_pred - y_true)))

    @staticmethod
    def calc_mape(y_true: Any, y_pred: Any) -> float:
        y_true, y_pred = _regression_arrays(y_true, y_pred)
        residuals = y_pred - y_true
        if np.any(non_zero := ~np.isclose(y_true, 0.0)):
            return float(np.mean(np.abs(residuals[non_zero] / y_true[non_zero])))
        return 0.0 if np.allclose(y_pred, y_true) else float("inf")

    @staticmethod
    def calc_r2(y_true: Any, y_pred: Any) -> float:
        y_true, y_pred = _regression_arrays(y_true, y_pred)
        ss_res = float(np.sum((y_pred - y_true) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    @staticmethod
    def calc_aic(y_true: Any, y_pred: Any, num_parameters: int) -> float:
        y_true, y_pred = _regression_arrays(y_true, y_pred)
        ss_res = float(np.sum((y_pred - y_true) ** 2))
        if np.isfinite(ss_res) and ss_res > 0:
            n_samples = int(y_true.size)
            log_likelihood = -n_samples / 2 * (
                np.log(2 * np.pi) + np.log(ss_res / n_samples) + 1
            )
            return float(2 * num_parameters - 2 * log_likelihood)
        return float("-inf") if ss_res == 0 else float("nan")

    @staticmethod
    def calc_bic(y_true: Any, y_pred: Any, num_parameters: int) -> float:
        y_true, y_pred = _regression_arrays(y_true, y_pred)
        ss_res = float(np.sum((y_pred - y_true) ** 2))
        if np.isfinite(ss_res) and ss_res > 0:
            n_samples = int(y_true.size)
            log_likelihood = -n_samples / 2 * (
                np.log(2 * np.pi) + np.log(ss_res / n_samples) + 1
            )
            return float(num_parameters * np.log(n_samples) - 2 * log_likelihood)
        return float("-inf") if ss_res == 0 else float("nan")

    @staticmethod
    def calc_pearson_r(y_true: Any, y_pred: Any) -> float:
        y_true, y_pred = _regression_arrays(y_true, y_pred)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.count_nonzero(finite) < 2:
            return float("nan")
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.corrcoef(y_true[finite], y_pred[finite])[0, 1])

    @staticmethod
    def calc_spearman_r(y_true: Any, y_pred: Any) -> float:
        y_true, y_pred = _regression_arrays(y_true, y_pred)
        finite = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.count_nonzero(finite) < 2:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(stats.spearmanr(y_true[finite], y_pred[finite]).statistic)

    @staticmethod
    def calc_complexity(expression: engine.Expression) -> int:
        return len(expression)


def regression_metrics(expression: engine.Expression, target: Any, prediction: Any) -> dict[str, Any]:
    """Return the standard metrics used by formula-producing tools."""
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
