"""Reusable evaluator infrastructure kept outside evaluator contracts."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp

import sr_harness_engine as engine

if TYPE_CHECKING:
    from ...core import AgentContext


def _regression_arrays(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        y_pred, y_true = np.broadcast_arrays(np.asarray(y_pred), np.asarray(y_true))
    except ValueError as exc:
        raise ValueError(
            f"Prediction shape {np.shape(y_pred)} and target shape {np.shape(y_true)} "
            "cannot be broadcast to a common shape."
        ) from exc
    return y_true.astype(float, copy=False).reshape(-1), y_pred.astype(float, copy=False).reshape(-1)


def calc_MSE(y_true: Any, y_pred: Any) -> float:
    """Calculate mean squared error.

    Args:
        y_true: Target values.
        y_pred: Predicted values broadcastable to ``y_true``.

    Returns:
        Mean squared error over flattened broadcast arrays.
    """
    y_true, y_pred = _regression_arrays(y_true, y_pred)
    return float(np.mean((y_pred - y_true) ** 2))


def calc_RMSE(y_true: Any, y_pred: Any) -> float:
    """Calculate root mean squared error.

    Args:
        y_true: Target values.
        y_pred: Predicted values broadcastable to ``y_true``.

    Returns:
        Root mean squared error.
    """
    return float(np.sqrt(calc_MSE(y_true, y_pred)))


def calc_MAE(y_true: Any, y_pred: Any) -> float:
    """Calculate mean absolute error.

    Args:
        y_true: Target values.
        y_pred: Predicted values broadcastable to ``y_true``.

    Returns:
        Mean absolute error.
    """
    y_true, y_pred = _regression_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_pred - y_true)))


def calc_MAPE(y_true: Any, y_pred: Any) -> float:
    """Calculate mean absolute percentage error over nonzero targets.

    Args:
        y_true: Target values.
        y_pred: Predicted values broadcastable to ``y_true``.

    Returns:
        Mean absolute percentage error, or NaN when every target is zero.
    """
    y_true, y_pred = _regression_arrays(y_true, y_pred)
    nonzero = y_true != 0
    if not np.any(nonzero):
        return float("nan")
    return float(np.mean(np.abs((y_pred[nonzero] - y_true[nonzero]) / y_true[nonzero])))


def calc_R2(y_true: Any, y_pred: Any) -> float:
    """Calculate the coefficient of determination.

    Args:
        y_true: Target values.
        y_pred: Predicted values broadcastable to ``y_true``.

    Returns:
        The R-squared score.
    """
    y_true, y_pred = _regression_arrays(y_true, y_pred)
    residual = float(np.sum((y_pred - y_true) ** 2))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1 - residual / total) if total else (1.0 if residual == 0 else float("-inf"))


def calc_AIC(y_true: Any, y_pred: Any, num_parameters: int) -> float:
    """Calculate the Gaussian Akaike information criterion.

    Args:
        y_true: Target values.
        y_pred: Predicted values broadcastable to ``y_true``.
        num_parameters: Number of fitted parameters.

    Returns:
        AIC value, negative infinity for an exact fit, or NaN for invalid loss.
    """
    y_true, y_pred = _regression_arrays(y_true, y_pred)
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    if np.isfinite(ss_res) and ss_res > 0:
        n = int(y_true.size)
        likelihood = -n / 2 * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
        return float(2 * num_parameters - 2 * likelihood)
    return float("-inf") if ss_res == 0 else float("nan")


def calc_BIC(y_true: Any, y_pred: Any, num_parameters: int) -> float:
    """Calculate the Gaussian Bayesian information criterion.

    Args:
        y_true: Target values.
        y_pred: Predicted values broadcastable to ``y_true``.
        num_parameters: Number of fitted parameters.

    Returns:
        BIC value, negative infinity for an exact fit, or NaN for invalid loss.
    """
    y_true, y_pred = _regression_arrays(y_true, y_pred)
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    if np.isfinite(ss_res) and ss_res > 0:
        n = int(y_true.size)
        likelihood = -n / 2 * (np.log(2 * np.pi) + np.log(ss_res / n) + 1)
        return float(num_parameters * np.log(n) - 2 * likelihood)
    return float("-inf") if ss_res == 0 else float("nan")


def calc_PearsonR(y_true: Any, y_pred: Any) -> float:
    """Calculate Pearson's correlation coefficient over finite pairs.

    Args:
        y_true: Target values.
        y_pred: Predicted values broadcastable to ``y_true``.

    Returns:
        Pearson correlation, or NaN when fewer than two finite pairs exist.
    """
    y_true, y_pred = _regression_arrays(y_true, y_pred)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(np.corrcoef(y_true[finite], y_pred[finite])[0, 1])


def calc_SpearmanR(y_true: Any, y_pred: Any) -> float:
    """Calculate Spearman's rank correlation over finite pairs.

    Args:
        y_true: Target values.
        y_pred: Predicted values broadcastable to ``y_true``.

    Returns:
        Spearman correlation, or NaN when fewer than two finite pairs exist.
    """
    y_true, y_pred = _regression_arrays(y_true, y_pred)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.spearmanr(y_true[finite], y_pred[finite]).statistic)


def calc_complexity(expression: engine.Expression) -> int:
    """Count nodes in a symbolic expression.

    Args:
        expression: Symbolic expression to inspect.

    Returns:
        Expression node count.
    """
    return len(expression)


def regression_metrics(expression: engine.Expression, y_true: Any, y_pred: Any) -> dict[str, float | int]:
    """Calculate the standard regression metric bundle.

    Args:
        expression: Fitted expression used for complexity and parameter counts.
        y_true: Target values.
        y_pred: Predicted values.

    Returns:
        Numeric prediction, information-criterion, correlation, and complexity metrics.
    """
    parameters = engine.count_parameters(expression)
    return {
        "mse": calc_MSE(y_true, y_pred), "rmse": calc_RMSE(y_true, y_pred),
        "mae": calc_MAE(y_true, y_pred), "mape": calc_MAPE(y_true, y_pred),
        "r2": calc_R2(y_true, y_pred), "aic": calc_AIC(y_true, y_pred, parameters),
        "bic": calc_BIC(y_true, y_pred, parameters),
        "pearson_r": calc_PearsonR(y_true, y_pred),
        "spearman_r": calc_SpearmanR(y_true, y_pred),
        "complexity": calc_complexity(expression),
    }


def select_random_indices(*, seed: int, n_samples: int, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    """Select reproducible random training and validation indices.

    Args:
        seed: Random seed.
        n_samples: Total sample count.
        n_train: Number of training samples.

    Returns:
        Training and validation index arrays.
    """
    indices = np.random.RandomState(seed).permutation(n_samples)
    return indices[:n_train], indices[n_train:]


def select_OOD_indices(*, seed: int, n_samples: int, n_train: int, value: Any) -> tuple[np.ndarray, np.ndarray]:
    """Select an ordered out-of-distribution split.

    Args:
        seed: Seed used to break ties reproducibly.
        n_samples: Total sample count.
        n_train: Number of low-ranked training samples.
        value: One-dimensional ordering variable.

    Returns:
        Training and validation index arrays.
    """
    value = np.asarray(value)
    if value.ndim != 1 or len(value) != n_samples:
        raise ValueError("split_ood_variable must have shape (N,)")
    shuffled = np.random.RandomState(seed).permutation(n_samples)
    ordered = shuffled[np.argsort(value[shuffled], kind="stable")]
    return ordered[:n_train], ordered[n_train:]


def split_indices(context: AgentContext, *, chronological: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
    """Select split indices according to evaluator arguments.

    Args:
        context: Unsplit agent context.
        chronological: Whether to preserve input order rather than use configured
            random or OOD splitting.

    Returns:
        Training and validation indices, or ``None`` when splitting is disabled.
    """
    n_samples = len(context.data[context.target])
    n_validation = int(round(n_samples * context.args.validation_fraction))
    n_train = n_samples - n_validation
    if n_train <= 0 or n_validation <= 0:
        return None
    if chronological:
        indices = np.arange(n_samples)
        return indices[:n_train], indices[n_train:]
    if context.args.split_by == "random":
        return select_random_indices(seed=context.args.split_random_state, n_samples=n_samples, n_train=n_train)
    if context.args.split_by == "ood":
        name = context.args.split_ood_variable
        if not name:
            raise ValueError("split_ood_variable is required when split_by='ood'")
        if name not in context.data:
            raise ValueError(f"split_ood_variable {name!r} not found in context.data")
        return select_OOD_indices(seed=context.args.split_random_state, n_samples=n_samples, n_train=n_train, value=context.data[name])
    raise ValueError(f"invalid split_by value: {context.args.split_by}")


def split_aligned_context(context: AgentContext, *, chronological: bool = False) -> dict[str, AgentContext]:
    """Split every aligned context array along its leading dimension.

    Args:
        context: Unsplit agent context.
        chronological: Whether to preserve input order.

    Returns:
        Training and validation context views.
    """
    indices = split_indices(context, chronological=chronological)
    if indices is None:
        return {"train": context.with_data(context.data), "validation": context.with_data(context.data)}
    train_indices, validation_indices = indices
    return {
        "train": context.with_data({name: value[train_indices] for name, value in context.data.items()}),
        "validation": context.with_data({name: value[validation_indices] for name, value in context.data.items()}),
    }


def integrate_ODE(f: engine.Expression, context: AgentContext, *, time: str = "t", state: str = "x") -> np.ndarray:
    """Integrate a one-dimensional ODE over observed sample times.

    Args:
        f: Fitted derivative expression.
        context: Context containing time and observed state arrays.
        time: Time-variable name.
        state: State-variable name.

    Returns:
        Integrated state values at the observed times.
    """
    t = np.asarray(context.data[time], dtype=float)
    observed = np.asarray(context.data[state], dtype=float)
    if t.ndim != 1 or observed.ndim != 1 or t.shape != observed.shape:
        raise ValueError("time and state must be equally shaped one-dimensional arrays")
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("time must contain at least two strictly increasing values")

    def rhs(current_time: float, current_state: np.ndarray) -> np.ndarray:
        value = engine.evaluate(f, {time: np.asarray([current_time]), state: np.asarray([current_state[0]])})
        return np.asarray([np.asarray(value, dtype=float).reshape(-1)[0]])

    solution = solve_ivp(rhs, (float(t[0]), float(t[-1])), [float(observed[0])], t_eval=t, rtol=1e-8, atol=1e-10)
    if not solution.success or solution.y.shape != (1, len(t)):
        raise RuntimeError(f"ODE integration failed: {solution.message}")
    return solution.y[0]


def calc_trajectory_rollout_RMSE(f: engine.Expression, context: AgentContext, *, time: str = "t", state: str = "x") -> float:
    """Calculate trajectory RMSE after integrating a derivative expression.

    Args:
        f: Fitted derivative expression.
        context: Context containing time and observed state arrays.
        time: Time-variable name.
        state: State-variable name.

    Returns:
        RMSE between observed and integrated state trajectories.
    """
    return calc_RMSE(context.data[state], integrate_ODE(f, context, time=time, state=state))
