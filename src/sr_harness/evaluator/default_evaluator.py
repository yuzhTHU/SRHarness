"""Default evaluation protocol for ordinary aligned data."""
from __future__ import annotations

from typing import Literal, TypeAlias

import sr_harness_engine as engine

from ..core import AgentContext
from . import utils

MetricValue: TypeAlias = float | int
MetricDict: TypeAlias = dict[str, MetricValue]
ContextSplits: TypeAlias = dict[Literal["train", "validation"], AgentContext]


class DefaultEvaluator:
    """Default implementation and extension point for formula evaluation."""

    @classmethod
    def split(cls, context: AgentContext) -> ContextSplits:
        """Split a context into training and validation views.

        Args:
            context: Complete unsplit agent context.

        Returns:
            Training and validation contexts keyed by split name.
        """
        return utils.split_aligned_context(context)

    @classmethod
    def fit(cls, f: engine.Expression, y: engine.Expression, context: AgentContext) -> engine.Expression:
        """Fit parameters in a general expression equality.

        Args:
            f: Expression whose parameters will be fitted.
            y: Expression providing target values.
            context: Training context containing all referenced variables.

        Returns:
            A copy of ``f`` with fitted parameter values.
        """
        target = engine.evaluate(y, context.data)
        return engine.fit(f, context.data, target).expression

    @classmethod
    def evaluate(cls, f: engine.Expression, y: engine.Expression, context: AgentContext) -> MetricDict:
        """Evaluate a general expression equality.

        Args:
            f: Fitted expression to evaluate.
            y: Expression providing target values.
            context: Context containing all referenced variables.

        Returns:
            Numeric regression and complexity metrics.
        """
        target = engine.evaluate(y, context.data)
        prediction = engine.evaluate(f, context.data)
        return utils.regression_metrics(f, target, prediction)

    @classmethod
    def fit_candidate(cls, f: engine.Expression, context: AgentContext) -> engine.Expression:
        """Fit an eligible candidate against the configured target variable.

        Args:
            f: Candidate expression whose parameters will be fitted.
            context: Training context with a configured target.

        Returns:
            A copy of ``f`` with fitted parameter values.
        """
        if context.target is None:
            raise ValueError("context.target must be configured before fitting a candidate")
        return cls.fit(f, engine.Symbol(context.target), context)

    @classmethod
    def evaluate_candidate(cls, f: engine.Expression, context: AgentContext) -> MetricDict:
        """Evaluate an eligible candidate against the configured target.

        Args:
            f: Fitted candidate expression.
            context: Context with a configured target.

        Returns:
            Numeric regression and complexity metrics.
        """
        if context.target is None:
            raise ValueError("context.target must be configured before evaluating a candidate")
        return cls.evaluate(f, engine.Symbol(context.target), context)
