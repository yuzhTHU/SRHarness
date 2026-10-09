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
        return utils.split_aligned_context(context)

    @classmethod
    def fit(cls, f: engine.Expression, y: engine.Expression, context: AgentContext) -> engine.Expression:
        target = engine.evaluate(y, context.data)
        return engine.fit(f, context.data, target).expression

    @classmethod
    def evaluate(cls, f: engine.Expression, y: engine.Expression, context: AgentContext) -> MetricDict:
        target = engine.evaluate(y, context.data)
        prediction = engine.evaluate(f, context.data)
        return utils.regression_metrics(f, target, prediction)

    @classmethod
    def fit_candidate(cls, f: engine.Expression, context: AgentContext) -> engine.Expression:
        if context.target is None:
            raise ValueError("context.target must be configured before fitting a candidate")
        return cls.fit(f, engine.Symbol(context.target), context)

    @classmethod
    def evaluate_candidate(cls, f: engine.Expression, context: AgentContext) -> MetricDict:
        if context.target is None:
            raise ValueError("context.target must be configured before evaluating a candidate")
        return cls.evaluate(f, engine.Symbol(context.target), context)
