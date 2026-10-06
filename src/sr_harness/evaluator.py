"""Public protocol for evaluating candidate formula strings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Evaluator(ABC):
    """Fit and score one candidate formula under a user-defined protocol.

    Implementations may use regression, numerical integration, trajectory
    matching, network simulation, or another scientific protocol. The
    interface deliberately uses formula strings and ordinary dictionaries so
    custom evaluators do not depend on SRHarness agent internals.
    """

    @abstractmethod
    def fit(
        self,
        formula: str,
        data: dict[str, Any],
        target: Any,
    ) -> dict[str, Any]:
        """Fit formula parameters and return their values or other fit state.

        Args:
            formula: Symbolic formula string.
            data: Data arrays keyed by variable name.
            target: Target name or target values.

        Returns:
            Fitted parameters and any reusable evaluator state.
        """

    @abstractmethod
    def evaluate(
        self,
        formula: str,
        data: dict[str, Any],
        target: Any,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return metrics and diagnostics for a formula on the supplied data.

        Args:
            formula: Symbolic formula string.
            data: Data arrays keyed by variable name.
            target: Target name or target values.
            parameters: Fitted parameter values keyed by parameter name.

        Returns:
            Metrics and diagnostics for candidate ranking and inspection.
        """
