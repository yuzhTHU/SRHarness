"""Deterministic cost-aware routing between base and strong LLM backends."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRoute:
    """Selected provider/model route and its rationale."""
    tier: str
    provider: str
    model: str
    score: int
    reason: str


class ModelRouter:
    """Choose a cheap base model or an optional stronger model per request."""

    COMPLEX_PATTERNS = {
        r"\b(?:pde|ode|differential equation|network dynamics|graph dynamics)\b": 3,
        r"\b(?:implicit|piecewise|singular|stiff|chaotic|high[- ]dimensional)\b": 2,
        r"\b(?:noise|noisy|numerical stability|floating[- ]point|cancellation)\b": 1,
        r"(?:微分方程|网络动力学|图动力学|隐式|分段|奇异|混沌|高维)": 3,
        r"(?:噪声|数值稳定|浮点|消减误差)": 1,
    }

    def __init__(
        self,
        *,
        enabled: bool,
        base_provider: str,
        base_model: str,
        strong_provider: str | None = None,
        strong_model: str | None = None,
    ):
        for name, value in (("base_provider", base_provider), ("base_model", base_model)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name, value in (("strong_provider", strong_provider), ("strong_model", strong_model)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be None or a non-empty string")
        self.enabled = enabled
        self.base_provider = base_provider
        self.base_model = base_model
        self.strong_provider = base_provider if strong_provider is None else strong_provider
        self.strong_model = strong_model

    @property
    def has_strong_backend(self) -> bool:
        """Run the ``has strong backend`` operation.

        Returns:
            bool: The operation result.
        """
        return bool(
            self.strong_model
            and (self.strong_provider, self.strong_model)
            != (self.base_provider, self.base_model)
        )

    def assess(self, task: str, feature_count: int) -> tuple[int, list[str]]:
        """Run the ``assess`` operation.

        Args:
            task: The task value.
            feature_count: The feature count value.

        Returns:
            tuple[int, list[str]]: The operation result.
        """
        text = task.lower()
        score = max(0, feature_count - 3)
        reasons = [f"{feature_count} features"] if feature_count > 3 else []
        for pattern, weight in self.COMPLEX_PATTERNS.items():
            if re.search(pattern, text):
                score += weight
                reasons.append(pattern)
        return score, reasons

    def route(
        self,
        *,
        task_score: int,
        task_reasons: list[str],
        refinement_step: int,
    ) -> ModelRoute:
        """Run the ``route`` operation.

        Args:
            task_score: The task score value.
            task_reasons: The task reasons value.
            refinement_step: The refinement step value.

        Returns:
            ModelRoute: The operation result.
        """
        if not self.enabled:
            return self._base(task_score, "automatic routing disabled")
        if not self.has_strong_backend:
            return self._base(task_score, "no distinct strong backend configured")
        if task_score >= 3:
            reason = "complex task signals: " + ", ".join(task_reasons or [str(task_score)])
            return self._strong(task_score, reason)
        if refinement_step >= 3:
            return self._strong(task_score, "base backend did not converge within two rounds")
        return self._base(task_score, "simple task or early exploration")

    def _base(self, score: int, reason: str) -> ModelRoute:
        return ModelRoute("base", self.base_provider, self.base_model, score, reason)

    def _strong(self, score: int, reason: str) -> ModelRoute:
        return ModelRoute(
            "strong",
            self.strong_provider,
            self.strong_model or self.base_model,
            score,
            reason,
        )
