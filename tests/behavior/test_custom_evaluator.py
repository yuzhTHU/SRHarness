"""用户自定义评估器的最小示例。"""

from typing import Any

import numpy as np
import pytest

from sr_harness import AgentContext, Evaluator
import sr_harness_engine as engine


class MeanSquaredErrorEvaluator(Evaluator):
    """一个只依赖公式字符串、字典和数组的自定义评估器。"""

    def fit(
        self,
        formula: str,
        data: dict[str, Any],
        target: Any,
    ) -> dict[str, Any]:
        result = engine.parse(formula).fit(data, target)
        return result.parameters

    def evaluate(
        self,
        formula: str,
        data: dict[str, Any],
        target: Any,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        prediction = engine.parse(formula).evaluate(data, parameters=parameters)
        error = np.asarray(prediction) - np.asarray(target)
        return {"mse": float(np.mean(error**2))}


def test_an_evaluator_can_define_its_own_fitting_and_scoring_protocol():
    evaluator = MeanSquaredErrorEvaluator()
    formula = "param('slope') * x + param('bias')"
    data = {"x": np.linspace(-2.0, 2.0, 21)}
    target = 3.0 * data["x"] - 1.0

    parameters = evaluator.fit(formula, data, target)
    metrics = evaluator.evaluate(formula, data, target, parameters)

    assert parameters == {"slope": pytest.approx(3.0), "bias": pytest.approx(-1.0)}
    assert metrics["mse"] < 1e-10


def test_an_evaluator_can_be_shared_with_agents_and_tools_through_context():
    evaluator = MeanSquaredErrorEvaluator()
    context = AgentContext(evaluator=evaluator)

    assert context.evaluator is evaluator
    assert context["evaluator"] is evaluator
