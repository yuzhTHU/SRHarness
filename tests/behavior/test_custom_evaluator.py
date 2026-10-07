"""用户自定义评估器的最小示例。"""

import argparse
import ast
import inspect
import numpy as np
import pytest

import sr_harness
from sr_harness import AgentContext, BaseEvaluator, DefaultEvaluator, GraphEvaluator, TemplateCustomEvaluator
from sr_harness.tools.evaluate_formula import EvaluateTool
import sr_harness_engine as engine


def test_base_evaluator_is_abstract_and_exposes_common_metric_helpers():
    with pytest.raises(TypeError):
        BaseEvaluator()

    assert BaseEvaluator.calc_mse([1.0, 2.0], [1.0, 3.0]) == pytest.approx(0.5)
    assert BaseEvaluator.calc_rmse([1.0, 2.0], [1.0, 3.0]) == pytest.approx(np.sqrt(0.5))
    assert BaseEvaluator.calc_mae([1.0, 2.0], [1.0, 3.0]) == pytest.approx(0.5)
    assert BaseEvaluator.calc_r2([1.0, 2.0], [1.0, 3.0]) == pytest.approx(-1.0)
    assert not hasattr(BaseEvaluator, "calculate_metrics")
    assert not hasattr(BaseEvaluator, "fit_constants")
    assert not hasattr(BaseEvaluator, "predict")


def test_public_evaluator_classes_have_the_new_names_and_minimal_template():
    assert not hasattr(sr_harness, "Evaluator")
    assert issubclass(DefaultEvaluator, BaseEvaluator)
    assert issubclass(GraphEvaluator, DefaultEvaluator)
    assert issubclass(TemplateCustomEvaluator, DefaultEvaluator)
    tree = ast.parse(inspect.getsource(TemplateCustomEvaluator))
    assert isinstance(tree.body[0], ast.ClassDef)
    assert len(tree.body[0].body) == 1
    assert isinstance(tree.body[0].body[0], ast.Pass)


def test_agent_context_keeps_runtime_arguments_separate_from_data():
    args = argparse.Namespace(validation_fraction=0)
    context = AgentContext(args=args, data={"x": np.arange(3.0)}, target="x")

    assert context.args is args
    assert context.data["x"] is not context.args
    assert not hasattr(context.args, "data")


class MeanSquaredErrorEvaluator(DefaultEvaluator):
    """一个只依赖表达式、上下文和目标数组的自定义评估器。"""

    @staticmethod
    def fit(expression, context, target):
        return expression.fit(context.data, target, num_nodes=context.num_nodes).expression

    @staticmethod
    def evaluate(expression, context, target):
        prediction = expression.evaluate(context.data, num_nodes=context.num_nodes)
        error = np.asarray(prediction) - np.asarray(target)
        return {"mse": float(np.mean(error**2))}


def test_an_evaluator_can_define_its_own_fitting_and_scoring_protocol():
    evaluator = MeanSquaredErrorEvaluator()
    formula = engine.parse("param('slope') * x + param('bias')")
    data = {"x": np.linspace(-2.0, 2.0, 21)}
    target = 3.0 * data["x"] - 1.0
    context = AgentContext(data=data | {"y": target}, target="y", evaluator=evaluator)

    fitted = evaluator.fit(formula, context, target)
    metrics = evaluator.evaluate(fitted, context, target)

    assert engine.parameter_values(fitted) == {
        "slope": pytest.approx(3.0), "bias": pytest.approx(-1.0),
    }
    assert metrics["mse"] < 1e-10


def test_default_evaluator_returns_complete_split_metrics():
    x = np.linspace(-2.0, 2.0, 21)
    target = 3.0 * x - 1.0
    evaluator = DefaultEvaluator()
    context = AgentContext(data={"x": x, "y": target}, target="y", evaluator=evaluator)

    fitted = evaluator.fit(engine.parse("param('slope') * x + param('bias')"), context, target)
    metrics = evaluator.evaluate(fitted, context, target)

    assert {"complexity", "mse", "rmse", "mae", "r2", "aic", "bic"} <= metrics.keys()
    assert metrics["complexity"] == len(fitted)
    assert metrics["mse"] < 1e-10


def test_graph_context_selects_graph_evaluator_and_splits_only_sample_axes():
    time = np.arange(5.0)
    edges = np.array([[0, 1], [1, 2]])
    node_values = np.arange(15.0).reshape(5, 3)
    edge_values = np.arange(10.0).reshape(5, 2)
    context = AgentContext(
        data={
            "time": time,
            "node": np.arange(3),
            "edge": np.arange(2),
            "endpoint": np.arange(2),
            "A": edges,
            "x": node_values,
            "w": edge_values,
            "y": node_values + 1,
        },
        target="y",
        variable_axes={
            "A": ("edge", "endpoint"),
            "x": ("time", "node"),
            "w": ("time", "edge"),
            "y": ("time", "node"),
        },
        variable_structures={"w": "A"},
        num_nodes=3,
    )
    context.args.validation_fraction = 0.4

    assert isinstance(context.evaluator, GraphEvaluator)
    assert context.train_data()["x"].shape == (3, 3)
    assert context.evaluation_data()["w"].shape == (2, 2)
    np.testing.assert_array_equal(context.train_data()["A"], edges)
    np.testing.assert_array_equal(context.evaluation_data()["node"], np.arange(3))


def test_formula_tool_rejects_an_evaluator_that_leaves_parameters_unbound():
    class BrokenEvaluator(DefaultEvaluator):
        @staticmethod
        def fit(expression, context, target):
            return expression

    context = AgentContext(
        data={"x": np.arange(5.0), "y": np.arange(5.0)},
        target="y",
        evaluator=BrokenEvaluator(),
    )

    result = EvaluateTool(context=context)(f="param('slope') * x", fit=True)

    assert not result.ok
    assert "unbound parameters: slope" in result.result["error"]


def test_an_evaluator_can_be_shared_with_agents_and_tools_through_context():
    evaluator = MeanSquaredErrorEvaluator()
    context = AgentContext(evaluator=evaluator)

    assert context.evaluator is evaluator
    assert context.evaluator is evaluator


class NetworkDerivativeEvaluator(DefaultEvaluator):
    """Fit indexed network laws using metadata carried by AgentContext."""

    @staticmethod
    def fit(expression, context, target):
        return expression.fit(context.data, target, num_nodes=context.num_nodes).expression

    @staticmethod
    def evaluate(expression, context, target):
        prediction = expression.evaluate(context.data, num_nodes=context.num_nodes)
        error = np.asarray(prediction) - np.asarray(target)
        return {"mse": float(np.mean(error**2))}


def test_formula_tool_uses_a_custom_evaluator_for_indexed_network_laws():
    num_nodes = 3
    data = {
        "A": np.array([[0, 1], [1, 2]], dtype=int),
        "theta": np.array([[0.2, 0.6, -0.4], [0.3, 0.8, -0.1]]),
        "omega": np.array([0.1, -0.2, 0.3]),
    }
    interaction = engine.parse(
        "sum[j](A[i, j], sin(theta[j] - theta[i]))"
    ).evaluate(data, num_nodes=num_nodes)
    data["dtheta_dt"] = data["omega"] + 0.65 * interaction
    data["edge_signal"] = np.zeros((2, 2))
    context = AgentContext(
        data=data,
        target="dtheta_dt",
        evaluator=NetworkDerivativeEvaluator(),
        num_nodes=num_nodes,
        variable_structures={"edge_signal": "A"},
    )

    result = EvaluateTool(context=context)(
        f=(
            "omega[i] + param('coupling', value=0.3) * "
            "sum[j](A[i, j], sin(theta[j] - theta[i]))"
        ),
        fit=True,
        show_diagnostics=False,
    )

    assert result.ok
    assert result.result["is_candidate"]
    assert result.result["fitted_parameters"]["coupling"] == pytest.approx(0.65)
    fitted_formula = engine.parse(result.result["formula"])
    fitted_parameter = next(
        node for node in fitted_formula.iter_preorder()
        if isinstance(node, engine.Parameter)
    )
    assert fitted_parameter.value == pytest.approx(0.65)
    assert result.result["data_split_results"]["train"]["metrics"]["mse"] < 1e-12


def test_formula_tool_passes_context_num_nodes_to_the_builtin_evaluator():
    num_nodes = 3
    data = {
        "A": np.array([[0, 1], [1, 2]], dtype=int),
        "theta": np.array([[0.2, 0.6, -0.4], [0.3, 0.8, -0.1]]),
        "omega": np.array([0.1, -0.2, 0.3]),
    }
    interaction = engine.parse(
        "sum[j](A[i, j], sin(theta[j] - theta[i]))"
    ).evaluate(data, num_nodes=num_nodes)
    data["dtheta_dt"] = data["omega"] + 0.65 * interaction
    data["edge_signal"] = np.zeros((2, 2))
    context = AgentContext(
        data=data,
        target="dtheta_dt",
        num_nodes=num_nodes,
        variable_structures={"edge_signal": "A"},
    )

    result = EvaluateTool(context=context)(
        f=(
            "omega[i] + param('coupling', value=0.3) * "
            "sum[j](A[i, j], sin(theta[j] - theta[i]))"
        ),
        fit=True,
        show_diagnostics=False,
    )

    assert result.ok
    assert result.result["is_candidate"]
    assert result.result["data_split_results"]["train"]["metrics"]["mse"] < 1e-12
