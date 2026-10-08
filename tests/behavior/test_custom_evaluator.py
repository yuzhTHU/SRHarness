"""用户自定义评估器的最小示例。"""

import argparse
import numpy as np
import pytest

import sr_harness
from sr_harness import AgentContext, DefaultEvaluator, GraphEvaluator, load_custom_evaluator
from sr_harness.evaluator import utils
from sr_harness.tools.evaluate_formula import EvaluateTool
import sr_harness_engine as engine


def test_evaluator_utils_exposes_common_metric_helpers():
    assert utils.calc_MSE([1.0, 2.0], [1.0, 3.0]) == pytest.approx(0.5)
    assert utils.calc_RMSE([1.0, 2.0], [1.0, 3.0]) == pytest.approx(np.sqrt(0.5))
    assert utils.calc_MAE([1.0, 2.0], [1.0, 3.0]) == pytest.approx(0.5)
    assert utils.calc_R2([1.0, 2.0], [1.0, 3.0]) == pytest.approx(-1.0)


def test_public_evaluator_classes_are_only_concrete_extension_points():
    assert not hasattr(sr_harness, "Evaluator")
    assert not hasattr(sr_harness, "TemplateCustomEvaluator")
    assert issubclass(GraphEvaluator, DefaultEvaluator)


def test_default_evaluator_contract_has_only_five_public_methods():
    assert {
        name for name, value in DefaultEvaluator.__dict__.items()
        if isinstance(value, classmethod) and not name.startswith("_")
    } == {"split", "fit", "evaluate", "fit_candidate", "evaluate_candidate"}


def test_load_custom_evaluator_supports_relative_imports_and_tracks_source(tmp_path):
    source = '''from . import utils
from .default_evaluator import DefaultEvaluator

class ProjectEvaluator(DefaultEvaluator):
    pass
'''
    evaluator = load_custom_evaluator(source=source)
    assert type(evaluator).__name__ == "ProjectEvaluator"
    assert type(evaluator).CUSTOM_EVALUATOR == {"source": source, "file": None}

    path = tmp_path / "project_evaluator.py"
    path.write_text(source)
    loaded = load_custom_evaluator(file=path)
    assert type(loaded).CUSTOM_EVALUATOR == {"source": source, "file": path.resolve()}


def test_load_custom_evaluator_requires_exactly_one_subclass():
    with pytest.raises(ValueError, match="exactly one"):
        load_custom_evaluator(source='''from .default_evaluator import DefaultEvaluator
class First(DefaultEvaluator): pass
class Second(DefaultEvaluator): pass
''')


def test_agent_context_keeps_runtime_arguments_separate_from_data():
    args = argparse.Namespace(validation_fraction=0)
    context = AgentContext(args=args, data={"x": np.arange(3.0)}, target="x")

    assert context.args is args
    assert context.data["x"] is not context.args
    assert not hasattr(context.args, "data")


class MeanSquaredErrorEvaluator(DefaultEvaluator):
    """一个只依赖表达式、上下文和目标数组的自定义评估器。"""

    @classmethod
    def fit(cls, f, y, context):
        target = y.evaluate(context.data, num_nodes=context.num_nodes)
        return f.fit(context.data, target, num_nodes=context.num_nodes).expression

    @classmethod
    def evaluate(cls, f, y, context):
        target = y.evaluate(context.data, num_nodes=context.num_nodes)
        prediction = f.evaluate(context.data, num_nodes=context.num_nodes)
        error = np.asarray(prediction) - np.asarray(target)
        return {"mse": float(np.mean(error**2))}


def test_an_evaluator_can_define_its_own_fitting_and_scoring_protocol():
    evaluator = MeanSquaredErrorEvaluator()
    formula = engine.parse("param('slope') * x + param('bias')")
    data = {"x": np.linspace(-2.0, 2.0, 21)}
    target = 3.0 * data["x"] - 1.0
    context = AgentContext(data=data | {"y": target}, target="y", evaluator=evaluator)

    fitted = evaluator.fit_candidate(formula, context)
    metrics = evaluator.evaluate_candidate(fitted, context)

    assert engine.parameter_values(fitted) == {
        "slope": pytest.approx(3.0), "bias": pytest.approx(-1.0),
    }
    assert metrics["mse"] < 1e-10


def test_default_evaluator_returns_complete_split_metrics():
    x = np.linspace(-2.0, 2.0, 21)
    target = 3.0 * x - 1.0
    evaluator = DefaultEvaluator()
    context = AgentContext(data={"x": x, "y": target}, target="y", evaluator=evaluator)

    fitted = evaluator.fit_candidate(engine.parse("param('slope') * x + param('bias')"), context)
    metrics = evaluator.evaluate_candidate(fitted, context)

    assert {"mse", "rmse", "mae", "r2", "aic", "bic"} <= metrics.keys()
    assert metrics["complexity"] == len(fitted)
    assert metrics["mse"] < 1e-10


def test_candidate_equations_use_candidate_only_hooks_and_evaluator_owns_complexity():
    class RolloutEvaluator(DefaultEvaluator):
        candidate_calls = 0

        @classmethod
        def evaluate_candidate(cls, f, context):
            cls.candidate_calls += 1
            return super().evaluate_candidate(f, context) | {"rollout_rmse": 0.125}

    x = np.linspace(0.0, 1.0, 8)
    context = AgentContext(
        data={"x": x, "dx_dt": 2 * x}, target="dx_dt", evaluator=RolloutEvaluator()
    )
    tool = EvaluateTool(context=context)

    candidate = tool.execute(f="2*x", show_diagnostics=False)
    non_candidate = tool.execute(f="2*x", y="dx_dt - x", show_diagnostics=False)

    assert RolloutEvaluator.candidate_calls == 2
    assert candidate["data_split_results"]["train"]["metrics"]["rollout_rmse"] == 0.125
    assert candidate["data_split_results"]["train"]["metrics"]["complexity"] == len(engine.parse("2*x"))
    assert "rollout_rmse" not in non_candidate["data_split_results"]["train"]["metrics"]


def test_evaluator_metrics_must_be_flat_numbers():
    class InvalidEvaluator(DefaultEvaluator):
        @classmethod
        def evaluate_candidate(cls, f, context):
            return {"mse": 0.0, "details": {"bad": True}}

    context = AgentContext(
        data={"x": np.arange(3.0), "y": np.arange(3.0)},
        target="y",
        evaluator=InvalidEvaluator(),
    )
    result = EvaluateTool(context=context)(f="x", show_diagnostics=False)
    assert not result.ok
    assert "must be a float or int" in result.result["error"]


def test_default_evaluator_provides_scalar_ode_rollout_infrastructure():
    t = np.linspace(0.0, 2.0, 101)
    x = np.exp(-0.4 * t)
    context = AgentContext(
        args=argparse.Namespace(validation_fraction=0.2),
        data={"t": t, "x": x, "dx_dt": -0.4 * x},
        target="dx_dt",
        variable_axes={"x": ("t",), "dx_dt": ("t",)},
    )

    splits = utils.split_aligned_context(context, chronological=True)
    rollout_rmse = utils.calc_trajectory_rollout_RMSE(
        engine.parse("-0.4*x"), splits["validation"]
    )

    assert np.all(np.diff(splits["train"].data["t"]) > 0)
    assert np.all(np.diff(splits["validation"].data["t"]) > 0)
    assert rollout_rmse < 1e-8


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
    assert context.train_split.data["x"].shape == (3, 3)
    assert context.validation_split.data["w"].shape == (2, 2)
    np.testing.assert_array_equal(context.train_split.data["A"], edges)
    np.testing.assert_array_equal(context.validation_split.data["node"], np.arange(3))


def test_formula_tool_rejects_an_evaluator_that_leaves_parameters_unbound():
    class BrokenEvaluator(DefaultEvaluator):
        @classmethod
        def fit(cls, f, y, context):
            return f

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

    @classmethod
    def fit(cls, f, y, context):
        target = y.evaluate(context.data, num_nodes=context.num_nodes)
        return f.fit(context.data, target, num_nodes=context.num_nodes).expression

    @classmethod
    def evaluate(cls, f, y, context):
        target = y.evaluate(context.data, num_nodes=context.num_nodes)
        prediction = f.evaluate(context.data, num_nodes=context.num_nodes)
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
