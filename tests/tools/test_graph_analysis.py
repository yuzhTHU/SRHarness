"""Graph symbolic-regression diagnostic tool tests."""

from __future__ import annotations

import argparse
import numpy as np

from sr_harness.core import AgentContext
from sr_harness.tools.call_pysr import PySRTool
from sr_harness.tools.call_sindy import SINDyTool
from sr_harness.tools.constant_fit import ConstantFitTool
from sr_harness.tools.graph_analysis import (
    GraphDataAnalysisTool,
    GraphFeatureFitTool,
    GraphResidualAnalysisTool,
    GraphStructureScreenTool,
    GraphTemplateFitTool,
)
from sr_harness.tools.polynomial_fit import PolynomialFitTool
from sr_harness.tools.power_law_fit import PowerLawFitTool
from sr_harness.tools.rational_fit import RationalFitTool
from sr_harness.tools.relationship_analysis import RelationshipAnalysisTool
from sr_harness.tools.statistics_analysis import StatisticsTool


def graph_context() -> AgentContext:
    relation = np.array([[0, 1], [0, 2], [1, 0], [2, 0]], dtype=np.int64)
    x = np.array([
        [0.2, 0.5, 0.9],
        [0.4, 0.8, 0.1],
        [0.7, 0.3, 0.6],
        [0.9, 0.2, 0.4],
    ])
    incoming = np.zeros_like(x)
    for target, source in relation:
        incoming[:, target] += x[:, source]
    y = 1.5 - 2 * x**2 + 3 * incoming
    return AgentContext(
        data={"R": relation, "x": x, "y": y},
        target="y",
        relation_names={"R"},
        num_nodes=3,
    )


def train_metrics(result):
    return result["data_split_results"]["train"]["metrics"]


def transformed_graph_context(transform) -> AgentContext:
    context = graph_context()
    incoming = np.zeros_like(context.data["x"])
    for target, source in context.data["R"]:
        incoming[:, target] += context.data["x"][:, source]
    context.data["y"] = transform(incoming)
    context.invalidate_splits()
    return context


def test_graph_data_analysis_reports_structure_and_ranked_probes():
    result = GraphDataAnalysisTool(context=graph_context()).execute(top_k=5)

    assert result["structure"]["num_nodes"] == 3
    assert result["structure"]["num_edges"] == 4
    assert result["structure"]["reciprocal_edge_ratio"] == 1.0
    assert result["fields"]["x"]["shape"] == [4, 3]
    assert len(result["top_affine_probes"]) == 5
    assert any("sum[j](R[i, j]" in item["formula"] for item in result["top_affine_probes"])
    assert "difference" in result["top_affine_probes_by_family"]


def test_graph_data_analysis_does_not_treat_axis_arrays_as_node_fields():
    context = graph_context()
    context = AgentContext(
        data=context.data | {
            "sample": np.arange(4), "node": np.arange(3),
            "edge": np.arange(4), "endpoint": np.arange(2),
        },
        target="y",
        variable_axes={"R": ("edge", "endpoint"), "x": ("sample", "node"), "y": ("sample", "node")},
        variable_descriptions={
            name: "" for name in ["R", "x", "y", "sample", "node", "edge", "endpoint"]
        },
        relation_names={"R"},
        num_nodes=3,
    )

    result = GraphDataAnalysisTool(context=context).execute(top_k=5)

    assert set(result["fields"]) == {"x"}


def test_graph_diagnostics_use_only_training_split():
    context = graph_context()
    context.args = argparse.Namespace(
        validation_fraction=0.5,
        split_by="random",
        split_random_state=7,
        split_ood_variable=None,
        data_revision=0,
    )
    train_rows = context.train_split.data["x"].shape[0]
    assert train_rows < context.data["x"].shape[0]

    data_result = GraphDataAnalysisTool(context=context).execute(top_k=5)
    residual_result = GraphResidualAnalysisTool(context=context).execute(
        candidate="0",
        probes=["x[i]"],
    )

    assert data_result["analysis_split"] == "train"
    assert data_result["fields"]["x"]["shape"][0] == train_rows
    assert residual_result["analysis_split"] == "train"
    expected_train_rmse = float(np.sqrt(np.mean(context.train_split.data["y"] ** 2)))
    assert residual_result["residual_summary"]["rmse"] == expected_train_rmse


def test_graph_feature_fit_recovers_sparse_indexed_formula():
    result = GraphFeatureFitTool(context=graph_context()).execute(
        features=[
            "x[i]",
            "x[i] ** 2",
            "x[i] ** 3",
            "sum[j](R[i,j], x[j])",
            "sum[j](R[i,j], x[j]-x[i])",
        ],
        max_terms=4,
    )

    assert train_metrics(result)["r2"] > 1 - 1e-12
    assert len(result["selected_features"]) <= 3
    assert "x[i] ** 2" in result["fitted_coefficients"]
    assert "sum[j](R[i, j], x[j])" in result["fitted_coefficients"]
    assert result["selection_trace"][-1]["rmse"] < 1e-12
    assert result["fit_configuration"]["beam_width"] == 4


def test_graph_feature_fit_aligns_bare_target_in_residual_expression():
    result = GraphFeatureFitTool(context=graph_context()).execute(
        features=["x[i] ** 2", "sum[j](R[i,j], x[j])"],
        y="y + 2 * x[i] ** 2",
        max_terms=2,
    )

    assert train_metrics(result)["r2"] > 1 - 1e-12
    assert result["target_expression"] == "y[i] + 2 * x[i] ** 2"


def test_graph_feature_fit_beam_recovers_joint_features_hidden_by_decoy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 3))
    b = rng.normal(size=(30, 3))
    y = a + b
    decoy = y + 0.35 * rng.normal(size=y.shape)
    context = AgentContext(
        data={
            "R": np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
            "a": a,
            "b": b,
            "decoy": decoy,
            "y": y,
        },
        target="y",
        relation_names={"R"},
        num_nodes=3,
    )
    greedy = GraphFeatureFitTool(context=context).execute(
        features=["a[i]", "b[i]", "decoy[i]"],
        include_bias=False,
        max_terms=2,
        min_relative_improvement=0,
        beam_width=1,
    )
    beam = GraphFeatureFitTool(context=context).execute(
        features=["a[i]", "b[i]", "decoy[i]"],
        include_bias=False,
        max_terms=2,
        min_relative_improvement=0,
        beam_width=3,
    )

    assert greedy["selected_features"][0] == "decoy[i]"
    assert train_metrics(greedy)["r2"] < 0.99
    assert set(beam["selected_features"]) == {"a[i]", "b[i]"}
    assert train_metrics(beam)["r2"] > 1 - 1e-12
    assert beam["fit_configuration"]["beam_width"] == 3


def test_graph_feature_fit_bic_lasso_recovers_joint_support():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(80, 3))
    b = rng.normal(size=(80, 3))
    y = 1.7 * a - 0.8 * b
    decoy = y + 0.4 * rng.normal(size=y.shape)
    context = AgentContext(
        data={
            "R": np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
            "a": a,
            "b": b,
            "decoy": decoy,
            "y": y,
        },
        target="y",
        relation_names={"R"},
        num_nodes=3,
    )

    result = GraphFeatureFitTool(context=context).execute(
        features=["a[i]", "b[i]", "decoy[i]"],
        include_bias=False,
        max_terms=2,
        selection_method="bic_lasso",
    )

    assert set(result["selected_features"]) == {"a[i]", "b[i]"}
    assert train_metrics(result)["r2"] > 1 - 1e-12
    assert result["fit_configuration"]["selection_method"] == "bic_lasso"


def test_graph_structure_screen_recovers_local_gated_saturation():
    context = graph_context()
    x = context.data["x"]
    static_a = np.array([0.7, 1.1, 1.5])
    static_b = np.array([0.2, 0.4, 0.6])
    incoming = np.zeros_like(x)
    for target, source in context.data["R"]:
        incoming[:, target] += x[:, source] ** 2 / (1 + x[:, source] ** 2)
    context.data |= {"a": static_a, "b": static_b}
    context.variable_descriptions |= {"a": "", "b": ""}
    context.variable_axes |= {"a": (), "b": ()}
    context.data["y"] = x * static_a - x**2 * static_b + x * incoming
    context.invalidate_splits()

    result = GraphStructureScreenTool(context=context).execute(max_terms=8)

    assert train_metrics(result)["r2"] > 1 - 1e-12
    assert result["field_roles"] == {
        "x": "sample-varying",
        "a": "sample-invariant",
        "b": "sample-invariant",
    }
    assert any("sum[j]" in feature for feature in result["selected_features"])


def test_graph_structure_screen_executes_bounded_sigmoid_refinement():
    context = graph_context()
    x = context.data["x"]
    incoming = np.zeros_like(x)
    for target, source in context.data["R"]:
        incoming[:, target] += 1 / (1 + np.exp(-4.2 * (x[:, source] - 0.55)))
    context.data["y"] = -0.7 * x + 1.3 * incoming
    context.invalidate_splits()

    result = GraphStructureScreenTool(context=context).execute(max_terms=8)

    assert result["nonlinear_refinement"] is not None
    assert train_metrics(result)["r2"] > 1 - 1e-10
    assert "sigmoid" in result["formula"]


def test_graph_template_fit_compares_caller_supplied_messages_and_gates():
    context = graph_context()
    x = context.data["x"]
    incoming = np.zeros_like(x)
    for target, source in context.data["R"]:
        incoming[:, target] += np.sin(1.3 * (x[:, source] + x[:, target]))
    context.data["y"] = 0.2 - 0.9 * x + 1.7 * x * incoming
    context.invalidate_splits()

    result = GraphTemplateFitTool(context=context).execute(
        messages=[
            "x[j]",
            "sin(param('omega', value=0.8)*(x[j] + x[i]))",
        ],
        local_features=["x[i]"],
        gates=["1", "x[i]"],
    )

    assert train_metrics(result)["r2"] > 1 - 1e-10
    assert "omega" in result["ranked_candidates"][0]["message"]
    assert result["ranked_candidates"][0]["gate"] == "x[i]"


def test_graph_residual_analysis_ranks_a_gated_probe():
    context = graph_context()
    x = context.data["x"]
    incoming = np.zeros_like(x)
    for target, source in context.data["R"]:
        incoming[:, target] += np.cos(x[:, source] - x[:, target])
    context.data["y"] = -0.4 * x + 2.2 * x * incoming
    context.invalidate_splits()

    result = GraphResidualAnalysisTool(context=context).execute(
        candidate="-0.4*x[i]",
        probes=[
            "sum[j](R[i,j], x[j])",
            "sum[j](R[i,j], cos(x[j]-x[i]))",
        ],
        normalizers=["1", "x[i]"],
    )

    best = result["ranked_residual_probes"][0]
    assert best["r2"] > 1 - 1e-12
    assert best["normalizer"] == "x[i]"
    assert "cos" in best["probe"]


def test_graph_residual_analysis_detects_conservative_residual():
    context = graph_context()
    x = context.data["x"]
    conservative = np.zeros_like(x)
    for target, source in context.data["R"]:
        conservative[:, target] += x[:, source] ** 2 - x[:, target] ** 2
    context.data["y"] = conservative
    context.invalidate_splits()

    result = GraphResidualAnalysisTool(context=context).execute(candidate="0")

    assert result["residual_summary"]["sample_node_sum_to_independent_scale"] < 1e-12


def test_existing_diagnostic_tools_accept_indexed_expressions():
    context = graph_context()
    aggregate = "sum[j](R[i,j], x[j])"

    statistics = StatisticsTool(context=context).execute(variables=["x[i]", aggregate])
    relationships = RelationshipAnalysisTool(context=context).execute(
        variables=["x[i]", aggregate], y="y", n_folds=2,
    )
    polynomial = PolynomialFitTool(context=context).execute(
        x=["x[i] ** 2", aggregate], max_degree=1, include_interactions=False,
    )

    assert statistics["exceptions"] == []
    assert relationships["exceptions"] == []
    assert set(relationships["relationships"]) == {"x[i]", aggregate}
    assert train_metrics(polynomial)["r2"] > 1 - 1e-12


def test_existing_model_fitters_accept_indexed_derived_features():
    aggregate = "sum[j](R[i,j], x[j])"

    rational = RationalFitTool(
        context=transformed_graph_context(lambda incoming: 1 / (1 + incoming)),
    ).execute(x=[aggregate], numerator_degree=0, denominator_degree=1)
    power = PowerLawFitTool(
        context=transformed_graph_context(lambda incoming: 2 * incoming**1.5),
    ).execute(x=[aggregate])
    constants = ConstantFitTool(
        context=transformed_graph_context(lambda incoming: 3 * incoming),
    ).execute(eq=f"3.01*({aggregate})")

    assert train_metrics(rational)["r2"] > 1 - 1e-12
    assert train_metrics(power)["r2"] > 1 - 1e-12
    assert any(item["train_r2"] > 1 - 1e-12 for item in constants["pareto_front"])


def test_external_search_wrappers_accept_indexed_derived_features(monkeypatch):
    aggregate = "sum[j](R[i,j], x[j])"
    context = transformed_graph_context(lambda incoming: 2 * incoming + 1)

    def fake_run_sindy(self, X, y_fit, x_names, poly_degree, include_trig, threshold):
        return "2*x1+1"

    def fake_run_pysr(self, X, y_fit, x_names, binary_ops, unary_ops, timeout, maxsize):
        return "2*x1+1", [], 3

    monkeypatch.setattr(SINDyTool, "_run_sindy", fake_run_sindy)
    monkeypatch.setattr(PySRTool, "_run_pysr", fake_run_pysr)
    sindy = SINDyTool(context=context).execute(x=[aggregate], poly_degree=1)
    pysr = PySRTool(context=context).execute(
        binary_operators=["+", "*"], unary_operators=[], x=[aggregate],
    )

    assert train_metrics(sindy)["r2"] > 1 - 1e-12
    assert train_metrics(pysr)["r2"] > 1 - 1e-12
