"""稀疏结构维、关系代数与需求驱动求值的行为示例。"""

import numpy as np
import pytest

import sr_harness_engine as engine


def test_num_nodes_is_explicit_and_preserves_isolated_nodes():
    data = {
        "x": np.array([2.0, 3.0, 5.0, 7.0]),
        "A": np.array([[0, 1]], dtype=int),
    }

    result = engine.parse("sum[j](A[i, j], x[j])").evaluate(data, num_nodes=4)

    assert np.array_equal(result, [3.0, 0.0, 0.0, 0.0])


def test_leading_dimensions_broadcast_and_the_last_dimension_is_structural():
    data = {
        "theta": np.array([[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]]),
        "omega": np.array([[0.1, 0.2, 0.3]]),
        "A": np.array([[0, 1], [1, 2]], dtype=int),
    }
    model = engine.parse(
        "omega[i] + 0.65 * sum[j](A[i, j], sin(theta[j] - theta[i]))"
    )

    expected = data["omega"] + 0.65 * np.array([
        [np.sin(1.0), np.sin(1.0), 0.0],
        [np.sin(1.0), np.sin(1.0), 0.0],
    ])
    assert np.allclose(model.evaluate(data, num_nodes=3), expected)


def test_relation_addition_retains_numeric_overlap_values():
    data = {
        "x": np.array([1.0, 2.0, 3.0]),
        "A1": np.array([[0, 1], [0, 2]], dtype=int),
        "A2": np.array([[0, 2], [1, 2]], dtype=int),
    }

    result = engine.parse(
        "sum[j](A1[i, j] + A2[i, j], x[j])"
    ).evaluate(data, num_nodes=3)

    assert np.array_equal(result, [8.0, 3.0, 0.0])


@pytest.mark.parametrize(
    ("relation", "expected"),
    [
        ("A[i, j] + 1", np.array([11.0, 6.0, 6.0])),
        (
            "exp(A[i, j])",
            np.array([np.e * 5.0 + 1.0, 6.0, 6.0]),
        ),
    ],
)
def test_nonzero_fill_values_remain_sparse(relation, expected):
    data = {
        "x": np.array([1.0, 2.0, 3.0]),
        "A": np.array([[0, 1], [0, 2]], dtype=int),
    }

    result = engine.parse(f"sum[j]({relation}, x[j])").evaluate(data, num_nodes=3)

    assert np.allclose(result, expected)


def test_edge_fields_use_the_relation_coordinate_table():
    data = {
        "x": np.array([1.0, 2.0, 3.0]),
        "A": np.array([[0, 1], [0, 2], [1, 2]], dtype=int),
        "w": engine.RelationField(np.array([10.0, 20.0, 30.0]), relation="A"),
    }

    result = engine.parse(
        "sum[j](A[i, j], w[i, j] * x[j])"
    ).evaluate(data, num_nodes=3)

    assert np.array_equal(result, [80.0, 90.0, 0.0])


def test_duplicate_relation_rows_keep_binary_incidence_semantics():
    data = {
        "x": np.array([1.0, 2.0]),
        "A": np.array([[0, 1], [0, 1]], dtype=int),
    }

    result = engine.parse("sum[j](A[i, j], x[j])").evaluate(data, num_nodes=2)

    assert np.array_equal(result, [2.0, 0.0])


def test_nested_sums_must_not_rebind_an_active_index():
    model = engine.parse("sum[j](x[j] * sum[j](x[j]))")

    with pytest.raises(ValueError, match="already active"):
        model.evaluate({"x": np.array([1.0, 2.0])}, num_nodes=2)


def test_nested_aggr_automatically_uses_fresh_internal_indices():
    data = {
        "x": np.array([1.0, 2.0, 3.0]),
        "A": np.array([[0, 1], [1, 2]], dtype=int),
    }
    model = engine.parse(
        "aggr(A, sour(aggr(A, targ(x) * sour(x))))"
    )

    assert np.array_equal(model.evaluate(data, num_nodes=3), [6.0, 0.0, 0.0])


def test_multiple_free_indices_create_multiple_dense_structural_axes():
    result = engine.parse("x[i] + x[j]").evaluate(
        {"x": np.array([1.0, 2.0])}, num_nodes=2
    )

    assert np.array_equal(result, [[2.0, 3.0], [3.0, 4.0]])


def test_unindexed_non_scalar_variables_are_rejected_in_indexed_models():
    with pytest.raises(ValueError, match="must be indexed"):
        engine.parse("x + sum[j](A[i, j], x[j])").evaluate(
            {"x": np.array([1.0, 2.0]), "A": np.array([[0, 1]])},
            num_nodes=2,
        )


def test_obvious_independent_factors_can_be_pulled_out_of_a_sum():
    x = np.arange(1.0, 1001.0)
    result = engine.parse("sum[j](x[i] * x[j])").evaluate(
        {"x": x}, num_nodes=len(x)
    )

    assert np.array_equal(result, x * x.sum())


def test_a_missing_reduction_index_multiplies_by_num_nodes():
    data = {
        "x": np.array([1.0, 2.0, 3.0]),
        "A": np.array([[0, 1], [1, 2]], dtype=int),
    }
    model = engine.parse(
        "sum[j](A[i, j], sum[k](x[i] * x[j]))"
    )

    assert np.array_equal(model.evaluate(data, num_nodes=3), [6.0, 18.0, 0.0])


def test_parameters_can_be_fitted_in_an_indexed_network_model():
    data = {
        "x": np.array([1.0, 2.0, 4.0]),
        "A": np.array([[0, 1], [1, 2]], dtype=int),
    }
    model = engine.parse("param('coupling') * sum[j](A[i, j], x[j] - x[i])")
    target = 0.65 * np.array([1.0, 2.0, 0.0])

    fitted = model.fit(data, target, num_nodes=3)

    assert fitted.parameters["coupling"] == pytest.approx(0.65, abs=1e-5)
    assert np.allclose(fitted.predict(data, num_nodes=3), target, atol=1e-5)
