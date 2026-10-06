"""Dense free-index outputs and relation-aligned gather behavior."""

import numpy as np
import pytest

import sr_harness_engine as engine


def test_multiple_free_indices_produce_dense_structural_axes():
    x = np.array([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]])

    result = engine.parse("x[i] + x[j]").evaluate({"x": x}, num_nodes=3)

    assert result.shape == (2, 3, 3)
    assert np.array_equal(result[0], [
        [2.0, 3.0, 5.0],
        [3.0, 4.0, 6.0],
        [5.0, 6.0, 8.0],
    ])


def test_gather_broadcasts_an_operand_with_fewer_free_indices():
    data = {
        "x": np.array([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]]),
        "A": np.array([[0, 1], [1, 2]], dtype=int),
    }

    result = engine.parse("gather(A[i, j], x[i])").evaluate(data, num_nodes=3)

    assert np.array_equal(result, [[1.0, 2.0], [10.0, 20.0]])


def test_gather_behaves_like_a_relation_aligned_array():
    data = {
        "x": np.array([1.0, 2.0, 4.0]),
        "w": np.array([10.0, 20.0]),
        "A": np.array([[0, 1], [1, 2]], dtype=int),
    }

    gathered = engine.parse("gather(A[i, j], x[i] + x[j])")

    assert np.array_equal(gathered.evaluate(data, num_nodes=3), [3.0, 6.0])
    assert np.array_equal(
        engine.parse("2 * gather(A[i, j], x[i] + x[j])").evaluate(
            data, num_nodes=3
        ),
        [6.0, 12.0],
    )
    assert np.allclose(
        engine.parse("sin(gather(A[i, j], x[i] + x[j]))").evaluate(
            data, num_nodes=3
        ),
        np.sin([3.0, 6.0]),
    )
    assert np.array_equal(
        engine.parse("gather(A[i, j], x[i] + x[j]) + w").evaluate(
            data, num_nodes=3
        ),
        [13.0, 26.0],
    )


def test_a_gathered_expression_can_be_lifted_back_to_relation_coordinates():
    data = {
        "x": np.array([1.0, 2.0, 4.0]),
        "A": np.array([[0, 1], [1, 2]], dtype=int),
    }
    model = engine.parse(
        "sum[i](A[i, j], "
        "gather(A[i, j], x[i] + x[j])[i, j] * x[i] * x[j])"
    )

    assert np.array_equal(model.evaluate(data, num_nodes=3), [0.0, 6.0, 48.0])


def test_gather_supports_hypergraph_relations():
    data = {
        "x": np.array([1.0, 2.0, 3.0, 4.0]),
        "T": np.array([[0, 1, 2], [1, 2, 3]], dtype=int),
    }

    result = engine.parse(
        "gather(T[i, j, k], x[i] * x[j] * x[k])"
    ).evaluate(data, num_nodes=4)

    assert np.array_equal(result, [6.0, 24.0])


def test_gather_multiplies_nonbinary_relation_values():
    data = {
        "x": np.array([1.0, 2.0, 4.0]),
        "A1": np.array([[0, 1], [1, 2]], dtype=int),
        "A2": np.array([[0, 1]], dtype=int),
    }

    result = engine.parse(
        "gather(A1[i, j] + 0.5 * A2[i, j], x[i] + x[j])"
    ).evaluate(data, num_nodes=3)

    assert np.array_equal(result, [4.5, 6.0])


def test_gather_preserves_direct_relation_row_order():
    data = {
        "x": np.array([1.0, 2.0, 4.0]),
        "A": np.array([[1, 2], [0, 1]], dtype=int),
    }

    result = engine.parse("gather(A[i, j], x[i] + x[j])").evaluate(
        data, num_nodes=3
    )

    assert np.array_equal(result, [6.0, 3.0])


def test_a_single_gathered_entry_keeps_its_entry_axis():
    data = {
        "x": np.array([1.0, 2.0]),
        "A": np.array([[0, 1]], dtype=int),
    }

    result = engine.parse("2 * gather(A[i, j], x[i] + x[j])").evaluate(
        data, num_nodes=2
    )

    assert result.shape == (1,)
    assert np.array_equal(result, [6.0])


def test_a_nonzero_relation_fill_gathers_the_full_structural_domain():
    data = {
        "x": np.array([1.0, 2.0]),
        "A": np.array([[0, 1]], dtype=int),
    }

    result = engine.parse("gather(A[i, j] + 1, x[i] + x[j])").evaluate(
        data, num_nodes=2
    )

    assert np.array_equal(result, [2.0, 6.0, 3.0, 4.0])


def test_gather_rejects_operand_indices_absent_from_the_relation():
    data = {
        "x": np.array([1.0, 2.0]),
        "A": np.array([[0, 1]], dtype=int),
    }

    with pytest.raises(ValueError, match="absent from its relation"):
        engine.parse("gather(A[i, j], x[k])").evaluate(data, num_nodes=2)


def test_one_expression_cannot_mix_fields_from_different_relations():
    data = {
        "x": np.array([1.0, 2.0]),
        "A": np.array([[0, 1]], dtype=int),
        "B": np.array([[1, 0]], dtype=int),
    }

    with pytest.raises(ValueError, match="different relations"):
        engine.parse(
            "gather(A[i, j], x[i]) + gather(B[i, j], x[i])"
        ).evaluate(data, num_nodes=2)


def test_relation_field_arithmetic_does_not_relax_other_indexed_expressions():
    data = {
        "x": np.array([1.0, 2.0]),
        "y": np.array([3.0, 4.0]),
    }

    with pytest.raises(ValueError, match="must be indexed"):
        engine.parse("sum[i](x[i]) + y").evaluate(data, num_nodes=2)
