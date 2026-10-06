"""指标缩并、网络和超图表达式的行为示例。"""

import numpy as np

import sr_harness_engine as engine


def test_a_free_index_keeps_an_axis_and_a_summed_index_removes_it():
    values = {"x": np.array([1.0, 2.0, 3.0])}

    assert np.array_equal(engine.parse("x[i]").evaluate(values, num_nodes=3), [1.0, 2.0, 3.0])
    assert engine.parse("sum[i](x[i])").evaluate(values, num_nodes=3) == 6.0


def test_an_edge_list_binds_indices_to_its_columns():
    """A 的列为 (target, source)，求和消去源指标 j，保留目标指标 i。"""
    data = {
        "x": np.array([1.0, 2.0, 3.0]),
        "A": np.array([[0, 1], [0, 2], [1, 2]], dtype=int),
    }
    model = engine.parse("sum[j](A[i, j], x[j])")

    # 1 -> 0, 2 -> 0, 2 -> 1; node 2 has no incoming edge.
    assert np.array_equal(model.evaluate(data, num_nodes=3), [5.0, 3.0, 0.0])


def test_node_dynamics_can_mix_local_and_aggregated_terms():
    data = {
        "x": np.array([1.0, 2.0, 3.0]),
        "A": np.array([[0, 1], [0, 2], [1, 2]], dtype=int),
    }
    model = engine.parse("x[i] + sum[j](A[i, j], x[i] * x[j])")

    assert np.array_equal(model.evaluate(data, num_nodes=3), [6.0, 8.0, 3.0])


def test_network_operations_preserve_the_feature_axis():
    data = {
        "x": np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
        "A": np.array([[0, 1], [0, 2], [1, 2]], dtype=int),
    }
    model = engine.parse("sum[j](A[i, j], x[j])")

    assert np.array_equal(
        model.evaluate(data, num_nodes=3),
        [[5.0, 3.0, 0.0], [50.0, 30.0, 0.0]],
    )


def test_edge_aligned_weights_can_scale_messages():
    data = {
        "x": np.array([1.0, 2.0, 3.0]),
        "A": np.array([[0, 1], [0, 2], [1, 2]], dtype=int),
        "weight": np.array([10.0, 20.0, 30.0]),
    }
    model = engine.parse("sum[j](A[i, j], weight[i, j] * x[j])")

    assert np.array_equal(model.evaluate(data, num_nodes=3), [80.0, 90.0, 0.0])


def test_relation_fields_can_store_one_weight_per_sample_and_edge():
    data = {
        "x": np.array([1.0, 2.0, 3.0]),
        "A": np.array([[0, 1], [0, 2], [1, 2]], dtype=int),
        "w": np.array([[10.0, 20.0, 30.0], [1.0, 2.0, 4.0]]),
    }
    model = engine.parse("sum[j](A[i, j], w[i, j] * x[j])")

    assert np.array_equal(
            model.evaluate(data, num_nodes=3),
            [[80.0, 90.0, 0.0], [8.0, 12.0, 0.0]],
    )


def test_an_independent_sum_can_appear_inside_a_network_term():
    data = {
        "x": np.array([1.0, 2.0, 3.0]),
        "A": np.array([[0, 1], [0, 2], [1, 2]], dtype=int),
    }
    model = engine.parse("x[i] + sum[j](A[i, j], x[i] * sum[k](x[k]) * x[j])")

    assert np.array_equal(model.evaluate(data, num_nodes=3), [31.0, 38.0, 3.0])


def test_the_same_index_language_extends_to_hyperedges():
    data = {
        "x": np.array([1.0, 2.0, 3.0, 4.0]),
        "T": np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3]], dtype=int),
    }
    model = engine.parse(
        "x[i] + sum[j, k](T[i, j, k], x[i] * x[j] * x[k])"
    )

    assert np.array_equal(model.evaluate(data, num_nodes=4), [19.0, 26.0, 3.0, 4.0])


def test_aggr_syntax_supports_source_to_target_message_passing():
    """便捷语法中的边表列顺序为 (target, source)。"""
    data = {
        "x": np.array([1.0, 2.0, 3.0]),
        "A": np.array([[1, 0], [2, 0], [2, 1]], dtype=int),
    }
    sugared = engine.parse("aggr(A * targ(x) * sour(x))")
    explicit = engine.parse("sum[j](A[i, j], x[i] * x[j])")
    inherited = engine.parse("aggr(A, targ(x) * sour(x))")

    assert str(sugared) == "sum[j](A[i, j], x[i] * x[j])"
    assert np.array_equal(sugared.evaluate(data, num_nodes=3), explicit.evaluate(data, num_nodes=3))
    assert np.array_equal(explicit.evaluate(data, num_nodes=3), [0.0, 2.0, 9.0])
    assert np.array_equal(
        inherited.evaluate(data, num_nodes=3),
        explicit.evaluate(data, num_nodes=3),
    )
