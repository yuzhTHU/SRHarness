"""指标缩并、网络和超图表达式的行为示例。"""

import numpy as np

import sr_harness_engine as engine


def test_a_free_index_keeps_an_axis_and_a_summed_index_removes_it():
    values = {"x": np.array([[1.0], [2.0], [3.0]])}

    assert np.array_equal(engine.parse("x[i]").evaluate(values), [[1.0], [2.0], [3.0]])
    assert np.array_equal(engine.parse("sum[i](x[i])").evaluate(values), [6.0])


def test_an_edge_list_binds_indices_to_its_columns():
    """A[i,j] 的第 0、1 列分别绑定 i、j，sum[j] 后结果按 i 聚合。"""
    data = {
        "x": np.array([[1.0], [2.0], [3.0]]),
        "A": np.array([[0, 1], [0, 2], [1, 2]], dtype=int),
    }
    model = engine.parse("sum[j](A[i, j] * x[j])")

    # i=0 receives x[1] + x[2], i=1 receives x[2], and i=2 is isolated.
    assert np.array_equal(model.evaluate(data), [[5.0], [3.0], [0.0]])


def test_node_dynamics_can_mix_local_and_aggregated_terms():
    data = {
        "x": np.array([[1.0], [2.0], [3.0]]),
        "A": np.array([[0, 1], [0, 2], [1, 2]], dtype=int),
    }
    model = engine.parse("x[i] + sum[j](A[i, j] * x[i] * x[j])")

    assert np.array_equal(model.evaluate(data), [[6.0], [8.0], [3.0]])


def test_network_operations_preserve_the_feature_axis():
    data = {
        "x": np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        "A": np.array([[0, 1], [0, 2], [1, 2]], dtype=int),
    }
    model = engine.parse("sum[j](A[i, j] * x[j])")

    assert np.array_equal(
        model.evaluate(data),
        [[5.0, 50.0], [3.0, 30.0], [0.0, 0.0]],
    )


def test_edge_aligned_weights_can_scale_messages():
    data = {
        "x": np.array([[1.0], [2.0], [3.0]]),
        "A": np.array([[0, 1], [0, 2], [1, 2]], dtype=int),
        "weight": np.array([10.0, 20.0, 30.0]),
    }
    model = engine.parse("sum[j](A[i, j] * weight * x[j])")

    assert np.array_equal(model.evaluate(data), [[80.0], [90.0], [0.0]])


def test_an_independent_sum_can_appear_inside_a_network_term():
    data = {
        "x": np.array([[1.0], [2.0], [3.0]]),
        "A": np.array([[0, 1], [0, 2], [1, 2]], dtype=int),
    }
    model = engine.parse("x[i] + sum[j](A[i, j] * sum[k](x[k]) * x[j])")

    assert np.array_equal(model.evaluate(data), [[31.0], [20.0], [3.0]])


def test_the_same_index_language_extends_to_hyperedges():
    data = {
        "x": np.array([[1.0], [2.0], [3.0], [4.0]]),
        "T": np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3]], dtype=int),
    }
    model = engine.parse(
        "x[i] + sum[j, k](T[i, j, k] * x[i] * x[j] * x[k])"
    )

    assert np.array_equal(model.evaluate(data), [[19.0], [26.0], [3.0], [4.0]])


def test_aggr_syntax_supports_source_to_target_message_passing():
    """便捷语法中的边表列顺序为 (source, target)。"""
    data = {
        "x": np.array([[1.0], [2.0], [3.0]]),
        "A": np.array([[1, 0], [2, 0], [2, 1]], dtype=int),
    }
    explicit = engine.parse("x + aggr(A, targ(A, x) * sour(A, x))")
    inherited = engine.parse("x + aggr(A, targ(x) * sour(x))")

    assert np.array_equal(explicit.evaluate(data), [[6.0], [8.0], [3.0]])
    assert np.array_equal(inherited.evaluate(data), explicit.evaluate(data))
