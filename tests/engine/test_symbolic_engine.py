import numpy as np
import pytest

import sr_harness_engine as engine


def test_parse_render_and_evaluate_algebra():
    expression = engine.parse("2.0 * sin(x1) + x2")

    assert str(expression) == "2.0 * sin(x1) + x2"
    assert np.allclose(
        expression.evaluate({"x1": np.array([0.0, np.pi / 2]), "x2": np.array([1.0, 1.0])}),
        [1.0, 3.0],
    )
    assert str(engine.parse(str(expression))) == str(expression)


def test_named_parameter_default_and_fit():
    x = np.linspace(0.0, 4.0, 20)
    expression = engine.parse(
        "param('alpha', value=0.3) * x / (1 - param('alpha')) + param('beta')"
    )

    with pytest.raises(ValueError, match="beta"):
        expression.evaluate({"x": x})

    result = expression.fit({"x": x}, 2.0 * x + 1.0)

    assert result.success
    assert np.allclose(result.predict({"x": x}), 2.0 * x + 1.0, atol=1e-4)
    assert set(result.parameters) == {"alpha", "beta"}


def test_grouped_parameter_fit():
    values = {
        "x": np.array([1.0, 2.0, 1.0, 2.0]),
        "s": np.array(["A", "A", "B", "B"], dtype=object),
    }
    expression = engine.parse("grouped_param(s) * x")
    result = expression.fit(values, np.array([2.0, 4.0, 3.0, 6.0]))

    assert result.success
    assert result.parameters["grouped:s"] == pytest.approx({"A": 2.0, "B": 3.0})
    assert np.allclose(result.predict(values), [2.0, 4.0, 3.0, 6.0])


def test_explicit_network_indices():
    values = {
        "x": np.array([[1.0], [2.0], [3.0]]),
        "A": np.array([[0, 1], [0, 2], [1, 2]], dtype=int),
    }
    expression = engine.parse("x[i] + sum[j](A[i, j], x[i] * x[j])")

    assert str(expression) == "x[i] + sum[j](A[i, j], x[i] * x[j])"
    assert np.allclose(expression.evaluate(values), [[6.0], [8.0], [3.0]])


def test_nested_global_sum_uses_an_independent_index():
    values = {
        "x": np.array([[1.0], [2.0], [3.0]]),
        "A": np.array([[0, 1], [0, 2], [1, 2]], dtype=int),
    }
    expression = engine.parse("x[i] + sum[j](A[i, j], x[i] * sum[k](x[k]) * x[j])")

    assert np.allclose(expression.evaluate(values), [[31.0], [38.0], [3.0]])


def test_hypergraph_indices():
    values = {
        "x": np.array([[1.0], [2.0], [3.0], [4.0]]),
        "T": np.array([[0, 1, 2], [0, 2, 3], [1, 2, 3]], dtype=int),
    }
    expression = engine.parse("x[i] + sum[j, k](T[i, j, k], x[i] * x[j] * x[k])")

    assert np.allclose(expression.evaluate(values), [[19.0], [26.0], [3.0], [4.0]])


@pytest.mark.parametrize(
    "source",
    [
        "x + aggr(A * targ(x) * sour(x))",
        "x + aggr(A, targ(A, x) * sour(A, x))",
        "x + aggr(A, targ(x) * sour(x))",
    ],
)
def test_aggr_compatibility_syntax(source):
    values = {
        "x": np.array([[1.0], [2.0], [3.0]]),
        "A": np.array([[1, 0], [2, 0], [2, 1]], dtype=int),
    }

    assert np.allclose(engine.parse(source).evaluate(values), [[1.0], [4.0], [12.0]])


def test_delay_uses_leading_time_axis():
    expression = engine.parse("x + delay(x, delta)")
    value = expression.evaluate(
        {"x": np.array([0.0, 1.0, 4.0, 9.0]), "delta": np.ones(4)},
        time=np.arange(4, dtype=float),
    )

    assert np.isnan(value[0])
    assert np.allclose(value[1:], [1.0, 5.0, 13.0])


def test_parser_does_not_execute_arbitrary_python():
    with pytest.raises(ValueError):
        engine.parse("__import__('os').system('echo unsafe')")
