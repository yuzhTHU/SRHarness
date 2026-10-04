"""普通参数和类别参数的拟合行为。"""

import numpy as np
import pytest

import sr_harness_engine as engine


def test_a_parameter_with_an_initial_value_can_be_evaluated_before_fitting():
    model = engine.parse("param('slope', value=2.0) * x")

    assert np.array_equal(model.evaluate({"x": np.array([1.0, 2.0])}), [2.0, 4.0])


def test_an_uninitialized_parameter_must_be_fitted_or_supplied():
    model = engine.parse("param('slope') * x")

    with pytest.raises(ValueError, match="slope.*no value"):
        model.evaluate({"x": np.array([1.0, 2.0])})

    prediction = model.evaluate(
        {"x": np.array([1.0, 2.0])},
        parameters={"slope": 3.0},
    )
    assert np.array_equal(prediction, [3.0, 6.0])


def test_repeated_parameter_names_share_one_fitted_value():
    x = np.linspace(-2.0, 2.0, 41)
    model = engine.parse("param('a', value=0.5) * x + param('a')")
    target = 2.0 * x + 2.0

    fitted = model.fit({"x": x}, target)

    assert set(fitted.parameters) == {"a"}
    assert fitted.parameters["a"] == pytest.approx(2.0, abs=1e-5)
    assert np.allclose(fitted.predict({"x": x}), target, atol=1e-5)


def test_fit_returns_a_result_without_mutating_the_source_expression():
    x = np.linspace(0.0, 3.0, 31)
    model = engine.parse("param('a') * x + param('b')")

    fitted = model.fit({"x": x}, 3.0 * x - 1.0, initial={"a": 1.0, "b": 0.0})

    with pytest.raises(ValueError, match="has no value"):
        model.evaluate({"x": x})
    assert fitted.parameters == pytest.approx({"a": 3.0, "b": -1.0}, abs=1e-5)
    assert np.allclose(fitted.evaluate({"x": x}), 3.0 * x - 1.0, atol=1e-5)


def test_numeric_literals_remain_fixed_during_normal_parameter_fitting():
    x = np.linspace(-1.0, 1.0, 21)
    model = engine.parse("2 * x + param('bias', value=0.0)")

    fitted = model.fit({"x": x}, 3.0 * x + 1.0)

    assert set(fitted.parameters) == {"bias"}
    assert fitted.parameters["bias"] == pytest.approx(1.0, abs=1e-5)
    assert str(model).startswith("2 * x")
    assert not np.allclose(fitted.predict({"x": x}), 3.0 * x + 1.0)


def test_grouped_parameter_assigns_one_value_to_each_category():
    model = engine.parse(
        "grouped_param(group, name='slope', value={'A': 2.0, 'B': 3.0}) * x"
    )
    data = {
        "x": np.array([1.0, 2.0, 1.0, 2.0]),
        "group": np.array(["A", "A", "B", "B"], dtype=object),
    }

    assert np.array_equal(model.evaluate(data), [2.0, 4.0, 3.0, 6.0])


def test_grouped_parameter_can_learn_category_specific_values():
    model = engine.parse("grouped_param(group, name='slope') * x")
    data = {
        "x": np.array([1.0, 2.0, 1.0, 2.0]),
        "group": np.array(["A", "A", "B", "B"], dtype=object),
    }

    fitted = model.fit(data, np.array([2.0, 4.0, -1.0, -2.0]))

    assert fitted.parameters["slope"] == pytest.approx({"A": 2.0, "B": -1.0})
    assert np.allclose(fitted.predict(data), [2.0, 4.0, -1.0, -2.0])
