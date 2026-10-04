"""普通符号表达式的公开行为示例。"""

import numpy as np

import sr_harness_engine as engine


def test_an_expression_can_be_parsed_rendered_and_evaluated():
    """字符串是模型的交换格式，求值数据通过变量名传入。"""
    model = engine.parse("2.0 * sin(x1) + x2**2")
    data = {
        "x1": np.array([0.0, np.pi / 2]),
        "x2": np.array([2.0, 3.0]),
    }

    assert str(model) == "2.0 * sin(x1) + x2 ** 2"
    assert np.allclose(model.evaluate(data), [4.0, 11.0])
    assert str(engine.parse(str(model))) == str(model)


def test_expressions_can_also_be_built_with_python_operators():
    """直接构造和字符串解析产生相同的用户可见模型。"""
    x = engine.Symbol("x")
    built = 1 + 2 * engine.exp(-x)
    parsed = engine.parse("1 + 2 * exp(-x)")

    values = {"x": np.array([0.0, 1.0])}
    assert str(built) == str(parsed)
    assert np.allclose(built.evaluate(values), parsed.evaluate(values))


def test_a_symbol_may_carry_a_default_value():
    """默认值适合小型交互示例；正式运行通常在 evaluate 时传数据。"""
    x = engine.Symbol("x", value=np.array([1.0, 2.0, 3.0]))

    assert np.array_equal((x + 1).evaluate(), [2.0, 3.0, 4.0])
    assert np.array_equal((x + 1).evaluate({"x": np.array([10.0])}), [11.0])


def test_scalar_values_broadcast_over_sample_arrays():
    model = engine.parse("2 * x + 1")

    assert np.array_equal(model.evaluate({"x": np.arange(4)}), [1, 3, 5, 7])


def test_common_nonlinear_functions_are_elementwise():
    model = engine.parse(
        "abs(x) + sqrt(y) + log(exp(z)) + tanh(w) + sigmoid(v)"
    )
    data = {
        "x": np.array([-2.0, 3.0]),
        "y": np.array([4.0, 9.0]),
        "z": np.array([0.5, 1.5]),
        "w": np.array([0.0, 0.0]),
        "v": np.array([0.0, 0.0]),
    }

    assert np.allclose(model.evaluate(data), [5.0, 8.0])


def test_known_constants_can_be_bound_while_parsing():
    model = engine.parse("pi * r**2", variables={"pi": np.pi})

    assert np.allclose(model.evaluate({"r": np.array([1.0, 2.0])}), [np.pi, 4 * np.pi])


def test_latex_rendering_is_available_for_user_interfaces():
    model = engine.parse("x / (1 + y)")

    assert model.to_str(latex=True) == r"\frac{x}{(1 + y)}"
