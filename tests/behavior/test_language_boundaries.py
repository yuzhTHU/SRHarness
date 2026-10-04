"""解析安全、错误信息和表达式树接口的行为示例。"""

import numpy as np
import pytest

import sr_harness_engine as engine


@pytest.mark.parametrize(
    "source",
    [
        "np.sin(x)",
        "__import__('os').system('echo unsafe')",
        "x[0:10]",
        "custom_python_function(x)",
        "[x for x in values]",
    ],
)
def test_the_parser_rejects_python_outside_the_symbolic_language(source):
    with pytest.raises((SyntaxError, ValueError)):
        engine.parse(source)


def test_a_missing_symbol_is_reported_by_name():
    model = engine.parse("x + missing")

    with pytest.raises(KeyError, match="missing"):
        model.evaluate({"x": np.array([1.0])})


def test_conflicting_initial_values_for_one_parameter_are_rejected():
    model = engine.parse("param('a', value=1.0) + param('a', value=2.0)")

    with pytest.raises(ValueError, match="Conflicting initial values.*'a'"):
        model.evaluate()


def test_expression_size_and_tree_are_available_for_diagnostics():
    model = engine.parse("sin(x) + 1")

    assert len(model) == 4
    assert [str(node) for node in model.iter_preorder()] == [
        "sin(x) + 1",
        "sin(x)",
        "x",
        "1",
    ]
    tree = model.to_tree()
    assert tree.splitlines() == [
        "sin(x) + 1",
        "├ sin(x)",
        "│ └ x",
        "└ 1",
    ]


def test_tree_replacement_returns_a_new_expression():
    model = engine.parse("x + 1")
    x_node = next(node for node in model.iter_preorder() if isinstance(node, engine.Symbol))
    replaced = model.replace(x_node, engine.Symbol("z"))

    assert str(model) == "x + 1"
    assert str(replaced) == "z + 1"
