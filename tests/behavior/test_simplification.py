"""常量折叠和参数计数的行为示例。"""

import numpy as np

import sr_harness_engine as engine


def test_parsing_folds_closed_constant_subexpressions():
    model = engine.parse("(2 + 3) * x")

    assert str(model) == "5 * x"


def test_folding_preserves_exact_fractions_and_named_functions():
    model = engine.parse("x ** (4 / 3) + sqrt(2)")

    assert str(model) == "x ** (4 / 3) + sqrt(2)"


def test_constant_folding_does_not_reorder_symbolic_terms():
    model = engine.parse("y + (2 * 3) * x")

    assert str(model) == "y + 6 * x"


def test_fixed_literals_are_not_fitted_parameters():
    model = engine.parse("2 * x + 1")

    assert engine.count_parameters(model) == 0
    assert model.count_parameters() == 0


def test_repeated_names_represent_one_parameter():
    model = engine.parse("param('a') * x + param('a') + param('b')")

    assert model.count_parameters() == 2


def test_grouped_parameters_count_one_value_per_observed_category():
    model = engine.parse("param('bias') + grouped_param(group, name='slope') * x")
    data = {"group": np.array(["A", "A", "B"], dtype=object)}

    assert model.count_parameters(data) == 3
