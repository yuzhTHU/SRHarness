"""SRHarness Engine: structured symbolic models for scientific discovery."""

from __future__ import annotations

from typing import Any, Callable

from .expression import (
    Aggregate,
    Binary,
    Expression,
    Function,
    Gather,
    GroupedParameter,
    Index,
    Indexed,
    Number,
    Parameter,
    Reduction,
    RelationLift,
    Symbol,
    Variable,
    aggr,
    function,
    gather,
    grouped_param,
    param,
    reduction,
    sour,
    targ,
)
from .analysis import count_parameters, fold_constants, parameter_values, unbound_parameters
from .desugar import desugar
from .evaluation import evaluate
from .optimize import FitResult, bind_parameters, fit
from .indexed_evaluation import RelationField
from .parser import parse
from .render import render


def _unary(name: str) -> Callable[[Any], Expression]:
    def constructor(value: Any) -> Expression:
        """Create a unary symbolic function expression.

        Args:
            value: Symbolic expression or value to wrap.

        Returns:
            A symbolic unary function call.
        """
        return function(name, value)

    constructor.__name__ = name
    constructor.__qualname__ = name
    return constructor


sin = _unary("sin")
cos = _unary("cos")
tan = _unary("tan")
tanh = _unary("tanh")
sinh = _unary("sinh")
cosh = _unary("cosh")
arcsin = _unary("arcsin")
arccos = _unary("arccos")
arctan = _unary("arctan")
exp = _unary("exp")
log = _unary("log")
log10 = _unary("log10")
sqrt = _unary("sqrt")
abs = _unary("abs")
sigmoid = _unary("sigmoid")
sign = _unary("sign")
sec = _unary("sec")
sech = _unary("sech")
csc = _unary("csc")
cot = _unary("cot")
inv = _unary("inv")


def delay(value: Any, delta: Any) -> Expression:
    """Create a delayed-value expression.

    Args:
        value: Time-dependent expression to sample from the past.
        delta: Scalar or sample-aligned delay interval.

    Returns:
        A symbolic ``delay(value, delta)`` call.
    """
    return function("delay", value, delta)


__all__ = [
    "Aggregate", "Binary", "Expression", "FitResult", "Function", "Gather",
    "GroupedParameter",
    "Index", "Indexed", "Number", "Parameter", "Reduction", "RelationField",
    "RelationLift", "Symbol",
    "Variable", "abs", "aggr", "arccos", "arcsin", "arctan", "cos", "cosh",
    "bind_parameters", "cot", "count_parameters", "csc", "delay", "desugar", "evaluate", "exp", "fit",
    "fold_constants", "function", "gather", "grouped_param", "inv", "log", "log10",
    "param", "parameter_values", "parse", "reduction", "render", "sec", "sech",
    "sigmoid", "sign", "sin",
    "sinh", "sour", "sqrt", "tan", "tanh", "targ", "unbound_parameters",
]
