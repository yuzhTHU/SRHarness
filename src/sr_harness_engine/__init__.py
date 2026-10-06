"""SRHarness Engine: structured symbolic models for scientific discovery."""

from .expression import (
    Aggregate,
    Binary,
    Expression,
    Function,
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
    grouped_param,
    param,
    reduction,
    sour,
    targ,
)
from .analysis import count_parameters, fold_constants
from .desugar import desugar
from .optimize import FitResult, fit
from .parser import parse
from .render import render


def _unary(name):
    return lambda value: function(name, value)


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


def delay(value, delta):
    """Create a delayed-value expression.

    Args:
        value: Time-dependent expression to sample from the past.
        delta: Scalar or sample-aligned delay interval.

    Returns:
        A symbolic ``delay(value, delta)`` call.
    """
    return function("delay", value, delta)


__all__ = [
    "Aggregate", "Binary", "Expression", "FitResult", "Function", "GroupedParameter",
    "Index", "Indexed", "Number", "Parameter", "Reduction", "RelationLift", "Symbol",
    "Variable", "abs", "aggr", "arccos", "arcsin", "arctan", "cos", "cosh",
    "cot", "count_parameters", "csc", "delay", "desugar", "exp", "fit",
    "fold_constants", "function", "grouped_param", "inv", "log", "log10",
    "param", "parse", "reduction", "render", "sec", "sech", "sigmoid", "sign", "sin",
    "sinh", "sour", "sqrt", "tan", "tanh", "targ",
]
