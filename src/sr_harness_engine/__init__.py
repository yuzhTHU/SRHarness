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
    aggr,
    function,
    grouped_param,
    param,
    reduction,
    sour,
    targ,
)
from .optimize import FitResult, fit
from .parser import parse
from .render import render


def _unary(name):
    return lambda value: function(name, value)


sin = _unary("sin")
cos = _unary("cos")
tan = _unary("tan")
tanh = _unary("tanh")
exp = _unary("exp")
log = _unary("log")
log10 = _unary("log10")
sqrt = _unary("sqrt")
abs = _unary("abs")
sigmoid = _unary("sigmoid")


def delay(value, delta):
    return function("delay", value, delta)


__all__ = [
    "Aggregate", "Binary", "Expression", "FitResult", "Function", "GroupedParameter",
    "Index", "Indexed", "Number", "Parameter", "Reduction", "RelationLift", "Symbol",
    "abs", "aggr", "cos", "delay", "exp", "fit", "function", "grouped_param", "log",
    "log10", "param", "parse", "reduction", "render", "sigmoid", "sin", "sour", "sqrt",
    "tan", "tanh", "targ",
]
