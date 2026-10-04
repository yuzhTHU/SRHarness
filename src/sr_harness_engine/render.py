"""Canonical string renderer for symbolic expressions."""

from __future__ import annotations

from .expression import (
    Aggregate,
    Binary,
    Expression,
    Function,
    GroupedParameter,
    Indexed,
    Number,
    Parameter,
    Reduction,
    RelationLift,
    Symbol,
    Unary,
)


PRECEDENCE = {"+": 10, "-": 10, "*": 20, "/": 20, "**": 30}


def render(expression: Expression, parent_precedence: int = 0, right: bool = False) -> str:
    if isinstance(expression, Number):
        return repr(expression.value)
    if isinstance(expression, Symbol):
        return expression.name
    if isinstance(expression, Parameter):
        suffix = "" if expression.value is None else f", value={expression.value!r}"
        return f"param({expression.name!r}{suffix})"
    if isinstance(expression, GroupedParameter):
        arguments = [render(expression.by)]
        if expression.name is not None:
            arguments.append(f"name={expression.name!r}")
        if expression.value is not None:
            arguments.append(f"value={dict(expression.value)!r}")
        if expression.default is not None:
            arguments.append(f"default={expression.default!r}")
        return f"grouped_param({', '.join(arguments)})"
    if isinstance(expression, Indexed):
        indices = ", ".join(str(index) for index in expression.indices)
        return f"{render(expression.base, 100)}[{indices}]"
    if isinstance(expression, Unary):
        text = f"{expression.operator}{render(expression.operand, 40)}"
        return f"({text})" if parent_precedence > 40 else text
    if isinstance(expression, Binary):
        precedence = PRECEDENCE[expression.operator]
        left = render(expression.left, precedence)
        right_precedence = precedence + (0 if expression.operator == "**" else 1)
        right_text = render(expression.right, right_precedence, right=True)
        text = f"{left} {expression.operator} {right_text}"
        return f"({text})" if precedence < parent_precedence else text
    if isinstance(expression, Function):
        return f"{expression.name}({', '.join(render(arg) for arg in expression.arguments)})"
    if isinstance(expression, Reduction):
        indices = ", ".join(str(index) for index in expression.indices)
        return f"sum[{indices}]({render(expression.operand)})"
    if isinstance(expression, Aggregate):
        return f"aggr({render(expression.relation)}, {render(expression.operand)})"
    if isinstance(expression, RelationLift):
        name = "targ" if expression.role == "target" else "sour"
        arguments = [expression.operand]
        if expression.relation is not None:
            arguments.insert(0, expression.relation)
        return f"{name}({', '.join(render(arg) for arg in arguments)})"
    raise TypeError(f"Cannot render {type(expression).__name__}.")
