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


def render(
    expression: Expression,
    parent_precedence: int = 0,
    right: bool = False,
    *,
    latex: bool = False,
    number_format: str = "",
) -> str:
    if isinstance(expression, Number):
        return format(expression.value, number_format) if number_format else repr(expression.value)
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
        base = render(expression.base, 100, latex=latex, number_format=number_format)
        return f"{base}[{indices}]"
    if isinstance(expression, Unary):
        operand = render(expression.operand, 40, latex=latex, number_format=number_format)
        text = f"{expression.operator}{operand}"
        return f"({text})" if parent_precedence > 40 else text
    if isinstance(expression, Binary):
        precedence = PRECEDENCE[expression.operator]
        left = render(expression.left, precedence, latex=latex, number_format=number_format)
        right_precedence = precedence + (0 if expression.operator == "**" else 1)
        right_text = render(
            expression.right,
            right_precedence,
            right=True,
            latex=latex,
            number_format=number_format,
        )
        if latex and expression.operator == "/":
            text = f"\\frac{{{left}}}{{{right_text}}}"
        elif latex and expression.operator == "**":
            text = f"{left}^{{{right_text}}}"
        else:
            operator = (
                r" \times "
                if latex and expression.operator == "*"
                else f" {expression.operator} "
            )
            text = f"{left}{operator}{right_text}"
        return f"({text})" if precedence < parent_precedence else text
    if isinstance(expression, Function):
        arguments = ", ".join(
            render(arg, latex=latex, number_format=number_format)
            for arg in expression.arguments
        )
        if latex and expression.name == "sqrt":
            return f"\\sqrt{{{arguments}}}"
        if latex and expression.name == "abs":
            return f"\\left|{arguments}\\right|"
        if latex:
            return f"{expression.name}\\left({arguments}\\right)"
        return f"{expression.name}({arguments})"
    if isinstance(expression, Reduction):
        indices = ", ".join(str(index) for index in expression.indices)
        operand = render(expression.operand, latex=latex, number_format=number_format)
        return f"\\sum_{{{indices}}} {operand}" if latex else f"sum[{indices}]({operand})"
    if isinstance(expression, Aggregate):
        relation = render(expression.relation, latex=latex, number_format=number_format)
        operand = render(expression.operand, latex=latex, number_format=number_format)
        return f"aggr({relation}, {operand})"
    if isinstance(expression, RelationLift):
        name = "targ" if expression.role == "target" else "sour"
        arguments = [expression.operand]
        if expression.relation is not None:
            arguments.insert(0, expression.relation)
        rendered = ", ".join(
            render(arg, latex=latex, number_format=number_format) for arg in arguments
        )
        return f"{name}({rendered})"
    raise TypeError(f"Cannot render {type(expression).__name__}.")
