"""Safe parser for the SRHarness symbolic model language."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any

from .analysis import fold_constants
from .desugar import desugar
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
    Unary,
)


FUNCTIONS = {
    "abs", "arccos", "arcsin", "arctan", "cos", "cosh", "cot", "csc", "exp", "inv",
    "log", "log10", "sec", "sech", "sigmoid", "sign", "sin", "sinh", "sqrt", "tan",
    "tanh", "pow2", "pow3",
}
BINARY_FUNCTIONS = {"max", "min"}


class ExpressionParser(ast.NodeVisitor):
    """Convert a restricted Python expression AST into engine nodes."""

    def __init__(self, symbols: Mapping[str, Any] | None = None):
        self.symbols = {}
        for name, value in (symbols or {}).items():
            if isinstance(value, Expression):
                self.symbols[name] = value
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Symbol override {name!r} must be an expression or number.")
            else:
                self.symbols[name] = Number(value)

    def parse(self, source: str) -> Expression:
        """Parse input into the canonical representation.

        Args:
            source: Source text to parse.

        Returns:
            The parsed, desugared, and constant-folded expression.
        """
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as error:
            raise SyntaxError(f"Invalid symbolic expression: {error.msg}.") from error
        return fold_constants(desugar(self.visit(tree.body)))

    def generic_visit(self, node: ast.AST):
        """Reject syntax that is outside the symbolic language.

        Args:
            node: Expression or syntax-tree node.
        """
        raise ValueError(f"Unsupported syntax: {type(node).__name__}.")

    def visit_Constant(self, node: ast.Constant) -> Expression:
        """Convert a numeric literal into a number node.

        Args:
            node: Expression or syntax-tree node.

        Returns:
            A numeric expression node.
        """
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"Only numerical literals are expressions, got {node.value!r}.")
        return Number(node.value)

    def visit_Name(self, node: ast.Name) -> Expression:
        """Resolve a name to a supplied or newly created symbol.

        Args:
            node: Expression or syntax-tree node.

        Returns:
            The expression associated with the name.
        """
        return self.symbols.setdefault(node.id, Symbol(node.id))

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Expression:
        """Convert a supported unary operator.

        Args:
            node: Expression or syntax-tree node.

        Returns:
            The converted operand or unary expression.
        """
        if isinstance(node.op, ast.USub):
            return Unary("-", self.visit(node.operand))
        if isinstance(node.op, ast.UAdd):
            return self.visit(node.operand)
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}.")

    def visit_BinOp(self, node: ast.BinOp) -> Expression:
        """Convert a supported binary arithmetic operator.

        Args:
            node: Expression or syntax-tree node.

        Returns:
            A binary expression node.
        """
        operators = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"
        }
        operator = operators.get(type(node.op))
        if operator is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}.")
        return Binary(operator, self.visit(node.left), self.visit(node.right))

    def visit_Subscript(self, node: ast.Subscript) -> Expression:
        """Convert symbolic indexing such as ``x[i]``.

        Args:
            node: Expression or syntax-tree node.

        Returns:
            An indexed expression node.
        """
        base = self.visit(node.value)
        return Indexed(base, self._indices(node.slice))

    def visit_Call(self, node: ast.Call) -> Expression:
        """Convert an approved symbolic function or reduction call.

        Args:
            node: Expression or syntax-tree node.

        Returns:
            The corresponding symbolic expression.
        """
        if isinstance(node.func, ast.Subscript) and self._name(node.func.value) == "sum":
            if node.keywords or len(node.args) not in {1, 2}:
                raise ValueError(
                    "sum[index](...) expects an operand, optionally preceded by a relation."
                )
            arguments = [self.visit(argument) for argument in node.args]
            relation, operand = (None, arguments[0]) if len(arguments) == 1 else arguments
            return Reduction(self._indices(node.func.slice), operand, relation)

        name = self._name(node.func)
        if name in FUNCTIONS:
            if node.keywords or len(node.args) != 1:
                raise ValueError(f"{name}(...) expects exactly one positional argument.")
            return Function(name, (self.visit(node.args[0]),))
        if name in BINARY_FUNCTIONS:
            if node.keywords or len(node.args) != 2:
                raise ValueError(f"{name}(...) expects exactly two positional arguments.")
            return Function(name, tuple(self.visit(argument) for argument in node.args))
        if name == "delay":
            if node.keywords or len(node.args) != 2:
                raise ValueError("delay(value, delta) expects exactly two arguments.")
            return Function(name, tuple(self.visit(argument) for argument in node.args))
        if name == "gather":
            if node.keywords or len(node.args) != 2:
                raise ValueError("gather(relation, expression) expects exactly two arguments.")
            return Gather(self.visit(node.args[0]), self.visit(node.args[1]))
        if name == "Number":
            if node.keywords or len(node.args) != 1:
                raise ValueError("Number(...) expects exactly one numerical literal.")
            return self.visit(node.args[0])
        if name == "param":
            return self._parameter(node)
        if name == "grouped_param":
            return self._grouped_parameter(node)
        if name == "aggr":
            if node.keywords or len(node.args) not in {1, 2}:
                raise ValueError(
                    "aggr expects aggr(relation * expression) or "
                    "aggr(relation, expression)."
                )
            if len(node.args) == 2:
                return Aggregate(self.visit(node.args[0]), self.visit(node.args[1]))
            relation, operand = _split_aggregation(self.visit(node.args[0]))
            return Aggregate(relation, operand)
        if name in {"targ", "sour"}:
            if node.keywords or len(node.args) not in {1, 2}:
                raise ValueError(f"{name} expects one value, optionally preceded by a relation.")
            arguments = [self.visit(argument) for argument in node.args]
            relation, operand = (None, arguments[0]) if len(arguments) == 1 else arguments
            return RelationLift("target" if name == "targ" else "source", operand, relation)
        raise ValueError(f"Unknown symbolic function: {name!r}.")

    def _parameter(self, node: ast.Call) -> Parameter:
        if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant):
            raise ValueError("param(...) requires a literal string name.")
        name = node.args[0].value
        if not isinstance(name, str) or not name:
            raise ValueError("Parameter names must be non-empty strings.")
        keywords = self._literal_keywords(node, {"value"})
        value = keywords.get("value")
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError("A parameter initial value must be numerical.")
        return Parameter(name, value)

    def _grouped_parameter(self, node: ast.Call) -> GroupedParameter:
        if len(node.args) != 1:
            raise ValueError("grouped_param(...) requires one grouping expression.")
        keywords = self._literal_keywords(node, {"name", "value", "default"})
        name = keywords.get("name")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("A grouped parameter name must be a non-empty string.")
        value = keywords.get("value")
        if value is not None and not isinstance(value, dict):
            raise ValueError("grouped_param value must be a category-to-number dictionary.")
        default = keywords.get("default")
        return GroupedParameter(self.visit(node.args[0]), name=name, value=value, default=default)

    @staticmethod
    def _name(node: ast.AST) -> str:
        if not isinstance(node, ast.Name):
            raise ValueError("Only direct symbolic function calls are allowed.")
        return node.id

    @staticmethod
    def _indices(node: ast.AST) -> tuple[Index, ...]:
        nodes = node.elts if isinstance(node, ast.Tuple) else (node,)
        if not nodes or not all(isinstance(item, ast.Name) for item in nodes):
            raise ValueError("Indices must be identifiers such as i, j, or k.")
        return tuple(Index(item.id) for item in nodes)

    @staticmethod
    def _literal_keywords(node: ast.Call, allowed: set[str]) -> dict[str, Any]:
        result = {}
        for keyword in node.keywords:
            if keyword.arg not in allowed:
                raise ValueError(f"Unexpected keyword argument: {keyword.arg!r}.")
            result[keyword.arg] = ast.literal_eval(keyword.value)
        return result


def parse(
    source: str,
    symbols: Mapping[str, Any] | None = None,
    *,
    variables: Mapping[str, Any] | None = None,
) -> Expression:
    """Parse *source* without using ``eval`` or executing user code.

    Args:
        source: Source text to parse.
        symbols: Optional predefined symbols or numeric constants.
        variables: Deprecated-compatible alias for ``symbols``.

    Returns:
        The parsed canonical expression.
    """
    if symbols is not None and variables is not None:
        raise TypeError("Use either symbols or variables, not both.")
    symbols = variables if variables is not None else symbols
    return ExpressionParser(symbols).parse(source)


def _split_aggregation(expression: Expression) -> tuple[Expression, Expression]:
    factors = _multiplication_factors(expression)
    if len(factors) < 2 or not isinstance(factors[0], Symbol):
        raise ValueError(
            "aggr(relation * expression) requires the relation symbol as its first factor."
        )
    operand = factors[1]
    for factor in factors[2:]:
        operand = Binary("*", operand, factor)
    return factors[0], operand


def _multiplication_factors(expression: Expression) -> list[Expression]:
    if isinstance(expression, Binary) and expression.operator == "*":
        return _multiplication_factors(expression.left) + _multiplication_factors(expression.right)
    return [expression]
