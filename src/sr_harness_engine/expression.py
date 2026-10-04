"""Expression nodes for the SRHarness symbolic engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping


class Expression:
    """Base class of every symbolic expression node."""

    def __add__(self, other: Any) -> Expression:
        return Binary("+", self, as_expression(other))

    def __radd__(self, other: Any) -> Expression:
        return Binary("+", as_expression(other), self)

    def __sub__(self, other: Any) -> Expression:
        return Binary("-", self, as_expression(other))

    def __rsub__(self, other: Any) -> Expression:
        return Binary("-", as_expression(other), self)

    def __mul__(self, other: Any) -> Expression:
        return Binary("*", self, as_expression(other))

    def __rmul__(self, other: Any) -> Expression:
        return Binary("*", as_expression(other), self)

    def __truediv__(self, other: Any) -> Expression:
        return Binary("/", self, as_expression(other))

    def __rtruediv__(self, other: Any) -> Expression:
        return Binary("/", as_expression(other), self)

    def __pow__(self, other: Any) -> Expression:
        return Binary("**", self, as_expression(other))

    def __rpow__(self, other: Any) -> Expression:
        return Binary("**", as_expression(other), self)

    def __neg__(self) -> Expression:
        return Unary("-", self)

    def __getitem__(self, index: Index | tuple[Index, ...] | str) -> Expression:
        if isinstance(index, tuple):
            indices = tuple(as_index(item) for item in index)
        else:
            indices = (as_index(index),)
        return Indexed(self, indices)

    def evaluate(
        self,
        values: Mapping[str, Any] | None = None,
        *,
        parameters: Mapping[str, Any] | None = None,
        time: Any = None,
        delay_resolver: Any = None,
    ) -> Any:
        """Evaluate this expression with NumPy values."""
        from .evaluation import evaluate

        return evaluate(
            self,
            values=values,
            parameters=parameters,
            time=time,
            delay_resolver=delay_resolver,
        )

    eval = evaluate

    @property
    def operands(self) -> tuple[Expression, ...]:
        """Child expressions, exposed as an immutable tuple."""
        from .tree import children

        return children(self)

    def iter_preorder(self):
        """Yield this node followed by its descendants."""
        from .tree import iter_preorder

        yield from iter_preorder(self)

    def iter_postorder(self):
        """Yield descendants followed by this node."""
        from .tree import iter_postorder

        yield from iter_postorder(self)

    def copy(self) -> Expression:
        return deepcopy(self)

    def replace(self, old: Expression, new: Expression, **_: Any) -> Expression:
        """Return a tree in which the exact *old* node is replaced by *new*."""
        from .tree import replace

        return replace(self, old, new)

    def to_str(
        self,
        *,
        latex: bool = False,
        number_format: str = "",
        **_: Any,
    ) -> str:
        from .render import render

        return render(self, latex=latex, number_format=number_format)

    def to_tree(self, *, number_format: str = "", **_: Any) -> str:
        """Render a compact preorder tree for diagnostics."""
        from .tree import children

        lines = [self.to_str(number_format=number_format)]

        def visit(node: Expression, prefix: str, connector: str) -> None:
            lines.append(f"{prefix}{connector}{node.to_str(number_format=number_format)}")
            values = children(node)
            continuation = "  " if connector == "└ " else "│ "
            for index, child in enumerate(values):
                last = index == len(values) - 1
                visit(child, prefix + continuation, "└ " if last else "├ ")

        values = children(self)
        for index, child in enumerate(values):
            visit(child, "", "└ " if index == len(values) - 1 else "├ ")
        return "\n".join(lines)

    def __len__(self) -> int:
        return sum(1 for _ in self.iter_preorder())

    def fit(
        self,
        values: Mapping[str, Any],
        target: Any,
        *,
        initial: Mapping[str, Any] | None = None,
        method: str = "BFGS",
        options: Mapping[str, Any] | None = None,
    ):
        """Fit named and grouped parameters against a target array."""
        from .optimize import fit

        return fit(self, values, target, initial=initial, method=method, options=options)

    def __str__(self) -> str:
        from .render import render

        return render(self)


@dataclass(frozen=True, slots=True)
class Number(Expression):
    value: int | float


@dataclass(frozen=True, slots=True)
class Symbol(Expression):
    name: str
    value: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class Parameter(Expression):
    name: str
    value: float | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class GroupedParameter(Expression):
    by: Expression
    name: str | None = None
    value: Mapping[Any, float] | None = field(default=None, compare=False)
    default: float | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class Index:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Unary(Expression):
    operator: str
    operand: Expression


@dataclass(frozen=True, slots=True)
class Binary(Expression):
    operator: str
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class Function(Expression):
    name: str
    arguments: tuple[Expression, ...]


@dataclass(frozen=True, slots=True)
class Indexed(Expression):
    base: Expression
    indices: tuple[Index, ...]


@dataclass(frozen=True, slots=True)
class Reduction(Expression):
    indices: tuple[Index, ...]
    operand: Expression


@dataclass(frozen=True, slots=True)
class Aggregate(Expression):
    relation: Expression
    operand: Expression


@dataclass(frozen=True, slots=True)
class RelationLift(Expression):
    role: str
    operand: Expression
    relation: Expression | None = None


def as_expression(value: Any) -> Expression:
    if isinstance(value, Expression):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected an expression or number, got {type(value).__name__}.")
    return Number(value)


Variable = Symbol


def as_index(value: Index | str) -> Index:
    if isinstance(value, Index):
        return value
    if isinstance(value, str) and value.isidentifier():
        return Index(value)
    raise TypeError(f"Expected an index name, got {value!r}.")


def param(name: str, value: float | None = None) -> Parameter:
    return Parameter(name, value)


def grouped_param(
    by: Expression,
    *,
    name: str | None = None,
    value: Mapping[Any, float] | None = None,
    default: float | None = None,
) -> GroupedParameter:
    return GroupedParameter(by, name=name, value=value, default=default)


def function(name: str, *arguments: Any) -> Function:
    return Function(name, tuple(as_expression(argument) for argument in arguments))


def reduction(indices: Index | tuple[Index, ...], operand: Any) -> Reduction:
    if not isinstance(indices, tuple):
        indices = (indices,)
    return Reduction(tuple(as_index(index) for index in indices), as_expression(operand))


def aggr(relation: Any, operand: Any) -> Aggregate:
    return Aggregate(as_expression(relation), as_expression(operand))


def targ(*arguments: Any) -> RelationLift:
    if len(arguments) == 1:
        return RelationLift("target", as_expression(arguments[0]))
    if len(arguments) == 2:
        return RelationLift("target", as_expression(arguments[1]), as_expression(arguments[0]))
    raise TypeError("targ expects targ(value) or targ(relation, value).")


def sour(*arguments: Any) -> RelationLift:
    if len(arguments) == 1:
        return RelationLift("source", as_expression(arguments[0]))
    if len(arguments) == 2:
        return RelationLift("source", as_expression(arguments[1]), as_expression(arguments[0]))
    raise TypeError("sour expects sour(value) or sour(relation, value).")
