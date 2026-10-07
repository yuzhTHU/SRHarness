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
        num_nodes: int | None = None,
    ) -> Any:
        """Evaluate this expression with NumPy values.

        Args:
            values: Values keyed by symbol name, or a context exposing ``data``
                and optional ``num_nodes`` metadata.
            parameters: Fitted parameter values keyed by parameter name.
            time: Optional sample times.
            delay_resolver: Optional callback that resolves delayed values.
            num_nodes: Explicit node count for indexed expressions.

        Returns:
            The evaluated scalar or array.
        """
        from .context import resolve_context
        from .evaluation import evaluate

        values, _, num_nodes = resolve_context(values, num_nodes=num_nodes)

        return evaluate(
            self,
            values=values,
            parameters=parameters,
            time=time,
            delay_resolver=delay_resolver,
            num_nodes=num_nodes,
        )

    eval = evaluate

    @property
    def operands(self) -> tuple[Expression, ...]:
        """Child expressions, exposed as an immutable tuple.

        Returns:
            The direct child nodes in expression order.
        """
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
        """Return an independent copy.

        Returns:
            A deep copy of this expression.
        """
        return deepcopy(self)

    def replace(self, old: Expression, new: Expression, **_: Any) -> Expression:
        """Return a tree in which the exact *old* node is replaced by *new*.

        Args:
            old: Existing expression node to replace.
            new: Replacement expression node.
            **_: Ignored compatibility options.

        Returns:
            A copied expression tree with matching nodes replaced.
        """
        from .tree import replace

        return replace(self, old, new)

    def to_str(
        self,
        *,
        latex: bool = False,
        number_format: str = "",
        **_: Any,
    ) -> str:
        """Render the expression as plain text or LaTeX.

        Args:
            latex: Whether to render LaTeX notation.
            number_format: Format specification for numeric literals.
            **_: Ignored compatibility options.

        Returns:
            The rendered expression.
        """
        from .render import render

        return render(self, latex=latex, number_format=number_format)

    def to_tree(self, *, number_format: str = "", **_: Any) -> str:
        """Render a compact preorder tree for diagnostics.

        Args:
            number_format: Format specification for numeric literals.
            **_: Ignored compatibility options.

        Returns:
            A multiline representation of the expression tree.
        """
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
        target: Any = None,
        *,
        initial: Mapping[str, Any] | None = None,
        method: str = "BFGS",
        options: Mapping[str, Any] | None = None,
        num_nodes: int | None = None,
    ):
        """Fit named and grouped parameters against a target array.

        Args:
            values: Values keyed by symbol name, or a context exposing ``data``,
                ``target``, and optional ``num_nodes`` metadata.
            target: Target name or target values. Omit when supplied by context.
            initial: Optional initial parameter values.
            method: Optimization method name.
            options: Optional optimizer settings.
            num_nodes: Explicit node count for indexed expressions.

        Returns:
            The fitted expression, parameter values, predictions, and loss.
        """
        from .context import resolve_context
        from .optimize import fit

        values, target, num_nodes = resolve_context(values, target, num_nodes)
        if target is None:
            raise ValueError("A target array or a context target is required for fitting.")

        return fit(
            self, values, target, initial=initial, method=method,
            options=options, num_nodes=num_nodes,
        )

    def fold_constants(self) -> Expression:
        """Return a copy with closed numerical subexpressions evaluated.

        Returns:
            A simplified expression with closed numeric branches folded.
        """
        from .analysis import fold_constants

        return fold_constants(self)

    def count_parameters(
        self,
        values: Mapping[str, Any] | None = None,
        *,
        parameters: Mapping[str, Any] | None = None,
    ) -> int:
        """Count independent fitted values represented by this expression.

        Args:
            values: Values keyed by symbol name.
            parameters: Fitted parameter values keyed by parameter name.

        Returns:
            The number of independent scalar parameter values.
        """
        from .analysis import count_parameters

        return count_parameters(self, values, parameters=parameters)

    def __str__(self) -> str:
        from .render import render

        return render(self)


@dataclass(frozen=True, slots=True)
class Number(Expression):
    """Fixed numeric literal."""
    value: int | float


@dataclass(frozen=True, slots=True)
class Symbol(Expression):
    """Named input symbol with an optional bound value."""
    name: str
    value: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class Parameter(Expression):
    """Named scalar parameter optimized during fitting."""
    name: str
    value: float | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class GroupedParameter(Expression):
    """Parameter with one fitted value per category."""
    by: Expression
    name: str | None = None
    value: Mapping[Any, float] | None = field(default=None, compare=False)
    default: float | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class Index:
    """Symbolic relation index."""
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class Unary(Expression):
    """Unary expression node."""
    operator: str
    operand: Expression


@dataclass(frozen=True, slots=True)
class Binary(Expression):
    """Binary expression node."""
    operator: str
    left: Expression
    right: Expression


@dataclass(frozen=True, slots=True)
class Function(Expression):
    """Named function-call expression node."""
    name: str
    arguments: tuple[Expression, ...]


@dataclass(frozen=True, slots=True)
class Indexed(Expression):
    """Expression annotated with symbolic indices."""
    base: Expression
    indices: tuple[Index, ...]


@dataclass(frozen=True, slots=True)
class Reduction(Expression):
    """Sum reduction with an optional relation binder."""
    indices: tuple[Index, ...]
    operand: Expression
    relation: Expression | None = None


@dataclass(frozen=True, slots=True)
class Gather(Expression):
    """Weighted collection of a structural expression on relation entries."""
    relation: Expression
    operand: Expression


@dataclass(frozen=True, slots=True)
class Aggregate(Expression):
    """Convenience aggregation node lowered to indexed syntax."""
    relation: Expression
    operand: Expression


@dataclass(frozen=True, slots=True)
class RelationLift(Expression):
    """Convenience source or target projection used inside an aggregation."""
    role: str
    operand: Expression
    relation: Expression | None = None


def as_expression(value: Any) -> Expression:
    """Convert a numeric literal or expression into an expression node.

    Args:
        value: Existing expression or numeric literal.

    Returns:
        The corresponding expression node.
    """
    if isinstance(value, Expression):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Expected an expression or number, got {type(value).__name__}.")
    return Number(value)


Variable = Symbol


def as_index(value: Index | str) -> Index:
    """Convert an index name into an index node.

    Args:
        value: Existing index or valid Python identifier.

    Returns:
        The corresponding symbolic index.
    """
    if isinstance(value, Index):
        return value
    if isinstance(value, str) and value.isidentifier():
        return Index(value)
    raise TypeError(f"Expected an index name, got {value!r}.")


def param(name: str, value: float | None = None) -> Parameter:
    """Create a named scalar parameter.

    Args:
        name: Parameter name shared by all matching occurrences.
        value: Optional initial or fixed value.

    Returns:
        A symbolic scalar parameter.
    """
    return Parameter(name, value)


def grouped_param(
    by: Expression,
    *,
    name: str | None = None,
    value: Mapping[Any, float] | None = None,
    default: float | None = None,
) -> GroupedParameter:
    """Create a parameter with one fitted value per category.

    Args:
        by: Symbol or expression containing category labels.
        name: Optional parameter-map name.
        value: Optional initial values keyed by category.
        default: Value used for categories absent from ``value``.

    Returns:
        A category-dependent parameter expression.
    """
    return GroupedParameter(by, name=name, value=value, default=default)


def function(name: str, *arguments: Any) -> Function:
    """Create a supported symbolic function call.

    Args:
        name: Function name recognized by the evaluator.
        *arguments: Function operands.

    Returns:
        A symbolic function node.
    """
    return Function(name, tuple(as_expression(argument) for argument in arguments))


def reduction(
    indices: Index | tuple[Index, ...],
    operand: Any,
    relation: Any = None,
) -> Reduction:
    """Create an indexed sum reduction.

    Args:
        indices: Symbolic indices.
        operand: Expression being reduced or transformed.
        relation: Relation expression that binds symbolic indices.

    Returns:
        A symbolic reduction node.
    """
    if not isinstance(indices, tuple):
        indices = (indices,)
    return Reduction(
        tuple(as_index(index) for index in indices),
        as_expression(operand),
        None if relation is None else as_expression(relation),
    )


def gather(relation: Any, operand: Any) -> Gather:
    """Collect structural values at the nonzero entries of a relation.

    Args:
        relation: Indexed relation or relation-valued expression.
        operand: Structural expression evaluated at relation coordinates.

    Returns:
        A relation-aligned symbolic field.
    """
    return Gather(as_expression(relation), as_expression(operand))


def aggr(relation: Any, operand: Any = None) -> Expression:
    """Build and lower target-wise graph aggregation syntax.

    Args:
        relation: Edge relation, or a product containing it when ``operand`` is omitted.
        operand: Message expression to aggregate by target node.

    Returns:
        The equivalent canonical indexed reduction.
    """
    from .desugar import desugar
    from .parser import _split_aggregation

    if operand is None:
        relation, operand = _split_aggregation(as_expression(relation))
    return desugar(Aggregate(as_expression(relation), as_expression(operand)))


def targ(*arguments: Any) -> RelationLift:
    """Project node values onto relation targets inside ``aggr``.

    Args:
        *arguments: Either ``value`` or ``relation, value``.

    Returns:
        A target projection used by aggregation desugaring.
    """
    if len(arguments) == 1:
        return RelationLift("target", as_expression(arguments[0]))
    if len(arguments) == 2:
        return RelationLift("target", as_expression(arguments[1]), as_expression(arguments[0]))
    raise TypeError("targ expects targ(value) or targ(relation, value).")


def sour(*arguments: Any) -> RelationLift:
    """Project node values onto relation sources inside ``aggr``.

    Args:
        *arguments: Either ``value`` or ``relation, value``.

    Returns:
        A source projection used by aggregation desugaring.
    """
    if len(arguments) == 1:
        return RelationLift("source", as_expression(arguments[0]))
    if len(arguments) == 2:
        return RelationLift("source", as_expression(arguments[1]), as_expression(arguments[0]))
    raise TypeError("sour expects sour(value) or sour(relation, value).")
