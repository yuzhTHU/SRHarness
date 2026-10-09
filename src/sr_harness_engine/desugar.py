"""Lower convenience syntax to the engine's canonical indexed language."""

from __future__ import annotations

from .expression import Aggregate, Expression, Index, Indexed, Reduction, RelationLift
from .tree import children, iter_preorder, with_children


def desugar(expression: Expression) -> Expression:
    """Compile ``aggr/targ/sour`` nodes into indexed reductions.

        Edge lists use ``(target, source)`` column order. A convenience aggregation
        therefore becomes ``sum[j](A[i, j], ...)``: ``j`` is the source index
        being reduced and ``i`` is the surviving target index.

    Args:
        expression: Symbolic expression to process.

    Returns:
        The equivalent expression using indexed reductions.
    """
    used = _index_names(expression)
    result = _desugar(expression, used)
    dangling = next(
        (node for node in iter_preorder(result) if isinstance(node, RelationLift)),
        None,
    )
    if dangling is not None:
        name = "targ" if dangling.role == "target" else "sour"
        raise ValueError(f"{name}(...) must appear inside aggr(...).")
    return result


def _desugar(node: Expression, used: set[str]) -> Expression:
    if isinstance(node, Aggregate):
        relation = _desugar(node.relation, used)
        target = _fresh_index("i", used)
        source = _fresh_index("j", used)
        operand = _lower_lifts(node.operand, relation, target, source, used)
        relation_binding = Indexed(relation, (target, source))
        return Reduction((source,), operand, relation_binding)
    if isinstance(node, RelationLift):
        return node
    return with_children(node, tuple(_desugar(child, used) for child in children(node)))


def _lower_lifts(
    node: Expression,
    relation: Expression,
    target: Index,
    source: Index,
    used: set[str],
) -> Expression:
    if isinstance(node, Aggregate):
        return _desugar(node, used)
    if isinstance(node, RelationLift):
        if node.relation is not None:
            explicit_relation = _desugar(node.relation, used)
            if explicit_relation != relation:
                raise ValueError("targ/sour relation must match its enclosing aggr relation.")
        operand = _desugar(node.operand, used)
        index = target if node.role == "target" else source
        return Indexed(operand, (index,))
    return with_children(
        node,
        tuple(
            _lower_lifts(child, relation, target, source, used)
            for child in children(node)
        ),
    )


def _index_names(expression: Expression) -> set[str]:
    names = set()
    for node in iter_preorder(expression):
        if isinstance(node, (Indexed, Reduction)):
            names.update(index.name for index in node.indices)
    return names


def _fresh_index(preferred: str, used: set[str]) -> Index:
    if preferred not in used:
        used.add(preferred)
        return Index(preferred)
    suffix = 1
    while f"{preferred}{suffix}" in used:
        suffix += 1
    name = f"{preferred}{suffix}"
    used.add(name)
    return Index(name)
