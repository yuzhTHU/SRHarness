"""Immutable expression-tree traversal and transformation helpers."""

from __future__ import annotations

from dataclasses import replace as dataclass_replace

from .expression import (
    Aggregate,
    Binary,
    Expression,
    Function,
    GroupedParameter,
    Indexed,
    Reduction,
    RelationLift,
    Unary,
)


def children(node: Expression) -> tuple[Expression, ...]:
    """Return the direct child expressions of a node.

    Args:
        node: Expression or syntax-tree node.

    Returns:
        The node's direct children in structural order.
    """
    if isinstance(node, Unary):
        return (node.operand,)
    if isinstance(node, Binary):
        return (node.left, node.right)
    if isinstance(node, Function):
        return node.arguments
    if isinstance(node, Indexed):
        return (node.base,)
    if isinstance(node, Reduction):
        return (node.operand,) if node.relation is None else (node.relation, node.operand)
    if isinstance(node, Aggregate):
        return (node.relation, node.operand)
    if isinstance(node, RelationLift):
        return (node.operand,) if node.relation is None else (node.relation, node.operand)
    if isinstance(node, GroupedParameter):
        return (node.by,)
    return ()


def with_children(node: Expression, values: tuple[Expression, ...]) -> Expression:
    """Return a copy of a node with new direct children.

    Args:
        node: Expression or syntax-tree node.
        values: Replacement children in structural order.

    Returns:
        The rebuilt expression node.
    """
    if len(values) != len(children(node)):
        raise ValueError(f"Expected {len(children(node))} children, got {len(values)}.")
    if isinstance(node, Unary):
        return dataclass_replace(node, operand=values[0])
    if isinstance(node, Binary):
        return dataclass_replace(node, left=values[0], right=values[1])
    if isinstance(node, Function):
        return dataclass_replace(node, arguments=values)
    if isinstance(node, Indexed):
        return dataclass_replace(node, base=values[0])
    if isinstance(node, Reduction):
        if node.relation is None:
            return dataclass_replace(node, operand=values[0])
        return dataclass_replace(node, relation=values[0], operand=values[1])
    if isinstance(node, Aggregate):
        return dataclass_replace(node, relation=values[0], operand=values[1])
    if isinstance(node, RelationLift):
        if node.relation is None:
            return dataclass_replace(node, operand=values[0])
        return dataclass_replace(node, relation=values[0], operand=values[1])
    if isinstance(node, GroupedParameter):
        return dataclass_replace(node, by=values[0])
    return node


def iter_preorder(node: Expression):
    """Yield an expression tree in preorder.

    Args:
        node: Expression or syntax-tree node.

    Yields:
        The node followed recursively by its children.
    """
    yield node
    for child in children(node):
        yield from iter_preorder(child)


def iter_postorder(node: Expression):
    """Yield an expression tree in postorder.

    Args:
        node: Expression or syntax-tree node.

    Yields:
        Descendants followed by their parent node.
    """
    for child in children(node):
        yield from iter_postorder(child)
    yield node


def transform(node: Expression, function) -> Expression:
    """Apply a bottom-up transformation to an expression tree.

    Args:
        node: Expression or syntax-tree node.
        function: Transformation callback.

    Returns:
        The transformed expression tree.
    """
    transformed = with_children(node, tuple(transform(child, function) for child in children(node)))
    return function(transformed)


def replace(node: Expression, old: Expression, new: Expression) -> Expression:
    """Replace a node by object identity throughout a tree.

    Args:
        node: Expression or syntax-tree node.
        old: Existing expression node to replace.
        new: Replacement expression node.

    Returns:
        The rebuilt expression tree.
    """
    if node is old:
        return new
    return with_children(node, tuple(replace(child, old, new) for child in children(node)))


def replace_at_path(node: Expression, path: tuple[int, ...], new: Expression) -> Expression:
    """Replace the node at a tuple of child indices.

    Args:
        node: Expression or syntax-tree node.
        path: Child indices from the root to the target node.
        new: Replacement expression node.

    Returns:
        The rebuilt expression tree.
    """
    if not path:
        return new
    values = list(children(node))
    index = path[0]
    values[index] = replace_at_path(values[index], path[1:], new)
    return with_children(node, tuple(values))
