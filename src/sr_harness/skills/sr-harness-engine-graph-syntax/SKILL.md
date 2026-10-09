---
name: sr-harness-engine-graph-syntax
description: Use this skill before writing, evaluating, or submitting SRHarness Engine formulas for graphs, networks, hypergraphs, node interactions, relation reductions, edge or hyperedge fields, gather operations, or message passing.
---

# Write graph and hypergraph formulas

Read `sr-harness-engine-syntax` first for ordinary arithmetic, functions, fitted parameters, and general parser restrictions. This skill covers the indexed structural language used for graph and hypergraph models.

Prefer structured Engine formulas over `evaluate_code`: indexed formulas retain interpretable structure, support parameter fitting and diagnostics, have measurable symbolic complexity, and can appear naturally in TopK and Pareto views.

## Indexed structural expressions

An expression containing symbolic indices such as `[i]`, `[i, j]`, or `[i, j, k]` uses the final array dimension as a structural dimension. Leading dimensions use ordinary element-wise broadcasting.

Indices must be symbolic identifiers; integer indices and slices such as `x[0]` or `x[0:10]` are invalid. In `theta[i]`, `i` names a node dimension rather than selecting the element stored in a Python variable named `i`.

The runtime must provide `num_nodes=N`. It is never inferred from a relation because isolated nodes may not occur in any relation row.

Typical value shapes are:

```text
node field:       (..., N)
edge field:       (..., E)
hyperedge field:  (..., H)
scalar field:     (..., 1)
binary relation:  (E, 2)
ternary relation: (H, 3)
```

Relations must be declared as relations in `context.data`. Do not infer relation semantics from names such as `A` or `T`.

Once an expression uses symbolic indices, index every non-singleton structural variable. Thus `x + x[i]` is invalid when `x.shape[-1] == N`; use `x[i] + x[i]`. A final singleton dimension may remain unindexed and broadcast naturally.

Free indices become dense trailing output axes in first-appearance order:

```text
x[i]                         -> (..., N)
x[i] + x[j]                  -> (..., N, N)
x[i] * x[j] * x[k]           -> (..., N, N, N)
```

Multiple free indices are legal, but their cost grows as `N**r`. Avoid dense multi-index intermediates when a relation-restricted reduction or `gather` can request only needed components.

## Relations and `sum[...]`

A binary relation stores rows in `(target, source)` order. Therefore:

```text
A[i, j]
```

binds `i` to the target column and `j` to the source column. A ternary relation stores `(target, source1, source2)` and is written `T[i, j, k]`.

Use `sum[index](operand)` to reduce a named structural index:

```text
sum[j](x[i] * x[j])
sum[j, k](x[i] * x[j] * x[k])
```

If the operand does not contain a reduced index, summing over that index multiplies the value by `N`.

Use the two-argument form to restrict the reduction to a sparse relation:

```text
sum[j](A[i, j], x[j] - x[i])
sum[j, k](T[i, j, k], x[i] * x[j] * x[k])
```

Mathematically, `sum[j](A[i, j], eq)` means `sum[j](A[i, j] * eq)`. The two-argument form is preferable because the evaluator can compute only relation-supported components instead of materializing a dense `N x N` tensor.

An edge-aligned field with shape `(..., E)` uses the same complete relation indices:

```text
sum[j](A[i, j], w[i, j] * sin(theta[j] - theta[i]))
```

A hyperedge-aligned field with shape `(..., H)` similarly uses all three indices. When several compatible relations make field association ambiguous, the data context must explicitly associate the field with its relation; changing the formula cannot resolve ambiguous data metadata.

Relation expressions are numerical sparse tensors and support ordinary algebra:

```text
sum[j](A1[i, j] + A2[i, j], x[j])
sum[j](A1[i, j] * A2[i, j], x[j])
sum[j](A1[i, j] + 0.5 * A2[i, j], x[j])
```

The first expression adds overlapping relation values; it is not merely a support union.

Do not bind the same index again inside its active scope. This is invalid:

```text
sum[j](A[i, j], sum[j](x[j]))
```

Rename the inner index, for example to `k`.

## Network and hypergraph examples

Unnormalized Kuramoto-like dynamics:

```text
omega[i] + param('K') * sum[j](A[i, j], sin(theta[j] - theta[i]))
```

Degree-normalized interaction:

```text
omega[i] + param('K') * sum[j](A[i, j], sin(theta[j] - theta[i])) / sum[j](A[i, j], 1)
```

Node, edge-weighted, and hypergraph terms in one RHS:

```text
(param('a') * x[i]
+ param('b') * sum[j](A[i, j], w[i, j] * tanh(x[j] - x[i]))
+ param('c') * sum[j, k](T[i, j, k], x[j] * x[k]))
```

Use parentheses when a denominator might be zero and inspect the data before submitting singular expressions.

## `gather`: produce relation-aligned values

`gather(relation, expression)` selects an indexed expression at the relation's stored coordinates and multiplies by relation values:

```text
gather(A[i, j], x[i] + x[j])          -> (..., E)
gather(T[i, j, k], x[i] * x[j] * x[k]) -> (..., H)
```

The operand may use only a subset of relation indices; missing indices broadcast:

```text
gather(A[i, j], x[i])
```

The result behaves like an ordinary edge or hyperedge field:

```text
2 * gather(A[i, j], x[i] + x[j])
sin(gather(A[i, j], x[i] + x[j]))
gather(A[i, j], x[i] + x[j]) + w
```

It can be indexed again with the same complete relation indices inside another structural expression:

```text
sum[j](A[i, j], gather(A[i, j], x[i] + x[j])[i, j] * x[j])
```

Use `sum[...]` when the desired output is node-aligned after reduction. Use `gather(...)` when the desired output is relation-row-aligned.

## `aggr`, `targ`, and `sour` convenience syntax

For ordinary binary message passing, these forms are accepted:

```text
aggr(A * targ(x) * sour(x))
aggr(A, targ(x) * sour(x))
aggr(A, targ(A, x) * sour(A, x))
```

They compile to canonical indexed syntax:

```text
sum[j](A[i, j], x[i] * x[j])
```

`targ` reads the target endpoint and `sour` reads the source endpoint. They are only valid inside `aggr`. Nested `aggr` is legal because the desugaring pass generates fresh indices automatically. Prefer explicit indexed syntax for hypergraphs, unusual reductions, and formulas that need precise index control.

## Graph and hypergraph checklist

Before calling `evaluate_formula` or `submit_formula`:

1. Confirm that `context.data` defines `num_nodes` explicitly.
2. Confirm relation metadata, endpoint order, structural-axis lengths, and any edge/hyperedge field association.
3. Index every non-singleton structural variable and keep reduction indices distinct across nested scopes.
4. Prefer relation-restricted `sum` over dense `N**r` expressions.
5. Use `gather` when relation-aligned output or an intermediate edge/hyperedge field is intended.
6. Inspect finite coverage, residual diagnostics, and train/validation metrics before submitting the formula.
