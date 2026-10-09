---
name: sr-harness-engine-syntax
description: Use this skill before writing, evaluating, or submitting SRHarness Engine formulas, especially formulas with fitted parameters, categorical grouped parameters, delays, network or hypergraph indices, relation reductions, edge fields, or gather operations.
---

# Write SRHarness Engine formulas

Use SRHarness Engine syntax for candidate models whenever it can express the model. Prefer a structured formula over `evaluate_code`: engine formulas can be parsed, fitted, diagnosed, ranked by complexity, and presented in TopK and Pareto views.

Pass only the right-hand-side expression to `evaluate_formula` or `submit_formula`. For a model such as `y = a*x + b`, submit `param('a') * x + param('b')`; pass the target separately through the tool's `y` argument when it is not the configured target.

## Ordinary element-wise expressions

Use these arithmetic operators:

```text
+  -  *  /  **
```

They follow normal precedence and NumPy-style broadcasting. Fixed numerical literals are not fitted:

```text
x1 + x2
2.0 * sin(x1)
x1**2 + exp(-x2)
max(x1, x2) / (1 + abs(x3))
```

Available one-argument functions are:

```text
abs  arccos  arcsin  arctan  cos  cosh  cot  csc  exp  inv
log  log10  pow2  pow3  sec  sech  sigmoid  sign  sin  sinh
sqrt  tan  tanh
```

`min(a, b)` and `max(a, b)` accept exactly two arguments. Function names must be bare names: use `sin(x)`, never `np.sin(x)` or `numpy.sin(x)`.

## Fitted parameters

Declare every fitted scalar explicitly with `param`:

```text
param('a') * x + param('b')
param('alpha', value=0.3) * x / (1 - param('alpha'))
```

The optional `value` is an initial/default numerical value. Repeated declarations with the same name refer to one shared parameter and must not provide conflicting initial values. When calling `evaluate_formula`, set `fit=true` to optimize unbound parameters.

Do not invent a bare variable name for a coefficient. Write `param('a') * x`, not `a * x`. Bare names denote data variables in the symbolic language.

For a coefficient shared within each category and allowed to differ between categories, use `grouped_param`:

```text
grouped_param(category) * x
grouped_param(category, name='slope', default=0.0) * x
grouped_param(category, name='slope', value={'A': 1.0, 'B': 2.0}) * x
```

Do not one-hot encode a categorical variable merely to express category-specific coefficients.

## Indexed structural expressions

An expression containing symbolic indices such as `[i]`, `[i, j]`, or `[i, j, k]` uses the final array dimension as a structural dimension. Leading dimensions use ordinary element-wise broadcasting.

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

## Delays

Use:

```text
x + delay(x, delta)
```

`delay(value, delta)` reads the value at `t - delta`. The default evaluator interpolates along the first array axis and returns `NaN` outside available history. A specialized ODE or DDE evaluator may supply its own history resolver and integration protocol. The expression describes only the RHS; it does not choose an integrator or loss.

## Syntax that is not allowed

The parser does not execute Python. Do not use:

```text
y = x + 1                 # assignment; provide only the RHS
np.sin(x)                 # attribute access
x[0:10]                   # slices; indices must be symbolic names
custom_function(x)        # unregistered function
lambda x: x               # lambda
__import__('os')           # arbitrary Python
```

Indices are symbolic identifiers, not integer array positions. In `theta[i]`, `i` names a node dimension; it does not mean “take the element at a Python variable named i.”

## Evaluation checklist

Before calling `evaluate_formula` or `submit_formula`:

1. Use names that exist in `context.data`; declare coefficients with `param`.
2. Use `fit=true` if the formula contains unbound `param` or `grouped_param` nodes.
3. For indexed models, confirm `num_nodes`, relation metadata, endpoint order, and structural-axis lengths.
4. Index every non-singleton structural variable and keep reduction indices distinct across nested scopes.
5. Prefer relation-restricted `sum` over dense `N**r` expressions.
6. Use `gather` only when relation-aligned output or an intermediate edge/hyperedge field is intended.
7. Inspect finite coverage, residual diagnostics, and train/validation metrics; a parsable formula is not necessarily a valid scientific model.
8. Submit structured formulas whenever possible. Use code-defined models only when the engine truly cannot express the candidate.
