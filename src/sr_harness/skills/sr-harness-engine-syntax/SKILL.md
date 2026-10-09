---
name: sr-harness-engine-syntax
description: Use this skill before writing, evaluating, or submitting ordinary SRHarness Engine formulas, especially formulas with fitted parameters, categorical grouped parameters, nonlinear functions, broadcasting, or delays.
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

## Graph and hypergraph expressions

Indexed node, edge, and hyperedge expressions are documented separately. Before writing a formula with symbolic indices, relations, `sum[...]`, `gather`, `aggr`, `targ`, or `sour`, use `read_skill` to read `sr-harness-engine-graph-syntax`.

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
custom_function(x)        # unregistered function
lambda x: x               # lambda
__import__('os')           # arbitrary Python
```

## Evaluation checklist

Before calling `evaluate_formula` or `submit_formula`:

1. Use names that exist in `context.data`; declare coefficients with `param`.
2. Use `fit=true` if the formula contains unbound `param` or `grouped_param` nodes.
3. Check domains and finite coverage before submitting expressions containing division, logarithms, roots, or delays.
4. Inspect residual diagnostics and train/validation metrics; a parsable formula is not necessarily a valid scientific model.
5. Submit structured formulas whenever possible. Use code-defined models only when the engine truly cannot express the candidate.
