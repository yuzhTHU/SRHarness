# SRHarness Engine

`sr_harness_engine` is the symbolic expression layer of SRHarness. It provides restricted parsing, canonical rendering, tree traversal, NumPy evaluation, parameter fitting, constant folding, and syntax for graphs, hypergraphs, and delays.

The Engine describes mathematics only. Train/validation splits, candidate ranking, and task-specific rollout metrics belong to Evaluators and the SRHarness runtime.

## Quick start

```python
import numpy as np
import sr_harness_engine as engine

x = np.linspace(-2.0, 2.0, 101)
target = 2.5 * np.sin(x) - 0.4
model = engine.parse("param('a') * sin(x) + param('b')")
fit = model.fit({"x": x}, target)

print(fit.expression)          # expression with bound parameters
print(fit.parameters)
print(fit.loss)
print(fit.evaluate({"x": x})[:3])
```

`parse()` uses a restricted Python AST. It does not call `eval` or execute arbitrary Python from a formula.

## Basic syntax

### Values, variables, and operators

```text
1.5
x
x1 + x2
x1 - x2
x1 * x2
x1 / x2
x ** 2
-x
```

The language supports `+`, `-`, `*`, `/`, `**`, and unary minus. Use `**`, not Python's bitwise-XOR operator `^`, when calling `engine.parse()` directly.

Built-in elementwise functions are:

```text
sin cos tan sinh cosh tanh
arcsin arccos arctan
exp log log10 sqrt abs
sigmoid sign sec sech csc cot inv
```

Expressions can also be constructed through Python:

```python
x = engine.Symbol("x")
expression = engine.exp(-x) * engine.sin(2 * x)
```

## Parameters

### Named parameters

```text
param('slope') * x + param('bias')
param('a', value=0.3) * x
```

Occurrences with the same name refer to one parameter. `value` is both a default and an optimization initial value. `FitResult.expression` contains bound parameter values after fitting.

```python
fit = engine.fit(
    engine.parse("param('a') * x + param('b')"),
    {"x": x},
    target,
    initial={"a": 1.0, "b": 0.0},
)
```

### Grouped parameters

```text
grouped_param(label, name='rate') * x
```

Samples in one category share a parameter, while categories receive separate values. Initial mappings and a default are supported:

```text
grouped_param(
    label,
    name='rate',
    value={'A': 1.0, 'B': 2.0},
    default=0.0,
) * x
```

## Parsing, rendering, and round trips

```python
expression = engine.parse("x ** (4 / 3) + sqrt(2)")
text = engine.render(expression)
latex = engine.render(expression, latex=True)
assert str(engine.parse(str(expression))) == str(expression)
```

Attribute access, slicing, comprehensions, lambdas, arbitrary calls, and non-literal keyword arguments are rejected. Examples of invalid input are `np.sin(x)`, `__import__('os')`, `x[0:10]`, and `custom_python_function(x)`.

## Evaluation

```python
expression = engine.parse("sin(x) + y ** 2")
value = expression.evaluate({"x": x, "y": y})
```

Variables follow NumPy broadcasting. Parameter values can be supplied separately:

```python
expression.evaluate({"x": x}, parameters={"a": 2.0, "b": -0.5})
```

The equivalent module function is `engine.evaluate(expression, values)`.

## Expression trees

All nodes derive from `Expression`:

| Node | Meaning |
|---|---|
| `Number` | Fixed numeric literal |
| `Symbol` / `Variable` | Data variable |
| `Parameter` | Named parameter |
| `GroupedParameter` | Category-dependent parameter |
| `Unary` / `Binary` | Arithmetic operation |
| `Function` | Function call |
| `Indexed` | Variable with free indices |
| `Reduction` | Index reduction |
| `Gather` / `Aggregate` / `RelationLift` | Relation operation |

```python
for node in expression.iter_preorder():
    print(type(node).__name__, node)

copy = expression.copy()
simplified = expression.fold_constants()
parameter_count = expression.count_parameters()
```

`operands` exposes direct children, while `replace(old, new)` creates a tree with identity-based replacement.

## Graph and hypergraph indices

Relation arrays use target-first ordering:

- graph `A.shape == (E, 2)`: `(target, source)` rows;
- ternary hypergraph `T.shape == (H, 3)`: `(target, source1, source2)` rows.

### Message aggregation

```text
sum[j](A[i, j], x[i] * x[j])
```

`A[i, j]` binds column 0 to target index `i` and column 1 to source index `j`. `sum[j]` eliminates the source index and aggregates by the remaining `i`. Supply `num_nodes` so isolated nodes remain represented:

```python
prediction = expression.evaluate(data, num_nodes=10)
```

Node variables normally have shape `(..., N)`, edge fields `(..., E)`, and hyperedge fields `(..., H)`. Leading dimensions broadcast under NumPy rules; the final dimension is structural.

Hypergraph aggregation uses the same language:

```text
sum[j, k](T[i, j, k], x[i] * x[j] * x[k])
```

### Gather and relation fields

`gather` evaluates at nonzero relation coordinates and returns edge- or hyperedge-aligned output:

```text
gather(A[i, j], x[i] + x[j])
gather(T[i, j, k], x[i] * x[j] * x[k])
```

An edge field can participate in aggregation:

```text
sum[j](A[i, j], w[i, j] * x[i] * x[j])
```

`w` may have shape `(E,)` or `(..., E)`. When multiple relations are present, `RelationField(w, relation="A")` identifies the matching coordinate table.

### Convenience syntax

```text
aggr(A * targ(x) * sour(x))
aggr(A, targ(x) * sour(x))
```

Both desugar to `sum[j](A[i, j], x[i] * x[j])`. Canonical rendering therefore emits index syntax rather than preserving the original sugar.

## Delays

```text
x + delay(x, delta)
```

The default evaluator interpolates along the first axis to obtain `x(t - delta)` and returns `NaN` outside available history:

```python
prediction = engine.parse("delay(x, delta)").evaluate(
    {"x": trajectory, "delta": lag},
    time=sample_times,
)
```

ODE/DDE Evaluators can provide a custom `delay_resolver` backed by their own history representation.

## Simplification and complexity

```python
assert str(engine.parse("(2 + 3) * x")) == "5 * x"
```

Constant folding preserves readable fractions, powers, and named functions. `count_parameters()` counts fitted parameters; `len(expression)` counts expression nodes. The default Evaluator uses node count as `complexity`.

## Engine and Evaluators

`DefaultEvaluator` exposes five extension points:

```python
split(context)
fit(f, y, context)
evaluate(f, y, context)
fit_candidate(f, context)
evaluate_candidate(f, context)
```

A general equality uses `fit/evaluate`; a target-eligible formula uses the candidate-specific entry points. An ODE Evaluator can therefore add integration and rollout metrics only for a formal `dx_dt = f(x, t)` candidate without treating every implicit equality as integrable dynamics.

Metrics, random/OOD/chronological splitting, and ODE integration infrastructure are available from `sr_harness.evaluator.utils`. See [SRHarness Core Abstractions](core-abstractions.md#evaluator) for the Evaluator design and extension workflow, and the [API Reference](/reference/) for signatures.

## Executable specification

Behavior tests provide runnable examples for basic expressions, parameters, relations, gather, delay, simplification, and custom Evaluators:

```bash
pytest tests/behavior
```

See the [API Reference](/reference/) for every public type and function.
