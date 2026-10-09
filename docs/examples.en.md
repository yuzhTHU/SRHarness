# Examples

## Synthetic static regression

```bash
sr-harness synthetic \
  -f 'y = 1 + x1 ** 2 + 2 * x1 * x2' \
  --features x1,x2 \
  --n-samples 240 \
  --x-low -2 --x-high 2 \
  --seed 42 \
  --validation-fraction 0.2 \
  --save-path ./logs/polynomial-example \
  -R 1 -C 1 -L 8 -K 1
```

Literal values are fixed constants. A candidate may instead contain `param(...)`; the Evaluator fits those parameters on the training split and reports both training and validation metrics.

## Fit parameters directly with the Engine

This example does not call a model service:

```python
import numpy as np
import sr_harness_engine as engine

x = np.linspace(-3.0, 3.0, 200)
y = 2.5 * np.sin(x) - 0.4
expression = engine.parse("param('a') * sin(x) + param('b')")
fit = expression.fit({"x": x}, y)

print(fit.expression)
print(fit.parameters)
print(fit.loss)
print(fit.evaluate({"x": x})[:5])
```

`fit.expression` has all parameters bound and can be evaluated directly or passed to an Evaluator.

## Discover an oscillatory ODE in the WebUI

1. Start `sr-harness run` and select the oscillatory ODE sample under Data Preparation.
2. The dataset contains `t`, `x`, and `dx_dt` without revealing the generating equation.
3. Under Task Setup, assign `dx_dt` as the target and `x`, `t` as features.
4. Use a research question such as: “Discover an interpretable differential equation `dx/dt = f(x, t)` for the observed oscillatory trajectory.”
5. Start with `DefaultEvaluator`, review the generated prompts, and run the search.

### Add a long-horizon rollout metric

Pointwise derivative accuracy does not guarantee stable long-term integration. Expand Evaluator Construction Agent and request:

> Create a TrajectoryRolloutEvaluator derived from DefaultEvaluator. Keep the default metrics and add rollout_rmse by integrating candidate ODEs in evaluate_candidate.

The Agent can inspect interfaces through `read_source`, write scripts below `context.evaluator/`, and test them with `validate_evaluator`. After saving, set `ranking_metric` to `rollout_rmse` and ask the Symbolic Regression Agent to continue.

A custom Evaluator must derive from `DefaultEvaluator`, return a flat `dict[str, float | int]` from evaluation methods, return a fully bound `Expression` from fitting methods, and perform splitting only in `split`. Reuse `sr_harness.evaluator.utils` for metrics, indices, and ODE infrastructure.

## Kuramoto graph dynamics

The Kuramoto BA sample contains:

| Variable | Shape | Meaning |
|---|---|---|
| `t` | `(T,)` | Time axis |
| `omega` | `(T, N)` | Natural frequency |
| `x` | `(T, N)` | Oscillator phase |
| `dx_dt` | `(T, N)` | Phase derivative |
| `A` | `(E, 2)` | Directed `(target, source)` edge list |

An Engine candidate can use relation-index syntax:

```text
omega[i] + param('K') * sum[j](A[i, j], sin(x[j] - x[i]))
```

Graph data selects `GraphEvaluator`; `num_nodes` and relation metadata come from `context.data/manifest.json`.

## Inspect and call tools

```bash
sr-harness tool list
sr-harness tool schema evaluate_formula
sr-harness tool call evaluate_formula \
  --context context.npz \
  --params '{"f": "param(\"a\") * x", "fit": true}'
```

A successful result exits with code 0; a failed `ToolCallResult` produces a nonzero exit code.

## Python API

```python
import numpy as np
from sr_harness import SRAgent

rng = np.random.default_rng(42)
x1 = rng.uniform(-2, 2, 200)
x2 = rng.uniform(-2, 2, 200)
agent = SRAgent(
    llm_provider="openrouter",
    llm_model="qwen/qwen3.5-flash-02-23",
    max_restart_loop=1,
    global_width=1,
    max_refinement_depth=5,
    local_sample_size=1,
    save_path="logs/python-example",
)
result = agent.run(
    X={"x1": x1, "x2": x2},
    y={"y": 1 + x1**2 + 2*x1*x2},
    problem_description="Discover y as a function of x1 and x2.",
)
```

See the [API Reference](/reference/) for the complete constructor and return types.
