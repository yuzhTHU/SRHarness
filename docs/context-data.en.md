# Structured Data and `context.data`

SRHarness stores structured data in a `context.data/` directory. Each variable and each sufficiently long axis is stored as an NPY file, while `manifest.json` describes variables, axes, and graph relationships.

This page distinguishes three kinds of rules:

- **hard constraints** cause loading to fail when violated;
- **semantic conventions** determine how users and Agents interpret the data but may not be fully machine-checkable;
- **recommendations** improve data quality and experimental interpretability without affecting loading.

!!! warning "Do not leak the answer in descriptions"
    Variable descriptions should not contain the formula to be discovered or hint at its functional form. The loader can verify only that `description` is a string, so this is an experimental-design recommendation rather than a hard constraint.

## Directory layout

```text
context.data/
├── manifest.json
├── x.npy
├── y.npy
├── time.npy       # long axes may also be stored as NPY
└── A.npy          # graph or hypergraph relations are variables
```

- A valid `manifest.json` is required. Every referenced NPY file must be at the directory root and its filename cannot contain a path.
- Each NPY file must contain a `numpy.ndarray` loadable with `allow_pickle=False`. Object dtype is not allowed; use NumPy Unicode or fixed-width string dtypes for text.
- Unreferenced `*.npy` files produce warnings rather than errors so that temporary artifacts can remain during preparation. Remove or register them before a production run.

## Root fields in `manifest.json`

| Field | Requirement | Meaning |
|---|---|---|
| `variables` | Required | Non-empty object containing every variable, including relation variables. |
| `axes` | Required | Object containing every axis referenced by variables; it may be empty. |
| `num_nodes` | Conditionally required | Positive integer required whenever a `kind: "relation"` variable exists, and forbidden otherwise. |

No other root fields are accepted. Unknown fields, missing required fields, and duplicate JSON keys invalidate the manifest. The target, feature selection, problem description, and Evaluator belong to a research task rather than the data and are therefore not stored in the manifest.

## Variables

```json
{
  "variables": {
    "theta": {
      "file": "theta.npy",
      "description": "Oscillator phase in radians.",
      "axes": ["time", "node"]
    }
  }
}
```

- Every variable requires `file`, `description`, and `axes`; only `kind` and `structure` are accepted as additional fields.
- A variable name must be non-empty, cannot be `.` or `..`, and cannot contain a path separator or NUL. Its file must be named exactly `<variable>.npy`.
- `axes` is an array of axis names. Its length must equal the array rank, and each dimension must match the corresponding axis length. Scalars use `[]`.
- After loading into `AgentContext`, variable names and axis names must be disjoint; together they form `context.data`.

## Axes

Every axis requires `description` and exactly one of `values`, `file`, or `size`:

```json
{
  "axes": {
    "time": {"values": [0.0, 0.1, 0.2], "description": "Time in seconds."},
    "node": {"values": ["node1", "node2", "node3"], "description": "Node label."},
    "sample": {"size": 100, "description": "Zero-based sample position."}
  }
}
```

- `values` must be a non-empty one-dimensional array of JSON scalars.
- `file` must be `<axis>.npy`, and the stored array must be one-dimensional.
- `size` must be a positive integer and produces `0 … size-1`.
- Every declared axis must be referenced by at least one variable, and variables cannot refer to undeclared axes.
- Use inline `values` for short axes and meaningful labels; use NPY files for long axes.

## Graph and hypergraph relations

```json
{
  "num_nodes": 10,
  "variables": {
    "A": {
      "file": "A.npy",
      "description": "Directed graph endpoint pairs.",
      "axes": ["edge", "endpoint"],
      "kind": "relation"
    },
    "weight": {
      "file": "weight.npy",
      "description": "Edge weight at each time.",
      "axes": ["time", "edge"],
      "structure": "A"
    }
  }
}
```

- A relation variable must declare `kind: "relation"`; SRHarness does not infer relations from names such as `A` or `T`.
- Relation arrays must have shape `(E, 2)` or `(H, 3)`, use an integer dtype, and contain endpoints in `[0, num_nodes)`.
- The semantic column order is `(target, source)` for `(E, 2)` and `(target, source1, source2)` for `(H, 3)`. Declare these labels in endpoint-axis `values` when possible.
- A variable whose final dimension aligns with relation length `E` or `H` uses `structure` to reference that relation, and the lengths must match.
- Ordinary node variables omit `structure`, and a relation never references itself. Relations in one manifest share one `num_nodes`.

## Descriptions

Every variable and axis requires a string `description`. Document real-world meaning, units, measurement or generation procedures, and material data-quality information. Preserve categorical values as strings unless the user explicitly requests an encoding such as one-hot.

Avoid descriptions such as “target generated by `y = 1 + x**2`” or “exponential-decay coordinate.” Use neutral descriptions such as “response variable,” or state only scientific meaning that is legitimately available to the search.

## Complete examples

### Tabular data

```json
{
  "variables": {
    "population": {"file": "population.npy", "description": "Annual population count.", "axes": ["year"]},
    "gdp": {"file": "gdp.npy", "description": "Annual gross domestic product in constant currency.", "axes": ["year"]}
  },
  "axes": {
    "year": {"values": [2018, 2019, 2020, 2021, 2022], "description": "Calendar year."}
  }
}
```

### Network dynamics

```json
{
  "num_nodes": 10,
  "variables": {
    "theta": {"file": "theta.npy", "description": "Oscillator phase in radians.", "axes": ["time", "node"]},
    "A": {"file": "A.npy", "description": "Directed interaction endpoints.", "axes": ["edge", "endpoint"], "kind": "relation"}
  },
  "axes": {
    "time": {"file": "time.npy", "description": "Time in seconds."},
    "node": {"values": ["node1", "node2", "node3", "node4", "node5", "node6", "node7", "node8", "node9", "node10"], "description": "Node label."},
    "edge": {"size": 32, "description": "Directed edge position."},
    "endpoint": {"values": ["target", "source"], "description": "Endpoint column order."}
  }
}
```

## Validation and loading

```python
from sr_harness.core import inspect_context_data, load_context_data

report = inspect_context_data("workspace/context.data")
if report["errors"]:
    print("invalid:", report["errors"])
else:
    loaded = load_context_data("workspace/context.data")
    print(loaded["data"].keys())
    print(loaded["variable_axes"])
    print(loaded["relation_names"], loaded["num_nodes"])
```

`inspect_context_data()` reports errors and warnings and is suitable for preparation-time checks. `load_context_data()` raises `ContextManifestError` for invalid data and returns arrays and metadata ready for `AgentContext`. Data Preparation Agent should invoke `validate_context_data` after every modification to `context.data/`.
