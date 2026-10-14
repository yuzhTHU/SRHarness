---
name: discover-graph-symbolic-laws
description: Discover compact symbolic laws from graph or network data using indexed expressions, topology checks, residual diagnosis, sparse composition, and caller-proposed message templates. Use when the target is node-aligned and relations constrain interactions.
---

# Discover graph symbolic laws

Read `sr-harness-engine-syntax` and `sr-harness-engine-graph-syntax` before
constructing formulas. Treat graph discovery as inference over three independent
choices:

```text
local node law + gate(target node) * reduce_relation(message(target, source))
```

Do not assume any component is present. Let held-out error and residual structure
decide which component to add.

For the rationale behind the workflow and how to interpret its diagnostics, read
`references/search-methods.md`. It contains no task-specific equations.

## Search workflow

1. Call `graph_data_analysis`. Establish relation orientation, degree variation,
   reciprocity, and which node fields are dynamic or static. Read both the global
   probe ranking and the best probe retained in each structural family.
2. Call `graph_structure_screen` before committing to the strongest one-feature
   probe. It jointly tests a bounded, generic factorized grammar and is especially
   useful when correlated trajectories make every true term weak in isolation.
   Its default run also fits a bounded set of generic parameterized message
   families. Compare the linear screen and nonlinear ranking; treat both as
   structural evidence until the complete formula is verified.
3. Fit a compact local baseline with `graph_feature_fit`. Use low-order terms and
   static-field products justified by the schema; do not start with a dense
   cross-product library. Use `selection_method="bic_lasso"` when forward search
   repeatedly selects correlated decoys.
4. Call `graph_residual_analysis` on that baseline. Supply a small set of probes
   and possible target-node normalizers. Use conservation, persistent node bias,
   and gated-probe improvement to decide the next structural change.
5. Fit additive and gated graph features together with `graph_feature_fit` when
   every feature is parameter-free. This tests compositions rather than isolated
   terms. The bounded beam keeps several partial combinations so that an
   individually strong decoy does not hide jointly useful features. Use
   `beam_width=1` only for a quick greedy screen, or at most 8 for an especially
   ambiguous compact library.
6. When the automatic parameterized refinement is accurate but non-exact, first
   simplify stable fitted constants and verify the complete expression. If the
   residual remains structured, use the reported refinement templates as a
   starting point for a few additional contrasts with `graph_template_fit`. Do
   not approximate a saturating or threshold-like residual by accumulating many
   polynomial terms.
7. When the residual otherwise suggests a nonlinear message, propose a few contrasting
   edge-message templates and pass them to `graph_template_fit`. The tool has no
   built-in kernel catalog: the proposed templates must follow from diagnostics
   or from a generic operator search, not a task label.
8. Simplify only stable fitted constants, then reevaluate the complete expression
   with `evaluate_formula`. Preserve the best compact candidate while exploring.

Graph-facing analysis tools accept a nodewise residual target such as
`y + u[i]`; they align the bare node field to `y[i]` automatically. Explicit
indexing remains preferable in final submitted formulas.

Typical calls are:

```text
graph_residual_analysis(
  candidate="local_candidate",
  probes=[
    "sum[j](A[i,j], u[j])",
    "sum[j](A[i,j], u[j]-u[i])"
  ],
  normalizers=["1", "u[i]", "v[i]"]
)

graph_template_fit(
  messages=[
    "u[j]-u[i]",
    "tanh(param('scale', value=1)*(u[j]-u[i]))"
  ],
  local_features=["u[i]", "v[i]"],
  gates=["1", "u[i]"]
)
```

These examples illustrate syntax, not preferred mechanisms. Change variables,
operators, and templates based on the observed residual.

## Decision rules

- Near-zero samplewise residual sums support a conservative or antisymmetric
  interaction; they do not identify its kernel.
- Persistent nodewise residual means support a missing static field, degree term,
  or node-specific local effect.
- A large improvement from `normalizer * probe` over `probe` supports a gate.
- Many weak polynomial terms usually indicate a missing coordinate,
  factorization, or nonlinear message. Diagnose before increasing degree.
- Reciprocal or regular graphs can make distinct formulas observationally
  equivalent. Report the ambiguity instead of interpreting split coefficients.
- A nonlinear parameter near a simple value is only a simplification hypothesis.
  Snap it and verify on held-out and full data before accepting it.

## Bound the search

- Compare hypotheses that answer different structural questions; avoid dozens of
  cosmetic variants of one expression.
- Keep feature libraries below 128 expressions and normally select at most six
  terms. Keep the beam bounded; widen it only to test a concrete
  combination-effect hypothesis.
- Do not emit free-index expressions unless a relation reduction or `gather`
  immediately binds the index.
- Do not inspect or print full arrays. Use shapes, summaries, ranked fits, and
  residual statistics.
- If progress stalls, continue from the best candidate's residual rather than
  restarting an unrelated broad search.
