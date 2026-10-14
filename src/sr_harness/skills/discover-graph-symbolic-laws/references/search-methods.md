# Evidence-driven graph-law search methods

This reference describes reusable operations for graph symbolic regression. It
intentionally contains no named benchmark, target equation, or coefficient.

## Separate topology from dynamics

First determine what the relation rows mean and whether direction is identifiable.
Reciprocal edges duplicate incoming and outgoing aggregates. Constant degree can
make a target-node term collinear with a relation sum of that same term. These are
properties of the observed graph, not evidence for a physical mechanism.

Classify node arrays by variation across leading samples. A node-static field may
act as a heterogeneous coefficient, but this is only a hypothesis to test in a
local library.

## Fit in coordinates that expose structure

Use sparse linear fitting as a diagnostic. A short exact fit is strong evidence;
a long high-scoring polynomial often means that the current coordinates hide a
factor or nonlinear transform. Inspect the sequence of residual reductions, not
only the final score.

Build graph features in layers:

1. local node expressions;
2. edge messages involving source and target values;
3. relation reductions of those messages;
4. optional target-node gates multiplying the reduction.

Fit plausible layers together. Testing each term only against the raw target can
hide a decisive feature behind a dominant local component.

`graph_structure_screen` provides a bounded first pass over these layers. Its
grammar consists of low-degree local products, source powers, power saturations,
source-target sine differences, and target-node gates. These are coordinate
families rather than named mechanisms. Use the screen to expose compositions;
do not interpret a long approximate fit as the discovered law.

## Use residual invariants

For a candidate `f`, study `target - f`.

- Sum the residual over nodes for each sample. Values consistently near numerical
  zero motivate conservative messages such as antisymmetric differences.
- Average over samples for each node. Persistent offsets motivate node-static
  fields, degree effects, or missing local heterogeneity.
- Compare a relational probe alone with the same probe multiplied by several
  target-node fields. A sharp held-out improvement motivates multiplicative
  gating.
- If error is concentrated in tails or plateaus, compare a few transforms with
  genuinely different asymptotic behavior rather than raising polynomial degree.

These diagnostics narrow a grammar; they do not uniquely determine a formula.

## Propose templates without embedding answers

`graph_template_fit` accepts arbitrary caller-proposed edge messages and wraps
them in the chosen relation. Use it only after diagnostics identify a structural
question. For example, contrast an unbounded message with a bounded one, or a raw
source transform with a target-source difference. Parameters inside a message
use Engine `param(...)` syntax.

The tool jointly fits the graph gain and the supplied local coefficients so that
a useful message is not rejected merely because the raw target also contains
strong local dynamics. Include both `1` and a small number of plausible
target-node gates when gating is unresolved.

The tool deliberately has no built-in list of preferred kernels. Operator choice
belongs to the search policy and should be auditable from preceding evidence.

## Simplify and falsify

Treat fitted constants as estimates. When a parameter is stably near an integer
or short rational value across splits or initializations, fix that value, refit
the remaining parameters, and compare held-out error. Remove negligible bias and
redundant features only if the simplified expression preserves the fit.

Check alternative explanations created by graph symmetry or regularity. Report
non-identifiability when the available data cannot distinguish them. Exact
floating-point recovery on one fixed topology does not by itself establish a
unique law.
