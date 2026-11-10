---
name: discover-hypergraph-known-edge-laws
description: Recover compact shared F/G/H on already-loaded pairwise and triplet tables that must not be edited. Call diagnostics, screen, refinement, optional node-indexed parameters, and full-trajectory reintegration from residuals. Use only with the known-edges agent.
---

# Recover compact symbolic laws on given hyperedges

The edges are given as input and must not be changed. The task is to recover compact kernels \(F,G,H\) shared by all nodes. \(F\) is the local node term and depends only on the state at the same node. When there is cross-trajectory evidence, \(F\) may add at most one node-indexed constant or one node-indexed linear parameter. Edges and their original weights are not unknowns. An empty pairwise channel is reported as \(G=0\); that is the shortest convention, not a claim that an unobserved pairwise kernel was identified. Without ground truth, symbolic equivalence cannot be self-certified. Compact-form review and full-trajectory reintegration are independent checks and must not replace each other.

## Data contract

The following data must already be loaded in context before any tool is called. Univariate problems provide `x` and `dx_dt`, both with shape `(trajectory, time, node)`. Multivariate problems provide named scalar fields `u, v, …` and derivatives `du_dt, dv_dt, …` on the same node axis. `num_nodes` counts hypergraph vertices; do not flatten state components into extra vertices. Pairwise and triplet tables are target-first; empty tables keep the correct column count. Univariate and multivariate problems share this table layout. The full original time grid and the original initial condition of every trajectory must be provided. Report missing fields to the data loader; do not read files to fill gaps. Tools read only the loaded observations and edges. Workflow caches must not rewrite the relation tables.

Univariate and multivariate problems use the same tool names. Multivariate problems write one equation per loaded node variable.

The caller chooses the next tool from each return value; this skill is not an automatic pipeline. `--check` validates the pipeline and does not replace the agent. Read this skill first. `sr-harness-engine-syntax` and `sr-harness-engine-graph-syntax` may be read afterwards if an indexed string must be written. Do not read `discover-hypergraph-laws`. Temporary `fit.py` files in logs are not required by this workflow.

## 1. Diagnose shape, empty channels, weights, and counting

Call `hypergraph_known_edge_diagnostics()`. Read the return value before deciding whether to screen.

- `status=blocked`: stop only when the univariate `x` data contract is broken. Do not call `hypergraph_formula_screen()`. Do not submit.
- `multivariate=true` or `state_dim>1`: write one equation per loaded node variable, \(\dot u,\dot v,\ldots\). The local node term \(F\) may use every variable at that vertex; \(G\) and \(H\) may be identically 0 on some variables. Call `hypergraph_formula_screen()`; do not pass `variables=["x"]`. Do not flatten variables onto the node axis. Later refinement, node-indexed fitting, and reintegration use the same `candidate_id`. Node-indexed parameter rules match the univariate case: at most one constant term or one slope term, attached to the equation that has evidence.
- Read `shape`, `state_range`, `derivative_rms`, `channels`, and `isolated_target_nodes`. Use weights when they are loaded; default missing weights to 1. A missing weight field does not prove that the original problem is unweighted. If the loader is known to have dropped weights, fix the loader first. Do not absorb edge-weight differences into kernel amplitudes.
- Counting follows the loaded target-first tables. `rows` counts each existing record once. An undirected triplet should have one neighbor pair at each endpoint. If the table contains both neighbor permutations and the loader declares `ordered_symmetric`, multiply the numerical aggregate by \(1/2\). The tool checks pairing and equal weights. It does not guess or edit the table.
- `node_clues` only hint at later inspection. Coupling has not been subtracted, so do not infer node-indexed parameters from these clues yet. Isolated targets receive no incoming edges; that is not a reason to delete their outgoing edges or edit the graph.
- If both tables are empty, a local-only formula may still be fit. Do not add edges in order to start later tools.

## 2. Screen proposes candidates; it does not authorize submission

For univariate problems call `hypergraph_formula_screen(variables=["x"], y="dx_dt", max_terms=6)`. For multivariate problems call `hypergraph_formula_screen()`. The default budget is small. Do not raise `max_terms` to the cap of 20 after a first failure.

Read `candidate_id`, `known_edges.channels`, absolute and relative derivative RMSE, per-node and per-trajectory residuals, rank, condition number, and the selected terms. `can_submit=false` is the normal screen result.

- Few shared terms, small residual, and no stable node structure: go to step 3 to refine, then step 5 to reintegrate.
- Main terms are identified but bias, log, exp, or similar terms have small amplitude: refine first. Do not treat those small terms as an identified mechanism. Judge by column amplitude and the error after deletion, not only by coefficient magnitude.
- Large residual, many terms, cancelling coefficients, or rank deficiency is not a reason to submit. Step 3 compares a small, evidence-backed set of families. If node clues exist, continue to step 4.
- Domain errors (log domain, vanishing denominator, overflow) mean the candidate is inapplicable. Do not mask them by truncating observations or deleting edges.

The screen `candidate_id` is the input to later tools. Do not copy a long coefficient matrix into later calls. The unweighted `rows` path keeps GraphFeatureFit and BIC-Lasso. The weighted path reuses the same selector and only replaces the aggregation column.

## 3. Refine the formula; replace the family only when needed

Call `hypergraph_formula_refine(candidate_id=screen_id)`.

The tool refits the current selected structure, removes exact redundancies in the design matrix, then deletes small terms one at a time and refits after each deletion. It proposes rounding of shared coefficients that are close to short decimals or small-denominator rationals. Do not pass `residual_allowance`, `coefficient_cutoff`, or `round_digits`: out-of-range values are clamped, and the default already maps \(0.999988\) to \(1\). If a node-indexed parameter was already accepted, it is re-estimated after shared amplitudes are held fixed; it is not replaced by a fixed template. These outputs are candidates, not a verification of the law.

Read `trace` (what was deleted or rounded), `baseline_metrics`, the new `metrics`, held-out trajectory results, `unresolved_small_terms`, `short_review`, and `rollback_id`. Residuals need not reach machine precision; the original `dx_dt` may contain differentiation error. Minimum BIC, small derivative MSE, or fewest terms must not replace reintegration.

If the selected family is misspecified, call the same tool with `replacement_terms=[{"channel":"F/G/H", "kernel":"scalar kernel in xi/xj/xk"}, ...]` and `evidence`. This replaces the entire short candidate library; it does not append kernels to a long expression. At most 12 terms. Coupling amplitudes remain shared. The tool does not accept an edge table. Keep the old `candidate_id` for comparison. Use the table below.

| Residual signature | Short families to contrast (not exclusive) | Stop expanding when |
| --- | --- | --- |
| Low-order polynomial leaves a few stable terms | Keep those terms; merge local redundancies | Deleting a term increases residual or integral error; then roll back |
| Residual is a per-node offset or scales with own state | Step 4 node-indexed parameters first; do not complicate \(G/H\) | Stop expanding the kernel if the node-indexed parameter passes its gates |
| Bilinear product, growth/competition | `product`: \(x_i x_j\); triplet \(x_i x_j x_k\) | Do not increase polynomial degree |
| Bounded contagion; vacancy times neighbor | `contagion`: \((1-x_i)x_j\); triplet \((1-x_i)x_j x_k\) | State outside \([0,1]\) may still be tried; drop the family if held-out fails |
| Signed phase difference | `phase_diff`; if overdamped, try `damped_phase` | If the phase-difference family already meets held-out, do not switch to a source sine |
| Linear diffusion / graph Laplacian | `linear_diff`: \(x_j-x_i\); triplet \(x_j+x_k-2x_i\) | Do not add local polynomials to mimic diffusion |
| Odd neighborhood source stronger than phase difference | `odd_source`: \(\sin(x_j)\); triplet may include \(\sin(x_i)\) | Do not switch if phase difference is stronger |
| Nonnegative, saturating neighborhood | `bounded_source` / `sat_fraction` | Vanishing denominator fails the family; try another |
| Positive decay / log growth | `exp_decay` / `log_growth` | Skip the log family when \(x\le-1\) |
| Empty \(G\), residual consistent with a missing middle index | `skip_middle`: \(x_i x_k\) | Do not use a missing-index kernel on undirected symmetric triplets |
| Triplet modulated by a neighbor difference | `gauss_mod` / `lorentz_diff` / `tanh_phase` | Do not copy \(H\) from the type of \(G\) |

Read `form_clues.suggested_form_families` (ranked by residual correlation; several families may appear). When the shared residual remains large or the parent expression is too long, `hypergraph_formula_refine` contrasts every short family that can be evaluated on the current domain. Switch families only if held-out error drops below half of the parent. Do not mix several families into one long expression. If `replacement_terms` are passed by hand, pass one family at a time. The screen library already contains these kernels for BIC subset selection.

Multivariate problems use the same tool with a per-variable catalog; do not change the univariate table above. Read per-variable RMSE, `zero_slots`, and `suggested_form_families`. Default coupling remains `linear_diff` / `phase_diff` / `product`. Local \(F\) may use every variable at the node. The automatic catalog `MULTIVARIATE_FORM_KERNEL_FAMILIES` contrasts one family per output variable when residual RMSE\(>10^{-2}\), coupling terms exceed 3, or total terms exceed 6. Switch only if held-out is full rank and error is less than half of the parent. Do not auto-switch without held-out. Cubic families appear only in this contrast; they are not in the default six coupling terms of screen. A zero slot in \(G\) or \(H\) is valid. Each hand-passed `replacement_terms` entry must include `component` (variable name). Listed outputs are replaced as a block; unlisted outputs are kept.

| Multivariate residual signature | Short family / action | Stop when |
| --- | --- | --- |
| One output residual varies with another same-node variable | `local_linear_cross` | Do not switch if held-out does not improve; stop raising local degree once held-out improves |
| Product of two variables after linear local terms | `local_bilinear_cross` | Drop if held-out does not fall or the 12-term budget is exceeded |
| Neighbor difference on one output; other outputs already explained by \(F\) | `linear_diff:<input variable>` | Do not switch unless held-out drops by half; keep other slots at 0 |
| Signed phase difference on one output | `phase_diff:<input variable>` | If that output already meets held-out, do not add sines to other outputs |
| Neighborhood product on one output | `product:<input variable>` | Do not increase polynomial degree |
| After low-order diffusion/product, a univariate residual remains consistent with a cubic polynomial | refine catalog `cubic_diff` / `cubic_ordered` / `cubic_pair_sum` (linear \(G\) allowed; multivariate triplets read `uj,uk` in neighbor order after removing the target from the original edge) | Not in default screen; contrast only if derivative RMSE\(>10^{-4}\); do not switch unless held-out drops by half; keep other output slots at 0 |
| After a compact shared kernel, only one variable shows a cross-trajectory node offset or slope | Step 4, attach to that variable only | Do not add if guards reject; do not give every variable its own node-indexed parameter |

These are conditional recommendations, not an exhaustive search per problem. The implementation fits only shared external linear amplitudes. If logs mention interior shape parameters of a nonlinear kernel, this pass does not optimize them automatically. If fixed-shape contrasts cannot explain the residual, report the gap; do not claim that this tool searched continuous parameters. `evaluate_code` is an explicitly recorded exception only. Do not require a `fit.py` to be written during the run. Any exception candidate must still re-enter the same short-candidate and reintegration gates; it cannot be submitted until it is attached.

After a family change, do not reuse the previous node-indexed authorization; re-test from the shared candidate. The tool already scans its catalog. The caller may hand-pass at most 3 additional replacement families that are backed by residuals. If there is no new evidence or the budget is exhausted, stop as unresolved. Do not submit the shortest candidate that failed the guards.

## 4. Add a node-indexed parameter only with evidence

Call `hypergraph_known_edge_diagnostics(candidate_id=...)`. Read `recommended_node_mode` (`constant` / `linear` / `none`) and `next_action`. For multivariate problems read `per_variable` (JSON still also writes `per_component`).

- `constant`: the residual is consistent with a per-node constant drive (frequency, injection).
- `linear`: the residual scales with own state (growth or decay \(r_i x_i\), \(\mu_i x_i\)).
- `none`: a node projection does not explain much residual; do not add a node-indexed parameter.

When a mode is recommended, call immediately:

`hypergraph_node_parameter_fit(candidate_id=shared_id, mode=<recommended_node_mode>, evidence="observed cross-trajectory or isolated-node clues")`.

Do not absorb node differences into a higher-order shared \(F\) (\(x_i^2,x_i^3\)) or into unrelated \(G/H\) terms. That represents \(F_i\) as a shared kernel; the guards reject it and submission yields an empty formula.

Allowed clues: in an overlapping state range, nodes differ by a constant drive or a slope, and the difference repeats across time and trajectories. Isolated nodes are a strong source of evidence. Estimate slopes by least squares; do not divide by \(x\) near 0. Allow only one node-indexed constant or one node-indexed slope. Do not keep two node-indexed arrays. Do not fit per time, per trajectory, or per edge. In the multivariate case the parameter may attach to \(F\) of only one variable. `hypergraph_formula_refine` does not add node-indexed parameters automatically.

The tool first tests the current shared candidate. If that candidate is a long shared expression and the node-indexed parameter is unidentifiable, it internally contrasts three short shared couplings — product \(x_i x_j\), gated contagion \((1-x_i)x_j\), and phase-difference sine — and tries constant and slope on each. Only fits that pass the same cross-trajectory guards are kept. This is a contrast over that class of dynamics, not a list of problem IDs.

Guards are not relaxed for a single problem:

- At least two trajectories; both training and held-out error improve relative to the shared version.
- Node-indexed parameters are stable across trajectories; between-node differences exceed estimation fluctuation; at least one pair of nodes has overlapping state coverage and different parameters.
- The design is full rank. A shared constant or slope must not be kept together with its node-indexed counterpart.

`status=rejected`: report unidentifiable; do not add quadratic or cubic node parameters. `accepted_for_refinement`: read `selected_mode` (it may differ from the requested `mode` after the contrast). Reintegrate this `candidate_id` immediately. If shared coefficients have not yet rounded to short decimals or small-denominator rationals, or `unresolved_small_terms` / `short_review` still fail, call `hypergraph_formula_refine` once more. Do not change the formula after integration has already returned `status=passed`.

## 5. Reintegration is an exact self-consistency check, not the sole submission criterion

Call `hypergraph_reintegrate(candidate_id=short_id)`. Use the same ID after final rounding and deletion. Run this at least once before submitting. The submitted `f` must be this ID, not a residual name or a formula string typed by the caller.

The tool integrates freely from each original initial condition to every original time, without inserting measured states of other nodes, without piecewise resets, and without fitting initial conditions. The same edges, weights, counting, and node-indexed parameters are used in fitting and in integration. Whether the given edges are true is unknown, so whole-trajectory RMSE \(\le 10^{-5}\) is not the only condition for submission.

- `status=blocked`: missing original grid or other loader problems; stop submission. Do not treat a cropped fitting grid as the full trajectory.
- `status=passed`: whole-trajectory RMSE \(\le 10^{-5}\), `can_submit=true`. This certifies an exact law. Immediately `submit_formula(f=<this candidate_id>)`.
- `status=failed`: the exact gate did not pass. Do not submit after the first failure. First return to `hypergraph_formula_refine` with `rollback_id` (do not pass `residual_allowance`), contrast other short families or a node-indexed parameter, and integrate again. Then read per-node residuals:
  - If one short kernel already meets derivative precision on at least two nodes, lock that kernel. Shared coefficients come from those nodes (submit 0.08 if those nodes give 0.08; do not replace the estimate by 1). Do not add channels, add terms, or retune amplitudes for nodes that do not match.
  - If the leftover residual is consistent with a per-node constant or linear drive, go to step 4 with the coupling kernel locked.
  - If there are no such clean nodes, submit the shortest already-integrated ID (if tied, the smaller integration error). Do not submit the last ID, and do not submit a longer expression whose integration error is only slightly smaller.
  `can_submit` remains false.
- Read per-trajectory RMSE as self-consistency of this given graph plus this kernel, not as a test of whether the graph is true. Do not edit edges from it.

## 6. Submit

Call `submit_formula(f=<candidate_id>)`. `f` must be an ID that has already been reintegrated. Submission is allowed whether or not the exact gate passed. Submission serializes the same candidate and keeps the integration RMSE. Any change to the formula, weights, edges, or observations invalidates the previous integration gate; verify again.

Keep the compact \(F,G,H\), the empty-channel convention, necessary runtime node-indexed parameters, the counting convention, and self-check metrics. `status=blocked` still does not count as a submission.

## Prohibitions

- Do not search for edges. Do not install, delete, or rewrite edges. Do not add edges so as to approach a complete graph. Do not run leave-one-edge.
- Do not call `hypergraph_support_screen`, `hypergraph_binary_support_search`, or `install_hypergraph_relations` on this path.
- Do not read files for hidden truth or trajectories. Do not use the public network. Do not branch on problem IDs. Do not preset node-indexed parameters.
- Do not fit differentiation error as an extra mechanism. Do not invent a kernel for an empty channel. Do not treat the undirected-triplet factor \(1/2\) as a new mechanism.
