# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Compact diagnostics and sparse feature fitting for graph symbolic regression."""
from __future__ import annotations

from functools import reduce
from itertools import combinations_with_replacement
from typing import Any, Dict, List
import warnings

import numpy as np
import sr_harness_engine as engine
from sklearn.linear_model import LassoLarsIC

from .base_tool import BaseTool, ToolMetadata, is_numeric_array


def _number(value: float) -> str:
    return f"{value:.6g}"


@BaseTool.register("graph_data_analysis")
class GraphDataAnalysisTool(BaseTool):
    """Summarize graph structure, node fields, and cheap local/neighbor probes."""

    metadata = ToolMetadata(name="graph_data_analysis")

    def execute(
        self,
        relation: str = None,
        variables: List[str] = None,
        y: str = None,
        top_k: int = 12,
    ) -> Dict[str, Any]:
        """Analyze graph topology and rank simple indexed features by affine target fit.

        This is a bounded first diagnostic for graph symbolic regression. It reports
        degree asymmetry, reciprocity, node-field variation, and affine fits of
        low-order local, incoming, outgoing, and difference probes.

        Args:
            relation: Binary relation name. Infer it when exactly one relation is configured.
            variables: Node-field names to probe. By default, infer numeric fields
                ending in the node axis.
            y: Target variable or indexed expression. Use the discovery target by default.
            top_k: Maximum number of ranked probes to return (1-30).
        """
        if self.context.num_nodes is None or not self.context.relation_names:
            raise ValueError("graph_data_analysis requires context.num_nodes and relation metadata")
        relation_names = sorted(self.context.relation_names)
        relation = relation or (relation_names[0] if len(relation_names) == 1 else None)
        if relation not in self.context.relation_names:
            raise ValueError(f"relation must be one of: {', '.join(relation_names)}")
        edges = np.asarray(self.context.data[relation])
        if edges.shape[1] != 2:
            raise ValueError("automatic graph probes currently require a binary relation")
        n = int(self.context.num_nodes)
        target_text = (y or self.context.target).strip().strip('"').strip("'")
        target_expression = self.parse_node_aligned_formula(target_text)
        analysis_context = self.context.train_split
        target = np.asarray(
            self.expression_values(
                target_expression,
                analysis_context.data,
                num_nodes=analysis_context.num_nodes,
            ),
            dtype=float,
        )
        inferred = []
        for name in self.context.feature_names():
            value = self.context.data[name]
            if (
                name not in self.context.relation_names
                and is_numeric_array(value)
                and np.asarray(value).ndim > 0
                and np.asarray(value).shape[-1] == n
            ):
                inferred.append(name)
        variables = variables or inferred
        invalid = [name for name in variables if name not in self.context.data]
        if invalid:
            raise ValueError(f"unknown variables: {', '.join(invalid)}")

        targets, sources = edges[:, 0], edges[:, 1]
        in_degree = np.bincount(targets, minlength=n)
        out_degree = np.bincount(sources, minlength=n)
        unique_edges = {tuple(map(int, row)) for row in edges}
        reverse_count = sum((source, target) in unique_edges for target, source in unique_edges)
        structure = {
            "relation": relation,
            "num_nodes": n,
            "num_edges": int(len(edges)),
            "density_without_self_loops": float(len(edges) / max(n * (n - 1), 1)),
            "self_loops": int(np.count_nonzero(targets == sources)),
            "duplicate_rows": int(len(edges) - len(unique_edges)),
            "reciprocal_edge_ratio": (
                float(reverse_count / len(unique_edges)) if unique_edges else 0.0
            ),
            "in_degree": self._degree_summary(in_degree),
            "out_degree": self._degree_summary(out_degree),
            "regular_in_degree": bool(np.all(in_degree == in_degree[0])),
            "regular_out_degree": bool(np.all(out_degree == out_degree[0])),
        }

        fields = {}
        probes = []
        exceptions = []
        for name in variables:
            value = np.asarray(analysis_context.data[name])
            if not is_numeric_array(value) or value.ndim == 0 or value.shape[-1] != n:
                exceptions.append(
                    f"Skipped {name!r}: expected a numeric node field ending in {n} nodes."
                )
                continue
            finite = np.asarray(value, dtype=float)[np.isfinite(value)]
            fields[name] = {
                "shape": list(value.shape),
                "minimum": float(np.min(finite)) if finite.size else float("nan"),
                "maximum": float(np.max(finite)) if finite.size else float("nan"),
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "std": float(np.std(finite)) if finite.size else float("nan"),
                "finite_ratio": float(finite.size / value.size) if value.size else 0.0,
                "constant_across_leading_samples": bool(
                    value.ndim == 1 or np.allclose(value, value.reshape((-1, n))[0], equal_nan=True)
                ),
            }
            formulas = [
                ("local", f"{name}[i]"),
                ("local", f"{name}[i] ** 2"),
                ("local", f"{name}[i] ** 3"),
                ("source", f"sum[j]({relation}[i, j], {name}[j])"),
                ("difference", f"sum[j]({relation}[i, j], {name}[j] - {name}[i])"),
                ("reverse_source", f"sum[j]({relation}[j, i], {name}[j])"),
            ]
            for family, formula in formulas:
                try:
                    expression = self.parse_formula(formula)
                    values = self.expression_values(
                        expression,
                        analysis_context.data,
                        num_nodes=analysis_context.num_nodes,
                    )
                    probes.append(self._affine_probe(formula, values, target) | {"family": family})
                except Exception as exc:
                    exceptions.append(f"Probe {formula!r} failed: {exc}")
        probes.sort(key=lambda item: (-item["r2"], item["rmse"], item["formula"]))
        by_family = {}
        for probe in probes:
            by_family.setdefault(probe["family"], [])
            if len(by_family[probe["family"]]) < 3:
                by_family[probe["family"]].append(probe)
        return {
            "analysis_split": "train",
            "target_expression": target_expression.to_str(),
            "structure": structure,
            "fields": fields,
            "top_affine_probes": probes[:max(1, min(int(top_k), 30))],
            "top_affine_probes_by_family": by_family,
            "exceptions": exceptions,
        }

    @staticmethod
    def _degree_summary(values: np.ndarray) -> Dict[str, Any]:
        return {
            "minimum": int(np.min(values)),
            "maximum": int(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "isolated_count": int(np.count_nonzero(values == 0)),
            "values": values.astype(int).tolist() if len(values) <= 32 else None,
        }

    @staticmethod
    def _affine_probe(formula: str, feature: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
        feature, target = np.broadcast_arrays(np.asarray(feature, dtype=float), target)
        x, yy = feature.ravel(), target.ravel()
        finite = np.isfinite(x) & np.isfinite(yy)
        if np.count_nonzero(finite) < 3:
            raise ValueError("fewer than three jointly finite values")
        design = np.column_stack([np.ones(np.count_nonzero(finite)), x[finite]])
        coefficients, _, rank, _ = np.linalg.lstsq(design, yy[finite], rcond=None)
        prediction = design @ coefficients
        residual = prediction - yy[finite]
        total = float(np.sum((yy[finite] - np.mean(yy[finite])) ** 2))
        return {
            "formula": formula,
            "intercept": float(coefficients[0]),
            "coefficient": float(coefficients[1]),
            "r2": float(1 - np.sum(residual**2) / total) if total else float("nan"),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "pearson": (
                float(np.corrcoef(x[finite], yy[finite])[0, 1])
                if np.std(x[finite]) else float("nan")
            ),
            "matrix_rank": int(rank),
        }

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        """Format graph-data diagnostics as concise model-facing text."""
        structure = result["structure"]
        lines = [
            f"Graph: relation={structure['relation']}, nodes={structure['num_nodes']}, "
            f"edges={structure['num_edges']}, "
            f"self-loops={structure['self_loops']}, duplicates={structure['duplicate_rows']}, "
            f"reciprocal={structure['reciprocal_edge_ratio']:.3g}.",
            f"In-degree min/mean/max={structure['in_degree']['minimum']}/"
            f"{structure['in_degree']['mean']:.3g}/{structure['in_degree']['maximum']}; "
            f"out-degree min/mean/max={structure['out_degree']['minimum']}/"
            f"{structure['out_degree']['mean']:.3g}/{structure['out_degree']['maximum']}.",
            "Node fields:",
        ]
        for name, field in result["fields"].items():
            lines.append(
                f"  - {name}: shape={tuple(field['shape'])}, range=[{_number(field['minimum'])}, "
                f"{_number(field['maximum'])}], std={_number(field['std'])}, "
                f"constant_across_samples={field['constant_across_leading_samples']}"
            )
        lines.append("Top one-feature affine probes (intercept + coefficient * feature):")
        for probe in result["top_affine_probes"]:
            lines.append(
                f"  - R2={probe['r2']:.6g}, RMSE={probe['rmse']:.6g}: "
                f"{_number(probe['intercept'])} + {_number(probe['coefficient'])} * "
                f"({probe['formula']})"
            )
        lines.append("Best probes retained within each structural family:")
        for family, probes in result["top_affine_probes_by_family"].items():
            best = probes[0]
            lines.append(
                f"  - {family}: R2={best['r2']:.6g}, RMSE={best['rmse']:.6g}: "
                f"{best['formula']}"
            )
        if result["exceptions"]:
            lines.append("Warnings: " + "; ".join(result["exceptions"][:5]))
        return "\n".join(lines)


@BaseTool.register("graph_structure_screen")
class GraphStructureScreenTool(BaseTool):
    """Screen a bounded factorized graph grammar without benchmark-specific templates."""

    metadata = ToolMetadata(name="graph_structure_screen")

    def execute(
        self,
        relation: str = None,
        variables: List[str] = None,
        local_degree: int = 3,
        source_powers: List[int] = None,
        message_families: List[str] = None,
        max_terms: int = 20,
        fit_refinements: bool = True,
    ) -> Dict[str, Any]:
        """Jointly screen local products, source messages, saturations, and gates.

        The generated library is deliberately generic: low-degree local monomials,
        source powers, power saturations, source-target sine differences, and
        multiplication of each aggregate by a sample-varying target-node field.
        A standardized BIC-Lasso screen considers the terms jointly, then ordinary
        least squares refits the retained support.

        Args:
            relation: Binary relation name. Infer it when exactly one is configured.
            variables: Node fields to use. Infer numeric node-aligned features by default.
            local_degree: Maximum total degree of local monomials (1-3).
            source_powers: Positive integer source powers to screen. Defaults to 1, 2, 3.
            message_families: Generic coordinate families to include. Choose from
                ``source_power``, ``power_saturation``, and ``sine_difference``.
            max_terms: Maximum number of jointly selected terms (1-20).
            fit_refinements: Whether to fit the bounded parameterized message
                families after the fixed-library screen.
        """
        if self.context.num_nodes is None or not self.context.relation_names:
            raise ValueError("graph_structure_screen requires graph relation metadata")
        relation_names = sorted(self.context.relation_names)
        relation = relation or (relation_names[0] if len(relation_names) == 1 else None)
        if relation not in self.context.relation_names:
            raise ValueError(f"relation must be one of: {', '.join(relation_names)}")
        if np.asarray(self.context.data[relation]).shape[1] != 2:
            raise ValueError("graph_structure_screen currently requires a binary relation")
        n = int(self.context.num_nodes)
        inferred = [
            name for name in self.context.feature_names()
            if name not in self.context.relation_names
            and is_numeric_array(self.context.data[name])
            and np.asarray(self.context.data[name]).ndim > 0
            and np.asarray(self.context.data[name]).shape[-1] == n
        ]
        variables = list(dict.fromkeys(variables or inferred))
        invalid = [name for name in variables if name not in inferred]
        if invalid:
            raise ValueError(
                "variables must be numeric node-aligned features: " + ", ".join(invalid)
            )
        if not variables:
            raise ValueError("No numeric node-aligned feature is available for screening.")
        local_degree = max(1, min(int(local_degree), 3))
        source_powers = list(dict.fromkeys(source_powers or [1, 2, 3]))
        if any(isinstance(power, bool) or not isinstance(power, int) or power < 1 or power > 5
               for power in source_powers):
            raise ValueError("source_powers must contain integers from 1 through 5")
        message_families = list(dict.fromkeys(message_families or [
            "source_power", "power_saturation", "sine_difference",
        ]))
        supported_families = {"source_power", "power_saturation", "sine_difference"}
        if unknown_families := set(message_families) - supported_families:
            raise ValueError(
                "unknown message_families: " + ", ".join(sorted(unknown_families))
            )

        field_roles = {}
        dynamic = []
        for name in variables:
            value = np.asarray(self.context.train_split.data[name], dtype=float)
            flat = value.reshape((-1, n))
            sample_varying = value.ndim > 1 and not np.allclose(
                flat, flat[0], equal_nan=True,
            )
            field_roles[name] = "sample-varying" if sample_varying else "sample-invariant"
            if sample_varying:
                dynamic.append(name)

        atoms = [f"{name}[i]" for name in variables]
        local_features = []
        for degree in range(1, local_degree + 1):
            local_features.extend(
                " * ".join(items) for items in combinations_with_replacement(atoms, degree)
            )
        gates = ["1", *(f"{name}[i]" for name in dynamic)]
        messages = []
        if "source_power" in message_families:
            for power in source_powers:
                messages.extend(f"{name}[j] ** {power}" for name in dynamic)
        if "power_saturation" in message_families:
            for power in source_powers:
                messages.extend(
                    f"{name}[j] ** {power} / (1 + abs({name}[j]) ** {power})"
                    for name in dynamic
                )
        if "sine_difference" in message_families:
            messages.extend(f"sin({name}[j] - {name}[i])" for name in dynamic)
        graph_features = []
        for message in messages:
            aggregate = f"sum[j]({relation}[i, j], {message})"
            graph_features.extend(
                aggregate if gate == "1" else f"({gate}) * ({aggregate})"
                for gate in gates
            )
        all_features = list(dict.fromkeys([*local_features, *graph_features]))
        if len(all_features) > 128:
            all_features = all_features[:128]
        screen = GraphFeatureFitTool(context=self.context).execute(
            features=all_features,
            max_terms=max_terms,
            selection_method="bic_lasso",
            show_diagnostics=True,
        )
        refinement_templates = []
        for name in dynamic:
            for power in source_powers:
                refinement_templates.append(
                    f"{name}[j] ** {power} / "
                    f"(param('scale', value=1) + abs({name}[j]) ** {power})"
                )
            refinement_templates.extend([
                f"sin(param('frequency', value=1) * ({name}[j] - {name}[i]))",
                f"sigmoid(param('slope', value=1) * "
                f"({name}[j] - param('center', value=0)))",
                f"tanh(param('slope', value=1) * "
                f"({name}[j] - param('center', value=0)))",
            ])
        refinement = None
        linear_metrics = screen["data_split_results"].get(
            "validation", screen["data_split_results"]["train"],
        )["metrics"]
        if fit_refinements and refinement_templates and linear_metrics["r2"] < 1 - 1e-10:
            # Start the nonlinear comparison from the lowest-complexity local
            # baseline.  High-order local terms selected to mimic a missing
            # message nonlinearity otherwise absorb its residual and degrade
            # parameter identification.
            retained_local = atoms[:8]
            bounded_templates = refinement_templates[:max(1, 96 // len(gates))]
            refinement = GraphTemplateFitTool(context=self.context).execute(
                messages=bounded_templates,
                local_features=retained_local,
                gates=gates,
                relation=relation,
                top_k=min(12, len(bounded_templates) * len(gates)),
            )
        best = screen
        if refinement is not None:
            screen_mse = linear_metrics["mse"]
            refinement_mse = refinement["data_split_results"].get(
                "validation", refinement["data_split_results"]["train"],
            )["metrics"]["mse"]
            if refinement_mse < screen_mse:
                best = refinement
        return best | {
            "field_roles": field_roles,
            "generated_library": {
                "local_feature_count": len(local_features),
                "message_count": len(messages),
                "gate_count": len(gates),
                "screened_feature_count": len(all_features),
                "truncated_at_limit": len(local_features) + len(graph_features) > 128,
            },
            "refinement_templates": refinement_templates,
            "linear_screen": {
                "formula": screen["formula"],
                "data_split_results": screen["data_split_results"],
                "selected_features": screen["selected_features"],
                "fitted_coefficients": screen["fitted_coefficients"],
            },
            "nonlinear_refinement": refinement,
            "screen_configuration": {
                "relation": relation,
                "variables": variables,
                "local_degree": local_degree,
                "source_powers": source_powers,
                "message_families": message_families,
                "fit_refinements": fit_refinements,
            },
        }

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        """Format the factorized structure screen and its next-step evidence."""
        lines = [GraphFeatureFitTool.format_evaluation_result(
            result, title="Generic factorized graph-structure screen",
        )]
        roles = ", ".join(f"{name}={role}" for name, role in result["field_roles"].items())
        library = result["generated_library"]
        lines.extend([
            f"Field roles: {roles}.",
            "Generated library: "
            f"{library['local_feature_count']} local terms, "
            f"{library['message_count']} messages, {library['gate_count']} gates; "
            f"screened {library['screened_feature_count']} joint features.",
            "Selected joint support:",
        ])
        for feature, coefficient in result["linear_screen"]["fitted_coefficients"].items():
            lines.append(f"  - {_number(coefficient)} * {feature}")
        metrics = result["data_split_results"].get(
            "validation", result["data_split_results"]["train"],
        )["metrics"]
        refinement = result["nonlinear_refinement"]
        if refinement is not None:
            lines.append("Bounded parameterized-message refinement was executed automatically.")
            for item in refinement["ranked_candidates"][:5]:
                candidate_metrics = item["validation_metrics"]
                lines.append(
                    f"  - R2={candidate_metrics['r2']:.8g}, "
                    f"RMSE={candidate_metrics['rmse']:.8g}: "
                    f"message={item['message']}, gate={item['gate']}"
                )
        if metrics["r2"] < 1 - 1e-10 and result["refinement_templates"]:
            lines.append(
                "Neither bounded screen is exact. Use residual evidence to propose a small "
                "contrasting set of additional parameterized messages for graph_template_fit:"
            )
            lines.extend(f"  - {item}" for item in result["refinement_templates"][:12])
        return "\n".join(lines)


@BaseTool.register("graph_template_fit")
class GraphTemplateFitTool(BaseTool):
    """Fit agent-supplied graph-message templates without a built-in kernel catalog."""

    metadata = ToolMetadata(name="graph_template_fit")

    def execute(
        self,
        messages: List[str],
        local_features: List[str] = None,
        gates: List[str] = None,
        relation: str = None,
        y: str = None,
        include_bias: bool = True,
        top_k: int = 6,
    ) -> Dict[str, Any]:
        """Fit and rank arbitrary edge-message templates with optional node gates.

        Each message is an expression over ``i`` and ``j`` that the tool wraps in
        the selected relation reduction. The caller, not the tool, chooses the
        operator family. Parameters written with ``param(...)`` are jointly fitted
        with local coefficients and graph gain. To test multiple initializations,
        include the same structure several times with different parameter values.

        Args:
            messages: Edge-message expressions, for example ``u[j]-u[i]``.
            local_features: Evidence-supported local terms whose coefficients are refit.
            gates: Multipliers outside the relation sum. Defaults to the ungated value ``1``.
            relation: Binary relation name. Infer it when exactly one is configured.
            y: Target variable or indexed residual expression. Use the target by default.
            include_bias: Whether every candidate has a fitted additive bias.
            top_k: Number of fitted message/gate combinations to return (1-20).
        """
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list of edge expressions")
        relation_names = sorted(self.context.relation_names)
        relation = relation or (relation_names[0] if len(relation_names) == 1 else None)
        if relation not in self.context.relation_names:
            raise ValueError(f"relation must be one of: {', '.join(relation_names)}")
        if np.asarray(self.context.data[relation]).shape[1] != 2:
            raise ValueError("graph_template_fit currently requires a binary relation")
        local_features = list(dict.fromkeys(local_features or []))
        gates = list(dict.fromkeys(gates or ["1"]))
        if len(messages) * len(gates) > 96:
            raise ValueError("messages * gates may not exceed 96")
        target_text = (y or self.context.target).strip().strip('"').strip("'")
        target_expression = self.parse_node_aligned_formula(target_text)
        for text in [*messages, *local_features, *gates]:
            self.parse_formula(text)

        fitted = []
        rejected = []
        for message in messages:
            aggregate = f"sum[j]({relation}[i, j], ({message}))"
            for gate in gates:
                graph_feature = (
                    aggregate if gate.strip() == "1" else f"({gate}) * ({aggregate})"
                )
                parts = []
                if include_bias:
                    parts.append("param('_gt_bias', value=0)")
                parts.extend(
                    f"param('_gt_local_{index}', value=0) * ({feature})"
                    for index, feature in enumerate(local_features)
                )
                parts.append(f"param('_gt_gain', value=1) * ({graph_feature})")
                formula_text = " + ".join(parts)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        evaluation = self.evaluate(
                            f=self.parse_formula(formula_text), y=target_expression,
                            fit=True, show_diagnostics=False,
                        )
                    split_results = evaluation["data_split_results"]
                    fitted.append({
                        "message": message,
                        "gate": gate,
                        "formula": evaluation["formula"],
                        "train_metrics": split_results["train"]["metrics"],
                        "validation_metrics": split_results.get(
                            "validation", split_results["train"],
                        )["metrics"],
                    })
                except Exception as exc:
                    rejected.append({"message": message, "gate": gate, "reason": str(exc)})
        if not fitted:
            raise ValueError("No graph message template could be fitted.")
        fitted.sort(key=lambda item: (
            item["validation_metrics"]["mse"], len(item["formula"]), item["message"],
        ))
        evaluation = self.evaluate(
            f=self.parse_formula(fitted[0]["formula"]), y=target_expression,
            fit=False, show_diagnostics=True,
        )
        return evaluation | {
            "ranked_candidates": fitted[:max(1, min(int(top_k), 20))],
            "rejected_candidates": rejected[:20],
            "search_configuration": {
                "messages": messages,
                "local_features": local_features,
                "gates": gates,
                "relation": relation,
                "include_bias": include_bias,
            },
        }

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        """Format ranked graph-message template fits for the model."""
        lines = [cls.format_evaluation_result(result, title="Best fitted graph template")]
        lines.append("Ranked caller-supplied message templates:")
        for item in result["ranked_candidates"]:
            metrics = item["validation_metrics"]
            lines.append(
                f"  - message={item['message']}, gate={item['gate']}: "
                f"R2={metrics['r2']:.8g}, RMSE={metrics['rmse']:.8g}; {item['formula']}"
            )
        if result["rejected_candidates"]:
            lines.append(f"Rejected candidate combinations: {len(result['rejected_candidates'])}")
        return "\n".join(lines)


@BaseTool.register("graph_residual_analysis")
class GraphResidualAnalysisTool(BaseTool):
    """Diagnose graph residual structure without selecting a mechanism in advance."""

    metadata = ToolMetadata(name="graph_residual_analysis")

    def execute(
        self,
        candidate: str,
        probes: List[str] = None,
        normalizers: List[str] = None,
        y: str = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Measure conservation, node bias, and gated probe fits for a residual.

        The tool makes no operator-family choice. Supply probes motivated by prior
        diagnostics; each is tested alone and multiplied by each proposed target-node
        normalizer. This turns residual inspection and factorization checks into one
        reproducible operation.

        Args:
            candidate: Current complete or local-only formula.
            probes: Candidate local or relation-reduced expressions for the residual.
            normalizers: Possible multiplicative gates. Defaults to the ungated value ``1``.
            y: Target variable or indexed expression. Use the discovery target by default.
            top_k: Number of affine residual-probe fits to return (1-30).
        """
        candidate_expression = self.parse_formula(candidate)
        target_text = (y or self.context.target).strip().strip('"').strip("'")
        target_expression = self.parse_node_aligned_formula(target_text)
        analysis_context = self.context.train_split
        target = np.asarray(
            self.expression_values(
                target_expression,
                analysis_context.data,
                num_nodes=analysis_context.num_nodes,
            ),
            dtype=float,
        )
        prediction = np.asarray(
            self.expression_values(
                candidate_expression,
                analysis_context.data,
                num_nodes=analysis_context.num_nodes,
            ),
            dtype=float,
        )
        prediction, target = np.broadcast_arrays(prediction, target)
        residual = target - prediction
        if residual.ndim == 0 or residual.shape[-1] != self.context.num_nodes:
            raise ValueError("candidate residual must end in the configured node dimension")
        finite = np.isfinite(residual)
        if not np.all(finite):
            raise ValueError("candidate residual contains non-finite values")
        rms = float(np.sqrt(np.mean(residual**2)))
        node_means = np.mean(residual.reshape((-1, residual.shape[-1])), axis=0)
        sample_sums = np.sum(residual, axis=-1)
        conservation_ratio = float(
            np.sqrt(np.mean(sample_sums**2))
            / max(np.sqrt(residual.shape[-1]) * rms, np.finfo(float).eps)
        )

        ranked = []
        rejected = []
        probes = list(dict.fromkeys(probes or []))
        normalizers = list(dict.fromkeys(normalizers or ["1"]))
        for probe in probes:
            for normalizer in normalizers:
                feature_text = probe if normalizer.strip() == "1" else (
                    f"({normalizer}) * ({probe})"
                )
                try:
                    values = self.expression_values(
                        self.parse_formula(feature_text),
                        analysis_context.data,
                        num_nodes=analysis_context.num_nodes,
                    )
                    ranked.append(
                        GraphDataAnalysisTool._affine_probe(feature_text, values, residual)
                        | {"probe": probe, "normalizer": normalizer}
                    )
                except Exception as exc:
                    rejected.append({
                        "probe": probe, "normalizer": normalizer, "reason": str(exc),
                    })
        ranked.sort(key=lambda item: (-item["r2"], item["rmse"], item["formula"]))
        evaluation = self.evaluate(
            f=candidate_expression, y=target_expression, show_diagnostics=False,
        )
        residual_expression = (
            f"({target_expression.to_str()}) - ({candidate_expression.to_str()})"
        )
        return evaluation | {
            "analysis_split": "train",
            "residual_expression": residual_expression,
            "residual_summary": {
                "rmse": rms,
                "mean": float(np.mean(residual)),
                "maximum_absolute": float(np.max(np.abs(residual))),
                "sample_node_sum_rmse": float(np.sqrt(np.mean(sample_sums**2))),
                "sample_node_sum_to_independent_scale": conservation_ratio,
                "node_mean_minimum": float(np.min(node_means)),
                "node_mean_maximum": float(np.max(node_means)),
                "node_mean_rmse": float(np.sqrt(np.mean(node_means**2))),
            },
            "ranked_residual_probes": ranked[:max(1, min(int(top_k), 30))],
            "rejected_probes": rejected[:20],
        }

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        """Format graph-residual invariants and ranked probes for the model."""
        summary = result["residual_summary"]
        lines = [
            f"Residual: {result['residual_expression']}",
            f"RMSE={summary['rmse']:.8g}, max_abs={summary['maximum_absolute']:.8g}, "
            f"mean={summary['mean']:.8g}.",
            "Samplewise node-sum RMSE="
            f"{summary['sample_node_sum_rmse']:.8g}; ratio to independent-node scale="
            f"{summary['sample_node_sum_to_independent_scale']:.8g}.",
            f"Node-mean range=[{summary['node_mean_minimum']:.8g}, "
            f"{summary['node_mean_maximum']:.8g}], "
            f"node-mean RMSE={summary['node_mean_rmse']:.8g}.",
        ]
        if result["ranked_residual_probes"]:
            lines.append("Best affine fits to the residual:")
            for probe in result["ranked_residual_probes"]:
                lines.append(
                    f"  - R2={probe['r2']:.8g}, RMSE={probe['rmse']:.8g}: "
                    f"normalizer={probe['normalizer']}, probe={probe['probe']}"
                )
        return "\n".join(lines)


@BaseTool.register("graph_feature_fit")
class GraphFeatureFitTool(BaseTool):
    """Fit a sparse linear combination of ordinary or indexed feature formulas."""

    metadata = ToolMetadata(name="graph_feature_fit")

    def execute(
        self,
        features: List[str],
        y: str = None,
        include_bias: bool = True,
        max_terms: int = 6,
        min_relative_improvement: float = 1e-8,
        beam_width: int = 4,
        selection_method: str = "beam",
        show_diagnostics: bool = True,
    ) -> Dict[str, Any]:
        """Select and fit a compact linear combination of formula features.

        Features may be ordinary formulas or graph formulas such as ``x[i]**3``
        and ``sum[j](R[i,j], sin(x[j]-x[i]))``. Use it to reproduce a custom
        NumPy design-matrix search without writing code. For nonlinear constants
        inside a chosen structure, follow with ``evaluate_formula(..., fit=true)``.

        Args:
            features: Candidate feature formulas. Supply structurally meaningful
                terms, not the target.
            y: Target variable or indexed expression. Use the discovery target by default.
            include_bias: Whether every fitted model includes an intercept.
            max_terms: Maximum selected non-bias features (1-20).
            min_relative_improvement: Stop when the fractional training-MSE
                improvement falls below this value.
            beam_width: Number of partial feature sets to retain at each step
                (1-32). Values above one can recover jointly useful features
                that greedy forward selection misses.
            selection_method: ``beam`` for bounded forward subset search or
                ``bic_lasso`` for a standardized Lasso-LARS screen followed by
                unpenalized refitting. The latter is useful for correlated
                trajectory libraries where no true term ranks well alone.
            show_diagnostics: Whether the final candidate includes residual diagnostics.
        """
        if not isinstance(features, list) or not features:
            raise ValueError("features must be a non-empty list of formula strings")
        if len(features) > 128:
            raise ValueError("features may contain at most 128 formulas")
        max_terms = max(1, min(int(max_terms), 20, len(features)))
        min_relative_improvement = max(0.0, float(min_relative_improvement))
        beam_width = max(1, min(int(beam_width), 32))
        if selection_method not in {"beam", "bic_lasso"}:
            raise ValueError("selection_method must be 'beam' or 'bic_lasso'")
        target_text = (y or self.context.target).strip().strip('"').strip("'")
        target_expression = self.parse_node_aligned_formula(target_text)
        train_context = self.context.train_split
        target = np.asarray(
            self.expression_values(
                target_expression, train_context.data, num_nodes=train_context.num_nodes,
            ),
            dtype=float,
        )
        expressions = []
        columns = []
        rejected = []
        for text in dict.fromkeys(features):
            try:
                expression = self.parse_formula(text)
                if engine.unbound_parameters(expression):
                    raise ValueError("feature contains unbound parameters")
                values = self.expression_values(
                    expression, train_context.data, num_nodes=train_context.num_nodes,
                )
                values, broadcast_target = np.broadcast_arrays(values, target)
                column = np.asarray(values, dtype=float).ravel()
                if not np.all(np.isfinite(column)):
                    raise ValueError("feature contains non-finite values")
                expressions.append(expression)
                columns.append(column)
                target = broadcast_target
            except Exception as exc:
                rejected.append({"feature": text, "reason": str(exc)})
        if not expressions:
            raise ValueError("No feature could be evaluated against the target.")
        yy = np.asarray(target, dtype=float).ravel()
        if not np.all(np.isfinite(yy)):
            raise ValueError("target contains non-finite values")
        matrix = np.column_stack(columns)
        total = float(np.sum((yy - np.mean(yy)) ** 2))
        if selection_method == "bic_lasso":
            selected, trace, beam_trace = self._bic_lasso_selection(
                matrix, yy, expressions, include_bias=include_bias, max_terms=max_terms,
            )
        else:
            selected, trace, beam_trace = self._beam_selection(
                matrix, yy, expressions, include_bias=include_bias,
                max_terms=max_terms, min_relative_improvement=min_relative_improvement,
                beam_width=beam_width, total=total,
            )

        design = matrix[:, selected]
        if include_bias:
            design = np.column_stack([np.ones(len(yy)), design])
        coefficients = np.linalg.lstsq(design, yy, rcond=None)[0]
        offset = 1 if include_bias else 0
        parts = []
        if include_bias and coefficients[0] != 0:
            parts.append(engine.Number(float(coefficients[0])))
        for coefficient, index in zip(coefficients[offset:], selected):
            if coefficient != 0:
                parts.append(engine.Number(float(coefficient)) * expressions[index])
        formula = reduce(lambda left, right: left + right, parts) if parts else engine.Number(0)
        evaluation = self.evaluate(
            f=formula, y=target_expression, show_diagnostics=show_diagnostics,
        )
        return evaluation | {
            "selected_features": [expressions[index].to_str() for index in selected],
            "fitted_coefficients": {
                **({"bias": float(coefficients[0])} if include_bias else {}),
                **{
                    expressions[index].to_str(): float(coefficient)
                    for coefficient, index in zip(coefficients[offset:], selected)
                },
            },
            "selection_trace": trace,
            "beam_trace": beam_trace,
            "rejected_features": rejected,
            "fit_configuration": {
                "candidate_count": len(expressions),
                "include_bias": include_bias,
                "max_terms": max_terms,
                "min_relative_improvement": min_relative_improvement,
                "beam_width": beam_width,
                "selection_method": selection_method,
            },
        }

    @staticmethod
    def _beam_selection(
        matrix: np.ndarray,
        yy: np.ndarray,
        expressions: list[engine.Expression],
        *,
        include_bias: bool,
        max_terms: int,
        min_relative_improvement: float,
        beam_width: int,
        total: float,
    ) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
        baseline = np.mean(yy) if include_bias else 0.0
        baseline_mse = float(np.mean((yy - baseline) ** 2))
        beam = [{
            "selected": [],
            "mse": baseline_mse,
            "trace": [],
        }]
        beam_trace = []
        for step in range(1, max_terms + 1):
            expanded = {}
            for state in beam:
                for index in range(matrix.shape[1]):
                    if index in state["selected"]:
                        continue
                    selected = state["selected"] + [index]
                    design = matrix[:, selected]
                    if include_bias:
                        design = np.column_stack([np.ones(len(yy)), design])
                    coefficients, _, rank, _ = np.linalg.lstsq(design, yy, rcond=None)
                    if rank < design.shape[1]:
                        continue
                    residual = yy - design @ coefficients
                    mse = float(np.mean(residual**2))
                    improvement = (state["mse"] - mse) / max(
                        state["mse"], np.finfo(float).eps,
                    )
                    if state["selected"] and improvement < min_relative_improvement:
                        continue
                    trace = state["trace"] + [{
                        "step": step,
                        "added_feature": expressions[index].to_str(),
                        "mse": mse,
                        "rmse": float(np.sqrt(mse)),
                        "r2": float(1 - mse * len(yy) / total) if total else float("nan"),
                        "relative_mse_improvement": float(improvement),
                    }]
                    candidate = {
                        "selected": selected,
                        "mse": mse,
                        "trace": trace,
                    }
                    subset = tuple(sorted(selected))
                    key = (
                        mse,
                        sum(len(expressions[item].to_str()) for item in selected),
                        tuple(expressions[item].to_str() for item in selected),
                    )
                    previous = expanded.get(subset)
                    if previous is None or key < previous[0]:
                        expanded[subset] = (key, candidate)
            if not expanded:
                break
            ranked = sorted(expanded.values(), key=lambda item: item[0])
            beam = [item[1] for item in ranked[:beam_width]]
            beam_trace.append({
                "step": step,
                "explored_model_count": len(expanded),
                "retained_model_count": len(beam),
                "best_mse": beam[0]["mse"],
                "best_r2": float(1 - beam[0]["mse"] * len(yy) / total)
                if total else float("nan"),
            })
            if beam[0]["mse"] <= np.finfo(float).eps * max(float(np.mean(yy**2)), 1.0):
                break

        best_state = beam[0]
        return best_state["selected"], best_state["trace"], beam_trace

    @staticmethod
    def _bic_lasso_selection(
        matrix: np.ndarray,
        yy: np.ndarray,
        expressions: list[engine.Expression],
        *,
        include_bias: bool,
        max_terms: int,
    ) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
        means = np.mean(matrix, axis=0)
        scales = np.std(matrix, axis=0)
        usable = scales > np.finfo(float).eps * np.maximum(np.abs(means), 1.0)
        if not np.any(usable):
            raise ValueError("All candidate features are constant on the training split.")
        standardized = (matrix[:, usable] - means[usable]) / scales[usable]
        noise_variance = None
        if standardized.shape[0] <= standardized.shape[1] + int(include_bias):
            noise_variance = max(
                float(np.var(yy)) * 1e-12,
                np.finfo(float).eps,
            )
        model = LassoLarsIC(
            criterion="bic",
            fit_intercept=include_bias,
            noise_variance=noise_variance,
        ).fit(standardized, yy)
        usable_indices = np.flatnonzero(usable)
        active = np.flatnonzero(np.abs(model.coef_) > np.finfo(float).eps)
        if not len(active):
            active = np.asarray([int(np.argmax(np.abs(model.coef_)))])
        ranked = sorted(active, key=lambda index: (-abs(model.coef_[index]), int(index)))
        selected = [int(usable_indices[index]) for index in ranked[:max_terms]]
        design = matrix[:, selected]
        if include_bias:
            design = np.column_stack([np.ones(len(yy)), design])
        coefficients = np.linalg.lstsq(design, yy, rcond=None)[0]
        offset = int(include_bias)
        contribution = np.abs(coefficients[offset:]) * np.std(matrix[:, selected], axis=0)
        material = contribution >= max(float(np.std(yy)) * 1e-8, np.finfo(float).eps)
        if np.any(material) and not np.all(material):
            selected = [index for index, keep in zip(selected, material) if keep]
            design = matrix[:, selected]
            if include_bias:
                design = np.column_stack([np.ones(len(yy)), design])
            coefficients = np.linalg.lstsq(design, yy, rcond=None)[0]
        prediction = design @ coefficients
        mse = float(np.mean((yy - prediction) ** 2))
        total = float(np.sum((yy - np.mean(yy)) ** 2))
        trace = [{
            "step": 1,
            "added_feature": "BIC-Lasso joint support",
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": float(1 - mse * len(yy) / total) if total else float("nan"),
            "relative_mse_improvement": float("nan"),
        }]
        return selected, trace, []

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        """Format sparse graph feature selection and its trace for the model."""
        lines = [cls.format_evaluation_result(result, title="Sparse graph feature fit")]
        lines.append("Feature-selection trace:")
        for step in result["selection_trace"]:
            lines.append(
                f"  - step {step['step']}: add {step['added_feature']}; "
                f"R2={step['r2']:.6g}, RMSE={step['rmse']:.6g}, "
                f"relative MSE improvement={step['relative_mse_improvement']:.6g}"
            )
        if result["rejected_features"]:
            lines.append(
                "Rejected features: "
                + "; ".join(
                    f"{item['feature']}: {item['reason']}"
                    for item in result["rejected_features"][:5]
                )
            )
        return "\n".join(lines)
