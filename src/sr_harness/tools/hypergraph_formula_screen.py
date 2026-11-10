# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Screen a compact pairwise/triplet formula library when edge tables are given."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base_tool import BaseTool, ToolMetadata
from .graph_analysis import GraphFeatureFitTool
from ._hypergraph_known_edge_common import (
    form_screen_features,
    host_state,
    inputs,
    is_multivariate,
    json_safe,
    known_edges_enabled,
    readable,
    relations,
    screen_handoff,
    weighted_screen,
)


def _relation_by_arity(context, arity: int) -> str:
    """Return the unique relation whose rows have the requested column count."""
    matches = []
    for name in sorted(context.relation_names):
        table = np.asarray(context.data[name])
        if table.ndim == 2 and table.shape[1] == arity:
            matches.append(name)
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {arity}-column relation, found {matches or 'none'}"
        )
    return matches[0]


def _node_field_names(context) -> List[str]:
    """Return numeric node-aligned feature names, excluding relation tables."""
    if context.num_nodes is None:
        raise ValueError("hypergraph_formula_screen requires context.num_nodes")
    n = int(context.num_nodes)
    names = []
    for name in context.feature_names():
        if name in context.relation_names:
            continue
        value = context.data[name]
        array = np.asarray(value)
        if array.ndim > 0 and array.shape[-1] == n and np.issubdtype(array.dtype, np.number):
            names.append(name)
    if not names:
        raise ValueError("No numeric node-aligned feature is available.")
    return names


@BaseTool.register("hypergraph_formula_screen")
class HypergraphFormulaScreenTool(BaseTool):
    """Screen a compact shared library on given pairwise and/or triplet tables."""

    metadata = ToolMetadata(
        name="hypergraph_formula_screen",
        description=(
            "Screen a compact local + pairwise + triplet formula library when pairwise "
            "and/or triplet edge tables are already installed. Either table may be empty. "
            "Does not search for unknown edges."
        ),
    )

    def execute(
        self,
        pairwise: str = None,
        triplet: str = None,
        variables: List[str] = None,
        max_terms: int = 6,
        y: str = None,
    ) -> Dict[str, Any]:
        """Select a sparse formula on fixed pairwise and/or triplet edges.

        Local, pairwise, and triplet messages enter one combined library. Terms that
        evaluate to non-finite values on the training split are dropped by
        ``graph_feature_fit``. Empty pairwise or triplet tables skip that side.

        Args:
            pairwise: Two-column relation name. Infer when exactly one exists.
            triplet: Three-column relation name. Infer when exactly one exists.
            variables: Node fields to use. Infer numeric node-aligned features by default.
            max_terms: Maximum selected non-bias features (1-20).
            y: Target variable or indexed expression. Use the discovery target by default.
        """
        if is_multivariate(self.context):
            from ._hypergraph_multivariate import screen_multivariate
            return screen_multivariate(self.context, max_terms=max_terms)
        known = known_edges_enabled(self.context)
        if known:
            inputs(self.context)
            if y not in (None, 'dx_dt') or variables not in (None, ['x']):
                raise ValueError('known-edges handoff supports x -> dx_dt only')
            variables = ['x']
        if self.context.num_nodes is None or not self.context.relation_names:
            raise ValueError(
                "hypergraph_formula_screen requires context.num_nodes and relation metadata"
            )
        pairwise = pairwise or _relation_by_arity(self.context, 2)
        triplet = triplet or _relation_by_arity(self.context, 3)
        if pairwise not in self.context.relation_names:
            raise ValueError(f"unknown pairwise relation: {pairwise}")
        if triplet not in self.context.relation_names:
            raise ValueError(f"unknown triplet relation: {triplet}")
        pair_table = np.asarray(self.context.data[pairwise])
        trip_table = np.asarray(self.context.data[triplet])
        if pair_table.ndim != 2 or pair_table.shape[1] != 2:
            raise ValueError(f"{pairwise} must be a two-column relation")
        if trip_table.ndim != 2 or trip_table.shape[1] != 3:
            raise ValueError(f"{triplet} must be a three-column relation")
        if known:
            rel = relations(self.context)
            if rel['G'][0] != pairwise or rel['H'][0] != triplet:
                raise ValueError('screen relation names must match known-edges config')
        pair_count = int(pair_table.shape[0])
        trip_count = int(trip_table.shape[0])
        if pair_count == 0 and trip_count == 0 and not known:
            raise ValueError("pairwise and triplet tables are both empty")

        variables = list(dict.fromkeys(variables or _node_field_names(self.context)))
        invalid = [name for name in variables if name not in self.context.data]
        if invalid:
            raise ValueError(f"unknown variables: {', '.join(invalid)}")

        features: List[str] = []
        for name in variables:
            features.extend([
                f"{name}[i]",
                f"{name}[i] ** 2",
                f"{name}[i] ** 3",
                f"{name}[i] * (1 - {name}[i])",
            ])
            if pair_count > 0:
                features.extend([
                    f"sum[j]({pairwise}[i, j], {name}[j] - {name}[i])",
                    f"sum[j]({pairwise}[i, j], sin({name}[j] - {name}[i]))",
                    f"sum[j]({pairwise}[i, j], {name}[i] * {name}[j])",
                    f"sum[j]({pairwise}[i, j], (1 - {name}[i]) * {name}[j])",
                    f"sum[j]({pairwise}[i, j], {name}[j] / (1 + {name}[j]))",
                    f"sum[j]({pairwise}[i, j], exp(-{name}[j]))",
                    f"sum[j]({pairwise}[i, j], log(1 + {name}[j]))",
                    (
                        f"sum[j]({pairwise}[i, j], "
                        f"sin({name}[j] - {name}[i]) / (1 + {name}[i] ** 2))"
                    ),
                ])
            if trip_count > 0:
                features.extend([
                    (
                        f"sum[j, k]({triplet}[i, j, k], "
                        f"{name}[j] + {name}[k] - 2 * {name}[i])"
                    ),
                    (
                        f"sum[j, k]({triplet}[i, j, k], "
                        f"sin({name}[j] + {name}[k] - 2 * {name}[i]))"
                    ),
                    (
                        f"sum[j, k]({triplet}[i, j, k], "
                        f"{name}[i] * {name}[j] * {name}[k])"
                    ),
                    (
                        f"sum[j, k]({triplet}[i, j, k], "
                        f"(1 - {name}[i]) * {name}[j] * {name}[k])"
                    ),
                    (
                        f"sum[j, k]({triplet}[i, j, k], "
                        f"exp(-({name}[j] - {name}[k]) ** 2) * "
                        f"({name}[j] + {name}[k] - 2 * {name}[i]))"
                    ),
                    (
                        f"sum[j, k]({triplet}[i, j, k], "
                        f"exp(-{name}[j] - {name}[k]))"
                    ),
                    (
                        f"sum[j, k]({triplet}[i, j, k], "
                        f"log(1 + {name}[j] * {name}[k]))"
                    ),
                    (
                        f"sum[j, k]({triplet}[i, j, k], "
                        f"tanh({name}[j] + {name}[k] - 2 * {name}[i]))"
                    ),
                    (
                        f"sum[j, k]({triplet}[i, j, k], "
                        f"{name}[j] * {name}[k] / "
                        f"((1 + {name}[j]) * (1 + {name}[k])))"
                    ),
                ])
            features.extend(form_screen_features(name, pairwise, triplet, pair_count, trip_count))
        features = list(dict.fromkeys(features))
        if len(features) > 128:
            features = features[:128]
        weights = host_state(self.context)['weights'] if known else {}
        weighted = any(
            name in weights and len(np.asarray(weights[name])) and not np.allclose(weights[name], 1.0)
            for name in (pairwise, triplet)
        )
        weighted |= bool(known and host_state(self.context)['config'].get('triplet_counting') == 'ordered_symmetric')
        if weighted:
            screen = weighted_screen(self, features, pairwise, triplet, max_terms)
        else:
            screen = GraphFeatureFitTool(context=self.context).execute(
                features=features, y=y, max_terms=max_terms,
                selection_method="bic_lasso", show_diagnostics=True,
            )
        payload = screen | {
            "screen_configuration": {
                "pairwise": pairwise,
                "triplet": triplet,
                "variables": variables,
                "max_terms": max_terms,
                "library": features,
            },
            "pairwise_count": pair_count,
            "triplet_count": trip_count,
        }
        if not known:
            return payload
        handoff = screen_handoff(self.context, screen, pairwise, triplet)
        screen['is_candidate'] = False
        for item in screen.get('all_formulas', []):
            item['is_candidate'] = False
        return json_safe(payload | {
            "known_edges": handoff,
            "candidate_id": handoff["candidate_id"],
            "can_submit": False,
        })

    @classmethod
    def format_result_dict(cls, result: Dict[str, Any]) -> str:
        """Format the hypergraph screen for the model."""
        if "known_edges" in result:
            if result.get("multivariate") or result.get("vector"):
                selected = {
                    name: {
                        "F": block.get("F"),
                        "G": block.get("G"),
                        "H": block.get("H"),
                        "derivative_rmse": block.get("derivative_rmse"),
                    }
                    for name, block in (result.get("components") or {}).items()
                }
                return readable({
                    "known_edges": result["known_edges"],
                    "screen_configuration": result["screen_configuration"],
                    "selected_support": selected,
                    "formula": result.get("formula"),
                    "candidate_id": result.get("candidate_id"),
                    "can_submit": False,
                })
            return readable({
                "known_edges": result["known_edges"],
                "screen_configuration": result["screen_configuration"],
                "selected_support": result["fitted_coefficients"],
                "can_submit": False,
            })
        lines = [
            GraphFeatureFitTool.format_evaluation_result(
                result,
                title="Hypergraph formula screen on given pairwise and/or triplet edges",
            )
        ]
        config = result["screen_configuration"]
        lines.append(
            f"Used relations: pairwise={config['pairwise']} "
            f"({result['pairwise_count']} rows), "
            f"triplet={config['triplet']} ({result['triplet_count']} rows)."
        )
        lines.append("Screened library:")
        for feature in config["library"]:
            lines.append(f"  - {feature}")
        lines.append("Selected support:")
        for feature, coefficient in result["fitted_coefficients"].items():
            lines.append(f"  - {coefficient:.6g} * {feature}")
        return "\n".join(lines)
