# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Commit agent-prepared tabular data into the shared runtime context."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base_tool import BaseTool, ToolMetadata


@BaseTool.register("commit_data")
class CommitDataTool(BaseTool):
    metadata = ToolMetadata(name="commit_data")

    def execute(
        self,
        path: str,
        target: str,
        features: list[str],
        variable_descriptions: dict[str, str] | None = None,
        provenance_note: str = "",
    ) -> dict[str, Any]:
        """Validate a prepared table and atomically publish it to symbolic regression.

        The source file remains in the workspace. Only the selected target and
        feature columns are committed, and every selected column must be finite
        and numeric. Use workspace_code_executor first for cleaning, joining,
        interpolation, encoding, and time alignment.

        Args:
            path: Relative path to a CSV or Excel file in the workspace.
            target: Column to use as the dependent variable.
            features: Columns to use as independent variables.
            variable_descriptions: Optional human-readable descriptions keyed by column name.
            provenance_note: Short note describing source and transformations.
        """
        workspace = self.context.workspace
        if workspace is None or (source := workspace.resolve(path)) is None or not source.is_file():
            raise ValueError("path must identify an existing workspace file")
        suffix = source.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(source)
        elif suffix == ".xlsx":
            frame = pd.read_excel(source)
        else:
            raise ValueError("path must use .csv or .xlsx")
        if not features:
            raise ValueError("features must contain at least one column")
        selected = list(dict.fromkeys([*features, target]))
        missing = [name for name in selected if name not in frame]
        if missing:
            raise ValueError(f"selected columns do not exist: {missing}")
        if target in features:
            raise ValueError("target cannot also be a feature")
        try:
            numeric = frame[selected].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("selected target and feature columns must be numeric") from exc
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("selected columns must contain finite values without missing data")
        if self.context.get("sr_active", False) and self.context.data:
            if target != self.context.target:
                raise ValueError("cannot change the target while symbolic regression is active")
            if len(numeric) != len(next(iter(self.context.data.values()))):
                raise ValueError("cannot change row alignment while symbolic regression is active")
            removed = [name for name in self.context.features if name not in features]
            changed = [
                name for name in [self.context.target, *self.context.features]
                if name in numeric
                and not np.array_equal(
                    numeric[name].to_numpy(dtype=float),
                    self.context.data[name],
                    equal_nan=True,
                )
            ]
            if removed or changed:
                raise ValueError(
                    "an active symbolic-regression run only accepts added features; "
                    f"removed={removed}, changed={changed}"
                )
        descriptions = self.context.variable_descriptions | dict(variable_descriptions or {})
        history = list(self.context.provenance.get("history", []))
        history.append({"source_path": path, "note": provenance_note.strip()})
        change = self.context.commit_data(
            {name: numeric[name].to_numpy(dtype=float) for name in selected},
            target=target,
            features=list(features),
            variable_descriptions=descriptions,
            provenance={"history": history},
        )
        return {"data_committed": True, **change}

    @classmethod
    def format_result_dict(cls, result: dict[str, Any]) -> str:
        return (
            f"Committed revision {result['revision']}: {result['rows']} rows, "
            f"target={result['target']}, features={result['features']}."
        )
