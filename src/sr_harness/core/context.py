# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Shared mutable context for agents and tools."""
from __future__ import annotations

import threading
from collections.abc import Iterator, MutableMapping
from typing import Any

import numpy as np

from .context_data import ContextData


class AgentContext(MutableMapping[str, Any]):
    """Authoritative shared data and resources for cooperating agents.

    The mapping interface keeps existing tools compatible while attribute access
    exposes the structured state used by agents and the Web session. ``data`` is
    the complete aligned dataset. Legacy ``context["data"]`` reads the active
    training split when one has been bound by :class:`SRAgent`.
    """

    _FIELDS = {
        "target",
        "features",
        "variable_descriptions",
        "workspace",
        "evaluation_data",
        "evaluator",
        "data_revision",
        "provenance",
        "axes",
        "variable_axes",
        "data_manifest_path",
    }

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        target: str | None = None,
        features: list[str] | None = None,
        variable_descriptions: dict[str, str] | None = None,
        workspace: Any = None,
        evaluation_data: dict[str, Any] | None = None,
        evaluator: Any = None,
        data_revision: int = 0,
        provenance: dict[str, Any] | None = None,
        **values: Any,
    ):
        self.data = self._normalize_data(data or {})
        self.axes = dict(getattr(self.data, "axes", {}))
        self.variable_axes = dict(getattr(self.data, "variable_axes", {}))
        self.data_manifest_path = (
            str(self.data.directory / "manifest.json")
            if getattr(self.data, "directory", None) is not None
            else None
        )
        self.training_data: dict[str, np.ndarray] | None = None
        self.evaluation_data = self._normalize_data(evaluation_data or {})
        self.evaluator = evaluator
        self.target = target
        self.features = list(features or self._default_features(self.data, target))
        self.variable_descriptions = dict(
            variable_descriptions or getattr(self.data, "descriptions", {})
        )
        self.workspace = workspace
        self.data_revision = int(data_revision)
        self.provenance = dict(provenance or {})
        self.last_change: dict[str, Any] | None = None
        self._values = dict(values)
        self._lock = threading.RLock()

    @staticmethod
    def _normalize_data(data: dict[str, Any]) -> ContextData:
        if isinstance(data, ContextData):
            return data
        return ContextData({str(name): np.asarray(value) for name, value in data.items()})

    @staticmethod
    def _default_features(data: dict[str, Any], target: str | None) -> list[str]:
        return [name for name in data if name != target]

    @property
    def tool_data(self) -> dict[str, np.ndarray]:
        """Return the active training split or the complete dataset.

        Returns:
            Data arrays exposed to scientific tools.
        """
        return self.training_data if self.training_data is not None else self.data

    @property
    def workspace_dir(self) -> str | None:
        """Return the active workspace directory.

        Returns:
            Workspace path, or ``None`` when no workspace is configured.
        """
        if self.workspace is None:
            return self._values.get("workspace_dir")
        return str(self.workspace.path)

    def commit_data(
        self,
        data: dict[str, Any],
        *,
        target: str,
        features: list[str] | None = None,
        variable_descriptions: dict[str, str] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate and atomically replace the structured dataset.

        Args:
            data: Data arrays keyed by variable name.
            target: Target name or target values.
            features: Ordered feature-column names; all non-target columns by default.
            variable_descriptions: Human-readable descriptions keyed by column name.
            provenance: Source and transformation metadata for the dataset.

        Returns:
            A description of the committed revision and column changes.
        """
        normalized = self._normalize_data(data)
        if not normalized:
            raise ValueError("data must contain at least one column")
        if target not in normalized:
            raise ValueError(f"target column does not exist: {target}")
        lengths = {len(value) for value in normalized.values()}
        if len(lengths) != 1:
            raise ValueError(f"all data columns must have equal length, got {sorted(lengths)}")
        if next(iter(lengths)) == 0:
            raise ValueError("data must contain at least one row")
        selected_features = list(features or self._default_features(normalized, target))
        if not selected_features:
            raise ValueError("at least one feature is required")
        if target in selected_features:
            raise ValueError("target cannot also be a feature")
        missing = [name for name in selected_features if name not in normalized]
        if missing:
            raise ValueError(f"feature columns do not exist: {missing}")

        with self._lock:
            previous_columns = list(self.data)
            previous_target = self.target
            self.data = normalized
            self.axes = {}
            self.variable_axes = {}
            self.data_manifest_path = None
            self.target = target
            self.features = selected_features
            self.variable_descriptions = dict(variable_descriptions or {})
            self.provenance = dict(provenance or {})
            self.training_data = None
            self.evaluation_data = {}
            self.data_revision += 1
            self.last_change = {
                "revision": self.data_revision,
                "target": target,
                "features": list(selected_features),
                "columns": list(normalized),
                "rows": next(iter(lengths)),
                "added_columns": [name for name in normalized if name not in previous_columns],
                "removed_columns": [name for name in previous_columns if name not in normalized],
                "target_changed": previous_target not in {None, target},
            }
            return dict(self.last_change)

    def add_features(
        self,
        features: dict[str, Any],
        *,
        descriptions: dict[str, str] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add aligned feature columns and create a new data revision.

        Args:
            features: New aligned columns keyed by name.
            descriptions: Descriptions for the new columns.
            provenance: Source and transformation metadata to merge.

        Returns:
            A description of the committed revision and column changes.
        """
        if not self.data or self.target is None:
            raise ValueError("structured data must be committed before adding features")
        normalized = self._normalize_data(features)
        collisions = sorted(set(normalized) & set(self.data))
        if collisions:
            raise ValueError(f"feature columns already exist: {collisions}")
        row_count = len(next(iter(self.data.values())))
        mismatched = {name: len(value) for name, value in normalized.items() if len(value) != row_count}
        if mismatched:
            raise ValueError(f"new features must contain {row_count} rows, got {mismatched}")
        merged_descriptions = self.variable_descriptions | dict(descriptions or {})
        merged_provenance = self.provenance | dict(provenance or {})
        return self.commit_data(
            self.data | normalized,
            target=self.target,
            features=[*self.features, *normalized],
            variable_descriptions=merged_descriptions,
            provenance=merged_provenance,
        )

    def commit_context_data(self, data: ContextData) -> dict[str, Any]:
        """Replace structured variables with a validated manifest-backed collection.

        Existing target and feature selections are retained only while their
        variables still exist. Axis metadata remains attached to ``data`` and is
        also exposed directly on the context for tools that need it.

        Args:
            data: Validated variables and axes loaded by ``ContextDataStore``.

        Returns:
            A description of the committed revision and variable changes.
        """
        if not isinstance(data, ContextData) or not data:
            raise ValueError("context data must contain at least one validated variable")
        with self._lock:
            previous = list(self.data)
            self.data = data
            self.axes = dict(data.axes)
            self.variable_axes = dict(data.variable_axes)
            self.variable_descriptions = {
                **dict(data.descriptions),
                **{
                    name: axis.description
                    for name, axis in data.axes.items()
                    if axis.description
                },
            }
            self.data_manifest_path = (
                str(data.directory / "manifest.json") if data.directory is not None else None
            )
            available = set(data) | set(data.axes)
            if self.target not in available:
                self.target = None
            self.features = [
                name for name in self.features
                if name in available and name != self.target
            ]
            self.training_data = None
            self.evaluation_data = ContextData({})
            self.data_revision += 1
            self.last_change = {
                "revision": self.data_revision,
                "variables": list(data),
                "added_variables": [name for name in data if name not in previous],
                "removed_variables": [name for name in previous if name not in data],
                "manifest": self.data_manifest_path,
            }
            return dict(self.last_change)

    def bind_split(
        self,
        training_data: dict[str, Any],
        evaluation_data: dict[str, Any],
    ) -> None:
        """Bind the split consumed by symbolic-regression tools.

        Args:
            training_data: Data exposed to fitting tools.
            evaluation_data: Held-out data exposed to evaluation tools.
        """
        with self._lock:
            self.training_data = self._normalize_data(training_data)
            self.evaluation_data = self._normalize_data(evaluation_data)

    def update_selection(
        self,
        *,
        target: str,
        features: list[str],
        variable_descriptions: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Update the variables consumed by symbolic regression.

        Args:
            target: Name of the selected target variable or axis.
            features: Ordered names of selected feature variables or axes.
            variable_descriptions: Updated human-readable descriptions.

        Returns:
            A description of the resulting data revision.
        """
        selected_features = list(dict.fromkeys(features))
        if not target or not selected_features:
            raise ValueError("a target and at least one feature are required")
        if target in selected_features:
            raise ValueError("target cannot also be a feature")

        def value(name: str) -> np.ndarray | None:
            if name in self.data:
                return np.asarray(self.data[name])
            axis = self.axes.get(name)
            return None if axis is None else np.asarray(axis.values)

        selected = {name: value(name) for name in [*selected_features, target]}
        missing = [name for name, array in selected.items() if array is None]
        if missing:
            raise ValueError(f"selected variables or axes do not exist: {missing}")
        arrays = {name: array for name, array in selected.items() if array is not None}
        if any(array.ndim != 1 for array in arrays.values()):
            raise ValueError("selected variables and axes must be one-dimensional")
        if len({len(array) for array in arrays.values()}) != 1:
            raise ValueError("selected variables and axes must have the same length")
        try:
            numeric = [array.astype(float, copy=False) for array in arrays.values()]
        except (TypeError, ValueError) as exc:
            raise ValueError("selected target and features must be numeric") from exc
        if any(not np.isfinite(array).all() for array in numeric):
            raise ValueError("selected target and features must contain finite values")

        descriptions = dict(variable_descriptions or {})
        available = set(self.data) | set(self.axes)
        unknown_descriptions = sorted(set(descriptions) - available)
        if unknown_descriptions:
            raise ValueError(
                f"descriptions reference unknown variables or axes: {unknown_descriptions}"
            )
        with self._lock:
            changed = (
                self.target != target
                or self.features != selected_features
                or any(
                    self.variable_descriptions.get(name, "") != description
                    for name, description in descriptions.items()
                )
            )
            self.target = target
            self.features = selected_features
            self.variable_descriptions.update(descriptions)
            if changed:
                self.training_data = None
                self.evaluation_data = ContextData({})
                self.data_revision += 1
            self.last_change = {
                "revision": self.data_revision,
                "target": target,
                "features": list(selected_features),
                "selection_changed": changed,
            }
            return dict(self.last_change)

    def schema(self) -> dict[str, Any]:
        """Return the current structured-data schema.

        Returns:
            Column names, roles, row count, revision, descriptions, and provenance.
        """
        with self._lock:
            rows = len(next(iter(self.data.values()))) if self.data else 0
            variables = {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "axes": list(self.variable_axes.get(name, ())),
                    "description": self.variable_descriptions.get(name, ""),
                }
                for name, value in self.data.items()
            }
            axes = {
                name: {
                    "size": len(axis.values),
                    "dtype": str(axis.values.dtype),
                    "description": axis.description,
                    "storage": axis.storage,
                }
                for name, axis in self.axes.items()
            }
            return {
                "revision": self.data_revision,
                "target": self.target,
                "features": list(self.features),
                "columns": list(self.data),
                "rows": rows,
                "variable_descriptions": dict(self.variable_descriptions),
                "provenance": dict(self.provenance),
                "variables": variables,
                "axes": axes,
                "manifest": self.data_manifest_path,
            }

    def __getitem__(self, key: str) -> Any:
        if key == "data":
            return self.tool_data
        if key == "workspace_dir":
            value = self.workspace_dir
            if value is None:
                raise KeyError(key)
            return value
        if key in self._FIELDS:
            return getattr(self, key)
        return self._values[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "data":
            self.data = self._normalize_data(value)
            self.axes = dict(getattr(self.data, "axes", {}))
            self.variable_axes = dict(getattr(self.data, "variable_axes", {}))
            self.training_data = None
        elif key == "workspace_dir":
            self._values[key] = value
        elif key in self._FIELDS:
            setattr(self, key, value)
        else:
            self._values[key] = value

    def __delitem__(self, key: str) -> None:
        if key in {"data", *self._FIELDS}:
            raise KeyError(f"cannot delete structured context field: {key}")
        del self._values[key]

    def __iter__(self) -> Iterator[str]:
        keys = ["data", *sorted(self._FIELDS)]
        if self.workspace_dir is not None:
            keys.append("workspace_dir")
        yield from dict.fromkeys([*keys, *self._values])

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __getstate__(self) -> dict[str, Any]:
        """Make tool contexts serializable for joblib worker processes."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()
