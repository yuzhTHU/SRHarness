# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Shared mutable context for agents and tools."""
from __future__ import annotations

import threading
from collections.abc import Iterator, MutableMapping
from typing import Any

import numpy as np


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
        "data_revision",
        "provenance",
    }

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        target: str | None = None,
        features: list[str] | None = None,
        variable_descriptions: dict[str, str] | None = None,
        workspace: Any = None,
        evaluation_data: dict[str, Any] | None = None,
        data_revision: int = 0,
        provenance: dict[str, Any] | None = None,
        **values: Any,
    ):
        self.data = self._normalize_data(data or {})
        self.training_data: dict[str, np.ndarray] | None = None
        self.evaluation_data = self._normalize_data(evaluation_data or {})
        self.target = target
        self.features = list(features or self._default_features(self.data, target))
        self.variable_descriptions = dict(variable_descriptions or {})
        self.workspace = workspace
        self.data_revision = int(data_revision)
        self.provenance = dict(provenance or {})
        self.last_change: dict[str, Any] | None = None
        self._values = dict(values)
        self._lock = threading.RLock()

    @staticmethod
    def _normalize_data(data: dict[str, Any]) -> dict[str, np.ndarray]:
        return {str(name): np.asarray(value) for name, value in data.items()}

    @staticmethod
    def _default_features(data: dict[str, Any], target: str | None) -> list[str]:
        return [name for name in data if name != target]

    @property
    def tool_data(self) -> dict[str, np.ndarray]:
        return self.training_data if self.training_data is not None else self.data

    @property
    def workspace_dir(self) -> str | None:
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
        """Validate and atomically replace the structured dataset."""
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
        """Add aligned feature columns and create a new data revision."""
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

    def bind_split(
        self,
        training_data: dict[str, Any],
        evaluation_data: dict[str, Any],
    ) -> None:
        """Bind the split consumed by symbolic-regression tools."""
        with self._lock:
            self.training_data = self._normalize_data(training_data)
            self.evaluation_data = self._normalize_data(evaluation_data)

    def schema(self) -> dict[str, Any]:
        with self._lock:
            rows = len(next(iter(self.data.values()))) if self.data else 0
            return {
                "revision": self.data_revision,
                "target": self.target,
                "features": list(self.features),
                "columns": list(self.data),
                "rows": rows,
                "variable_descriptions": dict(self.variable_descriptions),
                "provenance": dict(self.provenance),
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
