"""Shared runtime context for agents, evaluators, and tools."""
from __future__ import annotations

import argparse
import threading
from typing import TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    from ..evaluator import DefaultEvaluator
    from ..runtime.workspace import Workspace

_DEFAULT_EVALUATOR = object()


def _coerce_num_nodes(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, np.bool_):
        raise ValueError("num_nodes must be a positive integer")
    if isinstance(value, (int, np.integer)):
        number = int(value)
    elif isinstance(value, (float, np.floating)) and float(value) == int(value):
        number = int(value)
    else:
        raise ValueError("num_nodes must be a positive integer")
    if number < 1:
        raise ValueError("num_nodes must be a positive integer")
    return number


def _relation_array(rows: Any, width: int) -> np.ndarray:
    if rows is None:
        return np.zeros((0, width), dtype=np.int64)
    array = np.asarray(rows)
    if array.size == 0:
        return np.zeros((0, width), dtype=np.int64)
    if array.ndim == 1 and array.size == width:
        array = array.reshape(1, width)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"edges must have shape (E, {width})")
    if array.dtype.kind == "f":
        if not np.all(np.isfinite(array)) or not np.all(array == np.trunc(array)):
            raise ValueError("edge endpoints must be integers")
    elif array.dtype.kind not in "iu":
        raise ValueError("edge endpoints must be integers")
    return np.ascontiguousarray(array, dtype=np.int64)


class AgentContext:
    """One authoritative context containing data, metadata, and runtime arguments.

    Args:
        args: Runtime configuration shared by agents, evaluators, and tools.
        data: Arrays keyed by variable or axis name.
        target: Name of the current target variable.
        variable_descriptions: Human-readable descriptions keyed by data name.
        variable_axes: Axis names for every non-axis variable.
        variable_structures: Relation variable used by each structured variable.
        relation_names: Names of graph or hypergraph relation arrays.
        num_nodes: Number of nodes for graph or hypergraph data.
        evaluator: Evaluator instance used to fit, score, and split formulas.
        workspace: Optional managed workspace available to tools.
    """

    def __init__(self, *, args: argparse.Namespace | None = None, data: dict[str, Any] | None = None, target: str | None = None, variable_descriptions: dict[str, str] | None = None, variable_axes: dict[str, tuple[str, ...]] | None = None, variable_structures: dict[str, str] | None = None, relation_names: set[str] | None = None, num_nodes: int | None = None, evaluator: DefaultEvaluator | object = _DEFAULT_EVALUATOR, workspace: Workspace | None = None) -> None:
        if args is not None and not isinstance(args, argparse.Namespace):
            raise TypeError("args must be an argparse.Namespace")
        self.args = args if args is not None else argparse.Namespace()
        defaults = {
            "validation_fraction": 0.0,
            "split_by": "random",
            "split_random_state": 42,
            "split_ood_variable": None,
            "data_revision": 0,
        }
        for name, value in defaults.items():
            if not hasattr(self.args, name):
                setattr(self.args, name, value)
        self.workspace = workspace
        if any(not isinstance(name, str) or not name for name in (data or {})):
            raise TypeError("data keys must be non-empty strings")
        self.data = {name: np.asarray(value) for name, value in (data or {}).items()}
        self.target = target
        if variable_descriptions is None:
            self.variable_descriptions = {name: "" for name in self.data}
        else:
            self.variable_descriptions = dict(variable_descriptions)
        if variable_axes is None:
            self.variable_axes = {name: () for name in self.data}
        else:
            self.variable_axes = {name: tuple(axes) for name, axes in variable_axes.items()}
        self.variable_structures = dict(variable_structures or {})
        self.relation_names = set(relation_names or self.variable_structures.values())
        self.num_nodes = num_nodes
        from ..evaluator import DefaultEvaluator, GraphEvaluator
        if evaluator is _DEFAULT_EVALUATOR:
            evaluator = GraphEvaluator() if self.relation_names else DefaultEvaluator()
        if not isinstance(evaluator, DefaultEvaluator):
            raise TypeError("evaluator must be a DefaultEvaluator instance")
        self.evaluator: DefaultEvaluator = evaluator
        self._split_cache: dict[str, AgentContext] | None = None
        self._lock = threading.RLock()
        self._validate(allow_incomplete=not self.data or self.target is None)

    def _validate(self, *, allow_incomplete: bool = False) -> None:
        names = set(self.data)
        if not allow_incomplete and self.target not in names:
            raise ValueError("target must name an entry in data")
        if set(self.variable_descriptions) != names:
            raise ValueError("variable_descriptions keys must equal data keys")
        if any(not isinstance(value, str) for value in self.variable_descriptions.values()):
            raise TypeError("variable descriptions must be strings")
        if any(
            not isinstance(axis, str) or not axis
            for dimensions in self.variable_axes.values()
            for axis in dimensions
        ):
            raise TypeError("variable_axes values must contain non-empty axis names")
        non_axes = set(self.variable_axes)
        axes = {axis for dimensions in self.variable_axes.values() for axis in dimensions}
        if non_axes & axes or non_axes | axes != names:
            raise ValueError("variable_axes keys and values must partition data keys")
        for axis in axes:
            if self.data[axis].ndim != 1:
                raise ValueError(f"axis variable {axis!r} must be one-dimensional")
        if (self.num_nodes is None) != (not self.relation_names):
            raise ValueError("relation_names and num_nodes must appear together")
        if self.num_nodes is not None and (
            isinstance(self.num_nodes, bool)
            or not isinstance(self.num_nodes, int)
            or self.num_nodes < 1
        ):
            raise ValueError("num_nodes must be a positive integer")
        for relation_name in self.relation_names:
            if relation_name not in names:
                raise ValueError("relation_names must reference entries in data")
            relation = self.data[relation_name]
            if relation.ndim != 2 or relation.shape[1] not in {2, 3}:
                raise ValueError(f"relation variable {relation_name!r} must have shape (E, 2) or (H, 3)")
            if relation.dtype.kind not in "iu":
                raise ValueError(f"relation variable {relation_name!r} must contain integer endpoints")
            if np.any(relation < 0) or np.any(relation >= self.num_nodes):
                raise ValueError(f"relation variable {relation_name!r} endpoints must be in [0, {self.num_nodes})")
        for variable, structure in self.variable_structures.items():
            if variable not in names or structure not in names:
                raise ValueError("variable_structures must reference entries in data")
            if variable == structure:
                raise ValueError("a structured variable cannot reference itself")
            if structure not in self.relation_names:
                raise ValueError("variable_structures values must name relation variables")
            relation = self.data[structure]
            if relation.ndim != 2 or relation.shape[1] not in {2, 3}:
                raise ValueError(f"structure variable {structure!r} must have shape (E, 2) or (H, 3)")
            if self.data[variable].ndim == 0 or self.data[variable].shape[-1] != relation.shape[0]:
                raise ValueError(f"structured variable {variable!r} must end in the relation dimension")
            if relation.dtype.kind not in "iu":
                raise ValueError(f"structure variable {structure!r} must contain integer endpoints")
            if np.any(relation < 0) or np.any(relation >= self.num_nodes):
                raise ValueError(f"structure variable {structure!r} endpoints must be in [0, {self.num_nodes})")

    def invalidate_splits(self) -> None:
        """Discard cached training and validation context views."""
        with self._lock:
            self._split_cache = None

    def _sync_builtin_evaluator(self) -> None:
        from ..evaluator import DefaultEvaluator, GraphEvaluator
        if type(self.evaluator) not in {DefaultEvaluator, GraphEvaluator}:
            return
        self.evaluator = GraphEvaluator() if self.relation_names else DefaultEvaluator()

    def _splits(self) -> dict[str, AgentContext]:
        with self._lock:
            if self.target is None:
                raise ValueError("target must be configured before splitting AgentContext")
            if self._split_cache is None:
                splits = self.evaluator.split(self)
                if set(splits) != {"train", "validation"}:
                    raise ValueError("DefaultEvaluator.split() must return train and validation contexts")
                if any(not isinstance(split, AgentContext) for split in splits.values()):
                    raise TypeError("DefaultEvaluator.split() values must be AgentContext instances")
                self._split_cache = splits
            return self._split_cache

    def with_data(self, data: dict[str, np.ndarray]) -> AgentContext:
        """Create a context view over data while preserving runtime configuration.

        Args:
            data: Data mapping for the new context view.

        Returns:
            A context sharing arguments and evaluator configuration with this context.
        """
        return AgentContext(
            args=self.args, data=data, target=self.target,
            variable_descriptions={name: self.variable_descriptions[name] for name in data},
            variable_axes={name: axes for name, axes in self.variable_axes.items() if name in data},
            variable_structures={name: structure for name, structure in self.variable_structures.items() if name in data and structure in data},
            relation_names=self.relation_names & set(data),
            num_nodes=self.num_nodes, evaluator=self.evaluator, workspace=self.workspace,
        )

    @property
    def train_split(self) -> AgentContext:
        """Return the cached training context, computing both splits if needed."""
        return self._splits()["train"]

    @property
    def validation_split(self) -> AgentContext:
        """Return the cached validation context, computing both splits if needed."""
        return self._splits()["validation"]

    def axis_names(self) -> tuple[str, ...]:
        """Return axis-variable names in first-occurrence order."""
        return tuple(dict.fromkeys(axis for axes in self.variable_axes.values() for axis in axes))

    def variable_names(self) -> tuple[str, ...]:
        """Return names of non-axis variables in manifest order."""
        return tuple(self.variable_axes)

    def feature_names(self) -> tuple[str, ...]:
        """Return selected non-axis variables other than the target."""
        return tuple(name for name in self.variable_axes if name != self.target)

    def commit_context_data(self, loaded: dict[str, Any]) -> dict[str, Any]:
        """Replace live context data with a validated loader result.

        Args:
            loaded: Mapping returned by :func:`load_context_data`.

        Returns:
            New data revision and loaded variable names.
        """
        values = {str(name): np.asarray(value) for name, value in loaded["data"].items()}
        self.data = values
        self.variable_descriptions = dict(loaded["variable_descriptions"])
        self.variable_axes = {
            name: tuple(axes) for name, axes in loaded["variable_axes"].items()
        }
        self.variable_structures = dict(loaded["variable_structures"])
        self.relation_names = set(loaded["relation_names"])
        self.num_nodes = loaded["num_nodes"]
        self._sync_builtin_evaluator()
        self.target = self.target if self.target in values else None
        self.invalidate_splits()
        self.args.data_revision += 1
        self._validate(allow_incomplete=True)
        return {"revision": self.args.data_revision, "variables": list(values)}

    def commit_data(self, data: dict[str, Any], *, target: str, features: list[str] | None = None, variable_descriptions: dict[str, str] | None = None) -> dict[str, Any]:
        """Commit an ordinary aligned data selection to the live context.

        Args:
            data: Available arrays keyed by column name.
            target: Target column to retain.
            features: Feature columns to retain. Defaults to no features.
            variable_descriptions: Optional descriptions for retained columns.

        Returns:
            A summary containing revision, selection, columns, and row count.
        """
        selected = list(dict.fromkeys([*(features or []), target]))
        if target not in data:
            raise ValueError(f"target column does not exist: {target}")
        arrays = {name: np.asarray(data[name]) for name in selected}
        with self._lock:
            self.data = arrays
            self.target = target
            self.variable_descriptions = {name: str((variable_descriptions or {}).get(name, "")) for name in arrays}
            self.variable_axes = {name: () for name in arrays}
            self.variable_structures = {}
            self.relation_names = set()
            self.num_nodes = None
            self._sync_builtin_evaluator()
            self.invalidate_splits()
            self.args.data_revision += 1
            self._validate()
        return {"revision": self.args.data_revision, "target": target, "features": list(self.feature_names()), "columns": list(arrays), "rows": len(arrays[target])}

    def install_relations(self, *, pairwise: Any = None, triplet: Any = None, num_nodes: int | None = None) -> dict[str, Any]:
        """Replace the candidate pairwise and triplet tables and keep them for formula evaluation.

        Args:
            pairwise: Rows shaped ``(E, 2)`` in ``(target, source)`` order. ``None`` installs an empty table.
            triplet: Rows shaped ``(H, 3)`` in ``(target, source1, source2)`` order. ``None`` installs an empty table.
            num_nodes: Node count. When omitted, use the last dimension of ``x`` or ``dx_dt``.

        Returns:
            The installed node count and the number of rows in each table.

        Raises:
            ValueError: If an edge table has the wrong shape, a non-integer endpoint,
                or a node count that disagrees with the node axis of ``x`` or ``dx_dt``.
        """
        tables = {
            "pairwise": _relation_array(pairwise, 2),
            "triplet": _relation_array(triplet, 3),
        }
        descriptions = {
            "pairwise": "Candidate pairwise edges, columns (target, source).",
            "triplet": "Candidate triplet edges, columns (target, source1, source2).",
        }
        resolved_nodes = _coerce_num_nodes(num_nodes)
        with self._lock:
            saved = (
                dict(self.data),
                dict(self.variable_descriptions),
                dict(self.variable_axes),
                set(self.relation_names),
                self.num_nodes,
                self.evaluator,
            )
            try:
                for name in ("x", "dx_dt"):
                    if name not in self.data or np.asarray(self.data[name]).ndim < 1:
                        continue
                    observed = int(np.asarray(self.data[name]).shape[-1])
                    if resolved_nodes is None:
                        resolved_nodes = observed
                    elif resolved_nodes != observed:
                        raise ValueError(
                            f"num_nodes {resolved_nodes} does not match {name}.shape[-1] {observed}"
                        )
                if resolved_nodes is None:
                    raise ValueError("num_nodes is required when x and dx_dt are absent")
                data = {
                    name: value for name, value in self.data.items() if name not in tables
                }
                data.update(tables)
                self.data = data
                self.variable_descriptions = {
                    name: self.variable_descriptions.get(name, "")
                    for name in data
                    if name not in tables
                }
                self.variable_descriptions.update(descriptions)
                self.variable_axes = {
                    name: axes for name, axes in self.variable_axes.items() if name in data and name not in tables
                }
                for name in tables:
                    self.variable_axes[name] = ()
                self.relation_names = (self.relation_names - set(tables)) | set(tables)
                self.num_nodes = resolved_nodes
                self._sync_builtin_evaluator()
                self.invalidate_splits()
                self.args.data_revision += 1
                self._validate()
            except Exception:
                (
                    self.data,
                    self.variable_descriptions,
                    self.variable_axes,
                    self.relation_names,
                    self.num_nodes,
                    self.evaluator,
                ) = saved
                self.invalidate_splits()
                raise
        return {
            "num_nodes": self.num_nodes,
            "pairwise_count": int(tables["pairwise"].shape[0]),
            "triplet_count": int(tables["triplet"].shape[0]),
        }

    def add_features(self, features: dict[str, Any], *, descriptions: dict[str, str] | None = None) -> dict[str, Any]:
        """Add aligned feature arrays while preserving the current target.

        Args:
            features: New feature arrays keyed by name.
            descriptions: Optional descriptions for the new features.

        Returns:
            The commit summary produced for the updated data.

        Raises:
            ValueError: If the context has no configured target.
        """
        if self.target is None:
            raise ValueError("target must be configured before adding features")
        descriptions = self.variable_descriptions | {
            name: (descriptions or {}).get(name, "") for name in features
        }
        return self.commit_data(self.data | features, target=self.target, features=[*self.feature_names(), *features], variable_descriptions=descriptions)

    def update_selection(self, *, target: str, features: list[str], variable_descriptions: dict[str, str] | None = None) -> dict[str, Any]:
        """Select the variables used by symbolic regression.

        Args:
            target: Target variable name.
            features: Feature variable names.
            variable_descriptions: Description updates to apply before selection.

        Returns:
            New revision and selected target and features.
        """
        keep_variables = set(features) | {target}
        keep_axes = {axis for name, axes in self.variable_axes.items() if name in keep_variables for axis in axes}
        keep = keep_variables | keep_axes
        self.data = {name: value for name, value in self.data.items() if name in keep}
        self.target = target
        descriptions = self.variable_descriptions | dict(variable_descriptions or {})
        self.variable_descriptions = {name: descriptions.get(name, "") for name in self.data}
        self.variable_axes = {name: axes for name, axes in self.variable_axes.items() if name in keep_variables}
        self.variable_structures = {name: structure for name, structure in self.variable_structures.items() if name in self.data and structure in self.data}
        self.relation_names &= set(self.data)
        if not self.relation_names:
            self.num_nodes = None
        self._sync_builtin_evaluator()
        self.invalidate_splits()
        self.args.data_revision += 1
        self._validate()
        return {"revision": self.args.data_revision, "target": target, "features": list(self.feature_names()), "selection_changed": True}

    def schema(self) -> dict[str, Any]:
        """Return a serializable description of the active context data."""
        axes = set(self.axis_names())
        return {
            "revision": self.args.data_revision, "num_nodes": self.num_nodes,
            "target": self.target, "features": list(self.feature_names()), "columns": list(self.data),
            "rows": len(self.data[self.target]) if self.target in self.data else 0,
            "variable_descriptions": dict(self.variable_descriptions),
            "variables": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "axes": list(self.variable_axes.get(name, ())), "description": self.variable_descriptions[name], **({"kind": "relation"} if name in self.relation_names else {})} for name, value in self.data.items() if name not in axes},
            "axes": {name: {"size": len(self.data[name]), "dtype": str(self.data[name].dtype), "description": self.variable_descriptions[name]} for name in axes},
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

