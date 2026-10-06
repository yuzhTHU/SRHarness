"""Demand-driven evaluator for expressions with symbolic structural indices."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Mapping

import numpy as np

from .expression import (
    Binary,
    Expression,
    Function,
    Gather,
    GroupedParameter,
    Index,
    Indexed,
    Number,
    Parameter,
    Reduction,
    Symbol,
    Unary,
)
from .tree import children


@dataclass(frozen=True, slots=True)
class RelationField:
    """Values stored on the entries of a named relation.

    Args:
        values: Array whose last dimension enumerates relation entries.
        relation: Name of the coordinate-table symbol defining those entries.
    """

    values: Any
    relation: str


@dataclass(slots=True)
class _Demand:
    indices: tuple[str, ...]
    coordinates: np.ndarray

    def project(self, indices: tuple[str, ...]) -> np.ndarray:
        if not indices:
            return np.empty((len(self.coordinates), 0), dtype=np.intp)
        columns = [self.indices.index(index) for index in indices]
        return self.coordinates[:, columns]


@dataclass(slots=True)
class _SparseTensor:
    indices: tuple[str, ...]
    coordinates: np.ndarray
    values: np.ndarray
    fill_value: float
    num_nodes: int

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.num_nodes,) * len(self.indices)


class IndexedEvaluator:
    """Evaluate indexed expressions against a fixed structural demand."""

    def __init__(
        self,
        values: Mapping[str, Any],
        parameters: Mapping[str, Any],
        parameter_defaults: Mapping[str, float | None],
        num_nodes: int | None,
        time: Any,
        delay_resolver: Callable[..., Any] | None,
    ):
        if num_nodes is None:
            raise ValueError("num_nodes must be provided when evaluating indexed expressions.")
        if isinstance(num_nodes, bool) or int(num_nodes) != num_nodes or num_nodes <= 0:
            raise ValueError("num_nodes must be a positive integer.")
        self.num_nodes = int(num_nodes)
        self.values = dict(values)
        self.parameters = dict(parameters)
        self.parameter_defaults = dict(parameter_defaults)
        self.time = None if time is None else np.asarray(time, dtype=float)
        self.delay_resolver = delay_resolver
        self._free_cache: dict[int, tuple[str, ...]] = {}
        self._relation_cache: dict[int, _SparseTensor] = {}
        self._domain_cache: dict[int, Expression | None] = {}

    def __call__(self, expression: Expression) -> Any:
        self._validate_scopes(expression)
        self._validate_bare_symbols(expression)
        free = self.free_indices(expression)
        coordinates = self._root_coordinates(free)
        result = self._eval_at(expression, _Demand(free, coordinates))
        if free:
            result = np.asarray(result)
            return result.reshape(result.shape[:-1] + (self.num_nodes,) * len(free))
        if self.relation_domain(expression) is not None:
            return result
        if np.ndim(result) and np.shape(result)[-1:] == (1,):
            return np.asarray(result)[..., 0]
        return result

    def _root_coordinates(self, indices: tuple[str, ...]) -> np.ndarray:
        if not indices:
            return np.empty((1, 0), dtype=np.intp)
        return np.asarray(
            list(product(range(self.num_nodes), repeat=len(indices))), dtype=np.intp
        ).reshape(-1, len(indices))

    def free_indices(self, node: Expression) -> tuple[str, ...]:
        """Return free structural indices in stable expression order.

        Args:
            node: Expression subtree to inspect.

        Returns:
            Index names not bound by a reduction in this subtree.
        """
        cached = self._free_cache.get(id(node))
        if cached is not None:
            return cached
        if isinstance(node, Gather):
            result = ()
        elif isinstance(node, Indexed):
            result = tuple(index.name for index in node.indices)
        elif isinstance(node, Reduction):
            inputs = self.free_indices(node.operand)
            if node.relation is not None:
                inputs = _ordered_union(inputs, self.free_indices(node.relation))
            reduced = {index.name for index in node.indices}
            result = tuple(index for index in inputs if index not in reduced)
        else:
            result = ()
            for child in children(node):
                result = _ordered_union(result, self.free_indices(child))
        self._free_cache[id(node)] = result
        return result

    def relation_domain(self, node: Expression) -> Expression | None:
        """Return the relation whose entry axis is carried by an expression.

        Args:
            node: Expression subtree to inspect.

        Returns:
            The relation expression defining the entry axis, if present.
        """
        key = id(node)
        if key in self._domain_cache:
            return self._domain_cache[key]
        if isinstance(node, Gather):
            result = node.relation
        elif isinstance(node, Unary):
            result = self.relation_domain(node.operand)
        elif isinstance(node, Function):
            result = self._merge_domains(
                tuple(self.relation_domain(argument) for argument in node.arguments)
            )
        elif isinstance(node, Binary):
            result = self._merge_domains(
                (self.relation_domain(node.left), self.relation_domain(node.right))
            )
        else:
            result = None
        self._domain_cache[key] = result
        return result

    @staticmethod
    def _merge_domains(domains: tuple[Expression | None, ...]) -> Expression | None:
        present = [domain for domain in domains if domain is not None]
        if not present:
            return None
        first = present[0]
        if any(domain != first for domain in present[1:]):
            raise ValueError("One expression cannot combine fields from different relations.")
        return first

    def _eval_at(self, node: Expression, demand: _Demand) -> np.ndarray:
        if isinstance(node, Number):
            return np.asarray(node.value)
        if isinstance(node, Symbol):
            value = self._symbol_value(node)
            if isinstance(value, RelationField):
                value = value.values
            value = np.asarray(value)
            if value.ndim == 0:
                return value
            if not demand.indices:
                return value
            if value.shape[-1] != 1:
                raise ValueError(
                    f"Symbol {node.name!r} must be indexed in an indexed expression; "
                    "only a trailing singleton structural dimension may be unindexed."
                )
            return value
        if isinstance(node, Parameter):
            return np.asarray(self._parameter_value(node), dtype=float)
        if isinstance(node, GroupedParameter):
            labels = np.asarray(self._eval_at(node.by, demand), dtype=object)
            key = _grouped_parameter_key(node)
            fitted = self.parameters.get(key, node.value)
            fitted = {} if fitted is None else dict(fitted)
            missing = [label for label in _unique(labels) if label not in fitted]
            if missing and node.default is None:
                raise ValueError(f"Grouped parameter {key!r} has no values for {missing!r}.")
            flat = [fitted.get(label, node.default) for label in labels.reshape(-1)]
            return np.asarray(flat, dtype=float).reshape(labels.shape)
        if isinstance(node, Indexed):
            return self._indexed_at(node, demand)
        if isinstance(node, Gather):
            return self._gather_at(node, demand)
        if isinstance(node, Reduction):
            return self._reduction_at(node, demand)
        if isinstance(node, Unary):
            return -self._eval_at(node.operand, demand)
        if isinstance(node, Binary):
            left = self._eval_at(node.left, demand)
            right = self._eval_at(node.right, demand)
            with np.errstate(all="ignore"):
                return _binary(node.operator, left, right)
        if isinstance(node, Function):
            arguments = [self._eval_at(argument, demand) for argument in node.arguments]
            return self._function(node.name, arguments)
        raise TypeError(f"Cannot evaluate indexed node {type(node).__name__}.")

    def _indexed_at(self, node: Indexed, demand: _Demand) -> np.ndarray:
        if not isinstance(node.base, Symbol):
            domain = self.relation_domain(node.base)
            if domain is not None:
                return self._relation_field_at(node, domain, demand)
            indices = tuple(index.name for index in node.indices)
            base_indices = self.free_indices(node.base)
            if len(indices) != 1 or len(base_indices) != 1:
                raise ValueError(
                    "A computed expression can be reindexed only when it has one free "
                    "structural index."
                )
            requested = demand.project(indices)
            return self._eval_at(node.base, _Demand(base_indices, requested))
        name = node.base.name
        indices = tuple(index.name for index in node.indices)
        requested = demand.project(indices)
        raw = self._symbol_value(node.base)
        field = raw if isinstance(raw, RelationField) else None
        value = np.asarray(field.values if field is not None else raw)

        if self._is_relation_table(value, len(indices)):
            sparse = self._relation_from_table(name, value, indices)
            return self._sparse_lookup(sparse, requested)

        if len(indices) == 1:
            if value.ndim == 0 or value.shape[-1] != self.num_nodes:
                raise ValueError(
                    f"Node variable {name!r} must have shape (..., num_nodes); "
                    f"got {value.shape}."
                )
            return value[..., requested[:, 0]]

        relation_name = (
            field.relation
            if field is not None
            else self._infer_relation(name, value, len(indices))
        )
        relation = self._relation_table(relation_name, len(indices))
        if value.ndim == 0 or value.shape[-1] != len(relation):
            raise ValueError(
                f"Relation field {name!r} must have shape (..., {len(relation)}); "
                f"got {value.shape}."
            )
        positions = {_coordinate_key(row): position for position, row in enumerate(relation)}
        output = np.zeros(value.shape[:-1] + (len(requested),), dtype=value.dtype)
        selected = [
            (column, positions.get(_coordinate_key(row)))
            for column, row in enumerate(requested)
        ]
        for column, position in selected:
            if position is not None:
                output[..., column] = value[..., position]
        return output

    def _gather_at(self, node: Gather, demand: _Demand) -> np.ndarray:
        if demand.indices:
            raise ValueError(
                "A gathered relation field must be explicitly indexed before it is "
                "used in a structural expression."
            )
        relation_indices, coordinates, weights = self._gather_entries(node.relation)
        operand_indices = self.free_indices(node.operand)
        unknown = set(operand_indices) - set(relation_indices)
        if unknown:
            raise ValueError(
                f"gather operand uses indices absent from its relation: {sorted(unknown)!r}."
            )
        values = self._eval_at(node.operand, _Demand(relation_indices, coordinates))
        return np.asarray(values) * weights

    def _relation_field_at(
        self, node: Indexed, domain: Expression, demand: _Demand
    ) -> np.ndarray:
        indices = tuple(index.name for index in node.indices)
        relation_indices, coordinates, _ = self._gather_entries(domain)
        if len(indices) != len(relation_indices):
            raise ValueError(
                f"This relation field requires {len(relation_indices)} indices, "
                f"but {len(indices)} were supplied."
            )
        values = np.asarray(
            self._eval_at(node.base, _Demand((), np.empty((1, 0), dtype=np.intp)))
        )
        if values.ndim == 0 or values.shape[-1] != len(coordinates):
            raise ValueError("A relation-domain expression must end with its entry axis.")
        requested = demand.project(indices)
        positions = {
            _coordinate_key(row): position for position, row in enumerate(coordinates)
        }
        output = np.zeros(values.shape[:-1] + (len(requested),), dtype=values.dtype)
        for column, row in enumerate(requested):
            position = positions.get(_coordinate_key(row))
            if position is not None:
                output[..., column] = values[..., position]
        return output

    def _gather_entries(
        self, relation: Expression
    ) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
        if isinstance(relation, Indexed) and isinstance(relation.base, Symbol):
            indices = tuple(index.name for index in relation.indices)
            table = np.asarray(self._symbol_value(relation.base))
            self._relation_from_table(relation.base.name, table, indices)
            return indices, table.astype(np.intp, copy=False), np.ones(len(table))

        sparse = self._relation_tensor(relation)
        if sparse.fill_value == 0:
            nonzero = sparse.values != 0
            return (
                sparse.indices,
                sparse.coordinates[nonzero],
                sparse.values[nonzero],
            )

        coordinates = self._root_coordinates(sparse.indices)
        values = self._sparse_lookup(sparse, coordinates)
        nonzero = values != 0
        return sparse.indices, coordinates[nonzero], values[nonzero]

    def _reduction_at(self, node: Reduction, demand: _Demand) -> np.ndarray:
        free = self.free_indices(node)
        requested = demand.project(free)
        unique, inverse = _unique_rows(requested)
        local_demand = _Demand(free, unique)
        if node.relation is None:
            reduced = self._ordinary_reduction(node, local_demand)
        else:
            reduced = self._relation_reduction(node, local_demand)
        return np.asarray(reduced)[..., inverse]

    def _ordinary_reduction(self, node: Reduction, demand: _Demand) -> np.ndarray:
        reduced = tuple(index.name for index in node.indices)
        operand_indices = self.free_indices(node.operand)
        if set(reduced).isdisjoint(operand_indices):
            return self._eval_at(node.operand, demand) * self.num_nodes ** len(reduced)
        optimized = self._factor_reduction(node, demand, reduced)
        if optimized is not None:
            return optimized
        return self._enumerated_sum(node.operand, demand, reduced)

    def _factor_reduction(
        self, node: Reduction, demand: _Demand, reduced: tuple[str, ...]
    ) -> np.ndarray | None:
        operand = node.operand
        if not isinstance(operand, Binary) or operand.operator not in {"+", "-", "*"}:
            return None
        left_indices = set(self.free_indices(operand.left))
        right_indices = set(self.free_indices(operand.right))
        reduced_set = set(reduced)
        if operand.operator in {"+", "-"}:
            left = self._reduction_at(
                Reduction(tuple(Index(name) for name in reduced), operand.left), demand
            )
            right = self._reduction_at(
                Reduction(tuple(Index(name) for name in reduced), operand.right), demand
            )
            return _binary(operand.operator, left, right)
        if left_indices.isdisjoint(reduced_set):
            factor = self._eval_at(operand.left, demand)
            total = self._reduction_at(
                Reduction(tuple(Index(name) for name in reduced), operand.right), demand
            )
            return factor * total
        if right_indices.isdisjoint(reduced_set):
            factor = self._eval_at(operand.right, demand)
            total = self._reduction_at(
                Reduction(tuple(Index(name) for name in reduced), operand.left), demand
            )
            return factor * total
        return None

    def _relation_reduction(self, node: Reduction, demand: _Demand) -> np.ndarray:
        relation = self._relation_tensor(node.relation)
        reduced = tuple(index.name for index in node.indices)
        operand_indices = self.free_indices(node.operand)
        output_shape: tuple[int, ...] | None = None
        result: np.ndarray | None = None

        if relation.fill_value != 0:
            baseline = self._enumerated_sum(
                node.operand,
                demand,
                reduced,
            )
            result = relation.fill_value * baseline
            output_shape = np.shape(result)

        correction = self._sparse_correction(
            node.operand,
            demand,
            reduced,
            operand_indices,
            relation,
        )
        if result is None:
            result = correction
        else:
            result = result + correction
        if output_shape is not None and np.shape(result) != output_shape:
            result = np.broadcast_to(result, output_shape)
        return result

    def _enumerated_sum(
        self,
        operand: Expression,
        demand: _Demand,
        reduced: tuple[str, ...],
    ) -> np.ndarray:
        dependent = tuple(
            index for index in self.free_indices(operand) if index not in reduced
        )
        requested = demand.project(dependent)
        unique, inverse = _unique_rows(requested)
        local_demand = _Demand(dependent, unique)
        output_count = len(unique)
        combinations = np.asarray(
            list(product(range(self.num_nodes), repeat=len(reduced))), dtype=np.intp
        )
        if not len(combinations):
            combinations = np.empty((1, 0), dtype=np.intp)
        rows: list[list[int]] = []
        full_indices = _ordered_union(local_demand.indices, reduced)
        for output in local_demand.coordinates:
            fixed = dict(zip(local_demand.indices, output))
            for combination in combinations:
                bindings = fixed | dict(zip(reduced, combination))
                rows.append([bindings[index] for index in full_indices])
        values = self._eval_at(operand, _Demand(full_indices, np.asarray(rows, dtype=np.intp)))
        values = np.asarray(values)
        count = len(combinations)
        result = values.reshape(values.shape[:-1] + (output_count, count)).sum(axis=-1)
        return result[..., inverse]

    def _sparse_correction(
        self,
        operand: Expression,
        demand: _Demand,
        reduced: tuple[str, ...],
        operand_indices: tuple[str, ...],
        relation: _SparseTensor,
    ) -> np.ndarray:
        free_relation = tuple(index for index in relation.indices if index not in reduced)
        extra_reduced = tuple(index for index in reduced if index not in relation.indices)
        extra_combinations = list(product(range(self.num_nodes), repeat=len(extra_reduced))) or [()]
        rows: list[list[int]] = []
        weights: list[float] = []
        groups: list[int] = []
        eval_indices = _ordered_union(
            demand.indices, _ordered_union(relation.indices, extra_reduced)
        )
        relation_positions = [relation.indices.index(index) for index in free_relation]
        grouped_entries: dict[tuple[int, ...], list[tuple[np.ndarray, float]]] = {}
        for relation_row, stored in zip(relation.coordinates, relation.values):
            key = tuple(int(relation_row[position]) for position in relation_positions)
            grouped_entries.setdefault(key, []).append((relation_row, float(stored)))

        for group, output in enumerate(demand.coordinates):
            fixed = dict(zip(demand.indices, output))
            key = tuple(int(fixed[index]) for index in free_relation)
            for relation_row, stored in grouped_entries.get(key, ()):
                binding = dict(zip(relation.indices, relation_row))
                for extra in extra_combinations:
                    merged = fixed | binding | dict(zip(extra_reduced, extra))
                    rows.append([merged[index] for index in eval_indices])
                    weights.append(float(stored - relation.fill_value))
                    groups.append(group)

        if not rows:
            return np.zeros((len(demand.coordinates),), dtype=float)
        values = np.asarray(
            self._eval_at(operand, _Demand(eval_indices, np.asarray(rows, dtype=np.intp)))
        )
        weighted = values * np.asarray(weights)
        result = np.zeros(
            values.shape[:-1] + (len(demand.coordinates),),
            dtype=np.result_type(weighted, float),
        )
        np.add.at(result, (..., np.asarray(groups, dtype=np.intp)), weighted)
        return result

    def _relation_tensor(self, node: Expression) -> _SparseTensor:
        cached = self._relation_cache.get(id(node))
        if cached is not None:
            return cached
        result = self._compile_relation(node)
        self._relation_cache[id(node)] = result
        return result

    def _compile_relation(
        self, node: Expression, indices: tuple[str, ...] | None = None
    ) -> _SparseTensor:
        if isinstance(node, Indexed) and isinstance(node.base, Symbol):
            names = tuple(index.name for index in node.indices)
            value = np.asarray(self._symbol_value(node.base))
            return self._relation_from_table(node.base.name, value, names)
        if isinstance(node, Number):
            if indices is None:
                return _SparseTensor(
                    (), np.empty((0, 0), dtype=np.intp), np.empty(0),
                    float(node.value), self.num_nodes,
                )
            return _SparseTensor(
                indices,
                np.empty((0, len(indices)), dtype=np.intp),
                np.empty(0),
                float(node.value),
                self.num_nodes,
            )
        if isinstance(node, Parameter):
            value = float(self._parameter_value(node))
            return _SparseTensor(
                indices or (),
                np.empty((0, len(indices or ())), dtype=np.intp),
                np.empty(0),
                value,
                self.num_nodes,
            )
        if isinstance(node, Unary):
            operand = self._compile_relation(node.operand, indices)
            return _SparseTensor(
                operand.indices, operand.coordinates, -operand.values,
                -operand.fill_value, operand.num_nodes,
            )
        if isinstance(node, Function):
            if len(node.arguments) != 1:
                raise ValueError("Relation functions must be unary.")
            operand = self._compile_relation(node.arguments[0], indices)
            function = self._numpy_function(node.name)
            with np.errstate(all="ignore"):
                return _SparseTensor(
                    operand.indices,
                    operand.coordinates,
                    np.asarray(function(operand.values), dtype=float),
                    float(function(operand.fill_value)),
                    operand.num_nodes,
                )
        if isinstance(node, Binary):
            left = self._compile_relation(node.left, indices)
            right = self._compile_relation(node.right, left.indices or indices)
            if not left.indices:
                left = _constant_sparse(left.fill_value, right.indices, self.num_nodes)
            if not right.indices:
                right = _constant_sparse(right.fill_value, left.indices, self.num_nodes)
            if set(left.indices) != set(right.indices):
                raise ValueError("Relation operands must use the same structural indices.")
            right = _reorder_sparse(right, left.indices)
            return _combine_sparse(left, right, node.operator)
        raise ValueError(
            "The first argument of an indexed sum must be an indexed relation "
            "or algebra over indexed relations."
        )

    def _relation_from_table(
        self, name: str, value: np.ndarray, indices: tuple[str, ...]
    ) -> _SparseTensor:
        if not self._is_relation_table(value, len(indices)):
            raise ValueError(
                f"Relation {name!r} must be an integer coordinate table shaped "
                f"(entries, {len(indices)})."
            )
        table = value.astype(np.intp, copy=False)
        if np.any(table < 0) or np.any(table >= self.num_nodes):
            raise ValueError(
                f"Relation {name!r} contains an endpoint outside [0, {self.num_nodes})."
            )
        coordinates = np.unique(table, axis=0)
        return _SparseTensor(
            indices, coordinates, np.ones(len(coordinates)), 0.0, self.num_nodes
        )

    def _sparse_lookup(self, sparse: _SparseTensor, requested: np.ndarray) -> np.ndarray:
        entries = {
            _coordinate_key(row): value
            for row, value in zip(sparse.coordinates, sparse.values)
        }
        return np.asarray(
            [entries.get(_coordinate_key(row), sparse.fill_value) for row in requested],
            dtype=float,
        )

    def _infer_relation(self, field_name: str, value: np.ndarray, arity: int) -> str:
        candidates = []
        for name, raw in self.values.items():
            raw = raw.values if isinstance(raw, RelationField) else raw
            array = np.asarray(raw)
            if (
                self._is_relation_table(array, arity)
                and value.ndim
                and value.shape[-1] == len(array)
            ):
                candidates.append(name)
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot infer the relation for field {field_name!r}; wrap it in "
                "RelationField(values, relation='A')."
            )
        return candidates[0]

    def _relation_table(self, name: str, arity: int) -> np.ndarray:
        if name not in self.values:
            raise KeyError(f"No value was provided for relation {name!r}.")
        value = self.values[name]
        value = value.values if isinstance(value, RelationField) else value
        array = np.asarray(value)
        if not self._is_relation_table(array, arity):
            raise ValueError(f"{name!r} is not an integer relation table of arity {arity}.")
        return array.astype(np.intp, copy=False)

    @staticmethod
    def _is_relation_table(value: np.ndarray, arity: int) -> bool:
        return (
            value.ndim == 2
            and value.shape[1] == arity
            and arity >= 2
            and value.dtype.kind in "iu"
        )

    def _symbol_value(self, node: Symbol) -> Any:
        if node.name in self.values:
            return self.values[node.name]
        if node.value is not None:
            return node.value
        raise KeyError(f"No value was provided for symbol {node.name!r}.")

    def _parameter_value(self, node: Parameter) -> Any:
        value = self.parameters.get(node.name, self.parameter_defaults.get(node.name))
        if value is None:
            raise ValueError(f"Parameter {node.name!r} has no value. Fit it or provide it.")
        return value

    def _function(self, name: str, arguments: list[np.ndarray]) -> np.ndarray:
        if name == "delay":
            return self._delay(arguments[0], arguments[1])
        with np.errstate(all="ignore"):
            return self._numpy_function(name)(*arguments)

    @staticmethod
    def _numpy_function(name: str) -> Callable[..., Any]:
        functions = {
            "abs": np.abs, "arccos": np.arccos, "arcsin": np.arcsin,
            "arctan": np.arctan, "cos": np.cos, "cosh": np.cosh,
            "cot": lambda value: 1.0 / np.tan(value),
            "csc": lambda value: 1.0 / np.sin(value), "exp": np.exp,
            "inv": lambda value: 1.0 / value, "log": np.log,
            "log10": np.log10, "max": np.maximum, "min": np.minimum,
            "pow2": lambda value: value**2, "pow3": lambda value: value**3,
            "sec": lambda value: 1.0 / np.cos(value),
            "sech": lambda value: 1.0 / np.cosh(value), "sin": np.sin,
            "sinh": np.sinh, "sign": np.sign, "sqrt": np.sqrt,
            "tan": np.tan, "tanh": np.tanh,
            "sigmoid": lambda value: 1.0 / (1.0 + np.exp(-value)),
        }
        return functions[name]

    def _delay(self, value: Any, lag: Any) -> np.ndarray:
        value = np.asarray(value, dtype=float)
        lag = np.asarray(lag, dtype=float)
        if self.delay_resolver is not None:
            return np.asarray(self.delay_resolver(value, lag, self.time))
        if value.ndim == 0:
            return value
        time = np.arange(value.shape[0], dtype=float) if self.time is None else self.time
        if time.shape != (value.shape[0],):
            raise ValueError("time must match the leading data axis.")
        query = np.broadcast_to(time - lag, (value.shape[0],))
        flat = value.reshape(value.shape[0], -1)
        delayed = np.column_stack([
            np.interp(query, time, column, left=np.nan, right=np.nan) for column in flat.T
        ])
        return delayed.reshape(value.shape)

    def _validate_scopes(self, expression: Expression) -> None:
        def visit(node: Expression, bound: frozenset[str]) -> None:
            if isinstance(node, Reduction):
                names = tuple(index.name for index in node.indices)
                if len(set(names)) != len(names):
                    raise ValueError(f"A sum cannot bind an index more than once: {names}.")
                reused = bound.intersection(names)
                if reused:
                    raise ValueError(
                        "Nested sums cannot bind an index that is already active; "
                        f"rename {sorted(reused)!r}."
                    )
                if node.relation is not None:
                    visit(node.relation, bound)
                visit(node.operand, bound.union(names))
                return
            for child in children(node):
                visit(child, bound)

        visit(expression, frozenset())

    def _validate_bare_symbols(self, expression: Expression) -> None:
        def visit(node: Expression, allow_relation_field: bool) -> None:
            if isinstance(node, Symbol):
                value = self._symbol_value(node)
                value = value.values if isinstance(value, RelationField) else value
                array = np.asarray(value)
                if array.ndim and array.shape[-1] != 1 and not allow_relation_field:
                    raise ValueError(
                        f"Symbol {node.name!r} must be indexed because this expression "
                        "uses structural indices."
                    )
                return
            if isinstance(node, Indexed):
                if isinstance(node.base, Symbol):
                    return
                visit(node.base, self.relation_domain(node.base) is not None)
                return
            if isinstance(node, Gather):
                visit(node.relation, False)
                visit(node.operand, False)
                return
            carries_relation = self.relation_domain(node) is not None
            for child in children(node):
                visit(child, allow_relation_field or carries_relation)

        visit(expression, self.relation_domain(expression) is not None)


def _ordered_union(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    return left + tuple(item for item in right if item not in left)


def _binary(operator: str, left: Any, right: Any) -> np.ndarray:
    return {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "**": np.power}[
        operator
    ](left, right)


def _unique_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rows.shape[1] == 0:
        return rows[:1], np.zeros(len(rows), dtype=np.intp)
    return np.unique(rows, axis=0, return_inverse=True)


def _coordinate_key(row: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in row)


def _constant_sparse(
    value: float, indices: tuple[str, ...], num_nodes: int
) -> _SparseTensor:
    return _SparseTensor(
        indices, np.empty((0, len(indices)), dtype=np.intp),
        np.empty(0), value, num_nodes,
    )


def _reorder_sparse(tensor: _SparseTensor, indices: tuple[str, ...]) -> _SparseTensor:
    if tensor.indices == indices:
        return tensor
    columns = [tensor.indices.index(index) for index in indices]
    return _SparseTensor(
        indices, tensor.coordinates[:, columns], tensor.values,
        tensor.fill_value, tensor.num_nodes,
    )


def _combine_sparse(left: _SparseTensor, right: _SparseTensor, operator: str) -> _SparseTensor:
    function = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide, "**": np.power}[
        operator
    ]
    with np.errstate(all="ignore"):
        fill = float(function(left.fill_value, right.fill_value))
    left_values = {_coordinate_key(row): value for row, value in zip(left.coordinates, left.values)}
    right_values = {
        _coordinate_key(row): value
        for row, value in zip(right.coordinates, right.values)
    }
    keys = sorted(set(left_values).union(right_values))
    coordinates = []
    values = []
    for key in keys:
        with np.errstate(all="ignore"):
            value = float(
                function(
                    left_values.get(key, left.fill_value),
                    right_values.get(key, right.fill_value),
                )
            )
        if not np.isclose(value, fill, equal_nan=True):
            coordinates.append(key)
            values.append(value)
    return _SparseTensor(
        left.indices,
        np.asarray(coordinates, dtype=np.intp).reshape(-1, len(left.indices)),
        np.asarray(values, dtype=float),
        fill,
        left.num_nodes,
    )


def _unique(values: np.ndarray) -> list[Any]:
    result = []
    for value in values.reshape(-1):
        value = value.item() if isinstance(value, np.generic) else value
        if value not in result:
            result.append(value)
    return result


def _grouped_parameter_key(node: GroupedParameter) -> str:
    if node.name:
        return node.name
    if isinstance(node.by, Symbol):
        return f"grouped:{node.by.name}"
    return f"grouped:{node.by}"
