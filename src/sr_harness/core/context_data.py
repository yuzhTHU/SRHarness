"""Manifest-backed NumPy data stored in a workspace directory."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


class ContextManifestError(ValueError):
    """Raised when a context-data manifest cannot be validated."""


class _ContextDataReader:
    """Private implementation shared by the public loading functions."""

    MANIFEST_NAME = "manifest.json"
    REQUIRED_ROOT_FIELDS = {"variables", "axes"}
    OPTIONAL_ROOT_FIELDS = {"num_nodes"}
    REQUIRED_VARIABLE_FIELDS = {"file", "description", "axes"}
    OPTIONAL_VARIABLE_FIELDS = {"structure"}
    AXIS_SOURCE_FIELDS = {"values", "file", "size"}
    AXIS_FIELDS = {"description", *AXIS_SOURCE_FIELDS}

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()

    def inspect(self) -> dict[str, Any]:
        """Validate the store and return diagnostics without raising.

        Returns:
            A serializable report containing errors, warnings, and array summaries.
        """
        errors: list[str] = []
        warnings: list[str] = []
        manifest = self._read_manifest(errors)
        loaded = self._validate_manifest(manifest, errors, warnings) if manifest is not None else None
        report: dict[str, Any] = {
            "valid": not errors,
            "path": str(self.directory),
            "errors": errors,
            "warnings": warnings,
            "num_nodes": loaded["num_nodes"] if loaded is not None else None,
            "variables": {},
            "axes": {},
        }
        if loaded is not None:
            report["variables"] = {
                name: {
                    "file": manifest["variables"][name]["file"],
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "axes": list(loaded["variable_axes"][name]),
                    **(
                        {"structure": loaded["variable_structures"][name]}
                        if name in loaded["variable_structures"] else {}
                    ),
                    "description": loaded["variable_descriptions"][name],
                }
                for name, value in loaded["data"].items()
                if name in manifest["variables"]
            }
            report["axes"] = {
                name: {
                    "size": len(loaded["data"][name]),
                    "dtype": str(loaded["data"][name].dtype),
                    "storage": axis["storage"],
                    "description": loaded["variable_descriptions"][name],
                }
                for name, axis in loaded["axis_metadata"].items()
            }
        return report

    def load(self) -> dict[str, Any]:
        """Validate and load all variables and axes.

        Returns:
            An AgentContext-ready mapping containing arrays and metadata.

        Raises:
            ContextManifestError: If the manifest or referenced arrays are invalid.
        """
        errors: list[str] = []
        warnings: list[str] = []
        manifest = self._read_manifest(errors)
        loaded = self._validate_manifest(manifest, errors, warnings) if manifest is not None else None
        if errors or loaded is None:
            raise ContextManifestError("Invalid context.data:\n- " + "\n- ".join(errors))
        return loaded

    def _read_manifest(self, errors: list[str]) -> dict[str, Any] | None:
        manifest_path = self.directory / self.MANIFEST_NAME
        if not self.directory.is_dir():
            errors.append(f"directory does not exist: {self.directory}")
            return None
        if not manifest_path.is_file():
            errors.append(f"missing {self.MANIFEST_NAME}")
            return None

        def reject_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ContextManifestError(f"duplicate JSON key: {key}")
                result[key] = value
            return result

        try:
            with manifest_path.open(encoding="utf-8") as stream:
                manifest = json.load(stream, object_pairs_hook=reject_duplicates)
        except (ContextManifestError, json.JSONDecodeError, OSError) as exc:
            errors.append(f"cannot read {self.MANIFEST_NAME}: {exc}")
            return None
        if not isinstance(manifest, dict):
            errors.append("manifest root must be a JSON object")
            return None
        return manifest

    def _validate_manifest(
        self,
        manifest: dict[str, Any],
        errors: list[str],
        warnings: list[str],
    ) -> dict[str, Any] | None:
        self._check_fields(
            "manifest",
            manifest,
            self.REQUIRED_ROOT_FIELDS,
            errors,
            optional=self.OPTIONAL_ROOT_FIELDS,
        )
        variables = manifest.get("variables")
        axes = manifest.get("axes")
        num_nodes = manifest.get("num_nodes")
        if "num_nodes" in manifest and (
            isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 1
        ):
            errors.append("manifest.num_nodes must be a positive integer")
            num_nodes = None
        if not isinstance(variables, dict) or not variables:
            errors.append("manifest.variables must be a non-empty object")
            variables = {}
        if not isinstance(axes, dict):
            errors.append("manifest.axes must be an object")
            axes = {}

        loaded_axes: dict[str, dict[str, Any]] = {}
        referenced_files = {self.MANIFEST_NAME}
        for name, spec in axes.items():
            location = f"axes.{name}"
            if not self._valid_name(name):
                errors.append(f"{location}: name must be non-empty and contain no path separators")
                continue
            if not isinstance(spec, dict):
                errors.append(f"{location} must be an object")
                continue
            if "description" not in spec:
                errors.append(f"{location} is missing fields: ['description']")
            if extra := sorted(set(spec) - self.AXIS_FIELDS):
                errors.append(f"{location} contains unsupported fields: {extra}")
            description = self._description(location, spec, errors)
            sources = self.AXIS_SOURCE_FIELDS & set(spec)
            if len(sources) != 1:
                errors.append(f"{location} must contain exactly one of values, file, or size")
                continue
            source = next(iter(sources))
            try:
                if source == "values":
                    if not isinstance(spec["values"], list) or not spec["values"]:
                        raise ValueError("values must be a non-empty JSON array")
                    if any(isinstance(value, (list, dict)) for value in spec["values"]):
                        raise ValueError("values must contain scalar JSON values")
                    values = np.asarray(spec["values"])
                elif source == "size":
                    size = spec["size"]
                    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                        raise ValueError("size must be a positive integer")
                    values = np.arange(size)
                else:
                    expected = f"{name}.npy"
                    if spec["file"] != expected:
                        raise ValueError(f"file must be {expected!r}")
                    referenced_files.add(expected)
                    values = self._load_array(expected)
                    if values.ndim != 1:
                        raise ValueError(f"axis array must be one-dimensional, got shape {values.shape}")
                if values.ndim != 1:
                    raise ValueError("values must be one-dimensional")
                loaded_axes[name] = {"values": values, "description": description, "storage": source}
            except (OSError, ValueError) as exc:
                errors.append(f"{location}: {exc}")

        loaded_values: dict[str, np.ndarray] = {}
        variable_axes: dict[str, tuple[str, ...]] = {}
        variable_structures: dict[str, str] = {}
        descriptions: dict[str, str] = {}
        used_axes: set[str] = set()
        for name, spec in variables.items():
            location = f"variables.{name}"
            if not self._valid_name(name):
                errors.append(f"{location}: name must be non-empty and contain no path separators")
                continue
            if not isinstance(spec, dict):
                errors.append(f"{location} must be an object")
                continue
            self._check_fields(
                location,
                spec,
                self.REQUIRED_VARIABLE_FIELDS,
                errors,
                optional=self.OPTIONAL_VARIABLE_FIELDS,
            )
            description = self._description(location, spec, errors)
            structure = spec.get("structure")
            if structure is not None and not self._valid_name(structure):
                errors.append(f"{location}.structure must name an A or T variable")
                structure = None
            expected = f"{name}.npy"
            if spec.get("file") != expected:
                errors.append(f"{location}.file must be {expected!r}")
                continue
            declared_axes = spec.get("axes")
            if not isinstance(declared_axes, list) or any(
                not isinstance(axis, str) or not axis for axis in declared_axes
            ):
                errors.append(f"{location}.axes must be an array of non-empty axis names")
                continue
            missing_axes = [axis for axis in declared_axes if axis not in axes]
            if missing_axes:
                errors.append(f"{location}.axes references missing axes: {missing_axes}")
                continue
            referenced_files.add(expected)
            try:
                value = self._load_array(expected)
            except (OSError, ValueError) as exc:
                errors.append(f"{location}: {exc}")
                continue
            if value.ndim != len(declared_axes):
                errors.append(
                    f"{location}: array has {value.ndim} dimensions but declares "
                    f"{len(declared_axes)} axes"
                )
                continue
            mismatches = [
                f"{axis}={len(loaded_axes[axis]['values'])} (data={value.shape[index]})"
                for index, axis in enumerate(declared_axes)
                if axis in loaded_axes and len(loaded_axes[axis]["values"]) != value.shape[index]
            ]
            if mismatches:
                errors.append(f"{location}: axis length mismatch: {', '.join(mismatches)}")
                continue
            loaded_values[name] = value
            variable_axes[name] = tuple(declared_axes)
            if structure is not None:
                variable_structures[name] = structure
            descriptions[name] = description
            used_axes.update(declared_axes)

        unused_axes = sorted(set(axes) - used_axes)
        if unused_axes:
            errors.append(f"manifest.axes contains unreferenced axes: {unused_axes}")

        if bool(variable_structures) != (num_nodes is not None):
            errors.append("manifest.num_nodes and variable structure metadata must appear together")
        for variable, structure in variable_structures.items():
            location = f"variables.{variable}.structure"
            if variable == structure:
                errors.append(f"{location}: a structured variable cannot reference itself")
                continue
            if structure not in loaded_values:
                errors.append(f"{location} references missing variable {structure!r}")
                continue
            relation = loaded_values[structure]
            if relation.ndim != 2 or relation.shape[1] not in {2, 3}:
                errors.append(
                    f"{location}: {structure!r} must have shape (E, 2) or (H, 3), "
                    f"got {relation.shape}"
                )
                continue
            dependent = loaded_values.get(variable)
            if dependent is not None and (
                dependent.ndim == 0 or dependent.shape[-1] != relation.shape[0]
            ):
                errors.append(
                    f"{location}: {variable!r} must end in a dimension of "
                    f"length {relation.shape[0]}"
                )
            if relation.dtype.kind not in "iu":
                errors.append(f"{location}: relation endpoints must use an integer dtype")
            elif num_nodes is not None and (
                np.any(relation < 0) or np.any(relation >= num_nodes)
            ):
                errors.append(f"{location}: relation endpoints must be in [0, {num_nodes})")
        actual_npy = {path.name for path in self.directory.glob("*.npy") if path.is_file()}
        untracked = sorted(actual_npy - referenced_files)
        if untracked:
            warnings.append(f"unreferenced NPY files: {untracked}")
        if errors:
            return None
        data = {**{name: axis["values"] for name, axis in loaded_axes.items()}, **loaded_values}
        all_descriptions = {name: axis["description"] for name, axis in loaded_axes.items()} | descriptions
        return {
            "data": data,
            "variable_descriptions": all_descriptions,
            "variable_axes": variable_axes,
            "variable_structures": variable_structures,
            "num_nodes": num_nodes,
            "axis_metadata": loaded_axes,
        }

    def _load_array(self, filename: str) -> np.ndarray:
        path = self.directory / filename
        if not path.is_file() or path.parent != self.directory:
            raise ValueError(f"missing array file: {filename}")
        value = np.load(path, allow_pickle=False)
        if not isinstance(value, np.ndarray):
            raise ValueError(f"{filename} does not contain one NPY array")
        if value.dtype.hasobject:
            raise ValueError(f"{filename} uses object dtype, which is not allowed")
        return value

    @staticmethod
    def _valid_name(name: Any) -> bool:
        return (
            isinstance(name, str)
            and bool(name.strip())
            and name not in {".", ".."}
            and "/" not in name
            and "\\" not in name
            and "\0" not in name
        )

    @staticmethod
    def _description(location: str, spec: dict[str, Any], errors: list[str]) -> str:
        description = spec.get("description")
        if not isinstance(description, str):
            errors.append(f"{location}.description must be a string")
            return ""
        return description.strip()

    @staticmethod
    def _check_fields(
        location: str,
        value: dict[str, Any],
        expected: set[str],
        errors: list[str],
        *,
        optional: set[str] | None = None,
    ) -> None:
        optional = optional or set()
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected - optional)
        if missing:
            errors.append(f"{location} is missing fields: {missing}")
        if extra:
            errors.append(f"{location} contains unsupported fields: {extra}")


def load_context_data(directory: str | Path) -> dict[str, Any]:
    """Validate and load one manifest-backed ``context.data`` directory."""
    return _ContextDataReader(directory).load()


def inspect_context_data(directory: str | Path) -> dict[str, Any]:
    """Return validation diagnostics without raising for manifest errors."""
    return _ContextDataReader(directory).inspect()


def update_context_data_descriptions(
    directory: str | Path, descriptions: dict[str, str],
) -> dict[str, Any]:
    """Atomically update variable and axis descriptions in ``manifest.json``.

    The complete store is validated before the edit. Because only existing
    string-valued description fields are changed, its structural validity is
    preserved. The returned mapping matches :func:`load_context_data`.
    """
    directory = Path(directory).resolve()
    manifest_path = directory / _ContextDataReader.MANIFEST_NAME
    loaded = load_context_data(directory)
    if not isinstance(descriptions, dict) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in descriptions.items()
    ):
        raise ValueError("descriptions must map variable names to text")
    unknown = set(descriptions) - set(loaded["data"])
    if unknown:
        raise ValueError(f"unknown context.data variables: {sorted(unknown)}")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    for section in ("variables", "axes"):
        for name, spec in manifest[section].items():
            if name in descriptions:
                spec["description"] = descriptions[name].strip()
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=directory, prefix=".manifest-",
        suffix=".json.tmp", delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary_path, manifest_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    updated = dict(loaded)
    updated["variable_descriptions"] = loaded["variable_descriptions"] | {
        name: value.strip() for name, value in descriptions.items()
    }
    return updated
