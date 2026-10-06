"""Manifest-backed NumPy data stored in a workspace directory."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class ContextManifestError(ValueError):
    """Raised when a context-data manifest cannot be validated."""


@dataclass(frozen=True)
class ContextAxis:
    """One named axis shared by one or more structured variables.

    Args:
        name: Logical axis name used by variable declarations.
        values: Coordinate values, including generated positional coordinates.
        description: Human-readable meaning and units.
        storage: Manifest representation: ``values``, ``file``, or ``size``.
    """

    name: str
    values: np.ndarray
    description: str
    storage: str


class ContextData(dict[str, np.ndarray]):
    """Array mapping enriched with variable descriptions and named axes."""

    def __init__(
        self,
        values: dict[str, np.ndarray],
        *,
        axes: dict[str, ContextAxis] | None = None,
        variable_axes: dict[str, tuple[str, ...]] | None = None,
        descriptions: dict[str, str] | None = None,
        directory: Path | None = None,
    ):
        super().__init__(values)
        self.axes = dict(axes or {})
        self.variable_axes = dict(variable_axes or {})
        self.descriptions = dict(descriptions or {})
        self.directory = directory


class ContextDataStore:
    """Validate and load a flat NPY collection described by ``manifest.json``."""

    MANIFEST_NAME = "manifest.json"
    ROOT_FIELDS = {"variables", "axes"}
    VARIABLE_FIELDS = {"file", "description", "axes"}
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
            "variables": {},
            "axes": {},
        }
        if loaded is not None:
            report["variables"] = {
                name: {
                    "file": manifest["variables"][name]["file"],
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "axes": list(loaded.variable_axes[name]),
                    "description": loaded.descriptions[name],
                }
                for name, value in loaded.items()
            }
            report["axes"] = {
                name: {
                    "size": len(axis.values),
                    "dtype": str(axis.values.dtype),
                    "storage": axis.storage,
                    "description": axis.description,
                }
                for name, axis in loaded.axes.items()
            }
        return report

    def load(self) -> ContextData:
        """Validate and load all variables and axes.

        Returns:
            A mapping of variable names to arrays with attached axis metadata.

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
    ) -> ContextData | None:
        self._check_fields("manifest", manifest, self.ROOT_FIELDS, errors)
        variables = manifest.get("variables")
        axes = manifest.get("axes")
        if not isinstance(variables, dict) or not variables:
            errors.append("manifest.variables must be a non-empty object")
            variables = {}
        if not isinstance(axes, dict):
            errors.append("manifest.axes must be an object")
            axes = {}

        loaded_axes: dict[str, ContextAxis] = {}
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
                loaded_axes[name] = ContextAxis(name, values, description, source)
            except (OSError, ValueError) as exc:
                errors.append(f"{location}: {exc}")

        loaded_values: dict[str, np.ndarray] = {}
        variable_axes: dict[str, tuple[str, ...]] = {}
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
            self._check_fields(location, spec, self.VARIABLE_FIELDS, errors)
            description = self._description(location, spec, errors)
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
                f"{axis}={len(loaded_axes[axis].values)} (data={value.shape[index]})"
                for index, axis in enumerate(declared_axes)
                if axis in loaded_axes and len(loaded_axes[axis].values) != value.shape[index]
            ]
            if mismatches:
                errors.append(f"{location}: axis length mismatch: {', '.join(mismatches)}")
                continue
            loaded_values[name] = value
            variable_axes[name] = tuple(declared_axes)
            descriptions[name] = description
            used_axes.update(declared_axes)

        unused_axes = sorted(set(axes) - used_axes)
        if unused_axes:
            errors.append(f"manifest.axes contains unreferenced axes: {unused_axes}")
        actual_npy = {path.name for path in self.directory.glob("*.npy") if path.is_file()}
        untracked = sorted(actual_npy - referenced_files)
        if untracked:
            warnings.append(f"unreferenced NPY files: {untracked}")
        if errors:
            return None
        return ContextData(
            loaded_values,
            axes=loaded_axes,
            variable_axes=variable_axes,
            descriptions=descriptions,
            directory=self.directory,
        )

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
        if not isinstance(description, str) or not description.strip():
            errors.append(f"{location}.description must be a non-empty string")
            return ""
        return description.strip()

    @staticmethod
    def _check_fields(
        location: str,
        value: dict[str, Any],
        expected: set[str],
        errors: list[str],
    ) -> None:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        if missing:
            errors.append(f"{location} is missing fields: {missing}")
        if extra:
            errors.append(f"{location} contains unsupported fields: {extra}")
