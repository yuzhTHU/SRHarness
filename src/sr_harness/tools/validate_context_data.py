"""Validate a workspace ``context.data`` collection and explain repairs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core import inspect_context_data, load_context_data
from .base_tool import BaseTool, ToolMetadata


def _repair_hint(error: str) -> str:
    """Translate one validation error into a concrete repair action."""
    rules = (
        ("directory does not exist", "Create the context.data directory before validating it."),
        ("missing manifest.json", "Create context.data/manifest.json with non-empty variables and an axes object."),
        ("cannot read manifest.json", "Fix the JSON syntax, encoding, or duplicate keys in manifest.json."),
        ("manifest root must be", "Make the top level of manifest.json a JSON object."),
        ("manifest.variables must be a non-empty object", "Declare at least one variable under manifest.variables."),
        ("manifest.axes must be an object", "Set manifest.axes to a JSON object, even when no axes are needed."),
        ("is missing fields", "Add every listed required field at the reported manifest location."),
        ("contains unsupported fields", "Remove the listed fields; store scientific meaning and units in description."),
        ("must contain exactly one of values, file, or size", "Choose exactly one axis representation: inline values, <axis>.npy, or a positive size."),
        ("name must be non-empty", "Use a non-empty flat name without '/', '\\', '.' or '..'."),
        ("file must be", "Rename the NPY file and set file to the exact <name>.npy value shown in the error."),
        ("missing array file", "Create the referenced NPY file in context.data using numpy.save."),
        ("object dtype", "Convert object arrays to a numeric, Boolean, or fixed-width Unicode NumPy dtype."),
        ("axes references missing axes", "Declare every referenced axis under manifest.axes or correct the variable's axes list."),
        ("must be an array of non-empty axis names", "Set axes to a JSON array of non-empty axis-name strings in dimension order."),
        ("dimensions but declares", "Make the number of declared axes equal to the array rank, in dimension order."),
        ("axis length mismatch", "Make each array dimension equal to the size of its corresponding declared axis."),
        ("unreferenced axes", "Remove unused axes or reference them from at least one variable."),
        ("num_nodes and variable structure", "Provide both num_nodes and structure metadata for graph data, or omit both for ordinary data."),
        ("must have shape (E, 2) or (H, 3)", "Store graph edges as (E, 2) or hyperedges as (H, 3)."),
        ("must end in a dimension", "Make the structured variable's final dimension match the number of edges or hyperedges."),
        ("relation endpoints must use an integer dtype", "Save relation endpoints with an integer NumPy dtype."),
        ("relation endpoints must be in", "Correct relation indices or num_nodes so every endpoint lies in the reported range."),
        ("description must be a string", "Set description to a JSON string explaining meaning and units."),
    )
    return next((hint for fragment, hint in rules if fragment in error), "Correct the reported manifest field or NPY file, then run validate_context_data again.")


@BaseTool.register("validate_context_data")
class ValidateContextDataTool(BaseTool):
    """Validate context.data without mutating the live AgentContext."""

    metadata = ToolMetadata(name="validate_context_data")

    def execute(self, path: str = "context.data") -> dict[str, Any]:
        """Validate a manifest-backed context-data directory and explain every repair.

        This calls the production ``load_context_data`` loader, so a successful
        result guarantees that InteractiveSession can load the same collection.
        On failure it returns all detectable errors together with specific repair
        suggestions instead of stopping at an opaque exception.

        Args:
            path: Workspace-relative directory containing manifest.json and flat NPY files.

        Returns:
            Validation status, complete diagnostics, array summaries, and repair actions.
        """
        workspace = self.context.workspace
        directory = (workspace / path).resolve()
        if workspace not in directory.parents and directory != workspace:
            raise ValueError("path must identify a directory inside the workspace")
        try:
            load_context_data(directory)
        except ValueError:
            pass
        report = inspect_context_data(directory)
        errors = list(report.get("errors", []))
        report["repairs"] = [
            {"error": error, "action": _repair_hint(error)}
            for error in errors
        ]
        report["next_action"] = (
            "Fix every reported error and run validate_context_data again."
            if errors else
            "The collection is ready for InteractiveSession to load after this Agent turn."
        )
        return report

    @classmethod
    def format_result_dict(cls, result: dict[str, Any]) -> str:
        """Format complete, actionable diagnostics for the data-preparation Agent."""
        if not result.get("valid"):
            lines = [f"INVALID context.data: {result['path']}", "Required repairs:"]
            for index, repair in enumerate(result.get("repairs", []), 1):
                lines.extend((
                    f"{index}. {repair['error']}",
                    f"   Action: {repair['action']}",
                ))
            for warning in result.get("warnings", []):
                lines.append(f"Warning: {warning}")
            lines.append(result["next_action"])
            return "\n".join(lines)
        lines = [f"VALID context.data: {result['path']}"]
        for name, variable in result.get("variables", {}).items():
            lines.append(
                f"- variable {name}: shape={variable['shape']}, dtype={variable['dtype']}, "
                f"axes={variable['axes']}"
            )
        for name, axis in result.get("axes", {}).items():
            lines.append(
                f"- axis {name}: size={axis['size']}, dtype={axis['dtype']}, "
                f"storage={axis['storage']}"
            )
        if result.get("num_nodes") is not None:
            lines.append(f"- num_nodes={result['num_nodes']}")
        for warning in result.get("warnings", []):
            lines.append(f"Warning: {warning}")
        lines.append(result["next_action"])
        return "\n".join(lines)
