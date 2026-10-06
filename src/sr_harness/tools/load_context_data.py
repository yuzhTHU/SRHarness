"""Validate and load manifest-backed NumPy variables from the workspace."""
from __future__ import annotations

from typing import Any

from ..core import ContextDataStore
from .base_tool import BaseTool, ToolMetadata


@BaseTool.register("load_context_data")
class LoadContextDataTool(BaseTool):
    """Implementation of the context-data validation and loading boundary."""

    metadata = ToolMetadata(name="load_context_data")

    def execute(self, path: str = "context.data") -> dict[str, Any]:
        """Validate a context-data manifest and load it when valid.

        The directory must contain ``manifest.json`` and a flat collection of
        NPY files. This tool reports every detected manifest, filename, dtype,
        dimension, and axis-length problem in one call. A valid collection is
        atomically published as ``context.data`` for subsequent tools and agents.

        Args:
            path: Workspace-relative directory containing ``manifest.json``.

        Returns:
            Validation diagnostics and the committed context revision when valid.
        """
        workspace = self.context.workspace
        if workspace is None or (directory := workspace.resolve(path)) is None:
            raise ValueError("path must identify a directory inside the workspace")
        store = ContextDataStore(directory)
        report = store.inspect()
        if not report["valid"]:
            return {"data_committed": False, **report}
        change = self.context.commit_context_data(store.load())
        return {"data_committed": True, **report, **change}

    @classmethod
    def format_result_dict(cls, result: dict[str, Any]) -> str:
        """Format validation diagnostics for the language model.

        Args:
            result: Structured validation and commit result.

        Returns:
            Concise diagnostics or a loaded-variable summary.
        """
        if not result.get("valid"):
            lines = ["context.data is invalid:"]
            lines.extend(f"- {error}" for error in result.get("errors", []))
            lines.extend(f"- warning: {warning}" for warning in result.get("warnings", []))
            return "\n".join(lines)
        variables = ", ".join(result.get("variables", {}))
        axes = ", ".join(result.get("axes", {}))
        warnings = result.get("warnings", [])
        suffix = "" if not warnings else " Warnings: " + "; ".join(warnings)
        return f"Loaded context.data variables [{variables}] with axes [{axes}].{suffix}"
