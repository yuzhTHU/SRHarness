# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Execute Python with only the active conversation workspace mounted writable."""

from __future__ import annotations

import json
from typing import Any, Dict

from ..runtime import get_sandbox_runner
from .base_tool import BaseTool, ToolMetadata
from .code_executor import CodeExecutorTool


@BaseTool.register("workspace_code_executor")
class WorkspaceCodeExecutorTool(CodeExecutorTool):
    """Implementation of the workspace code executor tool."""

    metadata = ToolMetadata(name="workspace_code_executor")

    def execute(
        self,
        program: str,
        timeout_seconds: int = CodeExecutorTool.DEFAULT_TIMEOUT_SECONDS,
        memory_limit_mb: int = CodeExecutorTool.DEFAULT_MEMORY_LIMIT_MB,
        output_limit_bytes: int = CodeExecutorTool.DEFAULT_OUTPUT_LIMIT_BYTES,
    ) -> Dict[str, Any]:
        """Execute Python code in the workspace directory with file access.
        1) Code runs with cwd set to the workspace directory.
        2) Use `open("filename")` or `pandas.read_csv("filename")` to read workspace files.
        3) Use `open("output.csv", "w")` to write results back to the workspace.
        4) All file paths must be within the workspace. Absolute paths outside
           workspace are forbidden.
        5) Installed scientific Python libraries are available. Networking and
           child-process creation are unavailable.

        Args:
            program: Python code string to execute.
            timeout_seconds: Wall-clock timeout in seconds. The effective value is capped.
            memory_limit_mb: Address-space memory limit in MB. The effective value is capped.
            output_limit_bytes: Limit on the amount of output (in bytes) that can be produced.
        """
        timeout_seconds = self.bounded_int(
            timeout_seconds, self.DEFAULT_TIMEOUT_SECONDS, 1, self.MAX_TIMEOUT_SECONDS
        )
        memory_limit_mb = self.bounded_int(
            memory_limit_mb, self.DEFAULT_MEMORY_LIMIT_MB, 64, self.MAX_MEMORY_LIMIT_MB
        )
        output_limit_bytes = self.bounded_int(
            output_limit_bytes, self.DEFAULT_OUTPUT_LIMIT_BYTES, 1024, self.MAX_OUTPUT_LIMIT_BYTES
        )
        stdin_text = json.dumps(self.serialization(self.context.data), ensure_ascii=False)
        workspace_manager = getattr(self.context.args, "workspace_manager", None)
        result = get_sandbox_runner().run(
            self.extract_code(program),
            stdin=stdin_text,
            workspace=workspace_manager or self.context.workspace,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
            output_limit_bytes=output_limit_bytes,
            interruption_event=getattr(self, "cancel_event", None),
        )
        if result.exit_code != 0:
            raise RuntimeError(result.stderr or f"Sandboxed program exited with {result.exit_code}")
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": result.duration_seconds,
            "cpu_seconds": result.cpu_seconds,
            "peak_memory_bytes": result.peak_memory_bytes,
            "timeout_seconds": timeout_seconds,
            "memory_limit_mb": memory_limit_mb,
            "output_limit_bytes": output_limit_bytes,
        }
