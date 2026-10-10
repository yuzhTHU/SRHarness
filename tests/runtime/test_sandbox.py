"""Operating-system sandbox boundary tests."""

import pytest

from sr_harness.runtime.sandbox import SandboxExecutionError, SandboxRunner


def test_worker_output_protocol_rejects_symbolic_links():
    runner = SandboxRunner()
    program = """
import inspect
import os

frame = inspect.currentframe()
while frame is not None and "output_root" not in frame.f_locals:
    frame = frame.f_back
output_root = frame.f_locals["output_root"]
os.symlink("result.json", output_root / "arrays.json")
"""
    with pytest.raises(SandboxExecutionError, match="not a regular file"):
        runner.run(program)


def test_worker_cannot_start_child_process():
    runner = SandboxRunner()
    program = """
import subprocess

try:
    subprocess.run(["/bin/true"], check=True)
except PermissionError:
    print("blocked")
"""
    result = runner.run(program)
    assert result.exit_code == 0
    assert result.stdout.strip() == "blocked"
    assert result.duration_seconds >= 0
    assert result.cpu_seconds > 0
    assert result.peak_memory_bytes > 0
