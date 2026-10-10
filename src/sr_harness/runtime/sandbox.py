"""Operating-system sandbox used for untrusted Python execution."""

from __future__ import annotations

import io
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np

from .landlock import LandlockUnavailableError, landlock_abi
from .workspace import Workspace


def _fully_unlocked_workspace_subtrees(workspace: Workspace) -> list[Path]:
    """Return maximal directory trees containing no effectively locked entries.

    Landlock removal rights belong to directories and therefore cannot distinguish
    locked and unlocked siblings.  A directory is safe to grant recursively only
    when the directory itself and every ordinary descendant are unlocked.  Mounted
    inputs are governed by their own rules and are excluded from this calculation.
    """
    root = workspace.path
    mount_roots = set(workspace.mount_map)
    directories: list[Path] = []
    child_directories: dict[Path, list[Path]] = {}
    child_files: dict[Path, list[Path]] = {}
    for current_name, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_name)
        directory_names[:] = [
            name
            for name in directory_names
            if (current / name) not in mount_roots and not (current / name).is_symlink()
        ]
        directories.append(current)
        child_directories[current] = [current / name for name in directory_names]
        child_files[current] = [
            current / name
            for name in file_names
            if (current / name) not in mount_roots and not (current / name).is_symlink()
        ]

    fully_unlocked: dict[Path, bool] = {}
    for directory in reversed(directories):
        fully_unlocked[directory] = (
            not workspace.is_locked(directory)
            and all(not workspace.is_locked(path) for path in child_files[directory])
            and all(fully_unlocked[path] for path in child_directories[directory])
        )
    return [
        directory
        for directory in directories
        if fully_unlocked[directory]
        and (directory == root or not fully_unlocked.get(directory.parent, False))
    ]


class SandboxUnavailableError(RuntimeError):
    """Raised when the required operating-system sandbox is unavailable."""


class SandboxExecutionError(RuntimeError):
    """Raised when an isolated worker cannot complete its request."""


@dataclass(frozen=True)
class SandboxResult:
    """Observable result of executing a Python program in the sandbox.

    Args:
        exit_code: Program exit status; zero denotes successful execution.
        stdout: Captured standard output.
        stderr: Captured standard error.
        duration_seconds: Wall-clock execution duration in seconds.
        cpu_seconds: User and system CPU time consumed by the worker.
        peak_memory_bytes: Peak resident set size reported by the worker.
    """

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    cpu_seconds: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class _ProtocolResult:
    """Private result used by structured sandbox adapters."""

    payload: dict[str, Any]
    arrays: dict[str, dict[str, np.ndarray]]
    duration_seconds: float
    cpu_seconds: float
    peak_memory_bytes: int


def _write_array_groups(root: Path, groups: dict[str, dict[str, Any]]) -> None:
    """Serialize named array groups without pickle support."""
    manifest: dict[str, dict[str, str]] = {}
    array_root = root / "arrays"
    array_root.mkdir()
    index = 0
    for group_name, values in groups.items():
        manifest[group_name] = {}
        for name, value in values.items():
            filename = f"array-{index}.npy"
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise TypeError(f"Sandbox array {name!r} must not use object dtype")
            np.save(array_root / filename, array, allow_pickle=False)
            manifest[group_name][str(name)] = filename
            index += 1
    (root / "arrays.json").write_text(json.dumps(manifest), encoding="utf-8")


def _read_array_groups(root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load worker-produced arrays with object deserialization disabled."""
    manifest_file = root / "arrays.json"
    if not manifest_file.exists():
        return {}
    manifest = json.loads(_read_regular_bytes(manifest_file).decode("utf-8"))
    if not isinstance(manifest, dict):
        raise SandboxExecutionError("Sandbox array manifest must be a dictionary")
    groups: dict[str, dict[str, np.ndarray]] = {}
    for group_name, values in manifest.items():
        if not isinstance(group_name, str) or not isinstance(values, dict):
            raise SandboxExecutionError("Sandbox array manifest is malformed")
        group: dict[str, np.ndarray] = {}
        for name, filename in values.items():
            if not isinstance(name, str) or not isinstance(filename, str):
                raise SandboxExecutionError("Sandbox array manifest is malformed")
            path = root / "arrays" / filename
            if path.parent != root / "arrays":
                raise SandboxExecutionError("Sandbox returned an invalid array path")
            with io.BytesIO(_read_regular_bytes(path)) as stream:
                group[name] = np.load(stream, allow_pickle=False)
        groups[group_name] = group
    return groups


def _read_regular_bytes(path: Path, limit: int | None = None) -> bytes:
    """Read an ordinary file without following worker-created symlinks."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise SandboxExecutionError(f"Sandbox result file is missing: {path.name}") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise SandboxExecutionError(f"Sandbox result path is not a regular file: {path.name}")
    if limit is not None and metadata.st_size > limit:
        raise SandboxExecutionError(f"Sandbox result file exceeded its size limit: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read() if limit is None else stream.read(limit + 1)
    finally:
        os.close(descriptor)


class _SandboxProcessRunner:
    """Implement the private file protocol spoken by the sandbox worker."""

    RESULT_METADATA_LIMIT = 16 * 1024 * 1024

    def __init__(self) -> None:
        try:
            self.landlock_abi = landlock_abi()
        except LandlockUnavailableError as exc:
            raise SandboxUnavailableError(
                "Linux Landlock is unavailable; refusing to execute untrusted code."
            ) from exc
        if self.landlock_abi < 3:
            raise SandboxUnavailableError(
                "Linux Landlock ABI 3 or newer is required; refusing to execute "
                "untrusted code without file-truncation protection."
            )

    @staticmethod
    def _runtime_source_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _safe_environment(*, root: str, temporary: str) -> dict[str, str]:
        return {
            "HOME": "/tmp/home",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": f"{temporary}/matplotlib",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PATH": f"{root}/bin:/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": root,
            "TMPDIR": temporary,
            "VECLIB_MAXIMUM_THREADS": "1",
        }

    def _worker_command(
        self,
        *,
        input_directory: Path,
        output_directory: Path,
        temporary_directory: Path,
        workspace: Path | None,
        readonly_mounts: dict[str, Path],
        writable_mounts: dict[str, Path],
        protect_workspace_entries: bool,
        removable_workspace_subtrees: list[Path],
    ) -> tuple[list[str], dict[str, str], Path]:
        """Build a worker command whose first action installs the sandbox policy."""
        runtime = Path(sys.prefix).resolve()
        source = self._runtime_source_root().resolve()
        readable = [input_directory, runtime, source]
        executable = [runtime]
        for system_path in (Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64")):
            if system_path.exists():
                readable.append(system_path)
                executable.append(system_path)
        writable = [output_directory, temporary_directory]
        nonremovable_writable: list[Path] = []
        if workspace is not None:
            working_directory = workspace.resolve()
            if not working_directory.is_dir():
                raise FileNotFoundError(f"Workspace does not exist: {working_directory}")
            readable.append(working_directory)
            (nonremovable_writable if protect_workspace_entries else writable).append(
                working_directory
            )
            writable.extend(removable_workspace_subtrees)
            for name, source_path in readonly_mounts.items():
                if not name or Path(name).name != name:
                    raise ValueError(f"Invalid read-only mount name: {name!r}")
                readable.append(source_path.resolve())
            for name, source_path in writable_mounts.items():
                if not name or Path(name).name != name:
                    raise ValueError(f"Invalid writable mount name: {name!r}")
                writable.append(source_path.resolve())
        else:
            working_directory = temporary_directory
        environment = self._safe_environment(
            root=str(runtime),
            temporary=str(temporary_directory),
        )
        environment.update(
            {
                "SRH_SANDBOX_INPUT": str(input_directory),
                "SRH_SANDBOX_OUTPUT": str(output_directory),
                "SRH_SANDBOX_READABLE": json.dumps([str(path) for path in readable]),
                "SRH_SANDBOX_WRITABLE": json.dumps([str(path) for path in writable]),
                "SRH_SANDBOX_NONREMOVABLE_WRITABLE": json.dumps(
                    [str(path) for path in nonremovable_writable]
                ),
                "SRH_SANDBOX_EXECUTABLE": json.dumps([str(path) for path in executable]),
                "SRH_SANDBOX_CWD": str(working_directory),
            }
        )
        return (
            [
                str(runtime / "bin" / "python"),
                "-I",
                str(source / "sr_harness/runtime/sandbox_worker.py"),
            ],
            environment,
            working_directory,
        )

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

    def run_program(
        self,
        code: str,
        *,
        stdin: str,
        workspace: Workspace | str | Path | None,
        timeout_seconds: int,
        memory_limit_mb: int,
        output_limit_bytes: int,
        interruption_event: Event | None,
    ) -> SandboxResult:
        """Adapt plain Python execution to the private worker protocol."""
        result = self.run_protocol(
            operation="code",
            request={"program": code, "stdin_text": stdin},
            workspace=workspace,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
            output_limit_bytes=output_limit_bytes,
            interruption_event=interruption_event,
        )
        return SandboxResult(
            exit_code=int(result.payload.get("exit_code", 0)),
            stdout=str(result.payload.get("stdout", "")),
            stderr=str(result.payload.get("stderr", "")),
            duration_seconds=result.duration_seconds,
            cpu_seconds=result.cpu_seconds,
            peak_memory_bytes=result.peak_memory_bytes,
        )

    def run_protocol(
        self,
        *,
        operation: str,
        request: dict[str, Any],
        arrays: dict[str, dict[str, Any]] | None = None,
        workspace: Workspace | str | Path | None = None,
        timeout_seconds: int = 30,
        memory_limit_mb: int = 1024,
        output_limit_bytes: int = 64 * 1024,
        interruption_event: Event | None = None,
    ) -> _ProtocolResult:
        """Run one private protocol request in a confined worker.

        Args:
            operation: Worker operation identifier.
            request: JSON-compatible operation parameters.
            arrays: Named input array groups.
            workspace: Managed workspace or plain workspace path. Passing a managed
                workspace also applies its mounted-input and lock policies.
            timeout_seconds: Wall-clock timeout.
            memory_limit_mb: Additional address-space allowance in MB.
            output_limit_bytes: Maximum captured Python text output.
            interruption_event: Optional signal requesting prompt interruption.

        Returns:
            Private structured payload and safely deserialized array results.
        """
        with (
            tempfile.TemporaryDirectory(prefix="srh-sandbox-input-") as input_name,
            tempfile.TemporaryDirectory(prefix="srh-sandbox-output-") as output_name,
            tempfile.TemporaryDirectory(prefix="srh-sandbox-tmp-") as temporary_name,
        ):
            input_directory = Path(input_name)
            output_directory = Path(output_name)
            temporary_directory = Path(temporary_name)
            (temporary_directory / "home").mkdir()
            (temporary_directory / "matplotlib").mkdir()
            complete_request = {
                **request,
                "operation": operation,
                "memory_limit_mb": int(memory_limit_mb),
                "output_limit_bytes": int(output_limit_bytes),
                "timeout_seconds": int(timeout_seconds),
            }
            (input_directory / "request.json").write_text(
                json.dumps(complete_request, ensure_ascii=False), encoding="utf-8"
            )
            _write_array_groups(input_directory, arrays or {})
            workspace_manager = workspace if isinstance(workspace, Workspace) else None
            resolved_workspace = (
                workspace_manager.path
                if workspace_manager is not None
                else Path(workspace) if workspace is not None else None
            )
            resolved_mounts = {
                logical.name: source
                for logical, source in (
                    workspace_manager.mount_map.items()
                    if workspace_manager is not None
                    else ()
                )
            }
            readonly_mounts = {
                name: source
                for name, source in resolved_mounts.items()
                if workspace_manager is not None
                and workspace_manager.mount_mode(workspace_manager.path / name) == "read-only"
            }
            writable_mounts = {
                name: source
                for name, source in resolved_mounts.items()
                if name not in readonly_mounts
            }
            command, environment, working_directory = self._worker_command(
                input_directory=input_directory,
                output_directory=output_directory,
                temporary_directory=temporary_directory,
                workspace=resolved_workspace,
                readonly_mounts=readonly_mounts,
                writable_mounts=writable_mounts,
                protect_workspace_entries=bool(
                    workspace_manager is not None and workspace_manager.has_locks
                ),
                removable_workspace_subtrees=(
                    _fully_unlocked_workspace_subtrees(workspace_manager)
                    if workspace_manager is not None and workspace_manager.has_locks
                    else []
                ),
            )
            start = time.monotonic()
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
                env=environment,
                cwd=working_directory,
            )
            deadline = start + timeout_seconds
            try:
                while process.poll() is None:
                    if interruption_event is not None and interruption_event.is_set():
                        self._terminate(process)
                        raise InterruptedError("Sandbox execution was stopped by the user")
                    if time.monotonic() >= deadline:
                        self._terminate(process)
                        raise TimeoutError(
                            "Sandbox subprocess did not return result before "
                            f"timeout={timeout_seconds} seconds and has been terminated."
                        )
                    time.sleep(0.02)
                stderr = (process.stderr.read() if process.stderr is not None else b"")[
                    :output_limit_bytes
                ]
            finally:
                if process.poll() is None:
                    self._terminate(process)
            duration = time.monotonic() - start
            result_file = output_directory / "result.json"
            if not result_file.exists():
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise SandboxExecutionError(
                    f"Sandbox worker exited with code {process.returncode} without a result"
                    + (f": {detail}" if detail else "")
                )
            result = json.loads(
                _read_regular_bytes(
                    result_file,
                    self.RESULT_METADATA_LIMIT,
                ).decode("utf-8")
            )
            if not isinstance(result, dict):
                raise SandboxExecutionError("Sandbox result must be a dictionary")
            if error := result.pop("error", None):
                raise SandboxExecutionError(str(error))
            usage = result.pop("_sandbox_usage", {})
            return _ProtocolResult(
                payload=result,
                arrays=_read_array_groups(output_directory),
                duration_seconds=duration,
                cpu_seconds=float(usage.get("cpu_seconds", 0.0)),
                peak_memory_bytes=int(usage.get("peak_memory_bytes", 0)),
            )


class SandboxRunner:
    """Execute Python code with optional workspace access in the OS sandbox."""

    def __init__(self) -> None:
        self._process_runner = _SandboxProcessRunner()

    def run(
        self,
        code: str,
        *,
        stdin: str = "",
        workspace: Workspace | str | Path | None = None,
        timeout_seconds: int = 30,
        memory_limit_mb: int = 1024,
        output_limit_bytes: int = 64 * 1024,
        interruption_event: Event | None = None,
    ) -> SandboxResult:
        """Execute a Python program and capture its observable process result.

        Args:
            code: Python source code to execute.
            stdin: Text exposed as standard input.
            workspace: Optional working directory. When omitted, user code receives no
                persistent filesystem access. A managed workspace also enforces its
                mount and lock policies.
            timeout_seconds: Wall-clock timeout.
            memory_limit_mb: Address-space memory limit in MB.
            output_limit_bytes: Per-stream capture limit for standard output and error.
            interruption_event: Optional signal requesting prompt interruption.

        Returns:
            Captured process output and resource usage.
        """
        return self._process_runner.run_program(
            code,
            stdin=stdin,
            workspace=workspace,
            timeout_seconds=timeout_seconds,
            memory_limit_mb=memory_limit_mb,
            output_limit_bytes=output_limit_bytes,
            interruption_event=interruption_event,
        )


_DEFAULT_RUNNER: SandboxRunner | None = None


def get_sandbox_runner() -> SandboxRunner:
    """Return the process-wide sandbox runner."""
    global _DEFAULT_RUNNER
    if _DEFAULT_RUNNER is None:
        _DEFAULT_RUNNER = SandboxRunner()
    return _DEFAULT_RUNNER


def _run_sandbox_protocol(
    operation: str,
    request: dict[str, Any],
    *,
    arrays: dict[str, dict[str, Any]] | None = None,
    workspace: Workspace | str | Path | None = None,
    timeout_seconds: int = 30,
    memory_limit_mb: int = 1024,
    output_limit_bytes: int = 64 * 1024,
    interruption_event: Event | None = None,
) -> _ProtocolResult:
    """Run a structured internal protocol for trusted SRHarness adapters."""
    return get_sandbox_runner()._process_runner.run_protocol(
        operation=operation,
        request=request,
        arrays=arrays,
        workspace=workspace,
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
        output_limit_bytes=output_limit_bytes,
        interruption_event=interruption_event,
    )
