"""Small Linux Landlock and seccomp bindings for sandbox workers."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
from pathlib import Path


class LandlockUnavailableError(RuntimeError):
    """Raised when the kernel cannot enforce the requested Landlock policy."""


_LIBC = ctypes.CDLL(None, use_errno=True)
_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2
_SECCOMP_RET_KILL_PROCESS = 0x80000000
_AUDIT_ARCHITECTURES = {
    "x86_64": 0xC000003E,
    "aarch64": 0xC00000B7,
}

_FS_EXECUTE = 1 << 0
_FS_WRITE_FILE = 1 << 1
_FS_READ_FILE = 1 << 2
_FS_READ_DIR = 1 << 3
_FS_REMOVE_DIR = 1 << 4
_FS_REMOVE_FILE = 1 << 5
_FS_MAKE_CHAR = 1 << 6
_FS_MAKE_DIR = 1 << 7
_FS_MAKE_REG = 1 << 8
_FS_MAKE_SOCK = 1 << 9
_FS_MAKE_FIFO = 1 << 10
_FS_MAKE_BLOCK = 1 << 11
_FS_MAKE_SYM = 1 << 12
_FS_REFER = 1 << 13
_FS_TRUNCATE = 1 << 14
_FS_IOCTL_DEV = 1 << 15


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ushort),
        ("filters", ctypes.POINTER(_SockFilter)),
    ]


def _syscall(number: int, *args) -> int:
    result = int(_LIBC.syscall(number, *args))
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def landlock_abi() -> int:
    """Return the supported Landlock ABI version.

    Returns:
        Positive ABI version.
    """
    if platform.system() != "Linux":
        raise LandlockUnavailableError("Landlock is available only on Linux")
    try:
        return _syscall(
            _SYS_LANDLOCK_CREATE_RULESET,
            ctypes.c_void_p(),
            ctypes.c_size_t(0),
            ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION),
        )
    except OSError as exc:
        raise LandlockUnavailableError(f"Landlock is unavailable: {exc}") from exc


def _handled_rights(abi: int) -> int:
    rights = (
        _FS_EXECUTE
        | _FS_WRITE_FILE
        | _FS_READ_FILE
        | _FS_READ_DIR
        | _FS_REMOVE_DIR
        | _FS_REMOVE_FILE
        | _FS_MAKE_CHAR
        | _FS_MAKE_DIR
        | _FS_MAKE_REG
        | _FS_MAKE_SOCK
        | _FS_MAKE_FIFO
        | _FS_MAKE_BLOCK
        | _FS_MAKE_SYM
    )
    if abi >= 2:
        rights |= _FS_REFER
    if abi >= 3:
        rights |= _FS_TRUNCATE
    if abi >= 5:
        rights |= _FS_IOCTL_DEV
    return rights


def _path_rights(
    path: Path,
    handled: int,
    *,
    write: bool,
    execute: bool,
    remove: bool,
) -> int:
    rights = _FS_READ_FILE
    if path.is_dir():
        rights |= _FS_READ_DIR
    if execute:
        rights |= _FS_EXECUTE
    if write:
        rights |= (
            _FS_WRITE_FILE
            | _FS_MAKE_DIR
            | _FS_MAKE_REG
            | _FS_MAKE_SOCK
            | _FS_MAKE_FIFO
            | _FS_MAKE_SYM
            | _FS_TRUNCATE
        )
        if remove:
            rights |= _FS_REMOVE_DIR | _FS_REMOVE_FILE | _FS_REFER
    return rights & handled


def restrict_filesystem(
    *,
    readable: list[str | Path],
    writable: list[str | Path],
    nonremovable_writable: list[str | Path],
    executable: list[str | Path],
) -> int:
    """Restrict the current process and descendants to explicit path trees.

    Args:
        readable: Read-only files and directory trees.
        writable: Read-write files and directory trees.
        nonremovable_writable: Read-write trees in which removal, rename, and
            cross-directory linking are denied.
        executable: Trees from which binaries or libraries may be executed.

    Returns:
        Enforced Landlock ABI version.
    """
    abi = landlock_abi()
    handled = _handled_rights(abi)
    ruleset_attr = _RulesetAttr(handled_access_fs=handled)
    ruleset_fd = _syscall(
        _SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        ctypes.c_uint(0),
    )
    try:
        policies: dict[Path, tuple[bool, bool, bool]] = {}
        for raw_path, write, execute_path, remove in [
            *((path, False, False, False) for path in readable),
            *((path, True, False, True) for path in writable),
            *((path, True, False, False) for path in nonremovable_writable),
            *((path, False, True, False) for path in executable),
        ]:
            path = Path(raw_path).resolve()
            if not path.exists():
                continue
            previous = policies.get(path, (False, False, False))
            policies[path] = (
                previous[0] or write,
                previous[1] or execute_path,
                previous[2] or remove,
            )
        for path, (write, execute_path, remove) in policies.items():
            descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                attr = _PathBeneathAttr(
                    allowed_access=_path_rights(
                        path,
                        handled,
                        write=write,
                        execute=execute_path,
                        remove=remove,
                    ),
                    parent_fd=descriptor,
                    reserved=0,
                )
                _syscall(
                    _SYS_LANDLOCK_ADD_RULE,
                    ruleset_fd,
                    _LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(attr),
                    ctypes.c_uint(0),
                )
            finally:
                os.close(descriptor)
        if _LIBC.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        _syscall(_SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, ctypes.c_uint(0))
    except OSError as exc:
        raise LandlockUnavailableError(f"Could not enforce Landlock policy: {exc}") from exc
    finally:
        os.close(ruleset_fd)
    return abi


def _number_set(values: str) -> frozenset[int]:
    return frozenset(map(int, values.split()))


_BLOCKED_SYSCALLS = {
    "x86_64": _number_set(
        "29 30 31 34 41 53 56 57 58 59 62 64 65 66 67 68 69 70 "
        "71 76 77 90 91 92 93 94 101 103 132 133 141 142 144 155 159 161 "
        "163 164 165 166 167 168 169 170 171 172 173 175 176 179 188 189 190 197 "
        "198 199 200 203 220 227 234 246 248 249 250 256 259 260 261 268 272 279 "
        "280 298 300 301 302 303 304 305 308 310 311 313 321 322 323 424 425 426 "
        "427 434 435 438 440"
    ),
    "aarch64": _number_set(
        "5 6 7 14 15 16 33 39 40 41 45 46 51 52 53 54 55 88 "
        "89 97 104 105 106 117 118 119 122 129 130 131 142 161 162 170 171 186 "
        "187 188 189 190 191 192 193 194 195 196 197 198 199 217 218 219 220 221 "
        "224 225 238 239 241 261 262 263 265 266 268 270 271 273 280 281 282 294 "
        "424 425 426 427 434 435 438 440"
    ),
}


def restrict_system_calls() -> None:
    """Deny networking, child processes, and host-management syscalls."""
    architecture = platform.machine().lower()
    if architecture in {"amd64"}:
        architecture = "x86_64"
    if architecture in {"arm64"}:
        architecture = "aarch64"
    try:
        blocked = sorted(_BLOCKED_SYSCALLS[architecture])
    except KeyError as exc:
        raise LandlockUnavailableError(
            f"No seccomp syscall table for architecture {architecture!r}"
        ) from exc
    # classic BPF: validate seccomp_data.arch, load seccomp_data.nr, return
    # EACCES for blocked calls, and allow everything else.
    instructions = [
        _SockFilter(0x20, 0, 0, 4),
        _SockFilter(0x15, 1, 0, _AUDIT_ARCHITECTURES[architecture]),
        _SockFilter(0x06, 0, 0, _SECCOMP_RET_KILL_PROCESS),
        _SockFilter(0x20, 0, 0, 0),
    ]
    denied = 0x00050000 | errno.EACCES
    for syscall_number in blocked:
        instructions.append(_SockFilter(0x15, 0, 1, syscall_number))
        instructions.append(_SockFilter(0x06, 0, 0, denied))
    instructions.append(_SockFilter(0x06, 0, 0, 0x7FFF0000))
    array_type = _SockFilter * len(instructions)
    filters = array_type(*instructions)
    program = _SockFprog(len(instructions), filters)
    if _LIBC.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        code = ctypes.get_errno()
        raise LandlockUnavailableError(f"Could not set no_new_privs: {os.strerror(code)}")
    if _LIBC.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.byref(program)) != 0:
        code = ctypes.get_errno()
        raise LandlockUnavailableError(f"Could not install seccomp filter: {os.strerror(code)}")
