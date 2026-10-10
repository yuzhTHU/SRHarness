# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Persistent workspace management for interactive SRHarness sessions."""

from __future__ import annotations

import logging
import os
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from ..utils import log_exception

_logger = logging.getLogger(__name__)


class Workspace:
    """Manage a writable workspace with explicitly registered read-only inputs."""

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        """Create or open a workspace.

        Args:
            path: Existing or new workspace root. The caller owns the directory and
                decides where it is created and when it is removed.
        """
        self._path = Path(path).resolve()
        self._path.mkdir(parents=True, exist_ok=True)
        self._mount_map: dict[Path, Path] = {}
        self._readonly_mounts: set[Path] = set()
        self._lock_rules: dict[Path, bool] = {}
        _logger.info("Initialized workspace at %s", self._path)

    @property
    def path(self) -> Path:
        """Return the workspace root."""
        return self._path

    @property
    def mount_map(self) -> dict[Path, Path]:
        """Return logical workspace mount points mapped to their host sources."""
        return dict(self._mount_map)

    def mount_mode(self, path: str | Path) -> Literal["read-only", "read-write"] | None:
        """Return the registered mount mode containing a logical workspace path.

        Args:
            path: Absolute or workspace-relative logical path.

        Returns:
            ``"read-only"`` or ``"read-write"`` for a mounted path, otherwise
            ``None``.
        """
        logical = Path(path)
        if not logical.is_absolute():
            logical = self._path / logical
        try:
            logical.relative_to(self._path)
        except ValueError:
            raise ValueError("Mount queries must refer to a logical workspace path") from None
        for root in (logical, *logical.parents):
            if root in self._mount_map:
                return "read-only" if root in self._readonly_mounts else "read-write"
            if root == self._path:
                break
        return None

    @property
    def lock_rules(self) -> dict[str, bool]:
        """Return explicit lock overrides keyed by workspace-relative paths."""
        return {
            str(path.relative_to(self._path)): locked
            for path, locked in self._lock_rules.items()
        }

    @property
    def has_locks(self) -> bool:
        """Return whether any path is effectively protected by a lock rule."""
        return any(self._lock_rules.values())

    def is_locked(self, path: str | Path) -> bool:
        """Return the effective lock state after applying the nearest override."""
        logical = Path(path)
        if not logical.is_absolute():
            logical = self._path / logical
        try:
            logical.relative_to(self._path)
        except ValueError:
            raise ValueError("Lock queries must refer to a logical workspace path") from None
        for candidate in (logical, *logical.parents):
            if candidate == self._path.parent:
                break
            if candidate in self._lock_rules:
                return self._lock_rules[candidate]
            if candidate == self._path:
                break
        return False

    def contains_locked_paths(self, path: str | Path) -> bool:
        """Return whether a path is or contains an explicitly locked subtree."""
        logical = Path(path)
        if not logical.is_absolute():
            logical = self._path / logical
        try:
            logical.relative_to(self._path)
        except ValueError:
            raise ValueError("Lock queries must refer to a logical workspace path") from None
        return self.is_locked(logical) or any(
            locked and logical in rule.parents
            for rule, locked in self._lock_rules.items()
        )

    def set_locked(self, relative_path: str, locked: bool) -> Path:
        """Set an explicit lock rule on a workspace file or directory tree.

        Args:
            relative_path: Workspace-relative file or directory path.
            locked: Lock the path when true or explicitly unlock it when false.

        Returns:
            The affected workspace path.

        Raises:
            ValueError: If the path is invalid, missing, or belongs to a mounted input.
        """
        path = self.resolve(relative_path)
        if path is None or not path.exists():
            raise ValueError(f"File or directory not found: {relative_path}")
        logical = self._path / Path(relative_path)
        if self.mount_mode(logical) is not None:
            raise ValueError("Mounted inputs cannot be locked or unlocked")
        descendants = [rule for rule in self._lock_rules if logical in rule.parents]
        for rule in descendants:
            del self._lock_rules[rule]
        inherited = self.is_locked(logical.parent)
        if locked == inherited:
            self._lock_rules.pop(logical, None)
        else:
            self._lock_rules[logical] = locked
        items = [path]
        if path.is_dir():
            items.extend(item for item in path.rglob("*") if not item.is_symlink())
        if locked:
            items.reverse()
        for item in items:
            mode = stat.S_IMODE(item.stat().st_mode)
            item_locked = self.is_locked(item)
            updated = mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            if not item_locked:
                updated |= stat.S_IWUSR
            item.chmod(updated)
        return path

    def load_lock_rules(self, rules: dict[str, bool]) -> None:
        """Replace lock overrides from persisted workspace state.

        Args:
            rules: Workspace-relative paths mapped to explicit lock states.
        """
        parsed: dict[Path, bool] = {}
        for relative_path, locked in rules.items():
            if not isinstance(relative_path, str) or not isinstance(locked, bool):
                raise TypeError("Workspace lock rules must map path strings to booleans")
            requested = Path(relative_path)
            if requested.is_absolute() or not requested.parts or ".." in requested.parts:
                raise ValueError(f"Invalid workspace lock path: {relative_path!r}")
            logical = self._path / requested
            if not logical.exists() or logical.is_symlink() or self.mount_mode(logical) is not None:
                raise ValueError(f"Workspace lock path is unavailable: {relative_path!r}")
            parsed[logical] = locked
        self._lock_rules = parsed
        items = [self._path, *(item for item in self._path.rglob("*") if not item.is_symlink())]
        for item in reversed(items):
            mode = stat.S_IMODE(item.stat().st_mode)
            updated = mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            if not self.is_locked(item):
                updated |= stat.S_IWUSR
            item.chmod(updated)

    def resolve(
        self,
        relative_path: str,
        *,
        access: Literal["read", "write", "create", "remove"] = "read",
    ) -> Path | None:
        """Resolve a safe workspace-relative path.

        Args:
            relative_path: Relative path supplied by the caller.
            access: Intended operation. ``write`` changes existing content, ``create``
                adds a new entry, and ``remove`` deletes, moves, or replaces an
                existing entry.

        Returns:
            The resolved path, or ``None`` when the requested access is invalid.
        """
        if access not in {"read", "write", "create", "remove"}:
            raise ValueError(f"Unknown workspace access intent: {access}")
        if not relative_path:
            return self._path
        if os.path.isabs(relative_path):
            return None
        if len(relative_path) >= 2 and relative_path[0].isalpha() and relative_path[1] == ":":
            return None
        requested = Path(relative_path)
        if ".." in requested.parts:
            return None
        logical = self._path / requested
        if requested.parts:
            mount = self._path / requested.parts[0]
            if source := self._mount_map.get(mount):
                if access != "read" and mount in self._readonly_mounts:
                    return None
                if access == "remove" and len(requested.parts) == 1:
                    return None
                if source.is_file() and len(requested.parts) > 1:
                    return None
                candidate = (
                    (source / Path(*requested.parts[1:])).resolve()
                    if source.is_dir()
                    else source
                )
                try:
                    candidate.relative_to(source if source.is_dir() else source.parent)
                except ValueError:
                    return None
                return candidate
        candidate = logical.resolve()
        try:
            candidate.relative_to(self._path.resolve())
        except ValueError:
            return None
        if access != "read":
            if access == "create":
                probe = candidate.parent
                while not probe.exists() and probe != self._path:
                    probe = probe.parent
                logical_probe = self._path / probe.relative_to(self._path)
                if probe.exists() and self.is_locked(logical_probe):
                    return None
                return candidate
            if candidate.exists() and self.is_locked(logical):
                return None
            if access == "remove" and self.contains_locked_paths(logical):
                return None
            if not candidate.exists():
                probe = candidate.parent
                while not probe.exists() and probe != self._path:
                    probe = probe.parent
            else:
                probe = candidate
            logical_probe = self._path / probe.relative_to(self._path)
            if probe.exists() and self.is_locked(logical_probe):
                return None
        return candidate

    def mount(self, source: str | Path, *, readonly: bool = True) -> Path:
        """Expose a host file or directory through a registered link.

        Args:
            source: Existing host file or directory.
            readonly: Whether workspace operations and sandboxed code may modify it.

        Returns:
            The logical link inside the workspace.
        """
        source = Path(source).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Workspace input does not exist: {source}")
        destination = self._path / source.name
        if destination.is_symlink() and destination.resolve() == source:
            self._mount_map[destination] = source
            if readonly:
                self._readonly_mounts.add(destination)
            else:
                self._readonly_mounts.discard(destination)
            return destination
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Workspace input name {destination.name!r} is already in use; "
                "rename one of the conflicting inputs."
            )
        try:
            os.symlink(source, destination, target_is_directory=source.is_dir())
            self._mount_map[destination] = source
            if readonly:
                self._readonly_mounts.add(destination)
        except Exception as exc:
            _logger.error(
                "Failed to add %s to workspace: %s",
                source,
                log_exception(exc, with_traceback=False),
            )
            raise
        return destination

    def iter_files(self) -> Iterator[tuple[Path, Path]]:
        """Yield logical and resolved paths for every workspace file."""
        mounted = set(self._mount_map)
        for item in self._path.rglob("*"):
            if any(parent in mounted for parent in (item, *item.parents)):
                continue
            if item.is_symlink():
                continue
            if item.is_file():
                yield item.relative_to(self._path), item
        for logical, source in self._mount_map.items():
            prefix = Path(logical.name)
            if source.is_file():
                yield prefix, source
            else:
                for item in source.rglob("*"):
                    if item.is_file():
                        yield prefix / item.relative_to(source), item
