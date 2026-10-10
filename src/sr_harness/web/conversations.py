"""Persistent conversation registry for the interactive Web workbench."""
from __future__ import annotations

import json
import os
import stat
import tempfile
import threading
import uuid
import zipfile
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Body, HTTPException, Request, Response
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .session import InteractiveSession


CONVERSATION_COOKIE = "sr_harness_conversation"
CLIENT_COOKIE = "sr_harness_client"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationSessionProxy:
    """Resolve attribute access to the session bound to the current request."""

    def __init__(self, registry: "ConversationRegistry"):
        object.__setattr__(self, "_registry", registry)
        object.__setattr__(self, "_current", ContextVar("sr_harness_session", default=None))

    def bind(self, session: InteractiveSession):
        return self._current.set(session)

    def reset(self, token) -> None:
        self._current.reset(token)

    def current_session(self) -> InteractiveSession:
        session = self._current.get()
        if session is None:
            raise RuntimeError("No InteractiveSession is bound to the current request")
        return session

    def __getattr__(self, name: str):
        return getattr(self.current_session(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.current_session(), name, value)


class ConversationRegistry:
    """Persist conversation metadata and lazily own one session per conversation."""

    def __init__(
        self,
        workspace_dir: str | Path,
        *,
        workspace_files: list[str] | None = None,
        isolate_users: bool = False,
        initial_run_dir: str | Path | None = None,
        persist_sessions: bool = False,
        persistence_interval: float = 5.0,
    ):
        self.workspace_dir = Path(workspace_dir).expanduser().resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.workspace_dir / "conversations.json"
        self.workspaces_dir = self.workspace_dir / "workspaces"
        self.runs_dir = self.workspace_dir / "runs"
        self.workspaces_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        self.workspace_files = list(workspace_files or [])
        self.isolate_users = bool(isolate_users)
        self.persist_sessions = bool(persist_sessions)
        self.persistence_dir = (
            Path(initial_run_dir).expanduser().resolve()
            if self.persist_sessions and initial_run_dir is not None
            else None
        )
        self.persistence_interval = max(1.0, float(persistence_interval))
        self.lock = threading.RLock()
        self._persistence_stop = threading.Event()
        self._persistence_thread: threading.Thread | None = None
        self._persistence_error: Exception | None = None
        self._sessions: dict[str, InteractiveSession] = {}
        self._records = self._load_records()
        if not self._records:
            self._create_record(owner_id=None, run_dir=initial_run_dir)
        self.session_proxy = ConversationSessionProxy(self)
        if self.persist_sessions:
            self._persistence_thread = threading.Thread(
                target=self._persistence_loop,
                name="sr-harness-session-persistence",
                daemon=True,
            )
            self._persistence_thread.start()

    @property
    def default_session(self) -> InteractiveSession:
        with self.lock:
            record = next((item for item in self._records if not item["archived"]), None)
            if record is None:
                record = self._create_record(owner_id=None)
            return self._session_for(record)

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            self._write_records([])
            return []
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to load conversation registry: {exc}") from exc
        if not isinstance(payload, dict) or set(payload) != {"version", "conversations"}:
            raise ValueError("Conversation registry does not match the current schema")
        if payload["version"] != 1:
            raise ValueError("Unsupported conversation registry version")
        records = payload["conversations"]
        if not isinstance(records, list):
            raise ValueError("Conversation registry must contain a conversations list")
        required = {
            "id", "name", "owner_id", "workspace", "run_dir", "created_at",
            "updated_at", "archived",
        }
        for record in records:
            if not isinstance(record, dict) or set(record) != required:
                raise ValueError("Conversation registry contains an invalid record")
            if not all(
                isinstance(record[key], str) and record[key]
                for key in required - {"owner_id", "archived"}
            ):
                raise ValueError("Conversation registry record fields must be non-empty strings")
            if record["owner_id"] is not None and not isinstance(record["owner_id"], str):
                raise ValueError("Conversation owner_id must be a string or None")
            if not isinstance(record["archived"], bool):
                raise ValueError("Conversation archived must be a boolean")
        return records

    def _write_records(self, records: list[dict[str, Any]] | None = None) -> None:
        payload = {"version": 1, "conversations": records if records is not None else self._records}
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    def _create_record(
        self,
        *,
        owner_id: str | None,
        name: str | None = None,
        run_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        if name is None:
            name = f"Conversation {len(self._records) + 1}"
        else:
            name = name.strip()
            if not name:
                raise ValueError("Conversation name cannot be empty")
        conversation_id = uuid.uuid4().hex
        created_at = _now()
        record = {
            "id": conversation_id,
            "name": name,
            "owner_id": owner_id,
            "workspace": str(Path("workspaces") / conversation_id),
            "run_dir": str(Path(run_dir).expanduser().resolve()) if run_dir else str(Path("runs") / conversation_id),
            "created_at": created_at,
            "updated_at": created_at,
            "archived": False,
        }
        workspace = self.workspace_dir / record["workspace"]
        workspace.mkdir(parents=True, exist_ok=False)
        resolved_run_dir = self._record_path(record, "run_dir")
        resolved_run_dir.mkdir(parents=True, exist_ok=True)
        self._records.append(record)
        self._write_records()
        return record

    def _record_path(self, record: dict[str, Any], field: str) -> Path:
        path = Path(record[field])
        return path if path.is_absolute() else self.workspace_dir / path

    def _session_for(self, record: dict[str, Any]) -> InteractiveSession:
        conversation_id = record["id"]
        session = self._sessions.get(conversation_id)
        if session is None:
            workspace = self._record_path(record, "workspace")
            workspace.mkdir(parents=True, exist_ok=True)
            session = InteractiveSession(
                self._record_path(record, "run_dir").parent,
                workspace_files=self.workspace_files,
                run_dir=self._record_path(record, "run_dir"),
                workspace_path=workspace,
            )
            snapshot_path = self._session_snapshot_path(record)
            if self.persist_sessions and snapshot_path.is_file():
                try:
                    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                    session.restore_persistent_state(snapshot)
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"Unable to restore conversation {conversation_id}: {exc}"
                    ) from exc
            self._sessions[conversation_id] = session
        return session

    def _session_snapshot_path(self, record: dict[str, Any]) -> Path:
        if self.persistence_dir is not None:
            return self.persistence_dir / "sessions" / f"{record['id']}.json"
        return self._record_path(record, "run_dir") / "interactive-session.json"

    def persist(self) -> None:
        """Atomically persist every materialized InteractiveSession."""
        if not self.persist_sessions:
            return
        with self.lock:
            pairs = [
                (record, self._sessions[record["id"]])
                for record in self._records
                if record["id"] in self._sessions
            ]
        for record, session in pairs:
            snapshot_path = self._session_snapshot_path(record)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = snapshot_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(session.export_persistent_state(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(snapshot_path)
        self._persistence_error = None

    def _persistence_loop(self) -> None:
        while not self._persistence_stop.wait(self.persistence_interval):
            try:
                self.persist()
            except Exception as exc:
                self._persistence_error = exc

    def _visible(self, record: dict[str, Any], client_id: str) -> bool:
        return (
            not record["archived"]
            and (not self.isolate_users or record["owner_id"] == client_id)
        )

    def _owned(self, record: dict[str, Any], client_id: str) -> bool:
        return not self.isolate_users or record["owner_id"] == client_id

    def _raise_persistence_error(self) -> None:
        if self._persistence_error is not None:
            raise RuntimeError("InteractiveSession persistence failed") from self._persistence_error

    def resolve(self, client_id: str, selected_id: str | None) -> tuple[dict[str, Any], InteractiveSession]:
        with self.lock:
            self._raise_persistence_error()
            visible = [record for record in self._records if self._visible(record, client_id)]
            if self.isolate_users and not visible:
                unowned = next((record for record in self._records if record["owner_id"] is None), None)
                if unowned is not None:
                    unowned["owner_id"] = client_id
                    unowned["updated_at"] = _now()
                    self._write_records()
                    visible = [unowned]
            if not visible:
                visible = [self._create_record(owner_id=client_id if self.isolate_users else None)]
            record = next((item for item in visible if item["id"] == selected_id), visible[0])
            return record, self._session_for(record)

    def list(self, client_id: str, selected_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            records = [record for record in self._records if self._visible(record, client_id)]
            return {
                "isolate_users": self.isolate_users,
                "selected": selected_id,
                "conversations": [
                    {key: record[key] for key in ("id", "name", "created_at", "updated_at")}
                    for record in records
                ],
            }

    def create(self, client_id: str, name: str | None = None) -> dict[str, Any]:
        with self.lock:
            return self._create_record(owner_id=client_id if self.isolate_users else None, name=name)

    def select(self, client_id: str, conversation_id: str) -> dict[str, Any]:
        with self.lock:
            record = next(
                (item for item in self._records if item["id"] == conversation_id and self._visible(item, client_id)),
                None,
            )
            if record is None:
                raise KeyError(conversation_id)
            self._session_for(record)
            return record

    def rename(self, client_id: str, conversation_id: str, name: str) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("Conversation name cannot be empty")
        if len(name) > 120:
            raise ValueError("Conversation name cannot exceed 120 characters")
        with self.lock:
            record = self.select(client_id, conversation_id)
            record["name"] = name
            record["updated_at"] = _now()
            self._write_records()
            return record

    def archive(
        self, client_id: str, conversation_id: str, selected_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Archive a conversation and choose the conversation that remains selected."""
        with self.lock:
            record = next(
                (
                    item for item in self._records
                    if item["id"] == conversation_id and self._owned(item, client_id)
                    and not item["archived"]
                ),
                None,
            )
            if record is None:
                raise KeyError(conversation_id)
            record["archived"] = True
            record["updated_at"] = _now()
            visible = [item for item in self._records if self._visible(item, client_id)]
            if not visible:
                visible = [self._create_record(
                    owner_id=client_id if self.isolate_users else None,
                )]
            selected = next((item for item in visible if item["id"] == selected_id), visible[0])
            self._write_records()
            return record, selected

    @staticmethod
    def _write_tree(archive: zipfile.ZipFile, root: Path, prefix: str) -> None:
        if not root.exists():
            return
        for path in sorted(root.rglob("*")):
            relative = Path(prefix) / path.relative_to(root)
            if path.is_symlink():
                info = zipfile.ZipInfo(relative.as_posix())
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, str(path.readlink()))
            elif path.is_file():
                archive.write(path, relative.as_posix())

    def export(self, client_id: str, conversation_id: str) -> tuple[Path, str]:
        """Create a temporary ZIP containing a conversation and its retained state."""
        with self.lock:
            record = self.select(client_id, conversation_id)
            session = self._session_for(record)
            metadata = {
                key: record[key]
                for key in ("id", "name", "created_at", "updated_at")
            }
            snapshot = session.export_persistent_state()
            workspace = self._record_path(record, "workspace")
            run_dir = self._record_path(record, "run_dir")
        descriptor, archive_name = tempfile.mkstemp(suffix=".zip")
        os.close(descriptor)
        Path(archive_name).unlink(missing_ok=True)
        try:
            with zipfile.ZipFile(archive_name, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    "conversation.json",
                    json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                )
                archive.writestr(
                    "interactive-session.json",
                    json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n",
                )
                self._write_tree(archive, workspace, "workspace")
                self._write_tree(archive, run_dir, "run")
        except Exception:
            Path(archive_name).unlink(missing_ok=True)
            raise
        return Path(archive_name), f"conversation-{conversation_id}.zip"

    def close(self) -> None:
        self._persistence_stop.set()
        if self._persistence_thread is not None:
            self._persistence_thread.join(timeout=self.persistence_interval + 1)
        with self.lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.interrupt_active_work()
        for session in sessions:
            for thread in (session.thread, session.data_thread, session.evaluator_agent_thread):
                if thread:
                    thread.join(timeout=2)
        self.persist()


def mount_conversations(app, registry: ConversationRegistry) -> None:
    """Bind browser identity, conversation selection, and registry endpoints."""

    @app.middleware("http")
    async def bind_conversation(request: Request, call_next):
        client_id = request.cookies.get(CLIENT_COOKIE) or uuid.uuid4().hex
        selected_id = request.cookies.get(CONVERSATION_COOKIE)
        record, session = registry.resolve(client_id, selected_id)
        request.state.client_id = client_id
        request.state.conversation_id = record["id"]
        token = registry.session_proxy.bind(session)
        try:
            response = await call_next(request)
        finally:
            registry.session_proxy.reset(token)
        response.set_cookie(CLIENT_COOKIE, client_id, httponly=True, samesite="lax", max_age=31536000)
        selection_was_set = any(
            header.startswith(f"{CONVERSATION_COOKIE}=")
            for header in response.headers.getlist("set-cookie")
        )
        if selected_id != record["id"] and not selection_was_set:
            response.set_cookie(CONVERSATION_COOKIE, record["id"], httponly=True, samesite="lax", max_age=31536000)
        return response

    @app.get("/api/conversations")
    def conversations(request: Request):
        return registry.list(request.state.client_id, request.state.conversation_id)

    @app.post("/api/conversations")
    def create_conversation(request: Request, response: Response, payload: dict = Body(default={})):
        record = registry.create(request.state.client_id, str(payload.get("name", "")).strip() or None)
        response.set_cookie(CONVERSATION_COOKIE, record["id"], httponly=True, samesite="lax", max_age=31536000)
        return {key: record[key] for key in ("id", "name", "created_at", "updated_at")}

    @app.post("/api/conversations/{conversation_id}/select")
    def select_conversation(conversation_id: str, request: Request, response: Response):
        try:
            record = registry.select(request.state.client_id, conversation_id)
        except KeyError as exc:
            raise HTTPException(404, "Conversation not found") from exc
        response.set_cookie(CONVERSATION_COOKIE, record["id"], httponly=True, samesite="lax", max_age=31536000)
        return {"selected": record["id"]}

    @app.patch("/api/conversations/{conversation_id}")
    def rename_conversation(conversation_id: str, request: Request, payload: dict = Body(...)):
        try:
            record = registry.rename(request.state.client_id, conversation_id, str(payload.get("name", "")))
        except KeyError as exc:
            raise HTTPException(404, "Conversation not found") from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {key: record[key] for key in ("id", "name", "created_at", "updated_at")}

    @app.delete("/api/conversations/{conversation_id}")
    def archive_conversation(conversation_id: str, request: Request, response: Response):
        try:
            record, selected = registry.archive(
                request.state.client_id, conversation_id, request.state.conversation_id,
            )
        except KeyError as exc:
            raise HTTPException(404, "Conversation not found") from exc
        response.set_cookie(
            CONVERSATION_COOKIE, selected["id"], httponly=True,
            samesite="lax", max_age=31536000,
        )
        return {"archived": record["id"], "selected": selected["id"]}

    @app.get("/api/conversations/{conversation_id}/export")
    def export_conversation(conversation_id: str, request: Request):
        try:
            archive, filename = registry.export(request.state.client_id, conversation_id)
        except KeyError as exc:
            raise HTTPException(404, "Conversation not found") from exc
        return FileResponse(
            archive, filename=filename, media_type="application/zip",
            background=BackgroundTask(archive.unlink, missing_ok=True),
        )
