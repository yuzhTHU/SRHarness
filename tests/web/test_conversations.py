from __future__ import annotations

import io
import json
import zipfile

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from sr_harness.web.app import create_app
from sr_harness.web.conversations import ConversationRegistry


def test_conversations_are_persisted_and_can_be_renamed(tmp_path):
    registry = ConversationRegistry(tmp_path)
    app = create_app(tmp_path, session=registry.session_proxy, conversation_registry=registry)
    with TestClient(app) as client:
        initial = client.get("/api/conversations").json()
        assert len(initial["conversations"]) == 1
        created = client.post("/api/conversations", json={"name": "Experiment"}).json()
        renamed = client.patch(
            f"/api/conversations/{created['id']}", json={"name": "Oscillator"},
        ).json()
        assert renamed["name"] == "Oscillator"
        assert len(client.get("/api/conversations").json()["conversations"]) == 2
    payload = json.loads((tmp_path / "conversations.json").read_text())
    assert {item["name"] for item in payload["conversations"]} == {"Conversation 1", "Oscillator"}
    registry.close()
    restored = ConversationRegistry(tmp_path)
    assert {item["name"] for item in restored.list("browser")["conversations"]} == {
        "Conversation 1", "Oscillator",
    }
    restored.close()


def test_conversation_registry_rejects_records_from_another_schema(tmp_path):
    (tmp_path / "conversations.json").write_text(json.dumps({
        "version": 1,
        "conversations": [{
            "id": "conversation",
            "name": "Old record",
            "owner_id": None,
            "workspace": "workspaces/conversation",
            "run_dir": "runs/conversation",
            "created_at": "2026-10-09T00:00:00+00:00",
            "updated_at": "2026-10-09T00:00:00+00:00",
        }],
    }))

    with pytest.raises(ValueError, match="invalid record"):
        ConversationRegistry(tmp_path)


def test_conversation_can_be_exported_and_archived_without_deleting_state(tmp_path):
    registry = ConversationRegistry(tmp_path)
    app = create_app(tmp_path, session=registry.session_proxy, conversation_registry=registry)
    with TestClient(app) as client:
        created = client.post("/api/conversations", json={"name": "Export me"}).json()
        conversation_id = created["id"]
        assert client.put("/api/workspace/upload?path=notes.txt", content=b"retained").status_code == 200

        exported = client.get(f"/api/conversations/{conversation_id}/export")
        assert exported.status_code == 200
        with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
            assert json.loads(archive.read("conversation.json"))["name"] == "Export me"
            assert archive.read("workspace/notes.txt") == b"retained"
            assert "interactive-session.json" in archive.namelist()

        archived = client.delete(f"/api/conversations/{conversation_id}").json()
        assert archived["archived"] == conversation_id
        assert archived["selected"] != conversation_id
        assert conversation_id not in {
            item["id"] for item in client.get("/api/conversations").json()["conversations"]
        }

    payload = json.loads((tmp_path / "conversations.json").read_text())
    record = next(item for item in payload["conversations"] if item["id"] == conversation_id)
    assert record["archived"] is True
    assert (tmp_path / record["workspace"] / "notes.txt").read_bytes() == b"retained"
    assert conversation_id in registry._sessions
    registry.close()


def test_session_state_is_restored_and_active_work_is_interrupted(tmp_path):
    registry = ConversationRegistry(tmp_path, persist_sessions=True, persistence_interval=60)
    session = registry.default_session
    session.initial_prompt = "Persist this prompt"
    session.settings["ranking_metric"] = "rmse"
    session.sr_interaction_manager.command("message", "queued guidance")
    session.sr_interaction_manager.start_agent_execution()
    session.state = "running"
    session.sr_interaction_manager.publish_event(
        "prompt_added", {"message": {"role": "user", "content": "hello"}},
    )
    registry.persist()
    registry.close()

    restored = ConversationRegistry(tmp_path, persist_sessions=True, persistence_interval=60)
    recovered = restored.default_session
    assert recovered.initial_prompt == "Persist this prompt"
    assert recovered.settings["ranking_metric"] == "rmse"
    assert recovered.state == "idle"
    assert recovered.result["status"] == "interrupted"
    events = recovered.sr_interaction_manager.get_recent_events()["events"]
    assert any(event["kind"] == "prompt_added" for event in events)
    assert any(
        event["kind"] == "command_received"
        and event["payload"].get("restored_after_restart") is True
        for event in events
    )
    restored.close()


def test_session_snapshots_are_written_below_explicit_save_path(tmp_path):
    workspace_dir = tmp_path / "workspace-root"
    save_path = tmp_path / "saved-run"
    registry = ConversationRegistry(
        workspace_dir,
        initial_run_dir=save_path,
        persist_sessions=True,
        persistence_interval=60,
    )
    conversation_id = registry.list("browser")["conversations"][0]["id"]
    registry.default_session.initial_prompt = "durable"
    registry.persist()
    snapshot = save_path / "sessions" / f"{conversation_id}.json"
    assert json.loads(snapshot.read_text())["initial_prompt"] == "durable"
    registry.close()


def test_isolate_users_routes_workspace_requests_to_distinct_sessions(tmp_path):
    registry = ConversationRegistry(tmp_path, isolate_users=True)
    app = create_app(tmp_path, session=registry.session_proxy, conversation_registry=registry)
    with TestClient(app) as first, TestClient(app) as second:
        first_conversation = first.get("/api/conversations").json()
        second_conversation = second.get("/api/conversations").json()
        assert first_conversation["selected"] != second_conversation["selected"]

        assert first.put("/api/workspace/upload?path=first.txt", content=b"first").status_code == 200
        assert second.put("/api/workspace/upload?path=second.txt", content=b"second").status_code == 200
        first_files = {entry["name"] for entry in first.get("/api/workspace").json()["entries"]}
        second_files = {entry["name"] for entry in second.get("/api/workspace").json()["entries"]}
        assert first_files == {"first.txt"}
        assert second_files == {"second.txt"}
    registry.close()
