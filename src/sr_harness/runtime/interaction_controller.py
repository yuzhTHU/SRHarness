# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Thread-safe bidirectional control for interactive frontends."""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any


class InteractionController:
    """Coordinate pause/resume/stop, injected guidance, events, and replies."""

    _STREAM_DELTA_KINDS = {
        "assistant_delta",
        "data_assistant_delta",
        "evaluator_assistant_delta",
    }
    _STREAM_TERMINAL_KINDS = {
        "assistant",
        "assistant_error",
        "data_assistant",
        "data_assistant_error",
        "evaluator_assistant",
        "evaluator_assistant_error",
    }

    def __init__(self):
        self._condition = threading.Condition(threading.RLock())
        self._paused = False
        self._stopped = False
        self._force_stop = threading.Event()
        self._events: deque[dict[str, Any]] = deque(maxlen=1000)
        self._guidance: deque[str] = deque()
        self._search_transitions: deque[str] = deque()
        self._replies: dict[str, str] = {}
        self._sequence = 0
        self._evicted_through_seq = 0
        self._questions: dict[str, str] = {}
        self._waiting_at_boundary = False
        self._activity: dict[str, Any] = {"phase": "idle", "since": time.time()}

    def status(self) -> dict[str, Any]:
        """Run the ``status`` operation.

        Returns:
            dict[str, Any]: The operation result.
        """
        with self._condition:
            return {
                "paused": self._paused,
                "stopped": self._stopped,
                "force_stop_requested": self._force_stop.is_set(),
                "pending_guidance": len(self._guidance),
                "pending_transition": self._search_transitions[-1] if self._search_transitions else None,
                "last_event_seq": self._sequence,
                "questions": dict(self._questions),
                "waiting_at_boundary": self._waiting_at_boundary,
                "activity": dict(self._activity),
                "server_time": time.time(),
            }

    def command(self, action: str, message: str = "") -> dict[str, Any]:
        """Run the ``command`` operation.

        Args:
            action: The action value.
            message: Message text or provider message payload.

        Returns:
            dict[str, Any]: The operation result.
        """
        action = action.strip().lower()
        with self._condition:
            if action == "pause":
                self._paused = True
            elif action == "resume":
                self._paused = False
                self._force_stop.clear()
            elif action == "force_stop":
                self._paused = True
                self._force_stop.set()
            elif action == "stop":
                self._stopped = True
                self._paused = False
            elif action == "message":
                if not message.strip():
                    raise ValueError("message command requires non-empty message")
                self._guidance.append(message.strip())
                self._paused = False
                self._force_stop.clear()
            elif action in {"next_c", "next_r"}:
                self._search_transitions.clear()
                self._search_transitions.append(action)
                self._paused = False
            else:
                raise ValueError(
                    "action must be pause, resume, force_stop, stop, message, next_c, or next_r"
                )
            self._publish_locked("control", {"action": action, "message": message})
            self._condition.notify_all()
            return self.status()

    def request_pause(self) -> dict[str, Any]:
        """Pause at the next safe boundary without publishing a timeline event."""
        with self._condition:
            self._paused = True
            self._condition.notify_all()
            return self.status()

    @property
    def force_stop_event(self) -> threading.Event:
        """Cancellation event shared with the active model and tool calls."""
        return self._force_stop

    def consume_force_stop(self) -> bool:
        """Clear and report a pending forced-turn interruption."""
        with self._condition:
            pending = self._force_stop.is_set()
            self._force_stop.clear()
            return pending

    def wait_until_running(self) -> None:
        """Run the ``wait until running`` operation."""
        with self._condition:
            while self._paused and not self._stopped:
                self._waiting_at_boundary = True
                self._condition.wait(timeout=1.0)
            self._waiting_at_boundary = False
            if self._stopped:
                raise KeyboardInterrupt("Stopped through the interaction controller")

    def checkpoint(self) -> list[str]:
        """Run the ``checkpoint`` operation.

        Returns:
            list[str]: The operation result.
        """
        with self._condition:
            self.wait_until_running()
            guidance = list(self._guidance)
            self._guidance.clear()
            return guidance

    def take_search_transition(self) -> str | None:
        """Consume a request to advance to the next branch or restart.

        Returns:
            str | None: The operation result.
        """
        with self._condition:
            return self._search_transitions.popleft() if self._search_transitions else None

    def ask(self, message: str, timeout: float | None = None) -> str:
        """Run the ``ask`` operation.

        Args:
            message: Message text or provider message payload.
            timeout: Maximum wait time in seconds.

        Returns:
            str: The operation result.
        """
        event_id = uuid.uuid4().hex
        with self._condition:
            self._questions[event_id] = message
            self._publish_locked("question", {"message": message}, event_id=event_id)
            deadline = None if timeout is None else time.monotonic() + timeout
            while event_id not in self._replies and not self._stopped:
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining == 0:
                    self._questions.pop(event_id, None)
                    return "(No response before timeout)"
                self._condition.wait(timeout=remaining)
            if self._stopped:
                self._questions.pop(event_id, None)
                raise KeyboardInterrupt("Stopped while waiting for human input")
            self._questions.pop(event_id, None)
            return self._replies.pop(event_id)

    def reply(self, event_id: str, message: str) -> None:
        """Run the ``reply`` operation.

        Args:
            event_id: Identifier of a pending interaction event.
            message: Message text or provider message payload.
        """
        with self._condition:
            if event_id not in self._questions or event_id in self._replies:
                raise ValueError(f"unknown question event: {event_id}")
            self._replies[event_id] = message.strip() or "(No input)"
            self._publish_locked("reply", {"question_id": event_id, "message": message})
            self._condition.notify_all()

    def events(self, after_seq: int = 0) -> list[dict[str, Any]]:
        """Run the ``events`` operation.

        Args:
            after_seq: Last observed event sequence.

        Returns:
            list[dict[str, Any]]: The operation result.
        """
        with self._condition:
            return [dict(event) for event in self._events if event["seq"] > after_seq]

    def event_batch(self, after_seq: int = 0) -> dict[str, Any]:
        """Return events plus an explicit indication of true buffer eviction."""
        with self._condition:
            return {
                "events": [dict(event) for event in self._events if event["seq"] > after_seq],
                "truncated": after_seq < self._evicted_through_seq,
                "evicted_through_seq": self._evicted_through_seq,
            }

    def publish(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Publish .

        Args:
            kind: Event or resource kind.
            payload: Serializable event payload.

        Returns:
            dict[str, Any]: The operation result.
        """
        with self._condition:
            return self._publish_locked(kind, payload)

    def _publish_locked(
        self,
        kind: str,
        payload: dict[str, Any],
        event_id: str | None = None,
    ) -> dict[str, Any]:
        self._sequence += 1
        if kind == "activity":
            self._activity = {**payload, "since": time.time()}
        event = {
            "seq": self._sequence,
            "id": event_id or uuid.uuid4().hex,
            "kind": kind,
            "timestamp": time.time(),
            "payload": payload,
        }
        if (response_id := payload.get("response_id")) and kind in self._STREAM_DELTA_KINDS:
            # Stream callbacks contain cumulative snapshots. Keeping every
            # snapshot makes a page reload replay and render the same growing
            # response hundreds of times, while also evicting useful events.
            self._remove_stream_events(response_id, {kind})
        elif response_id and kind in self._STREAM_TERMINAL_KINDS:
            self._remove_stream_events(response_id, self._STREAM_DELTA_KINDS)
        if len(self._events) == self._events.maxlen and self._events:
            self._evicted_through_seq = max(self._evicted_through_seq, self._events[0]["seq"])
        self._events.append(event)
        return dict(event)

    def _remove_stream_events(self, response_id: str, kinds: set[str]) -> None:
        """Remove obsolete stream snapshots for one model response."""
        self._events = deque(
            (
                event
                for event in self._events
                if not (event["kind"] in kinds and event.get("payload", {}).get("response_id") == response_id)
            ),
            maxlen=self._events.maxlen,
        )
