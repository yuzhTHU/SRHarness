"""Thread-safe interaction state and event streams for one interactive agent."""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal, Mapping, TypeAlias, cast

from ..core import json_value


InteractionState: TypeAlias = Literal[
    "idle", "running", "pausing", "interrupting", "paused",
]
InteractionAction: TypeAlias = Literal["message", "pause", "force_pause"]
SRInteractionAction: TypeAlias = Literal[
    "message", "pause", "force_pause", "next_c", "next_r",
]
InteractionEventKind: TypeAlias = Literal[
    "prompt_added",
    "context",
    "assistant_started",
    "assistant_delta",
    "assistant_completed",
    "assistant_failed",
    "tool_started",
    "tool_completed",
    "settings_applied",
    "settings_failed",
    "workspace_changed",
    "execution_completed",
    "execution_failed",
    "command_received",
    "state_changed",
    "search_position_changed",
    "topk_updated",
]


@dataclass(frozen=True)
class PendingMessage:
    """A user message waiting to be inserted at an agent boundary."""

    content: str
    created_at: float


class _CancellationSignal:
    """Compatibility view for tools which expect ``cancel_event.is_set()``."""

    def __init__(self, manager: "InteractionManager") -> None:
        self._manager = manager

    def is_set(self) -> bool:
        return self._manager.is_interrupting


class InteractionManager:
    """Own the controls and observable event stream of exactly one agent."""

    _DELTA_KIND = "assistant_delta"
    _TERMINAL_KINDS = {"assistant_completed", "assistant_failed"}
    _EVENT_KINDS = {
        "prompt_added", "context", "assistant_started", "assistant_delta",
        "assistant_completed", "assistant_failed", "tool_started",
        "tool_completed", "settings_applied", "settings_failed",
        "workspace_changed", "execution_completed", "execution_failed",
        "command_received", "state_changed", "search_position_changed",
        "topk_updated",
    }

    def __init__(self, *, event_capacity: int = 1000) -> None:
        if event_capacity < 1:
            raise ValueError("event_capacity must be positive")
        self._condition = threading.Condition(threading.RLock())
        self._state: InteractionState = "idle"
        self._messages: deque[PendingMessage] = deque()
        self._events: deque[dict[str, Any]] = deque(maxlen=event_capacity)
        self._next_sequence = 1
        self._discarded_through = 0
        self._cancel_current: Callable[[], None] | None = None
        self._cancellation_signal = _CancellationSignal(self)

    @property
    def state(self) -> InteractionState:
        """Return the authoritative execution state."""
        with self._condition:
            return self._state

    @property
    def is_interrupting(self) -> bool:
        """Return whether the active operation should abort promptly."""
        return self.state == "interrupting"

    @property
    def cancellation_signal(self) -> _CancellationSignal:
        """Expose a read-only Event-like cancellation adapter to tools."""
        return self._cancellation_signal

    def status(self) -> dict[str, Any]:
        """Return a serializable control snapshot."""
        with self._condition:
            return {
                "interaction_state": self._state,
                "paused": self._state in {"pausing", "interrupting", "paused"},
                "waiting_at_boundary": self._state == "paused",
                "force_pause_requested": self._state == "interrupting",
                "pending_messages": len(self._messages),
                "last_event_seq": self._next_sequence - 1,
                "server_time": time.time(),
            }

    def command(self, action: InteractionAction, message: str | None = None) -> dict[str, Any]:
        """Apply a user control command to this agent."""
        if action not in {"message", "pause", "force_pause"}:
            raise ValueError("action must be message, pause, or force_pause")
        cancel: Callable[[], None] | None = None
        with self._condition:
            if action == "message":
                content = (message or "").strip()
                if not content:
                    raise ValueError("message command requires non-empty message")
                self._messages.append(PendingMessage(content, time.time()))
            elif action == "pause":
                if self._state == "running":
                    self._set_state_locked("pausing")
            elif self._state in {"running", "pausing"}:
                self._set_state_locked("interrupting")
                cancel = self._cancel_current
            if action != "message":
                self._publish_event_locked("command_received", {"action": action})
            self._condition.notify_all()
            result = self.status()
        if cancel is not None:
            cancel()
        return result

    def request_pause(self) -> None:
        """Request a quiet pause, such as after a tool-free response."""
        with self._condition:
            if self._state == "running":
                self._set_state_locked("pausing")
                self._condition.notify_all()

    def start_agent_execution(self) -> list[PendingMessage]:
        """Enter running state and consume messages that start this execution."""
        with self._condition:
            if self._state != "idle":
                raise RuntimeError(f"cannot start agent execution while {self._state}")
            messages = self._consume_messages_locked()
            self._set_state_locked("running")
            return messages

    def finish_agent_execution(self) -> None:
        """Mark a naturally completed agent execution as idle."""
        with self._condition:
            if self._state == "paused":
                raise RuntimeError("cannot finish an execution while paused")
            self._cancel_current = None
            self._set_state_locked("idle")

    @contextmanager
    def wait(self) -> Iterator[list[PendingMessage]]:
        """Yield boundary messages and resume only after the agent inserts them."""
        with self._condition:
            if self._state == "running":
                messages = self._consume_messages_locked()
                paused = False
            elif self._state in {"pausing", "interrupting", "paused"}:
                if self._state != "paused":
                    self._set_state_locked("paused")
                self._cancel_current = None
                while not self._messages:
                    self._condition.wait()
                messages = self._consume_messages_locked()
                paused = True
            else:
                raise RuntimeError(f"cannot wait at an agent boundary while {self._state}")
        try:
            yield messages
        finally:
            if paused:
                with self._condition:
                    if self._state == "paused":
                        self._set_state_locked("running")

    @contextmanager
    def cancellable(self, cancel: Callable[[], None]) -> Iterator[None]:
        """Register the cancellation callback for the current blocking operation."""
        with self._condition:
            if self._cancel_current is not None:
                raise RuntimeError("another cancellable operation is already active")
            self._cancel_current = cancel
            cancel_immediately = self._state == "interrupting"
        if cancel_immediately:
            cancel()
        try:
            yield
        finally:
            with self._condition:
                if self._cancel_current is cancel:
                    self._cancel_current = None

    def publish_event(self, kind: InteractionEventKind, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Append one observable event to this agent's timeline."""
        if kind not in self._EVENT_KINDS:
            raise ValueError(f"unsupported interaction event kind: {kind}")
        with self._condition:
            return self._publish_event_locked(kind, payload)

    def get_recent_events(self, after_sequence: int = 0) -> dict[str, Any]:
        """Return retained events newer than a consumer-owned sequence cursor."""
        with self._condition:
            latest_sequence = self._next_sequence - 1
            cursor_reset = after_sequence > latest_sequence
            effective_sequence = 0 if cursor_reset else after_sequence
            return {
                "events": [dict(event) for event in self._events if event["seq"] > effective_sequence],
                "truncated": effective_sequence < self._discarded_through,
                "discarded_through": self._discarded_through,
                "cursor_reset": cursor_reset,
                "server_time": time.time(),
            }

    def _consume_messages_locked(self) -> list[PendingMessage]:
        messages = list(self._messages)
        self._messages.clear()
        return messages

    def _set_state_locked(self, state: InteractionState) -> None:
        previous = self._state
        if previous == state:
            return
        self._state = state
        self._publish_event_locked("state_changed", {"previous": previous, "current": state})

    def _publish_event_locked(self, kind: InteractionEventKind, payload: Mapping[str, Any]) -> dict[str, Any]:
        payload_dict = cast(dict[str, Any], json_value(dict(payload)))
        sequence = self._next_sequence
        self._next_sequence += 1
        event = {
            "seq": sequence,
            "id": uuid.uuid4().hex,
            "kind": kind,
            "timestamp": time.time(),
            "payload": payload_dict,
        }
        response_id = payload_dict.get("response_id")
        if response_id and kind == self._DELTA_KIND:
            self._remove_stream_events_locked(response_id, {kind})
        elif response_id and kind in self._TERMINAL_KINDS:
            self._remove_stream_events_locked(response_id, {self._DELTA_KIND})
        if len(self._events) == self._events.maxlen and self._events:
            self._discarded_through = max(self._discarded_through, self._events[0]["seq"])
        self._events.append(event)
        return dict(event)

    def _remove_stream_events_locked(self, response_id: str, kinds: set[str]) -> None:
        self._events = deque(
            (
                event for event in self._events
                if not (
                    event["kind"] in kinds
                    and event.get("payload", {}).get("response_id") == response_id
                )
            ),
            maxlen=self._events.maxlen,
        )


class SRInteractionManager(InteractionManager):
    """Interaction manager with symbolic-regression branch commands."""

    def __init__(self, *, event_capacity: int = 1000) -> None:
        super().__init__(event_capacity=event_capacity)
        self._search_transitions: deque[Literal["next_c", "next_r"]] = deque()

    def command(self, action: SRInteractionAction, message: str | None = None) -> dict[str, Any]:
        """Apply a common command or queue an SR branch transition."""
        if action not in {"next_c", "next_r"}:
            return super().command(cast(InteractionAction, action), message)
        with self._condition:
            self._search_transitions.clear()
            self._search_transitions.append(action)
            self._publish_event_locked("command_received", {"action": action})
            self._condition.notify_all()
            return self.status()

    def consume_search_transition(self) -> Literal["next_c", "next_r"] | None:
        """Consume the newest queued branch transition."""
        with self._condition:
            return self._search_transitions.popleft() if self._search_transitions else None

    def status(self) -> dict[str, Any]:
        """Include the pending SR branch transition in the control snapshot."""
        with self._condition:
            status = super().status()
            status["pending_transition"] = (
                self._search_transitions[-1] if self._search_transitions else None
            )
            return status
