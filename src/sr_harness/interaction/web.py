# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Web adapter for an interactive SRHarness run."""
from __future__ import annotations

import shutil

from ..core import json_value
from .manager import InteractionManager


def add_variable_descriptions(messages, descriptions, variables):
    """Add variable descriptions.

    Args:
        messages: Conversation messages in provider-compatible order.
        descriptions: The descriptions value.
        variables: The variables value.
    """
    rows = [
        f"- {name}: {descriptions[name]}"
        for name in variables
        if descriptions.get(name)
    ]
    if rows:
        for message in messages:
            if message.get("role") == "user":
                message["content"] += "\n\nVariable descriptions:\n" + "\n".join(rows)
                break
    return messages


class WebInteractionManager(InteractionManager):
    """Connect one interactive agent to an :class:`InteractiveSession`."""

    def __init__(self, session):
        self.session = session

    def bind_run_state(self, run_state) -> None:
        """Bind run state.

        Args:
            run_state: The run state value.
        """
        with self.session.lock:
            self.session.run_state = run_state

    def bind_workspace(self, workspace) -> None:
        """Bind workspace.

        Args:
            workspace: The workspace value.
        """
        workspace.retain = True
        with self.session.lock:
            if self.session.workspace != workspace.path:
                for item in self.session.workspace.iterdir():
                    shutil.move(str(item), str(workspace.path / item.name))
                self.session.workspace = workspace.path

    def prepare_initial_prompt(self, messages, *, X, y):
        """Prepare initial prompt.

        Args:
            messages: Conversation messages in provider-compatible order.
            X: Input feature arrays keyed by variable name.
            y: Target data or target expression.
        """
        with self.session.lock:
            overrides = self.session.prompt_overrides.copy()
            descriptions = self.session.variable_descriptions.copy()
        add_variable_descriptions(messages, descriptions, [*X, *y])
        for message in messages:
            if message.get("role") in overrides:
                message["content"] = overrides[message["role"]]
        return messages

    def checkpoint(self) -> list[str]:
        """Run the ``checkpoint`` operation.

        Returns:
            list[str]: The operation result.
        """
        return self.session.controller.checkpoint()

    def take_search_transition(self) -> str | None:
        """Run the ``take search transition`` operation.

        Returns:
            str | None: The operation result.
        """
        return self.session.controller.take_search_transition()

    def wait_until_running(self) -> None:
        """Run the ``wait until running`` operation."""
        self.session.controller.wait_until_running()

    def force_stop_event(self):
        """Expose cancellation to streaming APIs and cancellable tools."""
        return self.session.controller.force_stop_event

    def consume_force_stop(self) -> bool:
        """Consume the forced interruption after recording it in conversation."""
        return self.session.controller.consume_force_stop()

    def take_runtime_settings(self):
        """Run the ``take runtime settings`` operation."""
        with self.session.lock:
            settings = self.session.pending_settings
            self.session.pending_settings = None
        return settings

    def commit_runtime_settings(self, settings) -> None:
        """Commit runtime settings.

        Args:
            settings: Runtime settings to validate or apply.
        """
        with self.session.lock:
            self.session.settings.update(settings)

    def ask_human(self, message: str) -> str:
        """Run the ``ask human`` operation.

        Args:
            message: Message text or provider message payload.

        Returns:
            str: The operation result.
        """
        return self.session.controller.ask(message)

    def should_request_guidance_after_tool_free_response(self) -> bool:
        """Avoid competing with an already queued pause or search transition."""
        status = self.session.controller.status()
        return not status["paused"] and status["pending_transition"] is None

    def pause_after_tool_free_response(self) -> bool:
        """Yield to the composer without presenting an artificial question card."""
        self.session.controller.request_pause()
        return True

    def publish(self, kind: str, payload) -> None:
        """Publish .

        Args:
            kind: Event or resource kind.
            payload: Serializable event payload.
        """
        self.session.controller.publish(kind, json_value(payload))
