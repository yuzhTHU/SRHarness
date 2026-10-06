"""Frontend boundary used by :class:`SRAgentInteractive`."""
from __future__ import annotations

from abc import ABC
from typing import Any


class InteractionManager(ABC):
    """Connect an interactive agent to a user interface.

    The default implementation is intentionally inert. Frontends may override
    control, prompt preparation, workspace ownership, and event publication
    without taking ownership of the search loop.
    """

    def bind_run_state(self, run_state) -> None:
        """Expose the authoritative in-memory run state to the frontend.

        Args:
            run_state: The run state value.
        """

    def bind_workspace(self, workspace) -> None:
        """Expose the active workspace to the frontend.

        Args:
            workspace: The workspace value.
        """

    def prepare_initial_prompt(self, messages, *, X, y):
        """Apply frontend-owned prompt additions or user overrides.

        Args:
            messages: Conversation messages in provider-compatible order.
            X: Input feature arrays keyed by variable name.
            y: Target data or target expression.
        """
        return messages

    def checkpoint(self) -> list[str]:
        """Wait at a safe boundary and return queued human guidance.

        Returns:
            list[str]: The operation result.
        """
        return []

    def take_search_transition(self) -> str | None:
        """Return a queued ``next_c`` or ``next_r`` transition.

        Returns:
            str | None: The operation result.
        """
        return None

    def wait_until_running(self) -> None:
        """Wait at a tool boundary while the frontend has paused the run."""

    def take_runtime_settings(self) -> dict[str, Any] | None:
        """Return and consume runtime settings queued by the frontend.

        Returns:
            dict[str, Any] | None: The operation result.
        """
        return None

    def commit_runtime_settings(self, settings: dict[str, Any]) -> None:
        """Tell the frontend that queued runtime settings were applied.

        Args:
            settings: Runtime settings to validate or apply.
        """

    def ask_human(self, message: str) -> str:
        """Ask the connected user for guidance.

        Args:
            message: Message text or provider message payload.

        Returns:
            str: The operation result.
        """
        raise RuntimeError("This interaction manager cannot ask a user for input.")

    def publish(self, kind: str, payload: Any) -> None:
        """Publish an observable event to the frontend.

        Args:
            kind: Event or resource kind.
            payload: Serializable event payload.
        """
