# Copyright (c) 2024-present, Yumeow. Licensed under the MIT License.
"""Base class and shared tool-call support for LLM provider adapters."""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from functools import cached_property
import os
from typing import Any, Callable, Dict, Generator, List, Literal, Mapping

from ..core import APICallResult, ToolCall
from ..parser import BaseParser
from ..tools import BaseTool
from ..utils import FactoryMixin, log_exception

_logger = logging.getLogger(f"sr_harness.{__name__}")

ToolParserName = Literal["text", "json", "xml", "openai"]
ToolList = list[BaseTool | type[BaseTool]]
StreamCallback = Callable[[dict[str, Any]], None]


class ModelResponseTruncatedError(RuntimeError):
    """Report a length-limited response while preserving its partial message.

    Args:
        message: Human-readable explanation of the truncation.
        partial_message: Provider response accumulated before truncation.
        tool_calls: Tool calls parsed from the partial response.
        usage: Token and price usage reported for the truncated response.
        sample: One-based local-sample index that was truncated.
    """

    def __init__(
        self,
        message: str,
        *,
        partial_message: dict[str, Any],
        tool_calls: list[ToolCall],
        usage: dict[str, dict[str, int | float]],
        sample: int,
    ) -> None:
        super().__init__(message)
        self.partial_message = partial_message
        self.tool_calls = tool_calls
        self.usage = usage
        self.sample = sample


class BaseAPI(ABC, FactoryMixin):
    """Common request, parser, and tool-call behavior for LLM providers."""

    supported_models: list[str] = []
    supports_streaming = False

    def __init__(
        self,
        model: str | None = None,
        tool_list: ToolList | None = None,
        tool_parser_name: ToolParserName = "text",
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.model = model
        self.tool_list = tool_list
        self.tool_parser_name = tool_parser_name
        self.environment = environment if environment is not None else {}
        self.tool_parser = self.build_parser(tool_parser_name)

    def getenv(self, name: str, default: str | None = None) -> str | None:
        """Read a session-scoped provider setting before the process environment."""
        return self.environment.get(name, os.environ.get(name, default))

    def require_env(self, name: str) -> str:
        """Return a required session-scoped provider setting."""
        value = self.getenv(name)
        if value is None:
            raise KeyError(name)
        return value

    def __call__(
        self,
        messages: list[dict[str, Any]] | str,
        stream_callback: StreamCallback | None = None,
        **kwargs: Any,
    ) -> APICallResult:
        """Request the provider and wrap its result generator."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if stream_callback is not None and self.supports_streaming:
            kwargs["stream_callback"] = stream_callback
        generator = self._request(messages, **kwargs)
        return APICallResult(generator, self.tool_parser)

    def cancel(self) -> None:
        """Request cancellation when a provider offers no stronger primitive."""

    @abstractmethod
    def _request(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Generator[dict[str, Any], None, dict[str, Any]]:
        """Yield provider responses and return aggregate response metadata."""
        raise NotImplementedError

    def build_parser(self, parser: ToolParserName) -> BaseParser | None:
        """Build parser.

        Args:
            parser: Argument parser to configure.

        Returns:
            BaseParser | None: The operation result.
        """
        if not self.tool_list or parser == "openai":
            return None
        if parser in {"text", "json", "xml"}:
            return BaseParser.create(parser, tool_list=self.tool_list)
        raise ValueError(f"Unsupported tool parser: {parser}")

    @cached_property
    def tool_description_text(self) -> str:
        """Format available tools for a model using a text-based parser.

        Returns:
            str: The operation result.
        """
        return (
            "Use the following tools when a tool call is needed. "
            "Return tool calls in the specified format.\n\n"
            f"{self.tool_parser.format_tools()}"
        )

    @cached_property
    def tool_description_json(self) -> List[Dict]:
        """Build OpenAI-compatible native function descriptions.

        Returns:
            List[Dict]: The operation result.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.metadata.name,
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.parameters,
                },
            }
            for tool in self.tool_list
        ]

    def add_tool_description(
        self,
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Add text-formatted tool instructions to the leading system message.

        Args:
            messages: Conversation messages in provider-compatible order.

        Returns:
            List[Dict[str, str]]: The operation result.
        """
        if (role := messages[0]["role"]) not in {"system", "developer"}:
            return [{"role": "system", "content": self.tool_description_text}] + messages
        if self.tool_description_text not in (content := messages[0]["content"]):
            return [
                {"role": role, "content": f"{content}\n\n{self.tool_description_text}"},
                *messages[1:],
            ]
        return messages

    def normalize_openai_tool_calls(self, tool_calls: List[Any]) -> List[ToolCall]:
        """Normalize provider-native function calls into internal ToolCall objects.

        Args:
            tool_calls: Tool calls returned by the model.

        Returns:
            List[ToolCall]: The operation result.
        """
        normalized = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                pass
            elif hasattr(tool_call, "to_dict"):
                tool_call = tool_call.to_dict()
            elif hasattr(tool_call, "model_dump"):
                tool_call = tool_call.model_dump()
            else:
                raise ValueError(f"Unrecognized tool call format: {tool_call}")
            try:
                normalized.append(self._parse_native_tool_call(tool_call))
            except json.JSONDecodeError as error:
                _logger.warning(
                    f"Skip tool call {tool_call!r} since it cannot be parsed as JSON: "
                    f"{log_exception(error)}."
                )
        return normalized

    def _parse_native_tool_call(self, tool_call: Dict[str, Any]) -> ToolCall:
        function = tool_call.get("function") or {}
        name = function.get("name") or tool_call.get("name") or None
        params = (
            function.get("arguments")
            or tool_call.get("arguments")
            or tool_call.get("arguments_json")
            or tool_call.get("args")
            or {}
        )
        if isinstance(params, str):
            params = json.loads(params) if params.strip() else {}
        return ToolCall(
            name=name,
            params=params,
            id=tool_call.get("id") or tool_call.get("call_id"),
            raw=tool_call,
        )

