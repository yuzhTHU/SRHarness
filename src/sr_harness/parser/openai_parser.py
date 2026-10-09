# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import re
import json
import inspect
from ast import literal_eval
from logging import getLogger
from typing import List, Dict, Any
from .base_parser import BaseParser
from ..core import ToolCall, ToolCallResult
from ..tools import BaseTool

_logger = getLogger(f'sr_harness.{__name__}')


@BaseParser.register('openai')
class OpenAIParser(BaseParser):
    """Parser for native OpenAI-compatible tool calls."""

    def format_tools(self) -> str:
        """Format tools.

        Returns:
            str: The operation result.
        """
        raise ValueError("OpenAIParser.format_tools() should not be called.")

    def parse_response(self, response: str) -> List[ToolCall]:
        """Parse response.

        Args:
            response: Provider response object.

        Returns:
            List[ToolCall]: The operation result.
        """
        raise ValueError("OpenAIParser.parse_response() should not be called.")

    def format_tool_calls(self, tool_calls: List[ToolCall]) -> str:
        """Format tool calls.

        Args:
            tool_calls: Tool calls returned by the model.

        Returns:
            str: The operation result.
        """
        raise ValueError("OpenAIParser.format_tool_calls() should not be called.")

    def format_tool_result_messages(
        self,
        tool_calls: List[ToolCall],
        results: List[ToolCallResult],
    ) -> List[Dict[str, Any]]:
        """Format tool result messages.

        Args:
            tool_calls: Tool calls returned by the model.
            results: Result records to process.

        Returns:
            List[Dict[str, Any]]: The operation result.
        """
        messages = []
        for tool_call, result in zip(tool_calls, results):
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": result.result_str,
            })
        return messages
