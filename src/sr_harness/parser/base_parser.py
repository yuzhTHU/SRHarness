# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""工具调用解析器模块。

负责将工具列表格式化为 LLM 可读的描述，以及从 LLM 响应中解析工具调用。
"""
from __future__ import annotations
from logging import getLogger
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..core import ToolCall, ToolCallResult
from ..tools import BaseTool
from ..utils import FactoryMixin

_logger = getLogger(f'sr_harness.{__name__}')


class BaseParser(ABC, FactoryMixin):
    """Base class for model tool-call parsers."""

    def __init__(self, tool_list: List[str] | List[Dict[str, Any]] | None = None):
        """初始化工具解析器。

        Args:
            tool_list: 可用的工具名称列表，或已加载的工具元数据列表。
                None 表示使用全部工具。
        """
        if tool_list is not None and all(isinstance(t, dict) for t in tool_list):
            self.tools = tool_list
        elif tool_list is not None and all(isinstance(t, BaseTool) for t in tool_list):
            self.tools = [
                {
                    "name": tool.metadata.name,
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.parameters,
                }
                for tool in tool_list
            ]
        elif tool_list is not None and all(isinstance(t, type) and issubclass(t, BaseTool) for t in tool_list):
            self.tools = [
                {
                    "name": tool.metadata.name,
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.parameters,
                }
                for tool in tool_list
            ]
        elif tool_list is not None:
            self.tools = [t for t in BaseTool.load_tool_list() if t['name'] in tool_list]
        else:
            self.tools = BaseTool.load_tool_list()

    @abstractmethod
    def format_tools(self) -> str:
        """Format tools.

        Returns:
            str: The operation result.
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> List[ToolCall]:
        """Parse response.

        Args:
            response: Provider response object.

        Returns:
            List[ToolCall]: The operation result.
        """
        pass

    @abstractmethod
    def format_tool_calls(self, tool_calls: List[ToolCall]) -> str:
        """Format tool calls.

        Args:
            tool_calls: Tool calls returned by the model.

        Returns:
            str: The operation result.
        """
        pass

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
        lines = []
        for tool_call, result in zip(tool_calls, results):
            lines.append(f"=== Results for `{tool_call.name}` with params `{tool_call.params}` ===")
            lines.append(result.result_str)
        return [{"role": "user", "content": "\n".join(lines)}] if lines else []
