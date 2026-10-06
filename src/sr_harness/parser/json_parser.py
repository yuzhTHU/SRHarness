# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
from __future__ import annotations
import re
import json
import inspect
from ast import literal_eval
from logging import getLogger
from typing import List, Dict, Any, Tuple
from .base_parser import BaseParser
from ..core import ToolCall
from ..tools import BaseTool

_logger = getLogger(f'sr_harness.{__name__}')


@BaseParser.register('json')
class JSONParser(BaseParser):
    """Parser for JSON-formatted tool calls."""

    def format_tools(self) -> str:
        """Format tools.

        Returns:
            str: The operation result.
        """
        lines = ["## Available Tools:", ""]

        for tool in self.tools:
            name = tool['name']
            desc = tool['description']

            # 获取工具类的签名
            try:
                if (tool_cls := BaseTool.REGISTRY_DICT.get(name)):
                    sig = inspect.signature(tool_cls.execute)
                    params = []
                    for param_name, param in sig.parameters.items():
                        if param_name in ('self', 'args', 'kwargs'):
                            continue
                        param_type = param.annotation.__name__ if param.annotation != inspect.Parameter.empty else 'Any'
                        if param.default is inspect.Parameter.empty:
                            params.append(f'"{param_name}": {param_type}')
                        else:
                            params.append(f'"{param_name}": {param_type} = {param.default}')
                    signature = "{" + ", ".join(params) + "}"
                else:
                    signature = "{}"
            except Exception:
                signature = "{}"

            lines.append(f"### {name}")
            lines.append(f"- **Description**: {desc}")
            lines.append(f"- **Params**: `{signature}`")
            lines.append("")

        lines.append("## Output Format:")
        lines.append("")
        lines.append("Respond with a JSON object in the following format:")
        lines.append("")
        lines.append("```json")
        lines.append('{"tool": "tool_name", "params": {"param1": "value1", "param2": "value2"}}')
        lines.append("```")
        lines.append("")
        lines.append("You can include multiple tool call blocks in one response if needed.")

        return "\n".join(lines)

    def parse_response(self, response: str) -> List[ToolCall]:
        """Parse response.

        Args:
            response: Provider response object.

        Returns:
            List[ToolCall]: The operation result.
        """
        tool_calls = []

        # 尝试从响应中提取 JSON
        if (json_match := re.search(r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL)):
            json_str = json_match.group(1)
        else:
            # 如果没有代码块标记，尝试直接解析整个响应
            json_str = response.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            _logger.warning(f"Failed to parse JSON: {e}")
            return tool_calls

        for action in data["actions"]:
            tool_calls.append(ToolCall(name=action["tool"], params=action["params"], raw_str=json_str))

        return tool_calls

    def format_tool_calls(self, tool_calls: List[ToolCall]) -> str:
        """Format tool calls.

        Args:
            tool_calls: Tool calls returned by the model.

        Returns:
            str: The operation result.
        """
        lines = []
        for call in tool_calls:
            action = {"tool": call.name, "params": call.params}
            lines.append("```json\n" + json.dumps(action) + "\n```")
        return "\n".join(lines)
