# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Common runtime mechanics shared by SRHarness agents."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from joblib import Parallel, delayed

from ..api import BaseAPI
from ..core import AgentContext, ToolCall, ToolCallResult
from ..parser import BaseParser
from ..tools import BaseTool
from ..utils import FactoryMixin

_logger = logging.getLogger(f"sr_harness.{__name__}")


def _execute_tool_call_in_subprocess(tool: BaseTool, tool_call: ToolCall) -> ToolCallResult:
    """Execute one tool call in a worker process."""
    return tool(**tool_call.params)


class Agent(ABC, FactoryMixin):
    """Base class for tool-using agents.

    Subclasses own their task loop and scientific state. This class only owns
    common API/tool initialization and tool execution mechanics.
    """

    context: AgentContext
    tools: list[BaseTool] | None
    parser: BaseParser | None
    api: BaseAPI | None

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the agent's task loop."""

    def initialize_tools(self, context: AgentContext) -> None:
        """Bind one shared context to every tool, parser, and API adapter."""
        self.context = context
        self.tools = [tool_cls(context=context) for tool_cls in self.tool_cls_list]
        self.parser = BaseParser.create(self.tool_parser, tool_list=self.tools)
        self.api = BaseAPI.create(
            self.llm_provider,
            model=self.llm_model,
            tool_list=self.tools,
            tool_parser_name=self.tool_parser,
        )

    def set_messages(self, messages: list[dict[str, Any]]) -> None:
        """Expose the current prompt through the shared tool context."""
        self.context["messages"] = deepcopy(messages)

    def execute_action(self, actions: list[ToolCall]) -> list[ToolCallResult | None]:
        """Execute tool calls serially."""
        results: list[ToolCallResult | None] = []
        for action in actions:
            tool = next((item for item in self.tools if item.metadata.name == action.name), None)
            if tool is None:
                _logger.trace(f"Unknown tool call: {action.name}. Skipping execution.")
                results.append(ToolCallResult(
                    ok=False,
                    result={},
                    result_str=f'Unknown tool calling for "{action.name}"',
                    meta_data={"tool": action.name},
                ))
                continue
            self.tools_counter.add(action.name)
            results.append(tool(**action.params))
        return results

    def execute_action_parallel(
        self,
        actions: list[ToolCall],
        max_workers: int,
    ) -> list[ToolCallResult]:
        """Execute independent tool calls in worker processes."""
        results: list[ToolCallResult | None] = [None] * len(actions)
        tasks = []
        for index, action in enumerate(actions):
            tool = next((item for item in self.tools if item.metadata.name == action.name), None)
            if tool is None:
                _logger.trace(f"Unknown tool call: {action.name}. Skipping execution.")
                results[index] = ToolCallResult(
                    ok=False,
                    result={},
                    result_str=f'Unknown tool calling for "{action.name}"',
                    meta_data={"tool": action.name},
                )
            else:
                tasks.append((index, delayed(_execute_tool_call_in_subprocess)(tool, action)))
                self.tools_counter.add(action.name)

        if tasks:
            workers = Parallel(n_jobs=max_workers, backend="loky")
            task_results = workers(task for _, task in tasks)
            for (index, _), result in zip(tasks, task_results):
                results[index] = result
        return results

