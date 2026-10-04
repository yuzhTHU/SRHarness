# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Persistent tool-using agent for preparing structured research data."""
from __future__ import annotations

import time
import uuid
from copy import deepcopy
from typing import Any

from ..core import AgentContext
from ..interaction import InteractionManager
from ..parser import BaseParser
from ..tools import BaseTool
from ..utils import ParallelTimer
from .agent import Agent


class DataPreparationAgent(Agent):
    """Turn workspace and web evidence into the shared structured dataset."""

    DEFAULT_TOOLS = [
        "workspace_shell",
        "workspace_code_executor",
        "web_search",
        "web_fetch",
        "read_pdf",
        "read_skill",
        "commit_data",
    ]

    def __init__(
        self,
        llm_provider: str,
        llm_model: str,
        *,
        context: AgentContext,
        tools: list[str] | None = None,
        tool_parser: str | BaseParser = "openai",
        llm_max_tokens: int = 4096,
        max_turns: int = 12,
        skills: list[str] | None = None,
        interaction_manager: InteractionManager | None = None,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.tool_parser = tool_parser
        self.llm_max_tokens = llm_max_tokens
        self.max_turns = max_turns
        self.skills = skills
        self.interaction_manager = interaction_manager or InteractionManager()
        self.tool_cls_list = BaseTool.load_tool_classes(tools or self.DEFAULT_TOOLS)
        self.tools_counter = ParallelTimer(unit="call")
        self.tools = None
        self.parser = None
        self.api = None
        self.context = context
        self.buffer: list[dict[str, Any]] = [{
            "role": "system",
            "content": (
                "You are the data-preparation agent in SRHarness. Work with the user's persistent "
                "workspace and conversation to create a clean table for symbolic regression. Inspect "
                "uploaded files, including PDF documents when relevant. Preserve useful identifiers "
                "such as year while cleaning and joining "
                "sources, and use web_search plus web_fetch when external evidence is requested. Use "
                "workspace_code_executor for reproducible transformations and save the resulting table. "
                "Call commit_data only when the target and selected features are numeric, finite, aligned, "
                "and ready. Explain material assumptions and sources. A later user request may extend the "
                "existing dataset, so retain and use this conversation and all workspace artifacts."
            ),
        }]
        self.initialize_tools(context)

    def initialize_tools(self, context: AgentContext) -> None:
        """Bind configured skills before constructing context-aware tools."""
        context["enabled_skills"] = self.skills
        super().initialize_tools(context)

    def run(self, instruction: str) -> dict[str, Any]:
        """Continue the persistent preparation conversation until it yields control."""
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("instruction must not be empty")
        self.context.update({
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_max_tokens": self.llm_max_tokens,
        })
        self.buffer.append({"role": "user", "content": instruction})
        self._publish("data_user", {"content": instruction})
        committed = False
        final_content = ""
        for turn in range(1, self.max_turns + 1):
            prompt = deepcopy(self.buffer)
            self.set_messages(prompt)
            self._publish("data_context", {"messages": prompt, "turn": turn})
            response_id = f"data:{uuid.uuid4().hex}"
            tool_schemas = {
                tool.metadata.name: {
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.parameters or {},
                }
                for tool in self.tools
            }
            self._publish("data_assistant_start", {
                "response_id": response_id,
                "turn": turn,
                "tool_schemas": tool_schemas,
                "provider": self.llm_provider,
                "model": self.llm_model,
            })
            last_stream_emit = 0.0

            def stream_callback(update: dict[str, Any]) -> None:
                nonlocal last_stream_emit
                now = time.monotonic()
                if update.get("type") == "delta" and now - last_stream_emit < 0.08:
                    return
                last_stream_emit = now
                self._publish("data_assistant_delta", {
                    **update,
                    "response_id": response_id,
                    "turn": turn,
                    "tool_schemas": tool_schemas,
                    "provider": self.llm_provider,
                    "model": self.llm_model,
                })

            try:
                call_result = self.api(
                    prompt,
                    n=1,
                    max_tokens=self.llm_max_tokens,
                    stream_callback=stream_callback,
                )
                responses = list(call_result)
            except Exception as exc:
                self._publish("data_assistant_error", {
                    "response_id": response_id,
                    "turn": turn,
                    "provider": self.llm_provider,
                    "model": self.llm_model,
                    "error": str(exc),
                })
                raise
            usage = call_result.returned.get("usage", {"token": {}, "price": {}})
            if not responses:
                raise RuntimeError("The model returned no usable response")
            content, calls, message = responses[0]
            final_content = content or ""
            self._publish("data_assistant", {
                "response_id": response_id,
                "content": content,
                "message": message,
                "tool_calls": calls,
                "tool_schemas": tool_schemas,
                "turn": turn,
                "usage": usage,
                "provider": self.llm_provider,
                "model": self.llm_model,
            })
            if not calls:
                self.buffer.append(message)
                break
            results = self._execute_with_events(calls)
            self.buffer.append(message)
            self.buffer.extend(self.parser.format_tool_result_messages(calls, results))
            committed = committed or any(result.result.get("data_committed") for result in results)
        else:
            final_content = (
                final_content
                or f"Stopped after the configured {self.max_turns} preparation turns."
            )
        result = {
            "message": final_content,
            "committed": committed,
            "context": self.context.schema(),
        }
        self._publish("data_complete", result)
        return result

    def _execute_with_events(self, calls):
        results = []
        for call in calls:
            schema = next((
                {
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.parameters or {},
                }
                for tool in self.tools
                if tool.metadata.name == call.name
            ), {})
            self._publish("data_tool_start", {"call": call, "tool_schema": schema})
            result = super().execute_action([call])[0]
            self._publish("data_tool_result", {
                "call": call,
                "tool_schema": schema,
                "result": result,
            })
            results.append(result)
        return results

    def _publish(self, kind: str, payload: Any) -> None:
        self.interaction_manager.publish(kind, payload)
