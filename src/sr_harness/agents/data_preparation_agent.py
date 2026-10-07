# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Persistent tool-using agent for preparing structured research data."""
from __future__ import annotations

import threading
import time
import uuid
from copy import deepcopy
from typing import Any

from ..core import AgentContext
from ..interaction import InteractionManager
from ..parser import BaseParser
from ..skills import SkillManager
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
        "load_context_data",
    ]
    DEFAULT_EXCLUDED_SKILLS = frozenset({"discover-symbolic-laws"})

    def __init__(
        self,
        llm_provider: str,
        llm_model: str,
        *,
        context: AgentContext,
        tools: list[str] | None = None,
        tool_parser: str | BaseParser = "openai",
        llm_max_tokens: int = 4096,
        skills: list[str] | None = None,
        interaction_manager: InteractionManager | None = None,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.tool_parser = tool_parser
        self.llm_max_tokens = llm_max_tokens
        self.turn_count = 0
        skill_manager = getattr(context.args, "skill_manager", None) or SkillManager()
        context.args.skill_manager = skill_manager
        self.skills = (
            list(skills)
            if skills is not None
            else [
                name for name in skill_manager.load_skills()
                if name not in self.DEFAULT_EXCLUDED_SKILLS
            ]
        )
        self.interaction_manager = interaction_manager or InteractionManager()
        self.tool_cls_list = BaseTool.load_tool_classes(tools or self.DEFAULT_TOOLS)
        self.tools_counter = ParallelTimer(unit="call")
        self.tools = None
        self.parser = None
        self.api = None
        self.context = context
        self._stop_requested = threading.Event()
        self.buffer: list[dict[str, Any]] = [{
            "role": "system",
            "content": (
                "You are the data-preparation agent in SRHarness. Work with the user's persistent "
                "workspace and conversation to create structured scientific data. Inspect uploaded files, "
                "including PDF documents when relevant, and use web_search plus web_fetch when external "
                "evidence is requested. If web_fetch reports that a site blocked automated access, do not "
                "retry the same URL; use web_search to locate an accessible authoritative alternative. Use "
                "workspace_code_executor for reproducible transformations.\n\n"
                "Store general structured results in the flat workspace directory context.data/. Put each "
                "variable and each file-backed axis in <name>.npy, using NumPy arrays "
                "that load with allow_pickle=False. The directory must contain manifest.json with variables "
                "and axes objects. For network or hypergraph data, also add a positive integer num_nodes at "
                "the manifest root; omit it for ordinary element-wise data. Each variable entry must contain "
                "file, description, and axes; its file must be <variable>.npy and its axes list must follow array "
                "dimension order. Each axis entry must contain description and exactly one of: values for a "
                "short inline JSON array, file for <axis>.npy, or size for a positive positional length. "
                "For each variable whose final dimension depends on an edge list A or hyperedge list T, "
                "set that variable's structure field to the name of A or T. The referenced relation must "
                "be an integer (E, 2) or (H, 3) array whose endpoints are in [0, num_nodes), and the "
                "dependent variable must have shape (..., E) or (..., H). Do not add structure to A or T "
                "itself, and never infer a relation from a variable name. "
                "Do not add format, version, revision, problem, attributes, shape, or dtype fields. Describe "
                "units and other meaning in description. Preserve existing variable files unless the user "
                "asks to replace them. NPY variables may contain numeric, Boolean, or string values. Preserve "
                "categorical and textual variables as Unicode string arrays rather than object arrays, unless "
                "the user explicitly requests an encoding such as one-hot encoding. Call load_context_data "
                "after writing or editing the collection; use "
                "all reported errors to repair the manifest before claiming completion. commit_data remains "
                "available for a simple finite numeric CSV/Excel table.\n\n"
                "Explain material assumptions and sources. A later request may extend the existing data, so "
                "retain and use this conversation and all workspace artifacts."
            ),
        }]
        self.initialize_tools(context)

    def reset_stop(self) -> None:
        """Clear a previous stop request before starting another user turn."""
        self._stop_requested.clear()

    def request_stop(self) -> None:
        """Request cooperative cancellation at the next safe boundary."""
        self._stop_requested.set()

    def _check_stop(self) -> None:
        if self._stop_requested.is_set():
            raise InterruptedError("Data preparation was stopped by the user")

    def initialize_tools(self, context: AgentContext) -> None:
        """Bind configured skills before constructing context-aware tools.

        Args:
            context: Shared agent and tool context.
        """
        context.args.enabled_skills = self.skills
        super().initialize_tools(context)

    def run(self, instruction: str) -> dict[str, Any]:
        """Continue the persistent preparation conversation until it yields control.

        Args:
            instruction: Natural-language instruction for the agent.

        Returns:
            dict[str, Any]: The operation result.
        """
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("instruction must not be empty")
        self.context.args.llm_provider = self.llm_provider
        self.context.args.llm_model = self.llm_model
        self.context.args.llm_max_tokens = self.llm_max_tokens
        self.buffer.append({"role": "user", "content": instruction})
        self._publish("data_user", {"content": instruction})
        committed = False
        final_content = ""
        while True:
            self._check_stop()
            self.turn_count += 1
            turn = self.turn_count
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
                self._check_stop()
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
                self._check_stop()
            except InterruptedError:
                raise
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
            self._check_stop()
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
            self._check_stop()
        return results

    def _publish(self, kind: str, payload: Any) -> None:
        self.interaction_manager.publish(kind, payload)
