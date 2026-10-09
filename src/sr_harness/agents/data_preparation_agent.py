# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Persistent tool-using agent for preparing structured research data."""
from __future__ import annotations

import time
import uuid
from copy import deepcopy
from typing import Any

from ..core import AgentContext
from ..runtime import InteractionManager
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
        "validate_context_data",
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
        interaction_manager: InteractionManager,
    ):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.tool_parser = tool_parser
        self.llm_max_tokens = llm_max_tokens
        self.turn_count = 0
        if hasattr(context.args, "skill_manager"):
            skill_manager = context.args.skill_manager
            if not isinstance(skill_manager, SkillManager):
                raise TypeError("context.args.skill_manager must be a SkillManager")
        else:
            skill_manager = SkillManager()
            context.args.skill_manager = skill_manager
        self.skills = (
            list(skills)
            if skills is not None
            else [
                name for name in skill_manager.load_skills()
                if name not in self.DEFAULT_EXCLUDED_SKILLS
            ]
        )
        if not isinstance(interaction_manager, InteractionManager):
            raise TypeError("interaction_manager must be an InteractionManager")
        self.interaction_manager = interaction_manager
        self.tool_cls_list = BaseTool.load_tool_classes(tools or self.DEFAULT_TOOLS)
        self.tools_counter = ParallelTimer(unit="call")
        self.tools = None
        self.parser = None
        self.api = None
        self.context = context
        self._force_recorded = False
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
                "Mark every edge-list or hyperedge-list variable itself with kind: relation; relation arrays "
                "must contain integer endpoint indices and require num_nodes at the manifest root. "
                "For each variable whose final dimension depends on an edge list A or hyperedge list T, "
                "set that variable's structure field to the name of A or T. The referenced relation must "
                "be an integer (E, 2) or (H, 3) array whose endpoints are in [0, num_nodes), and the "
                "dependent variable must have shape (..., E) or (..., H). Do not add structure to A or T "
                "itself, and never infer a relation from a variable name. Prefer explicit values such as "
                "['target', 'source'] for short semantic axes instead of an anonymous size. "
                "Do not add format, version, revision, problem, attributes, shape, or dtype fields. Describe "
                "units and other meaning in description. Preserve existing variable files unless the user "
                "asks to replace them. NPY variables may contain numeric, Boolean, or string values. Preserve "
                "categorical and textual variables as Unicode string arrays rather than object arrays, unless "
                "the user explicitly requests an encoding such as one-hot encoding. Create and transform "
                "context.data with workspace_code_executor, then call validate_context_data after every "
                "write or edit. Use every reported error and repair action before claiming completion. "
                "This validation "
                "does not mutate the live AgentContext; InteractiveSession loads valid files after your "
                "turn completes.\n\n"
                "Explain material assumptions and sources. A later request may extend the existing data, so "
                "retain and use this conversation and all workspace artifacts."
            ),
        }]
        self.initialize_tools(context)

    def _check_stop(self) -> None:
        if self.interaction_manager.is_interrupting:
            raise InterruptedError("用户强制中止")

    def _wait(self) -> bool:
        with self.interaction_manager.wait() as messages:
            for pending in messages:
                self._append_prompt({"role": "user", "content": pending.content})
        return bool(messages)

    def _record_force(self, update: dict[str, Any] | None = None) -> None:
        if self._force_recorded:
            return
        if update and (update.get("content") or update.get("reasoning")):
            message = {"role": "assistant", "content": update.get("content", "")}
            if update.get("reasoning"):
                message["reasoning"] = update["reasoning"]
            self.buffer.append(message)
        self._append_prompt({"role": "user", "content": "用户强制中止"})
        self._force_recorded = True

    def _append_prompt(self, message: dict[str, Any]) -> None:
        self.buffer.append(message)
        if message.get("role") in {"system", "user"}:
            self.interaction_manager.publish_event("prompt_added", {"message": message})

    def initialize_tools(self, context: AgentContext) -> None:
        """Bind configured skills before constructing context-aware tools.

        Args:
            context: Shared agent and tool context.
        """
        context.args.enabled_skills = self.skills
        super().initialize_tools(context)
        for tool in self.tools:
            tool.cancel_event = self.interaction_manager.cancellation_signal

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
        if not any(event["payload"].get("message") == self.buffer[0] for event in self.interaction_manager.get_recent_events()["events"] if event["kind"] == "prompt_added"):
            self.interaction_manager.publish_event("prompt_added", {"message": self.buffer[0]})
        self._append_prompt({"role": "user", "content": instruction})
        committed = False
        final_content = ""
        while True:
            self._check_stop()
            self.turn_count += 1
            turn = self.turn_count
            prompt = deepcopy(self.buffer)
            self.set_messages(prompt)
            self._publish("context", {"messages": prompt, "turn": turn})
            response_id = f"data:{uuid.uuid4().hex}"
            tool_schemas = {
                tool.metadata.name: {
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.parameters or {},
                }
                for tool in self.tools
            }
            self._publish("assistant_started", {
                "response_id": response_id,
                "turn": turn,
                "tool_schemas": tool_schemas,
                "provider": self.llm_provider,
                "model": self.llm_model,
            })
            last_stream_emit = 0.0
            latest_stream_update: dict[str, Any] = {}

            def stream_callback(update: dict[str, Any]) -> None:
                nonlocal last_stream_emit, latest_stream_update
                latest_stream_update = update
                self._check_stop()
                now = time.monotonic()
                if update.get("type") == "delta" and now - last_stream_emit < 0.08:
                    return
                last_stream_emit = now
                self._publish("assistant_delta", {
                    **update,
                    "response_id": response_id,
                    "turn": turn,
                    "tool_schemas": tool_schemas,
                    "provider": self.llm_provider,
                    "model": self.llm_model,
                })

            try:
                with self.interaction_manager.cancellable(self.api.cancel):
                    call_result = self.api(
                        prompt,
                        n=1,
                        max_tokens=self.llm_max_tokens,
                        stream_callback=stream_callback,
                    )
                responses = list(call_result)
                self._check_stop()
            except InterruptedError:
                self._record_force(latest_stream_update)
                self._publish("assistant_failed", {
                    **latest_stream_update,
                    "response_id": response_id,
                    "turn": turn,
                    "provider": self.llm_provider,
                    "model": self.llm_model,
                    "error": "用户强制中止",
                    "interrupted": True,
                })
                self._wait()
                continue
            except Exception as exc:
                self._publish("assistant_failed", {
                    **latest_stream_update,
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
            self._publish("assistant_completed", {
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
                if self._wait():
                    continue
                break
            results = self._execute_with_events(calls)
            self.buffer.append(message)
            self.buffer.extend(self.parser.format_tool_result_messages(calls, results))
            committed = committed or any(result.result.get("data_committed") for result in results)
            if self.interaction_manager.is_interrupting:
                self._record_force()
                self._check_stop()
            self._wait()
        result = {
            "message": final_content,
            "committed": committed,
            "context": self.context.schema(),
        }
        self._publish("execution_completed", result)
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
            self._publish("tool_started", {"call": call, "tool_schema": schema})
            started_at = time.monotonic()
            tool = next((item for item in self.tools if item.metadata.name == call.name), None)
            if tool is None:
                result = super().execute_action([call])[0]
            else:
                with self.interaction_manager.cancellable(tool.cancel):
                    result = super().execute_action([call])[0]
            self._publish("tool_completed", {
                "call": call,
                "tool_schema": schema,
                "result": result,
                "duration_seconds": time.monotonic() - started_at,
                "interrupted": self.interaction_manager.is_interrupting,
            })
            results.append(result)
        return results

    def _publish(self, kind: str, payload: Any) -> None:
        self.interaction_manager.publish_event(kind, payload)
