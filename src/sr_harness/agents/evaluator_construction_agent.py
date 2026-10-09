"""Restricted agent for constructing and repairing custom evaluators."""
from __future__ import annotations

import json
import math
import numbers
import time
import uuid
from typing import Any

from ..api import BaseAPI
from ..core import AgentContext, json_value
from ..parser import BaseParser
from ..runtime import InteractionManager
from ..tools import BaseTool
from ..utils import ParallelTimer
from .agent import Agent


class EvaluatorConstructionAgent(Agent):
    """Construct and validate evaluator scripts without mutating live context state."""

    def __init__(self, *, llm_provider: str, llm_model: str, context: AgentContext, tools: list[BaseTool], interaction_manager: InteractionManager, tool_parser: str = "openai", llm_max_tokens: int = 4096, max_steps: int = 16) -> None:
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.tool_parser = tool_parser
        self.llm_max_tokens = llm_max_tokens
        self.max_steps = max_steps
        self.context = context
        if not isinstance(interaction_manager, InteractionManager):
            raise TypeError("interaction_manager must be an InteractionManager")
        self.interaction_manager = interaction_manager
        self._force_recorded = False
        self.tools = tools
        for tool in self.tools:
            tool.cancel_event = self.interaction_manager.cancellation_signal
        self.tools_counter = ParallelTimer(unit="call")
        self.parser = BaseParser.create(tool_parser, tool_list=tools)
        self.api = BaseAPI.create(
            llm_provider,
            model=llm_model,
            tool_list=tools,
            tool_parser_name=tool_parser,
        )
        self.buffer: list[dict[str, Any]] = [{
            "role": "system",
            "content": (
                "You construct SRHarness DefaultEvaluator subclasses using only the tools and skills "
                "enabled for this session. Create evaluator scripts under context.evaluator/ with "
                "workspace_code_executor, inspect workspace files with workspace_shell, and test scripts "
                "with validate_evaluator. Never mutate the live AgentContext evaluator; InteractiveSession "
                "alone loads the selected file. Consult read_skill when useful. Use read_source to inspect "
                "src/sr_harness/evaluator/ and only the implementation symbols needed for the task. "
                "Keep custom subclasses short and reuse evaluator.utils infrastructure. "
                "DefaultEvaluator defines class methods split(context), fit(f, y, context), "
                "evaluate(f, y, context), fit_candidate(f, context), and "
                "evaluate_candidate(f, context). Override fit/evaluate to affect all equations, or "
                "override the candidate-only methods for behavior such as ODE trajectory rollouts. "
                "fit must return an Expression with every parameter bound. evaluate must return a "
                "flat dictionary whose values are only float or int. Preserve mse, rmse, mae, mape, r2, aic, "
                "bic, pearson_r, spearman_r, and complexity unless the user explicitly asks to remove "
                "one. Metric, split, "
                "index-selection, and ODE helpers live in `sr_harness.evaluator.utils`. New evaluator-specific "
                "helpers belong on the custom subclass. Never hide metric failures in broad "
                "try/except blocks or return placeholders such as strings or NaN: evaluate_formula "
                "must exercise the new metric and return a finite numeric value before you finish. "
                "Extract only necessary control values from context.args before passing them to other "
                "helpers. Test changes when data are available and repair failures before answering.\n\n"
                "The current dataset schema below is authoritative. Use its variable names, shapes, "
                "axis relationships, target, and descriptions when implementing dataset-specific "
                "metrics such as trajectory rollouts. A derivative target and its corresponding "
                "observed state trajectory are distinct arrays.\n\n"
                "For a trajectory rollout metric, preserve chronological order in split "
                "instead of applying a random row split, integrate the fitted RHS from the first "
                "observed state of each split, and compare the integrated state with the observed "
                "state. Prefer scipy.integrate.solve_ivp with t_eval over a low-accuracy Euler loop. "
                "candidate fit/evaluate receive an already-split context: do not call split from "
                "either method and do not refit inside evaluate because fit has already run. "
                "split should return chronological AgentContext views where practical.\n\n"
                "Before implementing numerical infrastructure yourself, inspect "
                "src/sr_harness/evaluator/utils/.\n\n"
                f"Current dataset schema:\n```json\n"
                f"{json.dumps(json_value(context.schema()), ensure_ascii=False, indent=2)}\n```\n\n"
                "Begin by inspecting src/sr_harness/evaluator/. If context.evaluator/ already exists, "
                "inspect its scripts too. Do not create that directory merely to inspect the workspace; "
                "create it only immediately before writing the first custom evaluator script."
            ),
        }]

    def _publish(self, kind: str, payload: dict[str, Any]) -> None:
        self.interaction_manager.publish_event(kind, payload)

    def _check_stop(self) -> None:
        if self.interaction_manager.is_interrupting:
            raise InterruptedError("用户强制中止")

    def _wait(self) -> bool:
        with self.interaction_manager.wait() as messages:
            for pending in messages:
                self._append_prompt({"role": "user", "content": pending.content})
        return bool(messages)

    def _append_prompt(self, message: dict[str, Any]) -> None:
        self.buffer.append(message)
        if message.get("role") in {"system", "user"}:
            self._publish("prompt_added", {"message": message})

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

    def run(self, instruction: str) -> dict[str, Any]:
        """Construct or repair a custom evaluator for one user instruction.

        Args:
            instruction: Natural-language evaluator requirements.

        Returns:
            The final assistant response and tool-event records produced during
            construction.

        Raises:
            ValueError: If ``instruction`` is empty.
        """
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Evaluator construction instruction must not be empty")
        if not any(event["kind"] == "prompt_added" for event in self.interaction_manager.get_recent_events()["events"]):
            self._publish("prompt_added", {"message": self.buffer[0]})
        self._append_prompt({"role": "user", "content": instruction})
        tool_events: list[dict[str, Any]] = []
        timeline_events: list[dict[str, Any]] = []
        update_needs_test = False
        require_custom_metric = any(
            marker in instruction.lower()
            for marker in ("metric", "rmse", "指标", "评测结果")
        )

        def test_has_finite_custom_metrics(result: Any) -> bool:
            if not result.ok or not isinstance(result.result, dict):
                return False
            standard = {
                "mse", "rmse", "mae", "mape", "r2", "aic", "bic",
                "pearson_r", "spearman_r", "complexity",
            }
            split_results = result.result.get("data_split_results", {})
            custom_values = []
            for split in split_results.values():
                metrics = split.get("metrics", {}) if isinstance(split, dict) else {}
                if not standard <= set(metrics):
                    return False
                custom_values.extend(
                    value for name, value in metrics.items() if name not in standard
                )
            return (not require_custom_metric or bool(custom_values)) and all(
                isinstance(value, numbers.Real)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in custom_values
            )

        for step in range(1, self.max_steps + 1):
            self._check_stop()
            response_id = f"evaluator:{uuid.uuid4().hex}"
            tool_schemas = {
                tool.metadata.name: {
                    "description": tool.metadata.description,
                    "parameters": tool.metadata.parameters or {},
                }
                for tool in self.tools
            }
            event_context = {
                "response_id": response_id,
                "turn": step,
                "tool_schemas": tool_schemas,
                "provider": self.llm_provider,
                "model": self.llm_model,
            }
            self._publish("context", {
                "messages": json_value(self.buffer),
                "turn": step,
            })
            self._publish("assistant_started", event_context)
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
                self._publish("assistant_delta", {**update, **event_context})

            try:
                with self.interaction_manager.cancellable(self.api.cancel):
                    call_result = self.api(
                        self.buffer,
                        n=1,
                        max_tokens=self.llm_max_tokens,
                        stream_callback=stream_callback,
                    )
                rows = list(call_result)
                self._check_stop()
            except InterruptedError:
                self._record_force(latest_stream_update)
                self._publish("assistant_failed", {
                    **latest_stream_update,
                    **event_context,
                    "error": "用户强制中止",
                    "interrupted": True,
                })
                self._wait()
                continue
            except Exception as exc:
                self._publish("assistant_failed", {
                    **latest_stream_update,
                    **event_context,
                    "error": str(exc),
                })
                raise
            if not rows:
                raise RuntimeError("EvaluatorConstructionAgent returned no usable response")
            content, calls, message = rows[0]
            self.buffer.append(message)
            assistant_event = {
                "kind": "assistant",
                "timestamp": time.time(),
                "content": content or "",
                "message": json_value(message),
                "tool_calls": [
                    {"name": call.name, "params": json_value(call.params), "id": call.id}
                    for call in calls
                ],
            }
            timeline_events.append(assistant_event)
            self._publish("assistant_completed", {
                **event_context,
                "content": content or "",
                "reasoning": message.get("reasoning", ""),
                "message": json_value(message),
                "tool_calls": assistant_event["tool_calls"],
                "usage": call_result.returned.get("usage", {"token": {}, "price": {}}),
            })
            if not calls:
                if update_needs_test:
                    self._append_prompt({
                        "role": "user",
                        "content": (
                            "Completion is blocked: call validate_evaluator after the latest evaluator "
                            "file edit. Repair the evaluator until validation succeeds and every "
                            "evaluator-specific metric it returns is a finite number."
                        ),
                    })
                    self._wait()
                    continue
                if self._wait():
                    continue
                return {
                    "message": content or "",
                    "tool_events": tool_events,
                    "timeline_events": timeline_events,
                    "streamed": True,
                }
            results = []
            for call in calls:
                tool = next((item for item in self.tools if item.metadata.name == call.name), None)
                self._publish("tool_started", {
                    "call": {"name": call.name, "params": json_value(call.params), "id": call.id},
                })
                if tool is None:
                    results.extend(self.execute_action([call]))
                else:
                    with self.interaction_manager.cancellable(tool.cancel):
                        results.extend(self.execute_action([call]))
            completed = [{
                "tool": call.name,
                "ok": result.ok,
                "call": {"name": call.name, "params": json_value(call.params), "id": call.id},
                "result": {
                    "ok": result.ok,
                    "result": json_value(result.result),
                    "result_str": result.result_str,
                    "meta_data": json_value(result.meta_data),
                },
                "timestamp": float(result.meta_data.get("timestamp", time.time())),
            } for call, result in zip(calls, results)]
            tool_events.extend(completed)
            timeline_events.extend({"kind": "tool_result", **event} for event in completed)
            for event in completed:
                self._publish("tool_completed", event)
            self.buffer.extend(self.parser.format_tool_result_messages(calls, results))
            if self.interaction_manager.is_interrupting:
                self._record_force()
                self._check_stop()
            for call, result in zip(calls, results):
                if call.name == "workspace_code_executor" and result.ok:
                    update_needs_test = True
                elif (
                    call.name == "validate_evaluator"
                    and update_needs_test
                    and test_has_finite_custom_metrics(result)
                ):
                    update_needs_test = False
            self._wait()
        raise RuntimeError(
            f"EvaluatorConstructionAgent exceeded the {self.max_steps}-step repair limit"
        )
