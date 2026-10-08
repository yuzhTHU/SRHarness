"""Restricted agent for constructing and repairing custom evaluators."""
from __future__ import annotations

import json
import math
import numbers
import threading
import time
import uuid
from typing import Any, Callable

from ..api import BaseAPI
from ..core import AgentContext, json_value
from ..parser import BaseParser
from ..tools import BaseTool
from ..utils import ParallelTimer
from .agent import Agent


class EvaluatorConstructionAgent(Agent):
    """Construct and validate evaluator scripts without mutating live context state."""

    def __init__(self, *, llm_provider: str, llm_model: str, context: AgentContext, tools: list[BaseTool], tool_parser: str = "openai", llm_max_tokens: int = 4096, max_steps: int = 16, event_callback: Callable[[str, dict[str, Any]], Any] | None = None):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.tool_parser = tool_parser
        self.llm_max_tokens = llm_max_tokens
        self.max_steps = max_steps
        self.context = context
        self.event_callback = event_callback
        self._stop_requested = threading.Event()
        self._pause_requested = threading.Event()
        self._force_recorded = False
        self.tools = tools
        for tool in self.tools:
            tool.cancel_event = self._stop_requested
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
        if self.event_callback is not None:
            self.event_callback(kind, payload)

    def request_stop(self, *, force: bool = False) -> None:
        """Pause after this turn, or force-interrupt its active operation."""
        self._pause_requested.set()
        if force:
            self._stop_requested.set()

    def _check_stop(self) -> None:
        if self._stop_requested.is_set():
            raise InterruptedError("用户强制中止")

    def _check_pause(self) -> None:
        if self._pause_requested.is_set():
            raise InterruptedError("Evaluator construction paused at a safe boundary")

    def _record_force(self, update: dict[str, Any] | None = None) -> None:
        if self._force_recorded:
            return
        if update and (update.get("content") or update.get("reasoning")):
            message = {"role": "assistant", "content": update.get("content", "")}
            if update.get("reasoning"):
                message["reasoning"] = update["reasoning"]
            self.buffer.append(message)
        self.buffer.append({"role": "user", "content": "用户强制中止"})
        self._force_recorded = True

    def run(self, instruction: str) -> dict[str, Any]:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Evaluator construction instruction must not be empty")
        self.buffer.append({"role": "user", "content": instruction})
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
            self._publish("evaluator_context", {
                "messages": json_value(self.buffer),
                "turn": step,
            })
            self._publish("evaluator_assistant_start", event_context)
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
                self._publish("evaluator_assistant_delta", {**update, **event_context})

            try:
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
                self._publish("evaluator_assistant_error", {
                    **latest_stream_update,
                    **event_context,
                    "error": "用户强制中止",
                })
                raise
            except Exception as exc:
                self._publish("evaluator_assistant_error", {
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
            self._publish("evaluator_assistant", {
                **event_context,
                "content": content or "",
                "reasoning": message.get("reasoning", ""),
                "message": json_value(message),
                "tool_calls": assistant_event["tool_calls"],
                "usage": call_result.returned.get("usage", {"token": {}, "price": {}}),
            })
            if not calls:
                if update_needs_test:
                    self.buffer.append({
                        "role": "user",
                        "content": (
                            "Completion is blocked: call validate_evaluator after the latest evaluator "
                            "file edit. Repair the evaluator until validation succeeds and every "
                            "evaluator-specific metric it returns is a finite number."
                        ),
                    })
                    self._check_pause()
                    continue
                self._check_pause()
                return {
                    "message": content or "",
                    "tool_events": tool_events,
                    "timeline_events": timeline_events,
                    "streamed": self.event_callback is not None,
                }
            results = []
            for call in calls:
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
                self._publish("evaluator_tool_result", event)
            self.buffer.extend(self.parser.format_tool_result_messages(calls, results))
            if self._stop_requested.is_set():
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
            self._check_pause()
        raise RuntimeError(
            f"EvaluatorConstructionAgent exceeded the {self.max_steps}-step repair limit"
        )
