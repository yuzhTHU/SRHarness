"""Restricted agent for constructing and repairing custom evaluators."""
from __future__ import annotations

from typing import Any

from ..api import BaseAPI
from ..core import AgentContext, json_value
from ..parser import BaseParser
from ..tools import BaseTool
from ..utils import ParallelTimer
from .agent import Agent


class EvaluatorConstructionAgent(Agent):
    """Construct a BaseEvaluator subclass with only update and evaluation tools."""

    def __init__(self, *, llm_provider: str, llm_model: str, context: AgentContext, tools: list[BaseTool], current_source: str, tool_parser: str = "openai", llm_max_tokens: int = 4096, max_steps: int = 8):
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.tool_parser = tool_parser
        self.llm_max_tokens = llm_max_tokens
        self.max_steps = max_steps
        self.context = context
        self.tools = tools
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
                "You construct restricted SRHarness BaseEvaluator subclasses. You may only update "
                "the custom evaluator through update_evaluator and test it through evaluate_formula. "
                "You have no filesystem, shell, network, or arbitrary code-execution tools. "
                "BaseEvaluator defines static abstract fit(expression, context, target), "
                "evaluate(expression, context, target), and split_data(context). DefaultEvaluator "
                "is the basic implementation. fit must return an Expression with every param bound; "
                "evaluate returns a flat metrics dictionary containing complexity and the ranking "
                "metric. BaseEvaluator exposes individual calc_mse, calc_rmse, calc_mae, calc_mape, "
                "calc_r2, calc_aic, calc_bic, calc_pearson_r, calc_spearman_r, and calc_complexity "
                "helpers. Extract only necessary control values from context.args before passing "
                "them to other helpers. Test changes when data are available and repair failures "
                "before answering.\n\n"
                f"Current source:\n```python\n{current_source}\n```"
            ),
        }]

    def run(self, instruction: str) -> dict[str, Any]:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Evaluator construction instruction must not be empty")
        self.buffer.append({"role": "user", "content": instruction})
        tool_events: list[dict[str, Any]] = []
        for _ in range(self.max_steps):
            call_result = self.api(
                self.buffer, n=1, max_tokens=self.llm_max_tokens,
            )
            rows = list(call_result)
            if not rows:
                raise RuntimeError("EvaluatorConstructionAgent returned no usable response")
            content, calls, message = rows[0]
            self.buffer.append(message)
            if not calls:
                return {"message": content or "", "tool_events": tool_events}
            results = self.execute_action(calls)
            tool_events.extend({
                "tool": call.name,
                "ok": result.ok,
                "result": json_value(result.result),
            } for call, result in zip(calls, results))
            self.buffer.extend(self.parser.format_tool_result_messages(calls, results))
        raise RuntimeError(
            f"EvaluatorConstructionAgent exceeded the {self.max_steps}-step repair limit"
        )
