import argparse
import json

import numpy as np
import pytest

from sr_harness._vendor.llmsr_bench.core import SEDTask
from sr_harness._vendor.llmsr_bench.algorithms import sr_harness
from sr_harness._vendor.llmsr_bench.algorithms.sr_harness import update_parser


def test_default_benchmark_tools_exclude_self_modification():
    parser = update_parser(argparse.ArgumentParser())
    tools = parser.get_default("tools")
    assert parser.get_default("split_by") == "random"
    assert "edit_tool" not in tools
    assert "harmonic_interaction_fit" not in tools
    tools_action = next(action for action in parser._actions if action.dest == "tools")
    assert "harmonic_interaction_fit" not in tools_action.choices


def test_failed_agent_with_formula_is_not_reported_as_success(monkeypatch, tmp_path):
    class FakeAgent:
        def __init__(self, **kwargs):
            self.named_timer = argparse.Namespace(named_time={})
            self.token_counter = argparse.Namespace(count=100)
            self.money_counter = argparse.Namespace(count=0.01)
            self.tools_counter = argparse.Namespace(named_count={})

        def run(self, **kwargs):
            return {
                "status": "failed",
                "best_formula": "x",
                "error": "API credits exhausted",
            }

    monkeypatch.setattr(sr_harness, "SRAgent", FakeAgent)
    monkeypatch.setattr(
        sr_harness,
        "_logger",
        argparse.Namespace(note=lambda *args: None, error=lambda *args: None),
    )
    parser = update_parser(argparse.ArgumentParser())
    args = parser.parse_args([])
    args.save_path = str(tmp_path)
    args.verbose = False
    args.debug = False
    task = SEDTask(
        name="test_problem",
        symbols=["y", "x"],
        symbol_descs=["target", "input"],
        symbol_properties=["O", "V"],
        train_X=np.array([[1.0], [2.0]]),
        train_y=np.array([1.0, 2.0]),
    )

    with pytest.raises(RuntimeError, match="API credits exhausted"):
        sr_harness.run(args, task)

    result_path = next(tmp_path.glob("experiments/*/result.jsonl"))
    saved_result = json.loads(result_path.read_text(encoding="utf-8").splitlines()[-1])
    assert saved_result["status"] == "failed"
    assert saved_result["best_formula"] == "x"
