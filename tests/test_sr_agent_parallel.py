from __future__ import annotations

import json
import numpy as np
import pytest
from types import SimpleNamespace

from sr_harness.core import (
    AgentContext,
    CandidateRecord,
    ToolCall,
    ToolCallResult,
    ToolMetadata,
)
from sr_harness.agents.sr_agent import SRAgent
from sr_harness.agents.sr_agent_interactive import SRAgentInteractive
from sr_harness.runtime import SRInteractionManager
from sr_harness.tools.base_tool import BaseTool
from sr_harness.tools.read_skill import ReadSkill
from sr_harness.tools.relationship_analysis import RelationshipAnalysisTool
from sr_harness.tools.statistics_analysis import StatisticsTool


@BaseTool.register("unit_parallel_tool")
class UnitParallelTool(BaseTool):
    metadata = ToolMetadata(name="unit_parallel_tool")

    def execute(self, value: int) -> dict:
        """Return a value with inherited context."""
        return {"value": value, "offset": self.context.args.offset}


@BaseTool.register("unit_messages_tool")
class UnitMessagesTool(BaseTool):
    metadata = ToolMetadata(name="unit_messages_tool")

    def execute(self) -> dict:
        """Return injected messages."""
        return {"messages": self.context.args.messages}


def make_agent(tmp_path):
    agent = SRAgent(
        llm_provider="unused",
        llm_model="unused",
        tools=["unit_parallel_tool"],
        save_path=str(tmp_path),
    )
    agent.tools = [UnitParallelTool(offset=10)]
    return agent


def test_interactive_agent_reuses_shared_run_loop():
    assert "run" not in SRAgentInteractive.__dict__
    assert SRAgentInteractive.run is SRAgent.run
    assert "create_initial_buffer" not in SRAgentInteractive.__dict__
    assert SRAgentInteractive.create_initial_buffer is SRAgent.create_initial_buffer
    assert "create_initial_prompt_messages" not in SRAgentInteractive.__dict__
    assert "prepare_model_messages" not in SRAgentInteractive.__dict__
    assert "update_conversation" not in SRAgentInteractive.__dict__
    assert not hasattr(SRAgent, "fit")


def test_interactive_initial_buffer_uses_narrow_customization_hooks(tmp_path):
    manager = SRInteractionManager()
    agent = SRAgentInteractive(
        llm_provider="unused",
        llm_model="unused",
        tools=[],
        skills=[],
        save_path=str(tmp_path),
        interaction_manager=manager,
    )
    agent.tools = []
    agent.variable_descriptions = {"x": "input", "y": "output"}

    messages = agent.create_initial_buffer(
        "Find y from x.", {"x": np.arange(3)}, {"y": np.arange(3)}, [],
    )

    assert "working with a human researcher" in messages[0]["content"]
    assert "- x: input" in messages[1]["content"]
    events = manager.get_recent_events()["events"]
    assert [event["payload"]["message"]["role"] for event in events] == [
        "system", "user", "user",
    ]


def test_initial_prompts_can_be_created_without_progress_or_buffer_events(tmp_path):
    manager = SRInteractionManager()
    agent = SRAgentInteractive(
        llm_provider="unused",
        llm_model="unused",
        tools=[],
        skills=[],
        save_path=str(tmp_path),
        interaction_manager=manager,
    )
    agent.variable_descriptions = {"x": "input", "y": "output"}

    messages = agent.create_initial_prompt_messages(
        "Find y from x.", {"x": np.arange(3)}, {"y": np.arange(3)}, [],
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "- x: input" in messages[1]["content"]
    assert manager.get_recent_events()["events"] == []


def test_interactive_guidance_is_added_before_prompt_construction():
    agent = object.__new__(SRAgentInteractive)
    agent.interaction_manager = SRInteractionManager()
    agent.interaction_manager.start_agent_execution()
    agent.interaction_manager.command("message", "compare against a power law")
    agent.runtime_settings_supplier = None
    agent.initial_messages = []
    agent.refresh_data = lambda *args, **kwargs: False
    buffer = [{"role": "user", "content": "Find a formula."}]

    agent.prepare_iteration(buffer, R=1, L=1, C=1)

    assert buffer[-1] == {
        "role": "user",
        "content": (
            "[Human guidance injected during the run]\n"
            "compare against a power law"
        ),
    }


def test_interactive_refreshes_data_before_queued_human_guidance():
    agent = object.__new__(SRAgentInteractive)
    agent.interaction_manager = SRInteractionManager()
    agent.interaction_manager.start_agent_execution()
    agent.interaction_manager.command("message", "continue with the new data")
    agent.runtime_settings_supplier = None
    agent.initial_messages = []
    agent.refresh_data = lambda: {
        "previous_revision": 3,
        "revision": 4,
        "target": "y",
        "features": ["x", "z"],
        "variable_descriptions": {"z": "new feature"},
    }
    buffer = []

    agent.prepare_iteration(buffer, R=1, L=2, C=1)

    assert "Data revision changed from 3 to 4" in buffer[0]["content"]
    assert "- z: new feature" in buffer[0]["content"]
    assert buffer[1]["content"].endswith("continue with the new data")
    events = agent.interaction_manager.get_recent_events()["events"]
    assert [event["kind"] for event in events].count("prompt_added") == 2
    assert all(event["kind"] != "data_revision" for event in events)


def test_interactive_agent_requests_pause_after_tool_free_response():
    agent = object.__new__(SRAgentInteractive)
    agent._last_iteration_had_tool_calls = False
    agent._forced_interruption_pending = False
    agent.best_candidate = lambda: None
    agent.interaction_manager = SRInteractionManager()
    agent.interaction_manager.start_agent_execution()
    buffer = [{"role": "assistant", "content": "I need more direction."}]

    status = agent.finish_iteration(buffer, R=1, L=2, C=1)

    assert status is None
    assert agent.interaction_manager.state == "pausing"
    assert buffer == [{"role": "assistant", "content": "I need more direction."}]


def test_interactive_agent_does_not_special_case_zero_mse_after_tool_call():
    agent = object.__new__(SRAgentInteractive)
    agent._last_iteration_had_tool_calls = True
    agent._forced_interruption_pending = False
    agent.best_candidate = lambda: SimpleNamespace(metric=lambda name, split: 0.0)
    agent.interaction_manager = SRInteractionManager()
    agent.interaction_manager.start_agent_execution()
    buffer = []

    assert agent.finish_iteration(buffer, R=1, L=2, C=1) is None
    assert agent.interaction_manager.state == "running"
    assert buffer == []


def test_web_tool_free_response_pauses_without_asking_a_question():
    agent = object.__new__(SRAgentInteractive)
    agent._last_iteration_had_tool_calls = False
    agent._forced_interruption_pending = False
    agent.best_candidate = lambda: None
    agent.interaction_manager = SRInteractionManager()
    agent.interaction_manager.start_agent_execution()

    assert agent.finish_iteration([], R=1, L=2, C=1) is None
    assert agent.interaction_manager.state == "pausing"


def test_safe_pause_does_not_block_tools_already_requested_this_turn(monkeypatch):
    agent = object.__new__(SRAgentInteractive)
    agent.tools = []
    agent.interaction_manager = SRInteractionManager()
    agent.interaction_manager.start_agent_execution()
    result = ToolCallResult(ok=True, result={"mse": 0.0}, result_str="MSE=0", meta_data={})
    monkeypatch.setattr(SRAgent, "execute_action", lambda self, actions: [result])

    returned = agent.execute_action([ToolCall("submit_formula", {"f": "x"}, id="call-1")])

    assert returned == [result]
    events = agent.interaction_manager.get_recent_events()["events"]
    assert [event["kind"] for event in events][-2:] == ["tool_started", "tool_completed"]


def test_interactive_agent_honors_pending_control_before_automatic_guidance():
    agent = object.__new__(SRAgentInteractive)
    agent._last_iteration_had_tool_calls = False
    agent._forced_interruption_pending = False
    agent.best_candidate = lambda: None
    agent.interaction_manager = SRInteractionManager()
    agent.interaction_manager.start_agent_execution()
    agent.interaction_manager.command("pause")
    buffer = []

    assert agent.finish_iteration(buffer, R=1, L=2, C=1) is None
    assert agent.interaction_manager.state == "pausing"
    assert buffer == []


def test_agent_exposes_only_skills_from_enabled_tools(tmp_path):
    enabled = SRAgent(
        llm_provider="unused",
        llm_model="unused",
        tools=["read_skill", "workspace_shell"],
        save_path=str(tmp_path / "enabled"),
    )
    disabled = SRAgent(
        llm_provider="unused",
        llm_model="unused",
        tools=["read_skill"],
        save_path=str(tmp_path / "disabled"),
    )

    enabled_reader = ReadSkill(skill_manager=enabled.skill_manager)
    disabled_reader = ReadSkill(skill_manager=disabled.skill_manager)

    runtime_skill = enabled_reader.skill_manager.load_skills()["workspace-shell"]
    skill_path = runtime_skill.skill_directory / "SKILL.md"
    assert skill_path.name == "SKILL.md"
    assert skill_path.parent.name == "workspace-shell"
    assert skill_path.read_text(encoding="utf-8") == enabled_reader.skill_manager.read_skill("workspace-shell")
    assert "workspace-shell" not in disabled_reader.skill_manager.load_skills()

    assert "<name>workspace-shell</name>" in enabled_reader.metadata.description
    assert "<name>workspace-shell</name>" not in disabled_reader.metadata.description
    result = enabled_reader(name="workspace-shell", file_path="SKILL.md", show_tree=True)
    assert result.ok is True
    assert "Supported commands" in result.result_str
    assert "Skill directory tree:\n- SKILL.md" in result.result_str

    enabled.tools = [enabled_reader]
    parallel_result = enabled.execute_action_parallel(
        [ToolCall(name="read_skill", params={"name": "workspace-shell"})],
        max_workers=2,
    )[0]
    assert parallel_result.ok is True
    assert "# Workspace Shell" in parallel_result.result_str


def test_execute_action_parallel_preserves_order_and_records_usage(tmp_path):
    agent = make_agent(tmp_path)
    actions = [
        ToolCall(name="unit_parallel_tool", params={"value": 1}),
        ToolCall(name="missing_tool", params={}),
        ToolCall(name="unit_parallel_tool", params={"value": 2}),
    ]

    results = agent.execute_action_parallel(actions, max_workers=2)

    assert [result.ok for result in results] == [True, False, True]
    assert results[0].result == {"value": 1, "offset": 10}
    assert results[1].result_str == 'Unknown tool calling for "missing_tool"'
    assert results[2].result == {"value": 2, "offset": 10}
    assert agent.tools_counter.named_count == {"unit_parallel_tool": 2}


def test_execute_tool_calls_uses_messages_already_injected_into_tool_context(tmp_path):
    agent = make_agent(tmp_path)
    agent.tools = [UnitMessagesTool()]
    agent.max_workers = 2
    prompt = [{"role": "system", "content": "Reusable context"}]
    agent.tools[0].context.args.messages = prompt
    response_list = [
        (
            "",
            [
                ToolCall(name="unit_messages_tool", params={}),
                ToolCall(name="unit_messages_tool", params={}),
            ],
            {},
        )
    ]

    results = agent.execute_tool_calls(response_list, R=1, L=1, C=1)
    assert results[0][0].result == {"messages": [{"role": "system", "content": "Reusable context"}]}


def test_create_initial_buffer_includes_refinement_budget_rule(tmp_path):
    agent = make_agent(tmp_path)
    agent.max_refinement_depth = 7

    prompt = agent.create_initial_buffer(
        problem_description="Find y from x.",
        X={"x": [1, 2, 3]},
        y={"y": [2, 4, 6]},
        restart_records=[],
    )

    system_content = prompt[0]["content"]
    assert "at most 7 refinement rounds" in system_content
    assert "At the final refinement round (L=7)" in system_content
    assert "final-answer mechanism" in system_content


def test_force_initial_diagnostics_runs_and_injects_results(tmp_path):
    agent = SRAgent(
        llm_provider="unused",
        llm_model="unused",
        tools=["statistics_analysis", "relationship_analysis", "read_skill"],
        save_path=str(tmp_path),
        force_initial_diagnostics=True,
    )
    data = {
        "x": np.arange(1.0, 11.0),
        "y": 2 * np.arange(1.0, 11.0),
    }
    context = {
        "data": data,
        "target": "y",
        "skill_manager": agent.skill_manager,
    }
    agent.tools = [
        StatisticsTool(**context),
        RelationshipAnalysisTool(**context),
        ReadSkill(**context),
    ]

    buffer = [{"role": "user", "content": "Find y=f(x)."}]
    prompt = agent.prepare_model_messages(buffer, R=1, L=1, C=1)

    diagnostic_message = prompt[-1]["content"]
    assert prompt[-1]["role"] == "user"
    assert "[Required initial diagnostics]" in diagnostic_message
    assert "## statistics_analysis" in diagnostic_message
    assert "## relationship_analysis" in diagnostic_message
    assert "## read_skill" in diagnostic_message
    assert "<skill_content name=\"discover-symbolic-laws\">" in diagnostic_message
    assert buffer[-1] == prompt[-1]
    assert agent.tools_counter.named_count == {
        "statistics_analysis": 1,
        "relationship_analysis": 1,
        "read_skill": 1,
    }
    records = [json.loads(line) for line in (tmp_path / "tool_calls.jsonl").read_text().splitlines()]
    assert [record["name"] for record in records] == [
        "statistics_analysis", "relationship_analysis", "read_skill",
    ]
    assert all(record["forced"] is True for record in records)


def test_force_initial_diagnostics_requires_all_tools(tmp_path):
    try:
        SRAgent(
            llm_provider="unused",
            llm_model="unused",
            tools=["statistics_analysis", "read_skill"],
            save_path=str(tmp_path),
            force_initial_diagnostics=True,
        )
    except ValueError as exc:
        assert "relationship_analysis" in str(exc)
    else:
        raise AssertionError("Expected missing required diagnostic tool validation to fail")


def test_split_data_keeps_validation_and_test_out_of_training_context(tmp_path):
    agent = SRAgent(
        llm_provider="unused", llm_model="unused", tools=["unit_parallel_tool"],
        save_path=str(tmp_path), validation_fraction=0.2,
        split_random_state=7,
    )
    train, validation = agent._split_data(
        {"x": np.arange(10.0)}, {"y": np.arange(10.0) * 2},
    )
    assert len(train["x"]) == 8
    assert len(validation["x"]) == 2
    assert set(train["x"]).isdisjoint(validation["x"])


def test_split_data_ood_uses_high_t_values_for_validation(tmp_path):
    agent = SRAgent(
        llm_provider="unused", llm_model="unused", tools=["unit_parallel_tool"],
        save_path=str(tmp_path), validation_fraction=0.2, split_by="ood",
        split_ood_variable="t",
    )
    train, validation = agent._split_data(
        {"x": np.arange(10.0), "t": np.array([8, 1, 5, 0, 9, 3, 7, 2, 6, 4])},
        {"y": np.arange(10.0) * 2},
    )
    assert train["t"].tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert validation["t"].tolist() == [8, 9]


def test_split_data_ood_uses_the_explicitly_selected_variable(tmp_path):
    agent = SRAgent(
        llm_provider="unused", llm_model="unused", tools=["unit_parallel_tool"],
        save_path=str(tmp_path), validation_fraction=0.25, split_by="ood",
        split_ood_variable="T",
    )
    _, validation_T = agent._split_data(
        {"x": np.arange(4.0), "T": np.array([300, 500, 200, 400])},
        {"y": np.arange(4.0)},
    )
    agent.split_ood_variable = "t"
    _, validation_t = agent._split_data(
        {"t": np.array([4, 1, 3, 2]), "T": np.array([100, 400, 300, 200])},
        {"y": np.arange(4.0)},
    )
    assert validation_T["T"].tolist() == [500]
    assert validation_t["t"].tolist() == [4]


def test_split_data_uses_manifest_structure_metadata_in_network_mode(tmp_path):
    theta = np.arange(30.0).reshape(10, 3)
    omega = np.array([0.1, 0.2, 0.3])
    links = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
    target = theta * 2
    data = {
        "theta": theta, "omega": omega, "links": links, "dtheta": target,
        "time": np.arange(10.0), "node": np.arange(3),
        "edge": np.arange(3), "endpoint": np.arange(2),
    }
    context = AgentContext(
        data=data,
        target="dtheta",
        variable_axes={
            "theta": ("time", "node"),
            "omega": ("node",),
            "links": ("edge", "endpoint"),
            "dtheta": ("time", "node"),
        },
        variable_structures={"theta": "links", "dtheta": "links"},
        num_nodes=3,
    )
    agent = SRAgent(
        llm_provider="unused", llm_model="unused", tools=["unit_parallel_tool"],
        save_path=str(tmp_path), validation_fraction=0.2, split_by="ood",
        split_ood_variable="time",
        context=context,
    )

    train, validation = agent._split_data(
        {"theta": theta, "omega": omega, "links": links},
        {"dtheta": target},
    )

    assert train["theta"].shape == (8, 3)
    np.testing.assert_array_equal(train["theta"], theta[:8])
    assert validation["dtheta"].shape == (2, 3)
    np.testing.assert_array_equal(train["omega"], omega)
    np.testing.assert_array_equal(validation["links"], links)


def test_split_data_ood_requires_an_explicit_variable(tmp_path):
    agent = SRAgent(
        llm_provider="unused", llm_model="unused", tools=["unit_parallel_tool"],
        save_path=str(tmp_path), validation_fraction=0.2, split_by="ood",
    )
    with pytest.raises(ValueError, match="split_ood_variable is required"):
        agent._split_data(
            {"x": np.array([8, 1, 5, 0, 9, 3, 7, 2, 6, 4])},
            {"y": np.arange(10.0)},
        )


def test_split_data_random_uses_requested_train_and_validation_sizes(tmp_path):
    agent = SRAgent(
        llm_provider="unused", llm_model="unused", tools=["unit_parallel_tool"],
        save_path=str(tmp_path), validation_fraction=0.2, split_by="random",
    )
    train, validation = agent._split_data(
        {"x": np.arange(10.0)}, {"y": np.arange(10.0)},
    )
    assert len(train["x"]) == 8
    assert len(validation["x"]) == 2


def test_split_data_zero_validation_fraction_reuses_training_data(tmp_path):
    agent = SRAgent(
        llm_provider="unused", llm_model="unused", tools=["unit_parallel_tool"],
        save_path=str(tmp_path), validation_fraction=0, split_by="ood",
    )
    train, validation = agent._split_data(
        {"t": np.arange(10.0)}, {"y": np.arange(10.0)},
    )
    assert len(train["t"]) == 10
    assert len(validation["t"]) == 10
    assert np.array_equal(train["t"], validation["t"])


def test_update_conversation_injects_iteration_status(tmp_path):
    agent = make_agent(tmp_path)
    agent.max_restart_loop = 2
    agent.global_width = 3
    agent.max_refinement_depth = 5
    buffer = [
        {"role": "system", "content": "Base system."},
        {"role": "user", "content": "Solve the task."},
    ]
    original_buffer = list(buffer)
    response_list = [("", [], {"role": "assistant", "content": ""})]
    results_list = [[]]

    updated, _ = agent.update_conversation(
        buffer, response_list, results_list, {}, R=1, L=2, C=1
    )

    assert updated is buffer
    assert len(buffer) == len(original_buffer) + 1
    assert buffer[0]["content"] == "Base system."
    assert "refinement round L=3/5" in buffer[-1]["content"]
    assert "From now on, 2 refinement round(s) remain" in buffer[-1]["content"]
    assert "No Pareto front yet" in buffer[-1]["content"]
    assert original_buffer[-1]["content"] == "Solve the task."


def test_update_conversation_injects_current_pareto_front(tmp_path):
    agent = make_agent(tmp_path)
    agent.max_refinement_depth = 5
    candidate = CandidateRecord(formula="x + y", node_id="candidate-node", details={
        "data_split_results": {
            "train": {"metrics": {"mse": 0.1, "r2": 0.98, "complexity": 3}},
            "validation": {"metrics": {"mse": 0.2, "r2": 0.97, "complexity": 3}},
        },
    })
    agent.push_candidate(candidate)
    buffer = [{"role": "user", "content": "Find a formula."}]

    agent.update_conversation(
        buffer,
        [("", [], {"role": "assistant", "content": ""})],
        [[]],
        {},
        R=1,
        L=1,
        C=1,
    )

    status = buffer[-1]["content"]
    assert "No Pareto front yet" not in status
    assert "Validation MSE=0.2" in status
    assert "Formula=x + y" in status


def test_update_conversation_final_round_tells_agent_to_submit(tmp_path):
    agent = make_agent(tmp_path)
    agent.max_refinement_depth = 4
    buffer = [
        {"role": "system", "content": "Base system."},
        {"role": "user", "content": "Solve the task."},
    ]
    response_list = [("", [], {"role": "assistant", "content": ""})]
    results_list = [[]]

    agent.update_conversation(buffer, response_list, results_list, {}, R=1, L=4, C=1)
    final_status = buffer[-1]["content"]

    assert "final refinement round" in final_status
    assert "Submit or state your best available target formula now" in final_status
    assert "Do not spend this response on new data exploration" in final_status


def test_get_ranking_metric_ignores_non_candidate_tool_results(tmp_path):
    agent = make_agent(tmp_path)
    diagnostic_result = ToolCallResult(
        True,
        {"content": "diagnostic output without formula metrics"},
        "diagnostic output without formula metrics",
        {},
    )

    assert agent.get_ranking_metric(diagnostic_result.result) == (None, None)
    assert agent.candidate_sort_key(diagnostic_result.result) is None


def test_update_conversation_sorts_unwrapped_tool_results(tmp_path):
    agent = make_agent(tmp_path)
    agent.parser = SimpleNamespace(format_tool_result_messages=lambda *args: [])
    buffer = [{"role": "user", "content": "Find a formula."}]
    response_list = [
        ("diagnostic", [], {"role": "assistant", "content": "diagnostic"}),
        ("candidate", [], {"role": "assistant", "content": "candidate"}),
    ]
    results_list = [
        [ToolCallResult(True, {"content": "diagnostic"}, "diagnostic", {})],
        [ToolCallResult(True, {
            "data_split_results": {"train": {"metrics": {"mse": 0.1, "complexity": 2}}},
        }, "candidate", {})],
    ]

    agent.update_conversation(buffer, response_list, results_list, {}, R=1, L=1, C=1)

    assert buffer[-2]["content"] == "candidate"


def test_collect_candidates_records_pareto_front(tmp_path):
    agent = make_agent(tmp_path)
    response_list = [
        ("", [
            ToolCall(name="evaluate_formula", params={"eq": "x"}),
            ToolCall(name="evaluate_formula", params={"eq": "x + y"}),
            ToolCall(name="evaluate_formula", params={"eq": "x + y + z"}),
        ], {}),
    ]
    results_list = [[
        ToolCallResult(True, {
            "formula": "x",
            "data_split_results": {
                "train": {"metrics": {"mse": 0.2, "complexity": 2}},
            },
            "is_candidate": True,
        }, "", {}),
        ToolCallResult(True, {
            "formula": "x + y",
            "data_split_results": {
                "train": {"metrics": {"mse": 0.1, "complexity": 3}},
            },
            "is_candidate": True,
        }, "", {}),
        ToolCallResult(True, {
            "formula": "x + y + z",
            "data_split_results": {
                "train": {"metrics": {"mse": 0.3, "complexity": 5}},
            },
            "is_candidate": True,
        }, "", {}),
    ]]

    candidates = agent.collect_candidates(response_list, results_list, R=1, L=1, C=1)
    result = agent.build_search_result("completed", R=1, L=1, C=1)

    assert [item.formula for item in candidates] == ["x + y", "x", "x + y + z"]
    assert result["pareto_front"] == [0, 1]
    assert [result["candidates"][index]["formula"] for index in result["pareto_front"]] == ["x + y", "x"]
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["pareto_front"] == [0, 1]


def test_no_file_mode_keeps_search_identity_in_memory():
    agent = SRAgent(
        llm_provider="unused",
        llm_model="unused",
        tools=["unit_parallel_tool"],
        save_path=None,
    )

    node_id = agent.run_state.node_id(R=1, C=1, L=1, K=1)
    assert node_id.startswith(f"{agent.run_state.run_id}:")
    assert agent.save_path is None


def test_collect_candidates_rejects_non_finite_candidate_metrics(tmp_path):
    agent = make_agent(tmp_path)
    response_list = [
        ("", [
            ToolCall(name="evaluate_formula", params={"f": "1.966*P*log(0.0423/P)"}),
            ToolCall(name="evaluate_formula", params={"f": "P"}),
        ], {}),
    ]
    results_list = [[
        ToolCallResult(True, {
            "formula": "7.2768821 * P * log(-111.24893 / P)",
            "data_split_results": {
                "train": {"metrics": {"mse": float("nan"), "complexity": 8}},
                "validation": {"metrics": {"mse": float("nan"), "complexity": 8}},
            },
            "is_candidate": True,
        }, "", {}),
        ToolCallResult(True, {
            "formula": "P",
            "data_split_results": {
                "train": {"metrics": {"mse": 0.2, "complexity": 1}},
                "validation": {"metrics": {"mse": 0.3, "complexity": 1}},
            },
            "is_candidate": True,
        }, "", {}),
    ]]

    topk_records = agent.collect_candidates(response_list, results_list, R=1, L=1, C=1)

    assert [entry.formula for entry in topk_records] == ["P"]
    assert [item.formula for item in agent.get_pareto_front()] == ["P"]
