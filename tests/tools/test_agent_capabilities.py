from __future__ import annotations

import threading

import numpy as np
import pytest

from sr_harness.runtime import InteractionManager, SRInteractionManager
from sr_harness.web.app import create_app
from sr_harness.runtime import ModelRouter
from sr_harness.core import AgentContext, SearchRunState, ToolCall
from sr_harness import SRAgent
from sr_harness.tools.evaluate_formula import EvaluateTool
from sr_harness.utils import ParallelTimer
from sr_harness.skills import SkillManager
from sr_harness.tools.eic import EICTool
from sr_harness.tools.nd2 import ND2Tool
from sr_harness.tools.sr4mdl import SR4MDLTool
from sr_harness.tools.subagent import SubagentTool
from sr_harness.tools.web_research import WebFetchTool, WebSearchTool


def test_sr_harness_engine_syntax_is_a_builtin_skill(tmp_path):
    manager = SkillManager(
        built_in_directory="src/sr_harness/skills",
        custom_directory=tmp_path / "custom",
    )
    skill = manager.get_skill("sr-harness-engine-syntax")
    content = manager.read_skill(skill.name)

    assert skill.readonly is True
    assert "sum[j](A[i, j]" in content
    assert "gather(A[i, j]" in content
    assert "grouped_param" in content


def test_eic_is_reproducible_and_finite():
    tool = EICTool(data={"x": np.linspace(1.0, 2.0, 64)})
    first = tool.execute("x + x**2", repeats=4, random_state=7)
    second = tool.execute("x + x**2", repeats=4, random_state=7)
    assert first["eic"] == second["eic"]
    assert np.isfinite(first["eic"])


def test_eic_reports_unstable_masked_subtree_in_annotated_tree():
    x = np.logspace(-12, -2, 128)
    tool = EICTool(data={"x": x})
    result = tool.execute("0*(sqrt(1+x)-1)+x", repeats=4, random_state=7)
    assert result["eic"] > result["output_eic"] + 3
    assert result["worst_subtree"] != "root"
    assert "likely redundant subtree" in result["tree"]
    assert "← EIC=" in result["tree"]


def test_eic_result_can_reenter_candidate_search_state():
    values = np.linspace(1.0, 2.0, 32)
    result = EICTool(
        data={"x": values, "y": values + values**2},
        target="y",
    ).execute("x + x**2", repeats=2)
    assert result["is_candidate"] is True
    assert result["data_split_results"]["train"]["metrics"]["mse"] == 0.0
    assert result["eic_diagnostics"]["worst_subtree"] == result["worst_subtree"]


def test_new_candidate_is_not_automatically_audited_with_eic():
    values = np.linspace(1.0, 2.0, 32)
    context = {"data": {"x": values, "y": values}, "target": "y"}
    candidate = EvaluateTool(**context)(f="x", show_diagnostics=False)
    agent = object.__new__(SRAgent)
    agent.ranking_metric = "mse"
    agent.larger_is_better = False
    agent.tools = [EICTool(**context)]
    agent.tools_counter = ParallelTimer(unit="call")
    agent.save_path = None
    agent.run_state = SearchRunState(
        save_path=None,
        ranking_metric="mse",
        larger_is_better=False,
    )
    topk = agent.collect_candidates(
        [("", [ToolCall(name="evaluate_formula", params={})], {})],
        [[candidate]],
        R=1,
        L=1,
        C=1,
    )
    assert "eic_diagnostics" not in topk[0].details
    assert agent.tools_counter.named_count.get("evaluate_eic", 0) == 0


def test_subagent_uses_isolated_callback_prompt():
    captured = {}

    def callback(messages):
        captured["messages"] = messages
        return "independent result"

    result = SubagentTool(
        data={"x": np.arange(3.0), "y": np.arange(3.0)},
        target="y",
        subagent_callback=callback,
    ).execute(
        "Critique this hypothesis.",
        mode="candidate_critique",
        candidate_formulas=["x"],
    )
    assert result["content"] == "independent result"
    assert "pearson_to_target=1" in captured["messages"][1]["content"]
    assert "Candidate formulas:\n- x" in captured["messages"][1]["content"]


def test_web_search_callback_is_bounded():
    tool = WebSearchTool(web_search_callback=lambda query, limit: [
        {"title": str(i), "url": f"https://example.com/{i}", "snippet": query}
        for i in range(20)
    ])
    result = tool.execute("symbolic regression", max_results=3)
    assert len(result["results"]) == 3


def test_web_fetch_rejects_private_network_addresses(monkeypatch):
    monkeypatch.setattr(
        "sr_harness.tools.web_research.socket.getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("127.0.0.1", 80))],
    )
    with pytest.raises(ValueError, match="public network"):
        WebFetchTool().execute("http://example.test/private")


def test_web_fetch_accepts_mixed_proxy_dns_but_not_mixed_direct_dns(monkeypatch):
    addresses = [
        (2, 1, 6, "", ("93.184.216.34", 443)),
        (10, 1, 6, "", ("2001::1", 443, 0, 0)),
    ]
    monkeypatch.setattr(
        "sr_harness.tools.web_research.socket.getaddrinfo",
        lambda *args, **kwargs: addresses,
    )
    monkeypatch.setattr(
        "sr_harness.tools.web_research.requests.utils.get_environ_proxies",
        lambda url: {},
    )
    with pytest.raises(ValueError, match="public network"):
        WebFetchTool().execute("https://example.test/page")

    class Response:
        is_redirect = False
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        encoding = "utf-8"

        def raise_for_status(self):
            return None

        def iter_content(self, size):
            yield b"<html><body>Public page</body></html>"

    monkeypatch.setattr(
        "sr_harness.tools.web_research.requests.utils.get_environ_proxies",
        lambda url: {"https": "http://127.0.0.1:6789"},
    )
    monkeypatch.setattr(
        "sr_harness.tools.web_research.requests.get",
        lambda *args, **kwargs: Response(),
    )
    result = WebFetchTool().execute("https://example.test/page")
    assert result["text"] == "Public page"


def test_web_fetch_reports_automated_access_blocks(monkeypatch):
    monkeypatch.setattr(
        "sr_harness.tools.web_research.socket.getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )

    class Response:
        is_redirect = False
        status_code = 403
        closed = False

        def close(self):
            self.closed = True

    response = Response()
    monkeypatch.setattr(
        "sr_harness.tools.web_research.requests.get",
        lambda *args, **kwargs: response,
    )
    result = WebFetchTool().execute("https://example.test/protected")
    assert result["blocked"] is True
    assert result["status_code"] == 403
    assert "authoritative source" in result["message"]
    assert response.closed


def test_eic_doc_is_registered_as_runtime_skill(tmp_path):
    manager = SkillManager(
        built_in_directory="src/sr_harness/skills",
        custom_directory=tmp_path / "custom",
    )
    manager.register_tool_docs([EICTool, ND2Tool, SR4MDLTool])
    names = manager.load_skills()
    assert "eic-structural-stability" in names
    assert "nd2-network-dynamics" in names
    assert "sr4mdl-search" in names


def test_sr4mdl_constructs_search_data_like_pysr(tmp_path, monkeypatch):
    root = tmp_path / "SR4MDL"
    root.mkdir()
    (root / "regressor.py").touch()
    checkpoint = root / "checkpoint.pth"
    checkpoint.touch()
    monkeypatch.setenv("SR4MDL_HOME", str(root))
    monkeypatch.setenv("SR4MDL_CHECKPOINT", str(checkpoint))
    captured = {}

    def fake_search(self, **kwargs):
        captured.update(kwargs)
        return "x1 + x2"

    monkeypatch.setattr(SR4MDLTool, "_run_sr4mdl", fake_search)
    values = np.linspace(0.0, 1.0, 32)
    result = SR4MDLTool(
        data={"a": values, "b": values**2, "y": values + values**2},
        target="y",
        train_indices=np.arange(24),
        validation_indices=np.arange(24, 32),
    ).execute(["+", "*"], [], x=["a", "b"], y="y", n_iter=7)

    assert set(captured["X"]) == {"x1", "x2"}
    assert captured["n_iter"] == 7
    assert result["formula"] == "a + b"
    assert result["method"] == "SR4MDL-MCTS"


def test_nd2_returns_search_candidate_instead_of_export(tmp_path, monkeypatch):
    root = tmp_path / "ND2"
    root.mkdir()
    (root / "search.py").touch()
    checkpoint = root / "checkpoint.pth"
    checkpoint.touch()
    monkeypatch.setenv("ND2_HOME", str(root))
    monkeypatch.setenv("ND2_CHECKPOINT", str(checkpoint))

    def fake_search(self, **kwargs):
        return {
            "formula": "omega + aggr(sin(sour(x) - targ(x)))",
            "prefix": ["add", "omega", "aggr", "sin", "sub", "sour", "x", "targ", "x"],
            "train_metrics": {"mse": 0.0, "rmse": 0.0, "r2": 1.0, "complexity": 9.0},
            "validation_metrics": None,
        }

    monkeypatch.setattr(ND2Tool, "_run_nd2", fake_search)
    adjacency = np.array([[0, 1], [1, 0]])
    x = np.arange(12.0).reshape(6, 2)
    result = ND2Tool(
        data={"A": adjacency, "x": x, "omega": np.ones_like(x), "dx": x},
        target="dx",
    ).execute(vars_node=["x", "omega"], y="dx")
    assert result["is_candidate"] is True
    assert result["method"] == "ND2-NDformer-MCTS"
    assert result["data_split_results"]["train"]["metrics"]["mse"] == 0.0


def test_network_split_preserves_graph_axes():
    agent = object.__new__(SRAgent)
    agent.validation_fraction = 0.25
    agent.split_by = "random"
    agent.split_random_state = 7
    adjacency = np.array([[0, 1], [1, 0]])
    edges = np.array([[0, 1], [1, 0]])
    x = np.arange(16.0).reshape(8, 2)
    agent.context = AgentContext(
        data={"A": adjacency, "G": edges, "x": x, "dx": x + 1},
        target="dx",
        variable_structures={"x": "A", "dx": "A"},
        num_nodes=2,
    )
    train, validation = agent._split_data(
        {"A": adjacency, "G": edges, "x": x},
        {"dx": x + 1},
    )
    assert train["x"].shape == (6, 2)
    assert validation["x"].shape == (2, 2)
    assert np.array_equal(train["A"], adjacency)
    assert np.array_equal(validation["G"], edges)


def test_interaction_manager_keeps_only_the_latest_stream_snapshot():
    manager = InteractionManager()
    manager.publish_event("assistant_started", {"response_id": "reply-1"})
    manager.publish_event(
        "assistant_delta",
        {"response_id": "reply-1", "content": "first"},
    )
    manager.publish_event("workspace_changed", {"path": "artifact.txt"})
    manager.publish_event(
        "assistant_delta",
        {"response_id": "reply-1", "content": "latest"},
    )

    events = manager.get_recent_events()["events"]

    assert [event["kind"] for event in events] == [
        "assistant_started",
        "workspace_changed",
        "assistant_delta",
    ]
    assert events[-1]["payload"]["content"] == "latest"
    batch = manager.get_recent_events(after_sequence=1)
    assert not batch["truncated"]
    assert [event["seq"] for event in batch["events"]] == [3, 4]

    manager.publish_event(
        "assistant_completed",
        {"response_id": "reply-1", "content": "complete"},
    )

    events = manager.get_recent_events()["events"]
    assert [event["kind"] for event in events] == [
        "assistant_started",
        "workspace_changed",
        "assistant_completed",
    ]


def test_interaction_manager_reports_only_true_buffer_eviction():
    manager = InteractionManager()
    for index in range(1002):
        manager.publish_event("workspace_changed", {"index": index})

    batch = manager.get_recent_events(after_sequence=1)

    assert batch["truncated"]
    assert batch["discarded_through"] == 2
    assert batch["events"][0]["seq"] == 3


def test_interaction_manager_rejects_snapshot_schema_drift():
    manager = InteractionManager()
    snapshot = manager.export_state()
    snapshot["deprecated_field"] = None

    with pytest.raises(ValueError, match="current schema"):
        manager.restore_state(snapshot)


def test_interaction_manager_rejects_inconsistent_snapshot_sequence():
    manager = InteractionManager()
    manager.publish_event("workspace_changed", {"path": "context.data"})
    snapshot = manager.export_state()
    snapshot["next_sequence"] = 1

    with pytest.raises(ValueError, match="next_sequence"):
        manager.restore_state(snapshot)


def test_model_router_uses_base_for_simple_task_and_strong_for_complex_task():
    router = ModelRouter(
        enabled=True,
        base_provider="openrouter",
        base_model="cheap",
        strong_model="strong",
    )
    score, reasons = router.assess("Find y = f(x).", feature_count=1)
    assert router.route(task_score=score, task_reasons=reasons, refinement_step=1).tier == "base"
    score, reasons = router.assess("Discover a noisy network dynamics ODE.", feature_count=5)
    assert router.route(task_score=score, task_reasons=reasons, refinement_step=1).tier == "strong"


def test_model_router_escalates_stagnated_search_and_respects_disable():
    router = ModelRouter(
        enabled=True,
        base_provider="base-provider",
        base_model="base-model",
        strong_provider="strong-provider",
        strong_model="strong-model",
    )
    route = router.route(task_score=0, task_reasons=[], refinement_step=3)
    assert (route.tier, route.provider, route.model) == (
        "strong", "strong-provider", "strong-model"
    )
    router.enabled = False
    assert router.route(task_score=9, task_reasons=[], refinement_step=4).tier == "base"


def test_subagent_prompt_contains_grounded_sr_mandate():
    captured = {}

    def callback(messages):
        captured["messages"] = messages
        return "review"

    SubagentTool(subagent_callback=callback).execute(
        "Check floating-point cancellation and recommend a discriminating test.",
        mode="candidate_critique",
        candidate_formulas=["exp(x)-1", "x"],
        evidence="Both formulas fit the observed interval.",
    )
    prompt = captured["messages"][1]["content"]
    assert "Available main-agent tools" in prompt
    assert "candidate_critique" in prompt
    assert "evaluate_eic" in prompt


def test_interaction_manager_waits_at_boundary_and_resumes_with_message():
    manager = InteractionManager()
    manager.command("message", "initial guidance")
    assert [message.content for message in manager.start_agent_execution()] == [
        "initial guidance"
    ]
    manager.command("pause")
    received = {}

    def wait_at_boundary():
        with manager.wait() as messages:
            received["messages"] = [message.content for message in messages]

    thread = threading.Thread(target=wait_at_boundary)
    thread.start()
    while manager.state != "paused":
        thread.join(0.01)
    manager.command("message", "try a power law")
    thread.join(2)

    assert received["messages"] == ["try a power law"]
    assert manager.state == "running"
    manager.finish_agent_execution()


def test_sr_interaction_manager_transition_resumes_a_paused_boundary():
    manager = SRInteractionManager()
    manager.start_agent_execution()
    manager.request_pause()
    received = {}

    def wait_at_boundary():
        with manager.wait() as messages:
            received["messages"] = messages

    thread = threading.Thread(target=wait_at_boundary)
    thread.start()
    while manager.state != "paused":
        thread.join(0.01)
    manager.command("next_r")
    thread.join(2)

    assert received["messages"] == []
    assert manager.consume_search_transition() == "next_r"
    assert manager.state == "running"
    manager.finish_agent_execution()


def test_create_app_owns_a_viewer_interaction_manager(tmp_path):
    app = create_app(tmp_path)
    assert isinstance(app.state.interaction_manager, InteractionManager)
