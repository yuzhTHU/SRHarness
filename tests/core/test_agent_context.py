import pickle
import argparse

import numpy as np

from sr_harness.agents.sr_agent import SRAgent
from sr_harness.core import AgentContext
from sr_harness.tools.commit_data import CommitDataTool
from sr_harness.tools.workspace_shell import Workspace


class CountingSplitter:
    def __init__(self):
        self.calls = 0

    def split_data(self, context):
        self.calls += 1
        return {
            "train": {name: value[:2] for name, value in context.data.items()},
            "evaluation": {name: value[2:] for name, value in context.data.items()},
        }


def test_agent_context_commits_data_and_caches_evaluator_split():
    context = AgentContext()
    change = context.commit_data(
        {"x": [1, 2, 3], "y": [2, 4, 6]},
        target="y",
        features=["x"],
    )
    assert change["revision"] == 1
    assert context.schema()["rows"] == 3
    context.args.validation_fraction = 1 / 3
    np.testing.assert_array_equal(context.train_data()["x"], [1, 2])
    np.testing.assert_array_equal(context.evaluation_data()["x"], [3])
    np.testing.assert_array_equal(context.data["x"], [1, 2, 3])


def test_split_data_is_cached_and_split_contexts_share_arrays_and_runtime_state(tmp_path):
    splitter = CountingSplitter()
    args = argparse.Namespace(run_label="shared")
    context = AgentContext(
        args=args,
        data={"x": np.arange(4), "y": np.arange(4) * 2},
        target="y",
        evaluator=splitter,
        workspace=tmp_path,
    )

    train_data = context.train_data()
    evaluation_data = context.evaluation_data()
    train_context = context.train_split()
    evaluation_context = context.evaluation_split()

    assert splitter.calls == 1
    assert train_context.args is args
    assert evaluation_context.evaluator is splitter
    assert train_context.workspace == context.workspace
    assert np.shares_memory(train_context.data["x"], context.data["x"])
    assert np.shares_memory(evaluation_context.data["x"], context.data["x"])
    assert train_context.data["x"] is train_data["x"]
    assert evaluation_context.data["x"] is evaluation_data["x"]


def test_agent_context_rejects_redundant_mapping_access_and_invalid_metadata():
    with np.testing.assert_raises(TypeError):
        AgentContext(data={"x": [1]}, target="x")["data"]
    with np.testing.assert_raises_regex(ValueError, "keys must equal data keys"):
        AgentContext(
            data={"x": [1]}, target="x", variable_descriptions={"other": "wrong"}
        )
    with np.testing.assert_raises_regex(ValueError, "partition data keys"):
        AgentContext(
            data={"time": [0], "x": [1]}, target="x", variable_axes={"x": ()}
        )


def test_agent_context_can_cross_worker_process_boundaries():
    context = AgentContext(data={"x": [1]}, target="x")
    restored = pickle.loads(pickle.dumps(context))
    np.testing.assert_array_equal(restored.data["x"], [1])
    restored.commit_data({"x": [2], "y": [3]}, target="y", features=["x"])
    assert restored.args.data_revision == 1


def test_sr_agent_refreshes_added_features_without_replacing_its_conversation():
    context = AgentContext()
    context.commit_data({"x": [1, 2, 3], "y": [2, 4, 6]}, target="y", features=["x"])
    agent = object.__new__(SRAgent)
    agent.context = context
    agent._data_revision = context.args.data_revision
    agent._active_X = {"x": context.data["x"]}
    agent._active_y = {"y": context.data["y"]}
    agent.validation_fraction = 0
    agent.split_by = "ood"
    agent.split_random_state = 42
    buffer = [{"role": "assistant", "content": "Earlier scientific evidence."}]

    context.add_features({"z": [3, 2, 1]})
    assert agent.refresh_data(buffer)

    assert list(agent._active_X) == ["x", "z"]
    assert buffer[0]["content"] == "Earlier scientific evidence."
    assert "available features are ['x', 'z']" in buffer[-1]["content"]
    assert context.train_data() is not None


def test_active_run_accepts_only_additive_data_commits(tmp_path):
    context = AgentContext(workspace=Workspace(path=tmp_path))
    context.commit_data({"x": [1, 2], "y": [2, 4]}, target="y", features=["x"])
    context.args.sr_active = True
    tool = CommitDataTool(context=context)

    (tmp_path / "added.csv").write_text("x,z,y\n1,5,2\n2,6,4\n")
    change = tool.execute("added.csv", target="y", features=["x", "z"])
    assert change["data_committed"]
    assert list(context.feature_names()) == ["x", "z"]

    (tmp_path / "changed.csv").write_text("x,z,y\n9,5,2\n2,6,4\n")
    with np.testing.assert_raises_regex(ValueError, "only accepts added features"):
        tool.execute("changed.csv", target="y", features=["x", "z"])
