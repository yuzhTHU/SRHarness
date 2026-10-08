import json

import numpy as np
import pytest

from sr_harness.core import (
    AgentContext,
    ContextManifestError,
    inspect_context_data,
    load_context_data,
    update_context_data_descriptions,
)
from sr_harness.tools.validate_context_data import ValidateContextDataTool
from sr_harness.tools.workspace_shell import Workspace


def write_manifest(directory, manifest):
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def test_context_data_loads_flat_variables_and_inline_file_and_size_axes(tmp_path):
    directory = tmp_path / "context.data"
    manifest = {
        "num_nodes": 2,
        "variables": {
            "population": {
                "file": "population.npy",
                "description": "Population in ten thousand people.",
                "axes": ["time"],
            },
            "category": {
                "file": "category.npy",
                "description": "Unencoded category label.",
                "axes": ["time"],
            },
            "node_label": {
                "file": "node_label.npy",
                "description": "Label attached to each node.",
                "axes": ["node"],
            },
            "edge_signal": {
                "file": "edge_signal.npy",
                "description": "Edge state over time.",
                "axes": ["time", "edge"],
                "structure": "A",
            },
            "A": {
                "file": "A.npy",
                "description": "Directed edge list.",
                "axes": ["edge", "edge_endpoint"],
            },
        },
        "axes": {
            "time": {
                "values": [2018, 2019, 2020],
                "description": "Observation year.",
            },
            "node": {
                "file": "node.npy",
                "description": "Node identifier.",
            },
            "edge": {"size": 2, "description": "Edge position."},
            "edge_endpoint": {
                "values": ["target", "source"],
                "description": "Directed endpoint order.",
            },
        },
    }
    write_manifest(directory, manifest)
    np.save(directory / "population.npy", np.array([1400.0, 1401.0, 1402.0]))
    np.save(directory / "category.npy", np.array(["urban", "rural", "urban"]))
    np.save(directory / "node_label.npy", np.array(["north", "south"]))
    np.save(directory / "edge_signal.npy", np.arange(6).reshape(3, 2))
    np.save(directory / "A.npy", np.array([[1, 0], [0, 1]]))
    np.save(directory / "node.npy", np.array(["Beijing", "Shanghai"]))

    report = inspect_context_data(directory)
    assert report["valid"]
    assert report["warnings"] == []
    assert report["num_nodes"] == 2
    assert report["variables"]["edge_signal"]["structure"] == "A"
    assert report["variables"]["edge_signal"]["shape"] == [3, 2]
    assert report["axes"]["time"]["storage"] == "values"
    assert report["axes"]["edge"]["storage"] == "size"

    data = load_context_data(directory)
    np.testing.assert_array_equal(data["data"]["category"], ["urban", "rural", "urban"])
    assert data["variable_axes"]["A"] == ("edge", "edge_endpoint")
    assert data["variable_structures"] == {"edge_signal": "A"}
    assert data["num_nodes"] == 2
    assert data["variable_descriptions"]["population"].startswith("Population")
    np.testing.assert_array_equal(data["data"]["node"], ["Beijing", "Shanghai"])


def test_context_data_descriptions_are_updated_atomically(tmp_path):
    directory = tmp_path / "context.data"
    write_manifest(directory, {
        "variables": {
            "x": {"file": "x.npy", "description": "Old input.", "axes": ["sample"]},
        },
        "axes": {
            "sample": {"values": [1, 2], "description": "Old axis."},
        },
    })
    np.save(directory / "x.npy", np.array([3.0, 4.0]))

    loaded = update_context_data_descriptions(
        directory, {"x": "Edited input.", "sample": ""},
    )

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["variables"]["x"]["description"] == "Edited input."
    assert manifest["axes"]["sample"]["description"] == ""
    assert loaded["variable_descriptions"] == {
        "x": "Edited input.", "sample": "",
    }
    assert load_context_data(directory)["variable_descriptions"] == loaded["variable_descriptions"]


def test_context_data_descriptions_update_unlocked_manifest_in_locked_directory(tmp_path):
    directory = tmp_path / "context.data"
    write_manifest(directory, {
        "variables": {
            "x": {"file": "x.npy", "description": "Old input.", "axes": ["sample"]},
        },
        "axes": {
            "sample": {"values": [1, 2], "description": "Sample."},
        },
    })
    np.save(directory / "x.npy", np.array([3.0, 4.0]))
    manifest = directory / "manifest.json"
    directory.chmod(0o555)
    manifest.chmod(0o644)
    try:
        loaded = update_context_data_descriptions(directory, {"x": "Edited input."})
    finally:
        directory.chmod(0o755)

    assert loaded["variable_descriptions"]["x"] == "Edited input."
    assert json.loads(manifest.read_text())["variables"]["x"]["description"] == "Edited input."


def test_context_data_reports_redundant_missing_and_misaligned_content(tmp_path):
    directory = tmp_path / "context.data"
    write_manifest(directory, {
        "variables": {
            "x": {
                "file": "wrong.npy",
                "description": "State.",
                "axes": ["time"],
                "dtype": "float64",
            },
            "y": {
                "file": "y.npy",
                "description": "Target.",
                "axes": ["time"],
            },
        },
        "axes": {
            "time": {"values": [1, 2], "description": "Time."},
            "unused": {"size": 2, "description": "Unused."},
        },
    })
    np.save(directory / "y.npy", np.arange(3))
    np.save(directory / "orphan.npy", np.arange(2))

    report = inspect_context_data(directory)
    assert not report["valid"]
    assert any("unsupported fields: ['dtype']" in error for error in report["errors"])
    assert any("x.npy" in error for error in report["errors"])
    assert any("axis length mismatch" in error for error in report["errors"])
    assert any("unreferenced axes" in error for error in report["errors"])
    assert report["warnings"] == ["unreferenced NPY files: ['orphan.npy']"]
    with pytest.raises(ContextManifestError):
        load_context_data(directory)


@pytest.mark.parametrize("num_nodes", [None, 0, -1, True, 2.5, "3"])
def test_context_data_rejects_invalid_num_nodes(tmp_path, num_nodes):
    directory = tmp_path / "context.data"
    write_manifest(directory, {
        "num_nodes": num_nodes,
        "variables": {
            "x": {"file": "x.npy", "description": "State.", "axes": ["node"]},
        },
        "axes": {"node": {"size": 2, "description": "Node position."}},
    })
    np.save(directory / "x.npy", np.arange(2.0))

    report = inspect_context_data(directory)

    assert not report["valid"]
    assert "manifest.num_nodes must be a positive integer" in report["errors"]


def test_relation_metadata_validates_shape_dtype_and_endpoint_range(tmp_path):
    directory = tmp_path / "context.data"
    write_manifest(directory, {
        "num_nodes": 3,
        "variables": {
                "edge_signal": {
                    "file": "edge_signal.npy",
                    "description": "Values on edges.",
                    "axes": ["edge"],
                    "structure": "links",
                },
                "links": {"file": "links.npy", "description": "Directed endpoint pairs.", "axes": ["edge", "endpoint"]},
        },
        "axes": {
            "edge": {"size": 2, "description": "Edge position."},
            "endpoint": {
                "values": ["target", "source"],
                "description": "Endpoint order.",
            },
        },
    })
    np.save(directory / "links.npy", np.array([[0, 1], [2, 3]], dtype=int))
    np.save(directory / "edge_signal.npy", np.array([1.0, 2.0]))

    report = inspect_context_data(directory)

    assert not report["valid"]
    assert any("relation endpoints must be in [0, 3)" in error for error in report["errors"])


def test_validate_context_data_tool_only_validates_workspace_files(tmp_path):
    directory = tmp_path / "context.data"
    write_manifest(directory, {
        "variables": {
            "x": {"file": "x.npy", "description": "Input.", "axes": ["sample"]},
            "y": {"file": "y.npy", "description": "Output.", "axes": ["sample"]},
        },
        "axes": {
            "sample": {"values": [1, 2, 3], "description": "Sample identifier."},
        },
    })
    np.save(directory / "x.npy", np.array([1.0, 2.0, 3.0]))
    np.save(directory / "y.npy", np.array([2.0, 4.0, 6.0]))
    workspace = Workspace(path=tmp_path)
    context = AgentContext(workspace=workspace)

    result = ValidateContextDataTool(context=context).execute()

    assert result["valid"]
    assert result["repairs"] == []
    assert "ready for InteractiveSession" in result["next_action"]
    assert not context.data
    assert not hasattr(context.args, "data_revision")


def test_validate_context_data_tool_returns_actionable_repairs(tmp_path):
    directory = tmp_path / "context.data"
    directory.mkdir()
    (directory / "manifest.json").write_text('{"variables": {}, "axes": {}}')
    context = AgentContext(workspace=Workspace(path=tmp_path))

    tool = ValidateContextDataTool(context=context)
    result = tool.execute()

    assert not result["valid"]
    assert result["repairs"]
    assert all(repair["error"] and repair["action"] for repair in result["repairs"])
    assert "run validate_context_data again" in result["next_action"]
    formatted = tool.format_result_dict(result)
    assert "INVALID context.data" in formatted
    assert "Action:" in formatted
