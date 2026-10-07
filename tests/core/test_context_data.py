import json

import numpy as np
import pytest

from sr_harness.core import AgentContext, ContextDataLoader, ContextManifestError
from sr_harness.tools.load_context_data import LoadContextDataTool
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

    store = ContextDataLoader(directory)
    report = store.inspect()
    assert report["valid"]
    assert report["warnings"] == []
    assert report["num_nodes"] == 2
    assert report["variables"]["edge_signal"]["structure"] == "A"
    assert report["variables"]["edge_signal"]["shape"] == [3, 2]
    assert report["axes"]["time"]["storage"] == "values"
    assert report["axes"]["edge"]["storage"] == "size"

    data = store.load()
    np.testing.assert_array_equal(data["data"]["category"], ["urban", "rural", "urban"])
    assert data["variable_axes"]["A"] == ("edge", "edge_endpoint")
    assert data["variable_structures"] == {"edge_signal": "A"}
    assert data["num_nodes"] == 2
    assert data["variable_descriptions"]["population"].startswith("Population")
    np.testing.assert_array_equal(data["data"]["node"], ["Beijing", "Shanghai"])


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

    report = ContextDataLoader(directory).inspect()
    assert not report["valid"]
    assert any("unsupported fields: ['dtype']" in error for error in report["errors"])
    assert any("x.npy" in error for error in report["errors"])
    assert any("axis length mismatch" in error for error in report["errors"])
    assert any("unreferenced axes" in error for error in report["errors"])
    assert report["warnings"] == ["unreferenced NPY files: ['orphan.npy']"]
    with pytest.raises(ContextManifestError):
        ContextDataLoader(directory).load()


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

    report = ContextDataLoader(directory).inspect()

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

    report = ContextDataLoader(directory).inspect()

    assert not report["valid"]
    assert any("relation endpoints must be in [0, 3)" in error for error in report["errors"])


def test_load_context_data_tool_commits_arrays_and_axis_metadata(tmp_path):
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

    result = LoadContextDataTool(context=context).execute()

    assert result["valid"] and result["data_committed"]
    assert context.args.data_revision == 1
    assert context.variable_axes == {"x": ("sample",), "y": ("sample",)}
    assert context.variable_descriptions == {
        "sample": "Sample identifier.",
        "x": "Input.",
        "y": "Output.",
    }
    assert context.schema()["variables"]["y"]["shape"] == [3]
