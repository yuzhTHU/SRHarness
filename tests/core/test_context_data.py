import json

import numpy as np
import pytest

from sr_harness.core import AgentContext, ContextDataStore, ContextManifestError
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
            "x": {
                "file": "x.npy",
                "description": "Node state over time.",
                "axes": ["time", "node"],
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
    np.save(directory / "x.npy", np.arange(6).reshape(3, 2))
    np.save(directory / "A.npy", np.array([[1, 0], [0, 1]]))
    np.save(directory / "node.npy", np.array(["Beijing", "Shanghai"]))

    store = ContextDataStore(directory)
    report = store.inspect()
    assert report["valid"]
    assert report["warnings"] == []
    assert report["variables"]["x"]["shape"] == [3, 2]
    assert report["axes"]["time"]["storage"] == "values"
    assert report["axes"]["edge"]["storage"] == "size"

    data = store.load()
    assert list(data) == ["population", "category", "x", "A"]
    np.testing.assert_array_equal(data["category"], ["urban", "rural", "urban"])
    assert data.variable_axes["A"] == ("edge", "edge_endpoint")
    assert data.descriptions["population"].startswith("Population")
    np.testing.assert_array_equal(data.axes["node"].values, ["Beijing", "Shanghai"])


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

    report = ContextDataStore(directory).inspect()
    assert not report["valid"]
    assert any("unsupported fields: ['dtype']" in error for error in report["errors"])
    assert any("x.npy" in error for error in report["errors"])
    assert any("axis length mismatch" in error for error in report["errors"])
    assert any("unreferenced axes" in error for error in report["errors"])
    assert report["warnings"] == ["unreferenced NPY files: ['orphan.npy']"]
    with pytest.raises(ContextManifestError):
        ContextDataStore(directory).load()


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
    assert context.data_revision == 1
    assert context.variable_axes == {"x": ("sample",), "y": ("sample",)}
    assert context.variable_descriptions == {
        "sample": "Sample identifier.",
        "x": "Input.",
        "y": "Output.",
    }
    assert context.schema()["variables"]["y"]["shape"] == [3]
