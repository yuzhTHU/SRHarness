import threading
import time

from sr_harness.core import AgentContext
from sr_harness.tools.workspace_code_executor import (
    WorkspaceCodeExecutorTool,
    WorkspaceSandBoxCodeExecutor,
)
from sr_harness.tools.workspace_shell import Workspace


def test_workspace_executor_supports_pathlib_open_opener(tmp_path):
    workspace = Workspace(path=tmp_path)
    tool = WorkspaceCodeExecutorTool(context=AgentContext(data={"x": [1, 2]}, workspace=workspace))
    result = tool(program=(
        "import pandas as pd\n"
        "pd.DataFrame({'x': [1], 'y': [2]}).to_csv('prepared.csv', index=False)\n"
        "print(open('prepared.csv').read())\n"
    ))
    assert result.ok, result.result_str
    assert (tmp_path / "prepared.csv").read_text() == "x,y\n1,2\n"


def test_workspace_executor_reads_but_cannot_modify_mounted_input(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x,y\n1,2\n")
    root = tmp_path / "workspace"
    workspace = Workspace(workspace_files=[str(source)], path=root)
    tool = WorkspaceCodeExecutorTool(
        context=AgentContext(data={"x": [1, 2]}, workspace=workspace)
    )

    result = tool(program="print(open('source.csv').read())")
    assert result.ok, result.result_str
    assert "x,y" in result.result_str

    result = tool(program="open('source.csv', 'w').write('changed')")
    assert not result.ok
    assert source.read_text() == "x,y\n1,2\n"


def test_workspace_executor_rejects_user_locked_files(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x,y\n1,2\n")
    source.chmod(0o444)

    assert WorkspaceSandBoxCodeExecutor.check_workspace_path(
        source, str(tmp_path), write=False,
    ) == str(source)
    try:
        WorkspaceSandBoxCodeExecutor.check_workspace_path(
            source, str(tmp_path), write=True,
        )
    except PermissionError as exc:
        assert "已锁定" in str(exc)
    else:
        raise AssertionError("locked file unexpectedly accepted for writing")


def test_workspace_executor_writes_unlocked_file_in_locked_directory(tmp_path):
    workspace = Workspace(path=tmp_path)
    directory = tmp_path / "context.data"
    directory.mkdir()
    manifest = directory / "manifest.json"
    manifest.write_text("old")
    workspace.set_locked("context.data", True)
    workspace.set_locked("context.data/manifest.json", False)
    tool = WorkspaceCodeExecutorTool(
        context=AgentContext(data={"x": [1, 2]}, workspace=workspace)
    )

    result = tool(program="open('context.data/manifest.json', 'w').write('new')")

    assert result.ok, result.result_str
    assert manifest.read_text() == "new"
    result = tool(program="import os\nos.remove('context.data/manifest.json')")
    assert not result.ok
    assert manifest.exists()


def test_workspace_executor_terminates_its_worker_when_cancelled(tmp_path):
    workspace = Workspace(path=tmp_path)
    tool = WorkspaceCodeExecutorTool(
        context=AgentContext(data={"x": [1, 2]}, workspace=workspace)
    )
    tool.cancel_event = threading.Event()
    outcome = {}

    def execute():
        try:
            outcome["result"] = tool(
                program="while True:\n    pass", timeout_seconds=30,
            )
        except Exception as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=execute)
    thread.start()
    time.sleep(.15)
    tool.cancel_event.set()
    thread.join(2)

    assert not thread.is_alive()
    assert "error" not in outcome
    assert outcome["result"].ok is False
    assert "InterruptedError" in outcome["result"].result_str
