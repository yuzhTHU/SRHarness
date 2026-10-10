import threading
import time

from sr_harness.core import AgentContext
from sr_harness.tools.workspace_code_executor import WorkspaceCodeExecutorTool
from sr_harness.runtime.workspace import Workspace


def make_tool(path, *, workspace_files=None):
    workspace = Workspace(path)
    for source in workspace_files or ():
        workspace.mount(source)
    return WorkspaceCodeExecutorTool(context=AgentContext(data={"x": [1, 2]}, workspace=workspace))


def test_workspace_executor_reads_and_writes_workspace_files(tmp_path):
    tool = make_tool(tmp_path)
    result = tool(
        program=(
            "import pandas as pd\n"
            "pd.DataFrame({'x': [1], 'y': [2]}).to_csv('prepared.csv', index=False)\n"
            "print(open('prepared.csv').read())\n"
        )
    )
    assert result.ok, result.result_str
    assert (tmp_path / "prepared.csv").read_text() == "x,y\n1,2\n"


def test_workspace_executor_reads_but_cannot_modify_mounted_input(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("x,y\n1,2\n")
    tool = make_tool(tmp_path / "workspace", workspace_files=[str(source)])

    result = tool(program="print(open('source.csv').read())")
    assert result.ok, result.result_str
    assert "x,y" in result.result_str

    result = tool(program="open('source.csv', 'w').write('changed')")
    assert not result.ok
    assert source.read_text() == "x,y\n1,2\n"

    original_mode = source.stat().st_mode
    result = tool(program="import os\nos.chmod('source.csv', 0o777)")
    assert not result.ok
    assert source.stat().st_mode == original_mode


def test_workspace_executor_can_modify_explicit_writable_mount(tmp_path):
    source = tmp_path / "shared"
    source.mkdir()
    workspace = Workspace(tmp_path / "workspace")
    workspace.mount(source, readonly=False)
    tool = WorkspaceCodeExecutorTool(
        context=AgentContext(data={"x": [1, 2]}, workspace=workspace)
    )

    result = tool(program="open('shared/result.txt', 'w').write('result')")

    assert result.ok, result.result_str
    assert (source / "result.txt").read_text() == "result"


def test_workspace_executor_cannot_access_host_paths_or_environment(tmp_path, monkeypatch):
    outside = tmp_path / "outside.txt"
    outside.write_text("host-secret")
    tool = make_tool(tmp_path / "workspace")
    monkeypatch.setenv("SR_HARNESS_SANDBOX_SECRET", "parent-secret")

    result = tool(program=f"print(open({str(outside)!r}).read())")
    assert not result.ok
    assert "host-secret" not in result.result_str

    result = tool(program=("import os\nprint(os.environ.get('SR_HARNESS_SANDBOX_SECRET'))\n"))
    assert result.ok, result.result_str
    assert result.result_str.splitlines()[0] == "None"


def test_workspace_executor_has_no_network_route(tmp_path):
    tool = make_tool(tmp_path)
    result = tool(
        program=("import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=1)\n"),
        timeout_seconds=3,
    )
    assert not result.ok
    assert "Network is unreachable" in result.result_str or "Permission denied" in result.result_str


def test_workspace_executor_writes_unlocked_file_in_locked_directory(tmp_path):
    workspace = Workspace(path=tmp_path)
    directory = tmp_path / "context.data"
    directory.mkdir()
    manifest = directory / "manifest.json"
    manifest.write_text("old")
    workspace.set_locked("context.data", True)
    workspace.set_locked("context.data/manifest.json", False)
    tool = WorkspaceCodeExecutorTool(context=AgentContext(data={"x": [1, 2]}, workspace=workspace))

    result = tool(program="open('context.data/manifest.json', 'w').write('new')")
    assert result.ok, result.result_str
    assert manifest.read_text() == "new"


def test_workspace_executor_enforces_explicit_locks(tmp_path):
    workspace = Workspace(path=tmp_path)
    locked = tmp_path / "locked.txt"
    unlocked = tmp_path / "unlocked.txt"
    locked.write_text("protected")
    unlocked.write_text("old")
    workspace.set_locked("locked.txt", True)
    tool = WorkspaceCodeExecutorTool(
        context=AgentContext(data={"x": [1, 2]}, workspace=workspace)
    )

    overwrite = tool(program="open('locked.txt', 'w').write('changed')")
    remove = tool(program="import os\nos.remove('locked.txt')")
    ordinary_write = tool(
        program=(
            "open('unlocked.txt', 'w').write('new')\n"
            "open('created.txt', 'w').write('created')\n"
        )
    )

    assert not overwrite.ok
    assert not remove.ok
    assert locked.read_text() == "protected"
    assert ordinary_write.ok, ordinary_write.result_str
    assert unlocked.read_text() == "new"
    assert (tmp_path / "created.txt").read_text() == "created"


def test_workspace_executor_can_remove_within_fully_unlocked_subtree(tmp_path):
    workspace = Workspace(path=tmp_path)
    locked = tmp_path / "locked.txt"
    locked.write_text("protected")
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "remove.txt").write_text("remove")
    (clean / "rename.txt").write_text("rename")
    workspace.set_locked("locked.txt", True)
    tool = WorkspaceCodeExecutorTool(
        context=AgentContext(data={"x": [1, 2]}, workspace=workspace)
    )

    result = tool(
        program=(
            "import os\n"
            "os.remove('clean/remove.txt')\n"
            "os.rename('clean/rename.txt', 'clean/renamed.txt')\n"
            "open('clean/created.txt', 'w').write('created')\n"
        )
    )

    assert result.ok, result.result_str
    assert not (clean / "remove.txt").exists()
    assert (clean / "renamed.txt").read_text() == "rename"
    assert (clean / "created.txt").read_text() == "created"
    assert locked.read_text() == "protected"


def test_workspace_executor_keeps_mixed_directory_removal_conservative(tmp_path):
    workspace = Workspace(path=tmp_path)
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    locked = mixed / "locked.txt"
    unlocked = mixed / "unlocked.txt"
    locked.write_text("protected")
    unlocked.write_text("ordinary")
    workspace.set_locked("mixed/locked.txt", True)
    tool = WorkspaceCodeExecutorTool(
        context=AgentContext(data={"x": [1, 2]}, workspace=workspace)
    )

    result = tool(
        program=(
            "import os\n"
            "open('mixed/created.txt', 'w').write('created')\n"
            "os.remove('mixed/unlocked.txt')\n"
        )
    )

    assert not result.ok
    assert locked.read_text() == "protected"
    assert unlocked.read_text() == "ordinary"
    assert (mixed / "created.txt").read_text() == "created"


def test_workspace_executor_honors_unlocked_subtree_override(tmp_path):
    workspace = Workspace(path=tmp_path)
    locked_parent = tmp_path / "locked-parent"
    unlocked_child = locked_parent / "unlocked-child"
    unlocked_child.mkdir(parents=True)
    removable = unlocked_child / "remove.txt"
    removable.write_text("remove")
    workspace.set_locked("locked-parent", True)
    workspace.set_locked("locked-parent/unlocked-child", False)
    tool = WorkspaceCodeExecutorTool(
        context=AgentContext(data={"x": [1, 2]}, workspace=workspace)
    )

    result = tool(
        program=(
            "import os\n"
            "os.remove('locked-parent/unlocked-child/remove.txt')\n"
            "open('locked-parent/unlocked-child/created.txt', 'w').write('created')\n"
        )
    )

    assert result.ok, result.result_str
    assert not removable.exists()
    assert (unlocked_child / "created.txt").read_text() == "created"


def test_workspace_executor_terminates_its_worker_when_cancelled(tmp_path):
    tool = make_tool(tmp_path)
    tool.cancel_event = threading.Event()
    outcome = {}

    def execute():
        outcome["result"] = tool(program="while True:\n    pass", timeout_seconds=30)

    thread = threading.Thread(target=execute)
    thread.start()
    time.sleep(0.15)
    tool.cancel_event.set()
    thread.join(3)

    assert not thread.is_alive()
    assert outcome["result"].ok is False
    assert "InterruptedError" in outcome["result"].result_str
