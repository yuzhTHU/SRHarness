from sr_harness.core import AgentContext
from sr_harness.tools.workspace_code_executor import WorkspaceCodeExecutorTool
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
