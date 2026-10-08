from sr_harness import AgentContext
from sr_harness.tools.read_source import ReadSourceTool


def test_read_source_lists_tree_and_extracts_symbol():
    tool = ReadSourceTool(context=AgentContext())
    listing = tool.execute("src/sr_harness/evaluator")
    assert "sr_harness/evaluator/default_evaluator.py" in listing["entries"]
    result = tool.execute(
        "src/sr_harness/evaluator/default_evaluator.py",
        "DefaultEvaluator.evaluate_candidate",
    )
    assert "def evaluate_candidate" in result["source"]


def test_read_source_queries_implementations_and_references_across_a_directory():
    result = ReadSourceTool(context=AgentContext()).execute(
        path="src/sr_harness",
        symbol="DefaultEvaluator",
        include_implementation=True,
        include_references=True,
    )
    assert any(item["qualified_name"] == "DefaultEvaluator" for item in result["definitions"])
    assert any(item["path"].endswith("graph_evaluator.py") for item in result["references"])


def test_read_source_can_query_references_without_returning_implementation():
    result = ReadSourceTool(context=AgentContext()).execute(
        symbol="load_custom_evaluator",
        include_implementation=False,
        include_references=True,
    )
    assert result["definitions"] == []
    assert result["references"]


def test_read_source_rejects_escape():
    result = ReadSourceTool(context=AgentContext())(path="../pyproject.toml")
    assert not result.ok
    assert "inside the installed SRHarness source tree" in result.result_str
