"""Search-state candidate projection tests."""

from sr_harness.core import SearchRunState, ToolCall, ToolCallResult


def _evaluation(*, mse: float, formula: str, is_candidate: bool) -> ToolCallResult:
    return ToolCallResult(
        True,
        {
            "formula": formula,
            "is_candidate": is_candidate,
            "data_split_results": {
                "train": {"metrics": {"mse": mse, "rmse": mse**0.5, "r2": 1 - mse}},
            },
        },
        "",
        {},
    )


def test_node_core_ignores_better_scoring_non_candidate_result():
    core = SearchRunState._build_core(
        "",
        [ToolCall(name="diagnostic", params={}), ToolCall(name="evaluate_formula", params={})],
        [
            _evaluation(mse=0.0, formula="alternate_target_fit", is_candidate=False),
            _evaluation(mse=0.2, formula="real_target_fit", is_candidate=True),
        ],
        {"role": "assistant", "content": ""},
    )

    assert core["is_candidate"] is True
    assert core["formula"] == "real_target_fit"
    assert core["score"]["mse"] == 0.2


def test_node_core_has_no_score_when_only_diagnostics_have_metrics():
    core = SearchRunState._build_core(
        "",
        [ToolCall(name="diagnostic", params={})],
        [_evaluation(mse=0.0, formula="alternate_target_fit", is_candidate=False)],
        {"role": "assistant", "content": ""},
    )

    assert core["is_candidate"] is False
    assert core["formula"] is None
    assert core["score"] is None
