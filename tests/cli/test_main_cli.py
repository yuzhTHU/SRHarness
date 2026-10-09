from __future__ import annotations

import pytest

from sr_harness.cli import entrypoint, main, setup_parser
from sr_harness.cli.run import _resolve_workspace_dir
from sr_harness.cli.synthetic import build_agent_options, make_dataset
from sr_harness.agents import SRAgent


def test_main_help_lists_subcommands(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["sr-harness", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        entrypoint()
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "SRHarness command-line interface" in output
    assert "run" in output
    assert "synthetic" in output
    assert "benchmark" in output
    assert "download-models" not in output
    assert "upload-models" not in output


def test_bare_command_prints_help(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["sr-harness"])
    assert entrypoint() == 0
    output = capsys.readouterr().out
    assert "SRHarness command-line interface" in output
    assert "run        Launch the SRHarness interactive workbench" in output
    assert "synthetic  Run SRHarness on a synthetic symbolic-regression problem" in output
    assert "benchmark  Evaluate SRHarness and baseline algorithms on LLM-SRBench" in output


def test_run_help_is_delegated(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["sr-harness", "run", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        entrypoint()
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "usage: sr-harness run" in output
    assert "--web" not in output
    assert "--anonymize" not in output
    assert "--workspace-dir" in output
    assert "--isolate-users" in output
    assert "--mount" in output
    assert "--reload" not in output
    assert "--no-browser" not in output
    assert "--llm-provider" not in output
    assert "--llm-model" not in output


def test_synthetic_help_is_delegated(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["sr-harness", "synthetic", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        entrypoint()
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "usage: sr-harness synthetic" in output
    assert "--equation" in output


def test_synthetic_defaults(monkeypatch):
    argv = ["sr-harness", "synthetic"]
    monkeypatch.setattr("sys.argv", argv)
    args = setup_parser().parse_args(argv[1:])
    assert args.local_sample_size == 1
    assert args.max_refinement_depth == 30
    assert args.global_width == 1
    assert args.max_restart_loop == 1
    assert args.split_by == "random"
    assert args.force_initial_diagnostics is True
    assert args.llm_model == "deepseek/deepseek-v4-flash-0731"
    assert args.tools is None
    assert args.ban_tools == []
    assert args.llm_max_tokens == 4096
    assert args.verbose is False
    assert args.debug is False

    options = build_agent_options(args)
    assert options["tools"] == list(SRAgent.DEFAULT_TOOLS)
    assert "workspace_code_executor" not in options["tools"]
    assert "validate_context_data" not in options["tools"]


def test_synthetic_banned_tools_override_selected_tools(monkeypatch):
    argv = [
        "sr-harness",
        "synthetic",
        "--tools",
        "evaluate_formula",
        "workspace_shell",
        "--ban-tools",
        "workspace_shell",
        "--llm-max-tokens",
        "2048",
    ]
    monkeypatch.setattr("sys.argv", argv)
    args = setup_parser().parse_args(argv[1:])
    options = build_agent_options(args)
    assert options["tools"] == ["evaluate_formula"]
    assert options["llm_max_tokens"] == 2048


def test_synthetic_features_are_space_separated_and_allow_nuisance_variables(monkeypatch):
    argv = [
        "sr-harness", "synthetic",
        "--equation", "y = x1 + x2",
        "--features", "x1", "x2", "x3",
        "--n-samples", "5",
        "--seed", "42",
    ]
    monkeypatch.setattr("sys.argv", argv)
    args = setup_parser().parse_args(argv[1:])
    features, target, _, data = make_dataset(args)

    assert features == ["x1", "x2", "x3"]
    assert target == "y"
    assert set(data) == {"x1", "x2", "x3", "y"}


def test_synthetic_features_may_hide_equation_variables(monkeypatch):
    argv = [
        "sr-harness", "synthetic",
        "--equation", "y = x1 + x2",
        "--features", "x1",
        "--seed", "42",
    ]
    monkeypatch.setattr("sys.argv", argv)
    args = setup_parser().parse_args(argv[1:])
    features, target, _, data = make_dataset(args)

    assert features == ["x1"]
    assert target == "y"
    assert set(data) == {"x1", "y"}
    assert data["y"].shape == (100,)


def test_workspace_dir_accepts_existing_files_and_reports_mounts(tmp_path, capsys):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "existing.txt").write_text("data")

    assert _resolve_workspace_dir(str(workspace), ["input.csv"]) == workspace
    warning = capsys.readouterr().err
    assert "every new conversation workspace" in warning


def test_benchmark_requires_algorithm(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["sr-harness", "benchmark"])
    with pytest.raises(SystemExit) as exc_info:
        entrypoint()
    assert exc_info.value.code == 2
    assert "the following arguments are required: --algorithm" in capsys.readouterr().err


def test_benchmark_boolean_optional_flags(monkeypatch):
    argv = [
        "sr-harness",
        "benchmark",
        "--algorithm",
        "linear",
        "--verbose",
        "--no-skip-successful",
        "--anonymize",
    ]
    monkeypatch.setattr("sys.argv", argv)
    args = setup_parser().parse_args(argv[1:])
    assert args.verbose is True
    assert args.skip_successful is False
    assert args.anonymize is True


def test_main_dispatches_parsed_namespace():
    parser = setup_parser()
    args = parser.parse_args(["tool", "list", "--json"])
    assert main(args) == 0
