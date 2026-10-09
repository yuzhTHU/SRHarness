from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from socket import gethostname

from sr_harness.utils import sanitize_filename


SCRIPT_NAME = "run"


def setup_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Configure the command-line argument parser.

    Args:
        parser: Argument parser to configure.

    Returns:
        argparse.ArgumentParser: The operation result.
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            prog="sr-harness run",
            description="Serve the SRHarness interactive workbench.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    else:
        parser.description = "Serve the SRHarness interactive workbench."
        parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser.add_argument("--name", default=SCRIPT_NAME, help=(
        "Experiment task name used when auto-generating exp_name."
    ))
    parser.add_argument("--exp-name", default=None, help=(
        "Experiment name. Defaults to a timestamped name."
    ))
    parser.add_argument("--save-dir", default=None, help=(
        "Root directory for logs and run artifacts. When omitted together with --save-path, a temporary directory is used."
    ))
    parser.add_argument("--save-path", default=None, help=(
        "Path to save agent logs and artifacts. Default is auto-generated from --save-dir and --exp-name."
    ))
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind.")
    parser.add_argument("--workspace-dir", default=None, metavar="DIRECTORY", help=(
        "Directory that stores the conversation registry and per-conversation workspaces."
    ))
    parser.add_argument("--isolate-users", action="store_true", help=(
        "Only show each browser the conversations created under its own persistent client cookie."
    ))
    parser.add_argument("--mount", nargs="+", default=[], metavar="PATH", help=(
        "Files or directories to mount read-only at the workspace root. Source basenames must be unique."
    ))
    return parser


def _resolve_workspace_dir(workspace_dir: str | None, mounts: list[str]) -> Path | None:
    if workspace_dir is None:
        return None
    path = Path(workspace_dir).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Workspace directory is not a directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    if mounts:
        print(
            "Read-only mount points will be created inside every new conversation workspace.",
            file=sys.stderr,
        )
    return path


def main(args: argparse.Namespace) -> int:
    """Run the command and return its process exit code.

    Args:
        args: Parsed command-line arguments.

    Returns:
        int: The operation result.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Please reinstall SRHarness with its default dependencies.") from exc

    from sr_harness.web.app import create_app
    from sr_harness.web.conversations import ConversationRegistry

    if args.exp_name is None:
        now = datetime.now()
        args.exp_name = sanitize_filename(
            f"{now:%Y%m%d}_{args.name}_{now:%H%M%S}_{gethostname()}"
        )
    else:
        args.exp_name = sanitize_filename(args.exp_name)
    requested_save_path = Path(args.save_path).expanduser() if args.save_path else (
        Path(args.save_dir).expanduser() / args.exp_name if args.save_dir else None
    )
    workspace_dir = _resolve_workspace_dir(args.workspace_dir, args.mount)
    if workspace_dir is None and requested_save_path is not None:
        workspace_dir = requested_save_path.resolve()
        workspace_dir.mkdir(parents=True, exist_ok=True)
    elif workspace_dir is None:
        workspace_dir = Path(tempfile.mkdtemp(prefix="sr_harness_workspace_")).resolve()
        print(
            "\033[31mWarning: Workspaces will be created in a temporary directory; "
            "conversation history may be lost. Specify --save-path and/or --workspace-dir.\033[0m",
            file=sys.stderr,
        )
    save_path = (
        requested_save_path.resolve()
        if requested_save_path is not None
        else workspace_dir / "runs" / args.exp_name
    )
    save_path.mkdir(parents=True, exist_ok=True)
    args.save_path = str(save_path)
    args.workspace_dir = str(workspace_dir)

    conversations = ConversationRegistry(
        workspace_dir,
        workspace_files=args.mount,
        isolate_users=args.isolate_users,
        initial_run_dir=save_path,
        persist_sessions=requested_save_path is not None,
    )
    session = conversations.session_proxy
    app = create_app(save_path.parent, session=session, conversation_registry=conversations)
    browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{browser_host}:{args.port}"
    print(f"SRHarness Interactive: {url}", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        conversations.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(setup_parser().parse_args()))
