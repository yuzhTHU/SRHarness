from __future__ import annotations

import argparse
import sys
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
    parser.add_argument("--save-dir", default=f"./logs/{SCRIPT_NAME}", help=(
        "Root directory for logs and run artifacts."
    ))
    parser.add_argument("--save-path", default=None, help=(
        "Path to save agent logs and artifacts. Default is auto-generated from --save-dir and --exp-name."
    ))
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind.")
    parser.add_argument("--workspace", default=None, metavar="DIRECTORY", help=(
        "Workspace directory that AI-operated tools may modify. Uses a temporary directory when omitted."
    ))
    parser.add_argument("--mount", nargs="+", default=[], metavar="PATH", help=(
        "Files or directories to mount read-only at the workspace root. Source basenames must be unique."
    ))
    return parser


def _resolve_workspace_path(workspace: str | None, mounts: list[str]) -> Path | None:
    if workspace is None:
        return None
    path = Path(workspace).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Workspace path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        print(
            f"Warning: workspace directory '{path}' is not empty; existing "
            "files may be modified or deleted by AI-operated tools.",
            file=sys.stderr,
        )
    if mounts:
        print(
            "Warning: read-only mount points will be created inside the "
            "selected workspace directory.",
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
        raise SystemExit("Please install web dependencies with: pip install -e .[web]") from exc

    from sr_harness.web.app import create_app
    from sr_harness.runtime import InteractionController
    from sr_harness.web.session import InteractiveSession

    if args.exp_name is None:
        now = datetime.now()
        args.exp_name = sanitize_filename(
            f"{now:%Y%m%d}_{args.name}_{now:%H%M%S}_{gethostname()}"
        )
    else:
        args.exp_name = sanitize_filename(args.exp_name)
    save_path = Path(args.save_path).expanduser() if args.save_path else (
        Path(args.save_dir).expanduser() / args.exp_name
    )
    save_path = save_path.resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    args.save_path = str(save_path)

    workspace_path = _resolve_workspace_path(args.workspace, args.mount)

    controller = InteractionController()
    session = InteractiveSession(
        save_path.parent,
        controller,
        workspace_files=args.mount,
        run_dir=save_path,
        workspace_path=workspace_path,
    )
    app = create_app(save_path.parent, controller=controller, session=session)
    browser_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{browser_host}:{args.port}"
    print(f"SRHarness Interactive: {url}", flush=True)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        controller.command("stop")
        if session.thread:
            session.thread.join(timeout=2)
        session.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(setup_parser().parse_args()))
