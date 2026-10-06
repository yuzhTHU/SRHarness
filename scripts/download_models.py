# Copyright (c) 2026-present, Yumeow. Licensed under the MIT License.
"""Download SRHarness model checkpoints."""
from __future__ import annotations
import argparse
import dotenv
from sr_harness.utils import tag2ansi, download_model, get_default

dotenv.load_dotenv()


def setup_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Configure the command-line argument parser.

    Args:
        parser: Argument parser to configure.

    Returns:
        argparse.ArgumentParser: The operation result.
    """
    default_repo = get_default("repo")
    default_release_tag = get_default("release_tag")
    default_token = get_default("token")

    description = "Download a model checkpoint from the SRHarness model store."
    if parser is None:
        parser = argparse.ArgumentParser(prog="python scripts/download_models.py", description=description)
    else:
        parser.description = description
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint path.")
    parser.add_argument("--name", required=True, help="Remote model name, e.g. property-scratch.")
    parser.add_argument("--repo", default=default_repo, help=f"GitHub repo in owner/name form.")
    parser.add_argument("--release-tag", default=default_release_tag, help=f"GitHub release tag.")
    parser.add_argument("--token", default=default_token, help="GitHub token.")
    return parser


def main(args: argparse.Namespace) -> int:
    """Run the command and return its process exit code.

    Args:
        args: Parsed command-line arguments.

    Returns:
        int: The operation result.
    """
    save_path = download_model(
        name=args.name,
        checkpoint=args.checkpoint,
        repo=args.repo,
        release_tag=args.release_tag,
        token=args.token,
    )
    print(tag2ansi(f"Downloaded [bold green]{args.name}[reset] to [bold blue]{save_path}[reset]"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(setup_parser().parse_args()))
