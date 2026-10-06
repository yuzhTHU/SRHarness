#!/usr/bin/env python3
"""Check public Python APIs for English Google-style docstrings."""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOTS = (Path("src/sr_harness"), Path("src/sr_harness_engine"))


def public_functions(tree: ast.Module):
    """Yield public top-level functions and direct public class methods.

    Args:
        tree: Parsed Python module.

    Yields:
        ast.FunctionDef | ast.AsyncFunctionDef: Public callable definitions.
    """
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                yield node
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for method in node.body:
                if (
                    isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not method.name.startswith("_")
                ):
                    yield method


def arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Return user-facing argument names for a callable definition.

    Args:
        node: Callable syntax-tree node.

    Returns:
        list[str]: Argument names excluding ``self`` and ``cls``.
    """
    values = [
        argument.arg
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        if argument.arg not in {"self", "cls"}
    ]
    if node.args.vararg:
        values.append(node.args.vararg.arg)
    if node.args.kwarg:
        values.append(node.args.kwarg.arg)
    return values


def main() -> int:
    """Validate repository docstrings and return a process exit code.

    Returns:
        int: Zero when every checked callable is documented; otherwise one.
    """
    failures: list[str] = []
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "utils" in path.parts or "_vendor" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in public_functions(tree):
                docstring = ast.get_docstring(node) or ""
                location = f"{path}:{node.lineno}:{node.name}"
                if not docstring:
                    failures.append(f"{location}: missing docstring")
                    continue
                if re.search(r"[\u4e00-\u9fff]", docstring):
                    failures.append(f"{location}: docstring must be English")
                if arguments(node) and "Args:" not in docstring:
                    failures.append(f"{location}: missing Google-style Args section")
    if failures:
        print("\n".join(failures))
        return 1
    print("All public callables have English Google-style docstrings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
