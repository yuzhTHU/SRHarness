#!/usr/bin/env python3
"""Generate the API sections embedded in the two long-form documentation pages."""

from __future__ import annotations

import ast
import re
from pathlib import Path


START = "<!-- API_REFERENCE_START -->"
END = "<!-- API_REFERENCE_END -->"


def signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a compact signature from a callable syntax-tree node.

    Args:
        node: Callable definition to render.

    Returns:
        str: Signature without a function body or return annotation prefix.
    """
    clone = ast.FunctionDef(
        name=node.name,
        args=node.args,
        body=[ast.Pass()],
        decorator_list=[],
        returns=node.returns,
        type_comment=None,
    )
    text = ast.unparse(ast.fix_missing_locations(clone))
    return text.split(":\n", 1)[0].removeprefix("def ")


def markdown_docstring(docstring: str) -> str:
    """Convert a Google-style docstring to compact Markdown.

    Args:
        docstring: Cleaned docstring text.

    Returns:
        str: Markdown representation.
    """
    lines = docstring.splitlines()
    output: list[str] = []
    section = None
    for line in lines:
        stripped = line.strip()
        if stripped in {"Args:", "Returns:", "Yields:", "Raises:", "Attributes:"}:
            section = stripped[:-1]
            output.extend(["", f"**{section}**", ""])
            continue
        match = re.match(r"^\s{4}([^:]+):\s*(.*)$", line)
        if section and match:
            output.append(f"- `{match.group(1)}`: {match.group(2)}")
        else:
            output.append(line)
    return "\n".join(output).strip()


def module_reference(path: Path, package_root: Path) -> str:
    """Generate one module section.

    Args:
        path: Python module path.
        package_root: Parent directory containing the package.

    Returns:
        str: Markdown for the module's public API.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative = path.relative_to(package_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    module_name = ".".join(parts)
    entries: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            entries.extend([
                f"### `{module_name}.{signature(node)}`",
                "",
                markdown_docstring(ast.get_docstring(node) or ""),
                "",
            ])
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            entries.extend([
                f"### `{module_name}.{node.name}`",
                "",
                markdown_docstring(ast.get_docstring(node) or ""),
                "",
            ])
            for method in node.body:
                if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if method.name.startswith("_"):
                    continue
                entries.extend([
                    f"#### `{node.name}.{signature(method)}`",
                    "",
                    markdown_docstring(ast.get_docstring(method) or ""),
                    "",
                ])
    if not entries:
        return ""
    return "\n".join([f"## `{module_name}`", "", *entries]).rstrip()


def package_reference(package: Path) -> str:
    """Generate a complete package reference from source docstrings.

    Args:
        package: Package directory below ``src``.

    Returns:
        str: Markdown containing every public module entry.
    """
    sections = []
    for path in sorted(package.rglob("*.py")):
        if "utils" in path.parts or "_vendor" in path.parts:
            continue
        section = module_reference(path, Path("src"))
        if section:
            sections.append(section)
    return "\n\n".join(sections)


def replace_generated_section(path: Path, content: str) -> None:
    """Replace the generated API block in a Markdown page.

    Args:
        path: Documentation page to update.
        content: Generated API reference Markdown.
    """
    source = path.read_text(encoding="utf-8")
    if START not in source or END not in source:
        raise ValueError(f"Missing API reference markers in {path}")
    before, remainder = source.split(START, 1)
    _, after = remainder.split(END, 1)
    path.write_text(
        f"{before}{START}\n\n{content}\n\n{END}{after}",
        encoding="utf-8",
    )


def main() -> int:
    """Regenerate both embedded API references.

    Returns:
        int: Zero after successful generation.
    """
    replace_generated_section(Path("docs/index.md"), package_reference(Path("src/sr_harness")))
    replace_generated_section(
        Path("docs/engine.md"),
        package_reference(Path("src/sr_harness_engine")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
