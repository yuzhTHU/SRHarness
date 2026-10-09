"""Read files, definitions, and references from installed SRHarness source."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .base_tool import BaseTool
from ..core import ToolMetadata

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_ALLOWED_SUFFIXES = {".py", ".md", ".html", ".json", ".toml", ".yaml", ".yml"}
_MAX_SOURCE_SIZE = 200_000
_MAX_SYMBOL_RESULTS = 200


def _resolve(path: str | None) -> Path:
    path = path or "."
    if path == "src":
        path = "."
    elif path.startswith("src/"):
        path = path[4:]
    target = (_SOURCE_ROOT / path).resolve()
    if target != _SOURCE_ROOT and _SOURCE_ROOT not in target.parents:
        raise ValueError("path must remain inside the installed SRHarness source tree")
    return target


def _python_files(target: Path) -> list[Path]:
    if target.is_file():
        if target.suffix != ".py":
            raise ValueError("symbol queries require a Python file or directory")
        return [target]
    return [item for item in sorted(target.rglob("*.py")) if "__pycache__" not in item.parts]


def _matches_symbol(candidate: str, query: str) -> bool:
    return candidate == query or candidate.endswith("." + query)


def _attribute_name(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _symbol_results(path: Path, source: str, symbol: str, *, implementation: bool, references: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], []
    lines = source.splitlines()
    relative = str(path.relative_to(_SOURCE_ROOT))
    definitions: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.scope: list[str] = []

        def _definition(self, node):
            qualified = ".".join([*self.scope, node.name])
            if implementation and _matches_symbol(qualified, symbol):
                definitions.append({
                    "path": relative, "line": node.lineno, "qualified_name": qualified,
                    "source": "\n".join(lines[node.lineno - 1:node.end_lineno]),
                })
            self.scope.append(node.name)
            self.generic_visit(node)
            self.scope.pop()

        visit_ClassDef = _definition
        visit_FunctionDef = _definition
        visit_AsyncFunctionDef = _definition

        def visit_Attribute(self, node):
            name = _attribute_name(node)
            if references and name and _matches_symbol(name, symbol):
                usages.append({
                    "path": relative, "line": node.lineno, "column": node.col_offset,
                    "reference": name, "source": lines[node.lineno - 1].strip(),
                })
            self.generic_visit(node)

        def visit_Name(self, node):
            if references and "." not in symbol and node.id == symbol:
                usages.append({
                    "path": relative, "line": node.lineno, "column": node.col_offset,
                    "reference": node.id, "source": lines[node.lineno - 1].strip(),
                })

        def visit_ImportFrom(self, node):
            if references:
                for alias in node.names:
                    imported = f"{node.module}.{alias.name}" if node.module else alias.name
                    if _matches_symbol(imported, symbol) or _matches_symbol(alias.name, symbol):
                        usages.append({
                            "path": relative, "line": node.lineno, "column": node.col_offset,
                            "reference": imported, "source": lines[node.lineno - 1].strip(),
                        })
            self.generic_visit(node)

    Visitor().visit(tree)
    unique = {(item["path"], item["line"], item["column"], item["reference"]): item for item in usages}
    return definitions, list(unique.values())


@BaseTool.register("read_source")
class ReadSourceTool(BaseTool):
    """Inspect installed source files and locate symbol definitions or references."""

    metadata = ToolMetadata(
        name="read_source",
        description=(
            "List a directory, read a source file, or query a Python symbol's implementation "
            "and references. Paths are relative to src; omit path to search all SRHarness source."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Optional path relative to src."},
                "symbol": {"type": "string", "description": "Optional symbol such as DefaultEvaluator.evaluate."},
                "include_implementation": {"type": "boolean", "description": "Return matching definitions; defaults to true."},
                "include_references": {"type": "boolean", "description": "Return matching usages; defaults to false."},
            },
            "additionalProperties": False,
        },
    )

    def execute(self, path: str | None = None, symbol: str | None = None, include_implementation: bool = True, include_references: bool = False) -> dict[str, Any]:
        """Read a source path or locate a Python symbol.

        Args:
            path: File or directory relative to the installed ``src`` tree.
            symbol: Qualified or unqualified Python symbol to locate.
            include_implementation: Whether symbol queries return definitions.
            include_references: Whether symbol queries return references.

        Returns:
            A directory listing, file source, or symbol-query result.
        """
        target = _resolve(path)
        if not target.exists():
            raise ValueError(f"Source path does not exist: {path}")
        if symbol:
            if not include_implementation and not include_references:
                raise ValueError("a symbol query must request implementation, references, or both")
            definitions, references = [], []
            for item in _python_files(target):
                source = item.read_text(encoding="utf-8")
                if len(source) > _MAX_SOURCE_SIZE:
                    continue
                found_definitions, found_references = _symbol_results(
                    item, source, symbol,
                    implementation=include_implementation,
                    references=include_references,
                )
                definitions.extend(found_definitions)
                references.extend(found_references)
                if len(definitions) + len(references) >= _MAX_SYMBOL_RESULTS:
                    break
            result = {
                "path": str(target.relative_to(_SOURCE_ROOT)), "type": "symbol", "symbol": symbol,
                "definitions": definitions[:_MAX_SYMBOL_RESULTS],
                "references": references[:max(0, _MAX_SYMBOL_RESULTS - len(definitions))],
            }
            if len(result["definitions"]) == 1:
                result["source"] = result["definitions"][0]["source"]
            return result
        if target.is_dir():
            entries = [
                str(item.relative_to(_SOURCE_ROOT)) for item in sorted(target.rglob("*"))
                if item.is_file() and item.suffix in _ALLOWED_SUFFIXES and "__pycache__" not in item.parts
            ]
            return {"path": str(target.relative_to(_SOURCE_ROOT)), "type": "directory", "entries": entries}
        if target.suffix not in _ALLOWED_SUFFIXES:
            raise ValueError(f"Unsupported source-file type: {target.suffix}")
        source = target.read_text(encoding="utf-8")
        if len(source) > _MAX_SOURCE_SIZE:
            raise ValueError(f"Source file exceeds {_MAX_SOURCE_SIZE} characters")
        return {"path": str(target.relative_to(_SOURCE_ROOT)), "type": "file", "source": source}

    @classmethod
    def format_result_dict(cls, result: dict[str, Any]) -> str:
        """Format a source inspection result for a language model.

        Args:
            result: Structured result returned by :meth:`execute`.

        Returns:
            Human-readable source text or query matches.
        """
        if result["type"] == "directory":
            return "Source tree: " + result["path"] + "\n" + "\n".join(result["entries"])
        if result["type"] == "file":
            return f"Source: {result['path']}\n```\n{result['source']}\n```"
        lines = [f"Symbol: {result['symbol']} (scope: {result['path'] or 'src'})"]
        for item in result["definitions"]:
            lines.append(f"Implementation: {item['path']}:{item['line']} · {item['qualified_name']}\n```python\n{item['source']}\n```")
        for item in result["references"]:
            lines.append(f"Reference: {item['path']}:{item['line']} · `{item['source']}`")
        if len(lines) == 1:
            lines.append("No matching implementation or reference found.")
        return "\n".join(lines)
