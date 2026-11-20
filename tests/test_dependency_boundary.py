"""Test: acoustic_firewall must not transitively import sap_stimulus.

Walks the AST of every .py file under packages/acoustic_firewall and asserts
that 'sap_stimulus' never appears as an import name or module path.
"""
from __future__ import annotations

import ast
from pathlib import Path

_FIREWALL_ROOT = Path(__file__).parent.parent / "packages" / "acoustic_firewall"
_FORBIDDEN = "sap_stimulus"


def _imports_in_file(path: Path) -> list[str]:
    """Return all module names imported in *path* (direct names only, no aliases)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


def test_acoustic_firewall_does_not_import_sap_stimulus():
    """No .py file under acoustic_firewall/ may import sap_stimulus."""
    violations: list[str] = []
    for py_file in _FIREWALL_ROOT.rglob("*.py"):
        for imported in _imports_in_file(py_file):
            if _FORBIDDEN in imported:
                violations.append(f"{py_file.relative_to(_FIREWALL_ROOT)}: imports '{imported}'")
    assert not violations, (
        f"acoustic_firewall imports sap_stimulus — dependency boundary violated:\n"
        + "\n".join(violations)
    )
