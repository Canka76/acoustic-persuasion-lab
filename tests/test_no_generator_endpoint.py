"""Test: no module under apps/ may import sap_stimulus.

The study app serves pre-rendered files from disk only.
The stimulus generator must never be reachable via a network surface.
"""
from __future__ import annotations

import ast
from pathlib import Path

_APPS_ROOT = Path(__file__).parent.parent / "apps"
_FORBIDDEN = "sap_stimulus"


def _imports_in_file(path: Path) -> list[str]:
    """Return all module names imported in *path*."""
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


def test_apps_do_not_import_sap_stimulus():
    """No .py file under apps/ may import sap_stimulus."""
    violations: list[str] = []
    for py_file in _APPS_ROOT.rglob("*.py"):
        for imported in _imports_in_file(py_file):
            if _FORBIDDEN in imported:
                violations.append(f"{py_file.relative_to(_APPS_ROOT)}: imports '{imported}'")
    assert not violations, (
        f"apps/ imports sap_stimulus — ethical gate violated (generator must not be a network service):\n"
        + "\n".join(violations)
    )
