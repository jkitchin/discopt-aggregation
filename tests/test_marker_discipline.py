"""RT1: enforce that the ``unit`` marker stays solver-free.

CONTRIBUTING requires ``unit`` tests to run without the solver.  A test that
drives the solver (``model.solve()``, the ``solve``/``_local_solve`` helpers)
must be tagged ``smoke`` or ``slow`` so ``pytest -m unit`` (which conftest's hook
keeps solver-free) never selects it.  This meta-test scans the suite statically
and fails if a solver-driving test would still be collected under ``-m unit`` —
catching a new test that forgets the ``smoke`` tag.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_TESTS_DIR = Path(__file__).resolve().parent


def _calls_solver(fn: ast.FunctionDef) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "solve":
                return True
            if isinstance(f, ast.Name) and f.id in ("solve", "_local_solve"):
                return True
    return False


def _tagged(fn: ast.FunctionDef, name: str) -> bool:
    return any(f"'{name}'" in ast.dump(d) for d in fn.decorator_list)


def test_unit_selection_is_solver_free():
    leaks: list[str] = []
    for path in sorted(_TESTS_DIR.glob("test_*.py")):
        text = path.read_text()
        tree = ast.parse(text)
        file_unit = "pytestmark = pytest.mark.unit" in text
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
                continue
            if not _calls_solver(node):
                continue
            if _tagged(node, "smoke") or _tagged(node, "slow"):
                continue  # correctly excluded from -m unit by conftest's hook
            if file_unit or _tagged(node, "unit"):
                leaks.append(f"{path.name}::{node.name}")
    assert not leaks, (
        "these solver-driving tests are selected by `pytest -m unit`; tag them "
        f"@pytest.mark.smoke (or slow): {leaks}"
    )
