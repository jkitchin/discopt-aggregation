"""RT3: the built wheel actually contains the skill and plugin assets.

``test_packaging.py`` and ``test_skill.py`` validate against the *source* tree,
so deleting ``[tool.setuptools.package-data]`` from pyproject would pass them
while shipping a wheel with no skill files — breaking every pip-installed
``discopt-aggregation-skill install``.  This test builds a real wheel and
inspects it, closing that gap.  It is skipped only when no wheel builder is
available (so it never blocks a minimal environment).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

_ROOT = Path(__file__).resolve().parent.parent


def _build_wheel(outdir: Path) -> Path:
    """Build a wheel into ``outdir`` with whatever builder is available."""
    builders: list[list[str]] = []
    try:
        import build  # noqa: F401

        builders.append([sys.executable, "-m", "build", "--wheel", "-o", str(outdir)])
    except ModuleNotFoundError:
        pass
    if shutil.which("uv"):
        builders.append(["uv", "build", "--wheel", "-o", str(outdir)])
    if not builders:
        pytest.skip("no wheel builder available (need `build` or `uv`)")
    last: subprocess.CompletedProcess | None = None
    for cmd in builders:
        last = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
        if last.returncode == 0:
            wheels = list(outdir.glob("*.whl"))
            assert wheels, f"builder produced no wheel: {cmd}"
            return wheels[0]
    raise AssertionError(f"wheel build failed: {last.stderr if last else 'no builder ran'}")


def test_wheel_ships_skill_and_plugin_assets(tmp_path: Path):
    wheel = _build_wheel(tmp_path)
    names = set(zipfile.ZipFile(wheel).namelist())
    required = {
        "discopt/aggregation/skill/SKILL.md",
        "discopt/aggregation/skill/README.md",
        "discopt/aggregation/skill/references/methods.md",
        "discopt/aggregation/skill/references/options.md",
        "discopt/aggregation/skill/references/troubleshooting.md",
        "discopt/aggregation/skill/references/recovery-and-implicit.md",
        "discopt/aggregation/skill/references/study-tools.md",
        "discopt/aggregation/skill/examples/quickstart.py",
        "discopt/aggregation/skill/examples/aggressive_safely.py",
    }
    missing = required - names
    assert not missing, f"wheel is missing packaged skill files: {sorted(missing)}"
