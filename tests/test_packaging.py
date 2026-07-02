"""Packaging sanity checks: version consistency and public-API import."""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib

import discopt.aggregation as agg

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parent.parent


def test_version_is_exposed():
    assert isinstance(agg.__version__, str)
    assert agg.__version__ == "0.2.0"


def test_version_matches_pyproject():
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    assert data["project"]["version"] == agg.__version__


def test_changelog_documents_current_version():
    changelog = (_ROOT / "CHANGELOG.md").read_text()
    assert f"[{agg.__version__}]" in changelog


def test_public_api_is_importable():
    # every name in __all__ resolves on the package
    for name in agg.__all__:
        assert hasattr(agg, name), name
