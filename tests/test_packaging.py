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
    assert agg.__version__ == "0.3.0"


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


def test_public_api_has_no_duplicates_and_expected_size():
    # __all__ has no duplicates and its size is pinned, so a drift (a new export
    # not reflected in the docs, or an accidental removal) is a loud failure
    # rather than a silently stale count.
    assert len(agg.__all__) == len(set(agg.__all__))
    assert len(agg.__all__) == 81
