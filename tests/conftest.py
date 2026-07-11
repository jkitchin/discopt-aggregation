"""Test-collection policy shared across the suite.

Marker discipline (CONTRIBUTING §Development): ``unit`` tests must be solver-free.
Many files carry a file-level ``pytestmark = pytest.mark.unit`` for their
pure-logic tests but also contain a handful of tests that drive the solver; those
are tagged ``@pytest.mark.smoke`` (or ``slow``) individually.  Because a
module-level mark applies to every test in the file, such a test ends up carrying
*both* ``unit`` and ``smoke`` — and ``pytest -m unit`` would still select it,
re-introducing a solver dependency into the "unit" set.

The hook below makes the more-specific mark win: when the selection is a plain
``unit`` run, any test that is *also* ``smoke``/``slow`` is deselected, so
``pytest -m unit`` is guaranteed solver-free even on a machine without pounce.
The full suite (no ``-m`` filter, as CI runs it) is unaffected and still runs
every test.
"""

from __future__ import annotations


def pytest_collection_modifyitems(config, items):
    markexpr = config.getoption("markexpr", default="") or ""
    # Only act on a unit-only selection; leave `-m smoke`, `-m slow`, combined
    # expressions, and the unfiltered full run untouched.
    if "unit" not in markexpr or "smoke" in markexpr or "slow" in markexpr:
        return
    kept, deselected = [], []
    for item in items:
        names = {m.name for m in item.iter_markers()}
        if "unit" in names and ("smoke" in names or "slow" in names):
            deselected.append(item)
        else:
            kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept
