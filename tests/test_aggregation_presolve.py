"""Tests for S3: native-presolve overlap audit + composition safety.

Confirms the documented overlap set, that plugin-aggregate-then-solve matches a
direct discopt solve for both write-only and used intermediates (composition is
exact, no double-elimination conflict), and the integrality-soundness invariant
(integer/binary variables are never eliminated).
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.api import aggregate
from discopt.aggregation.presolve import (
    native_presolve_overlap,
    recommended_ordering,
    verify_composition,
)

pytestmark = pytest.mark.unit


def test_overlap_covers_the_three_native_passes():
    overlap = native_presolve_overlap()
    names = {o.native_pass for o in overlap}
    assert any("eliminate_variables" in n for n in names)
    assert any("aggregate_variables" in n for n in names)
    assert any("factorable_eliminate" in n for n in names)
    # every entry classifies the relationship
    assert all(o.relationship in {"subset", "complementary"} for o in overlap)


def test_recommended_ordering_is_documented():
    text = recommended_ordering()
    assert "first" in text.lower()
    assert "verify_composition" in text


def _write_only():
    m = dm.Model("wo")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    u = m.continuous("u", lb=-50, ub=50)
    m.subject_to(u - (x + 2 * y) == 0.0)  # u appears only here (write-only)
    m.subject_to(x + y >= 1.0)
    m.minimize(x**2 + y**2)
    return m


def _used():
    m = dm.Model("us")
    x = m.continuous("x", lb=-5, ub=5)
    w = m.continuous("w", lb=-50, ub=50)
    m.subject_to(w - (2 * x + 1) == 0.0)
    m.minimize(w**2 + x)  # w used in the objective -> only the plugin eliminates it
    return m


@pytest.mark.parametrize("builder", [_write_only, _used])
@pytest.mark.parametrize("method", ["gr", "lm"])
def test_composition_matches_direct_solve(builder, method):
    check = verify_composition(builder(), method=method)
    assert check.matches
    assert check.direct_objective == pytest.approx(check.aggregated_objective, abs=1e-5)


def test_integer_variables_are_never_eliminated():
    # the integrality-soundness invariant behind the implied-integrality note
    m = dm.Model("mi")
    a = m.continuous("a", lb=-5, ub=5)
    b = m.integer("b", lb=0, ub=10)
    c = m.continuous("c", lb=-50, ub=50)
    m.subject_to(c - (a + b) == 0.0)
    m.minimize(c**2 + b)
    for method in ("d2", "gr", "lm"):
        assert "b" not in aggregate(m, method=method).eliminated
