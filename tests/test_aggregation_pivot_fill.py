"""Tests for F2: pivot tolerance and fill-in caps in aggregation selection.

- Pivot tolerance guards against eliminating a variable via a near-zero pivot
  (dividing the defining equation by a tiny coefficient — the ill-conditioning
  hazard Achterberg et al. flag). The default is on; ``pivot_tol=0`` recovers
  exact paper behaviour.
- The fill-in cap skips eliminations estimated to add too many Jacobian
  incidences, and dropping them keeps the set strictly lower triangular.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.api import aggregate
from discopt.aggregation.incidence import build_incidence, linear_vars
from discopt.aggregation.strategies import aggregation_set

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# pivot tolerance                                                             #
# --------------------------------------------------------------------------- #
def test_linear_vars_pivot_filters_tiny_coefficient():
    m = dm.Model("piv")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.subject_to(1e-8 * y + x == 1.0)
    c = m._constraints[0]
    names = {"x", "y"}
    # No tolerance: both are linear (y with a tiny coefficient).
    assert linear_vars(c, names, pivot_tol=0.0) == {"x": 1.0, "y": pytest.approx(1e-8)}
    # Default-scale tolerance drops the tiny pivot; x (coeff 1) survives.
    assert linear_vars(c, names, pivot_tol=1e-6) == {"x": 1.0}


def test_pivot_tolerance_is_relative_to_row_scale():
    # Coefficients 2 and 3 are both fine at 1e-6 (scale 3): neither is tiny.
    m = dm.Model("rel")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.subject_to(2 * x + 3 * y == 1.0)
    c = m._constraints[0]
    assert set(linear_vars(c, {"x", "y"}, pivot_tol=1e-6)) == {"x", "y"}


def test_build_incidence_applies_pivot_default():
    m = dm.Model("piv2")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.subject_to(1e-8 * y + x == 1.0)
    inc = build_incidence(m)  # default pivot_tol = 1e-6
    assert inc.constraint_linear_vars(0) == {"x"}  # y filtered as a tiny pivot
    inc0 = build_incidence(m, pivot_tol=0.0)
    assert inc0.constraint_linear_vars(0) == {"x", "y"}


def test_aggregate_avoids_tiny_pivot_by_default():
    # y is defined only by a tiny-pivot equation; eliminating it would divide by
    # ~1e-8. The default guard refuses; pivot_tol=0 permits it.
    def build():
        m = dm.Model("tiny")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-1e9, ub=1e9)
        m.subject_to(1e-8 * y - x**2 == 0.0)  # y = 1e8 * x^2; y is the only linear var
        m.minimize(x**2 + 0 * y)
        return m

    default = aggregate(build(), method="lm")
    assert "y" not in default.eliminated  # guarded

    permissive = aggregate(build(), method="lm", pivot_tol=0.0)
    assert "y" in permissive.eliminated  # tiny pivot allowed


# --------------------------------------------------------------------------- #
# fill-in cap                                                                 #
# --------------------------------------------------------------------------- #
def _fill_model():
    # y is the only linear variable in its 4-variable defining equation (a,b,c
    # appear squared), and y is used nonlinearly in two more constraints, so
    # eliminating y has a large fill; w1, w2 have zero fill.
    m = dm.Model("fill")
    a = m.continuous("a", lb=-5, ub=5)
    b = m.continuous("b", lb=-5, ub=5)
    c = m.continuous("c", lb=-5, ub=5)
    y = m.continuous("y", lb=-50, ub=50)
    w1 = m.continuous("w1", lb=-50, ub=50)
    w2 = m.continuous("w2", lb=-50, ub=50)
    m.subject_to(y - (a**2 + b**2 + c**2) == 0.0)  # cid 0, defines y
    m.subject_to(w1 - y * a == 0.0)  # cid 1, defines w1; y used
    m.subject_to(w2 - y * b == 0.0)  # cid 2, defines w2; y used
    m.minimize(a + b + c)
    return m


def test_fill_estimate_value():
    inc = build_incidence(_fill_model())
    # constraint 0 has vars {y,a,b,c} (3 others); y appears in c0,c1,c2 (deg 3).
    # fill = (4 - 1) * (3 - 1) = 6.
    assert inc.fill_estimate(0, "y") == 6
    # w1 appears only in its defining constraint -> zero fill.
    assert inc.fill_estimate(1, "w1") == 0


def test_max_fill_drops_high_fill_elimination():
    # Single pass so fill is evaluated once against the original structure.
    capped = aggregate(_fill_model(), method="gr", recursive=False, max_fill=5)
    assert "y" not in capped.eliminated  # fill 6 > 5 -> dropped
    assert {"w1", "w2"} <= set(capped.eliminated)  # zero-fill, kept

    uncapped = aggregate(_fill_model(), method="gr", recursive=False, max_fill=None)
    assert "y" in uncapped.eliminated


def test_fill_cap_preserves_lower_triangularity():
    inc = build_incidence(_fill_model(), max_fill=5)
    agg = aggregation_set(inc, method="gr")
    # Reuse the invariant: each elimination's defining constraint references,
    # among eliminated variables, only earlier ones.
    pos = {e.var: i for i, e in enumerate(agg.eliminations)}
    eliminated = set(pos)
    for i, e in enumerate(agg.eliminations):
        for w in inc.constraint_vars(e.cid) - {e.var}:
            if w in eliminated:
                assert pos[w] < i


def test_defaults_do_not_change_well_conditioned_models():
    # A well-conditioned model must aggregate identically with the new defaults.
    m = dm.Model("ok")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.subject_to(y - (x + 1) == 0.0)
    m.minimize(x**2 + y**2)
    res = aggregate(m, method="d2")
    assert res.n_eliminated == 1  # y eliminated, unchanged from paper behaviour
