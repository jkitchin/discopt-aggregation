"""Tests for D1: recursive-path + surfaced dual recovery.

``recover_duals_from_result`` recovers full-space multipliers from a whole
(recursive, multi-pass) ``AggregationResult`` by composing the per-pass
constraint-origin maps and back-substituting in the base model.  ``solve(...,
recover_duals=True)`` surfaces the result.  These tests certify correctness via
the full-space KKT stationarity residual (~0), cross-check against the original
model's own multipliers, and confirm method-independence and the documented
out-of-scope cases.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

from discopt.aggregation.api import aggregate, solve
from discopt.aggregation.duals import (
    kkt_stationarity_residual,
    recover_duals_from_result,
)

pytestmark = pytest.mark.unit


def _convex_recursive_model():
    """Convex model: an ld1 cascade (a->b) then a d2 elimination, + an inequality.

    Linear defining equalities keep it convex, so the reduced solve takes the
    convex fast path and exposes constraint multipliers.
    """
    m = dm.Model("d1c")
    a = m.continuous("a", lb=-10, ub=10)
    b = m.continuous("b", lb=-20, ub=20)
    w = m.continuous("w", lb=-20, ub=20)
    x = m.continuous("x", lb=-5, ub=5)
    z = m.continuous("z", lb=-5, ub=5)
    m.subject_to(a == 3.0)  # ld1
    m.subject_to(b - 2 * a == 1.0)  # ld1 cascade -> b = 7
    m.subject_to(w - (x + b) == 0.0)  # d2 (after b substituted) -> w = x + 7
    m.subject_to(x + z >= 1.0)  # inequality, survives
    m.minimize(x**2 + w**2 + z**2 + 0.1 * b)
    return m


def test_recursive_recovery_satisfies_kkt():
    result = aggregate(_convex_recursive_model(), method="d2")
    assert result.n_passes >= 2  # genuinely recursive (ld1 cascade + d2)
    reduced_result = result.reduced_model.solve()
    dr = recover_duals_from_result(result, reduced_result)
    assert dr.available
    resid = kkt_stationarity_residual(result.base_model, dr.full_x, dr.constraint_duals)
    assert resid < 1e-6


def test_recursive_recovery_matches_original_duals():
    m = _convex_recursive_model()
    result = aggregate(m, method="d2")
    dr = recover_duals_from_result(result, result.reduced_model.solve())
    assert dr.available
    original = _convex_recursive_model().solve()
    for cid, mu in dr.constraint_duals.items():
        want = float(np.asarray(original.constraint_duals[f"c{cid}"]).reshape(()))
        assert mu == pytest.approx(want, abs=1e-5)


@pytest.mark.parametrize("method", ["d2", "gr", "lm"])
def test_recovery_is_method_independent(method):
    result = aggregate(_convex_recursive_model(), method=method)
    dr = recover_duals_from_result(result, result.reduced_model.solve())
    assert dr.available
    resid = kkt_stationarity_residual(result.base_model, dr.full_x, dr.constraint_duals)
    assert resid < 1e-6
    # the full-space KKT point is unique, so every method recovers the same duals
    original = _convex_recursive_model().solve()
    for cid, mu in dr.constraint_duals.items():
        want = float(np.asarray(original.constraint_duals[f"c{cid}"]).reshape(()))
        assert mu == pytest.approx(want, abs=1e-5)


def test_solve_surfaces_duals_only_when_requested():
    plain = solve(_convex_recursive_model(), method="d2")
    assert plain.duals is None
    withd = solve(_convex_recursive_model(), method="d2", recover_duals=True)
    assert withd.duals is not None and withd.duals.available
    resid = kkt_stationarity_residual(
        aggregate(_convex_recursive_model(), method="d2").base_model,
        withd.duals.full_x,
        withd.duals.constraint_duals,
    )
    assert resid < 1e-6


def test_result_carries_dual_recovery_bookkeeping():
    result = aggregate(_convex_recursive_model(), method="d2")
    assert result.base_model is not None
    assert len(result.origin_maps) == result.n_passes
    # one elim record per eliminated variable, tagged with a valid pass index
    assert len(result.elim_records) == result.n_eliminated
    assert all(0 <= p < result.n_passes for _v, _c, p in result.elim_records)


def test_maximize_recovers_duals():
    """DU1: the recursive path recovers maximise duals in discopt's convention.

    discopt reports maximise multipliers in the ``minimise(-f)`` convention, so
    recovery scales the objective-gradient term by ``-1``.  With an eliminable
    ``y = x + 1`` and an active inequality ``x >= -0.3``, the recovered duals must
    match the original model's direct-solve duals and pass the KKT self-check.
    """

    def build():
        m = dm.Model("maxo")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-9, ub=9)
        m.subject_to(y - (x + 1) == 0.0)  # defining (eliminable)
        m.subject_to(x >= -0.3)  # surviving inequality, active at the optimum
        m.maximize(-(x**2) - y**2)
        return m

    result = aggregate(build(), method="d2")
    dr = recover_duals_from_result(result, result.reduced_model.solve())
    assert dr.available, dr.reason
    resid = kkt_stationarity_residual(result.base_model, dr.full_x, dr.constraint_duals)
    assert resid < 1e-6, f"KKT residual {resid}"
    original = build().solve()
    for cid, mu in dr.constraint_duals.items():
        want = float(np.asarray(original.constraint_duals[f"c{cid}"]).reshape(()))
        assert mu == pytest.approx(want, abs=1e-6)


def test_cross_pass_coupling_recovered_by_linear_solve():
    """C7-A: recursive passes whose defining bodies couple are solved correctly.

    ``b = 2a+1`` (pass 0) references ``a``, eliminated later; the combined defining
    system is not triangular, so back-substitution gave wrong duals.  The linear
    solve recovers multipliers that satisfy full-space KKT stationarity.
    """
    m = dm.Model("coupled")
    a = m.continuous("a", lb=-9, ub=9)
    b = m.continuous("b", lb=-9, ub=9)
    c = m.continuous("c", lb=-9, ub=9)
    d = m.continuous("d", lb=-9, ub=9)
    m.subject_to(b - (2 * a + 1) == 0.0)
    m.subject_to(c - (b + a) == 0.0)
    m.subject_to(d - (c - 2) == 0.0)
    m.subject_to(a + d >= 1.0)  # active inequality at the optimum
    m.minimize((a - 0.1) ** 2 + b**2 + c**2 + d**2)
    res = solve(m, method="d2", recover_duals=True)
    assert res.duals.available, res.duals.reason
    resid = kkt_stationarity_residual(
        res.aggregation.base_model, res.duals.full_x, res.duals.constraint_duals
    )
    assert resid < 1e-6


def test_active_eliminated_bound_reported_unavailable():
    """C7-B: an active bound on an eliminated variable is out of scope, not wrong."""
    m = dm.Model("activebnd")
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=2.0)  # y = x^2 hits ub=2 at the optimum
    m.subject_to(y - x**2 == 0.0)
    m.minimize((x - 5.0) ** 2)
    res = solve(m, method="d2", recover_duals=True)
    assert not res.duals.available
    assert res.duals.reason.startswith("active_eliminated_bound")


def test_available_recoveries_pass_self_check():
    """C7-C: any available=True recovery certifiably satisfies KKT stationarity."""
    res = solve(_convex_recursive_model(), method="d2", recover_duals=True)
    assert res.duals.available
    resid = kkt_stationarity_residual(
        res.aggregation.base_model, res.duals.full_x, res.duals.constraint_duals
    )
    assert resid < 1e-6
