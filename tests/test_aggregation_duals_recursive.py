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


def test_maximize_is_out_of_scope():
    m = dm.Model("maxo")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-9, ub=9)
    m.subject_to(y - (x + 1) == 0.0)
    m.maximize(-(x**2) - y**2)
    result = aggregate(m, method="d2")
    dr = recover_duals_from_result(result, result.reduced_model.solve())
    assert not dr.available  # maximise sign handling documented out of scope
