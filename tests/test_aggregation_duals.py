"""Tests for G3b: full-space Lagrange-multiplier recovery.

Validated two independent ways:

1. **KKT stationarity residual** — the recovered multipliers must satisfy the
   full-space stationarity ``∇f + Σ μ_c ∇body_c = 0`` at the recovered point.
2. **Independent oracle** — the recovered multipliers must match those from
   solving the *original* model directly.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

from discopt.aggregation.duals import kkt_stationarity_residual, recover_duals
from discopt.aggregation.strategies import aggregation_set
from discopt.aggregation.transform import apply_aggregation

pytestmark = pytest.mark.smoke


def _eq_model():
    # y = x + 1 (defining), w = x - z (surviving equality); interior optimum.
    m = dm.Model("k")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-9, ub=9)
    w = m.continuous("w", lb=-9, ub=9)
    z = m.continuous("z", lb=-5, ub=5)
    m.subject_to(y - (x + 1) == 0.0)
    m.subject_to(w - (x - z) == 0.0)
    m.minimize(x**2 + y**2 + w**2 + (z - 1) ** 2)
    return m


def _recover(model, method):
    agg = aggregation_set(model, method=method)
    rm = apply_aggregation(model, agg)
    rr = rm.model.solve()
    return agg, rm, recover_duals(rm, rr)


@pytest.mark.parametrize("method", ["ld2", "d2", "gr", "lm"])
def test_kkt_residual_and_oracle(method):
    model = _eq_model()
    agg, rm, dr = _recover(model, method)
    assert dr.available
    # (1) full-space KKT stationarity residual is ~0.
    res = kkt_stationarity_residual(model, dr.full_x, dr.constraint_duals)
    assert res < 1e-6, f"KKT residual {res}"
    # (2) recovered multipliers match the original model's duals (keyed c{cid}).
    original = _eq_model().solve()
    for cid, mu in dr.constraint_duals.items():
        want = float(np.asarray(original.constraint_duals[f"c{cid}"]).reshape(()))
        assert mu == pytest.approx(want, abs=1e-4)


def test_recovers_defining_multiplier_for_every_constraint():
    model = _eq_model()
    _agg, _rm, dr = _recover(model, "gr")
    # Both original constraints have a recovered multiplier.
    assert set(dr.constraint_duals) == {0, 1}


def test_with_active_inequality():
    # y = x + 1 (defining) and an active inequality x >= -0.3.
    def build():
        m = dm.Model("ineq")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-9, ub=9)
        m.subject_to(y - (x + 1) == 0.0)  # defining
        m.subject_to(x >= -0.3)  # surviving inequality (active at optimum)
        m.minimize(x**2 + y**2)
        return m

    model = build()
    agg = aggregation_set(model, method="d2")
    rm = apply_aggregation(model, agg)
    rr = rm.model.solve()
    dr = recover_duals(rm, rr)
    assert dr.available
    res = kkt_stationarity_residual(model, dr.full_x, dr.constraint_duals)
    assert res < 1e-5, f"KKT residual {res}"
    original = build().solve()
    for cid, mu in dr.constraint_duals.items():
        want = float(np.asarray(original.constraint_duals[f"c{cid}"]).reshape(()))
        assert mu == pytest.approx(want, abs=1e-4)


@pytest.mark.unit
def test_maximize_objective_is_unavailable():
    m = dm.Model("max")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-9, ub=9)
    m.subject_to(y - (x + 1) == 0.0)
    m.maximize(-(x**2 + y**2))
    agg = aggregation_set(m, method="d2")
    rm = apply_aggregation(m, agg)

    class _Fake:
        x = {"x": 0.0}
        constraint_duals: dict = {}

    dr = recover_duals(rm, _Fake())
    assert dr.available is False  # maximise sign handling out of scope
