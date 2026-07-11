"""RT2: coverage for interactions the review flagged as untested.

- guards (``max_fill`` / ``pivot_tol``) composed with ``integer_aggregation``;
- ``recover_duals`` on a *scalarized* array model (base-model-space bookkeeping);
- recovery's integrality tolerance for a slightly-off kept integer value.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.api import aggregate, solve

pytestmark = pytest.mark.unit


def _integer_chain():
    m = dm.Model("intchain")
    s0 = m.integer("s0", lb=0, ub=20)
    s1 = m.integer("s1", lb=0, ub=20)
    q = m.integer("q", lb=0, ub=10)
    d = m.integer("d", lb=0, ub=10)
    m.subject_to(s1 - (s0 + q - d) == 0)  # all-integer balance
    m.subject_to(s0 - 7 == 0)
    m.minimize((q - 2) ** 2 + (d - 1) ** 2 + s1 * s1)
    return m


@pytest.mark.parametrize("max_fill", [None, 0, 1, 5])
def test_max_fill_with_integer_aggregation_stays_sound(max_fill):
    """max_fill composed with integer_aggregation never eliminates more than the
    uncapped run and keeps stats coherent (n_eliminated matches eliminated)."""
    uncapped = aggregate(_integer_chain(), integer_aggregation=True).n_eliminated
    agg = aggregate(_integer_chain(), integer_aggregation=True, max_fill=max_fill)
    assert agg.n_eliminated == len(agg.eliminated)
    assert agg.n_eliminated <= uncapped


def test_pivot_tol_with_integer_aggregation_keeps_unit_pivots():
    """Integer edges require a ±1 pivot, so even a large pivot_tol (which would
    veto a small continuous pivot) does not veto an admitted integer elimination."""
    base = aggregate(_integer_chain(), integer_aggregation=True).n_eliminated
    tol = aggregate(_integer_chain(), integer_aggregation=True, pivot_tol=0.9).n_eliminated
    assert tol == base


@pytest.mark.smoke
def test_recover_duals_on_scalarized_array_model():
    """RT2: dual recovery on an array (scalarized) model recovers a multiplier for
    every scalar defining equality, in the base model's space."""
    m = dm.Model("arrdual")
    x = m.continuous("x", shape=(3,), lb=-5, ub=5)
    y = m.continuous("y", shape=(3,), lb=-20, ub=20)
    for t in range(3):
        m.subject_to(y[t] - (x[t] + 1) == 0.0)
    m.minimize(dm.sum(x**2 + y**2))
    res = solve(m, scalarize=True, recover_duals=True)
    assert res.status == "optimal"
    assert res.duals is not None and res.duals.available, res.duals and res.duals.reason
    assert len(res.duals.constraint_duals) == 3


def test_recovery_tolerates_slightly_offinteger_kept_value():
    """RT2: a kept integer returned slightly off-integral by a solver's integrality
    tolerance (e.g. 4e-7) is absorbed by recovery's 1e-6 check; the eliminated
    integer still recovers as an exact integer."""
    m = dm.Model("tol")
    m.integer("i", lb=0, ub=10)
    m.integer("y", lb=0, ub=20)
    m.subject_to(m._variables[1] - (m._variables[0] + 1) == 0.0)
    m.minimize((m._variables[0] - 4) ** 2)
    agg = aggregate(m, method="d2", integer_aggregation=True)
    (kept,) = [v for v in ("i", "y") if v not in agg.eliminated]
    off = 5.0000004 if kept == "y" else 4.0000004
    full = agg.recovery.recover_full({kept: off})
    (elim,) = agg.eliminated
    assert full[elim] == round(full[elim])  # recovered as an exact integer
