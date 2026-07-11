"""Tests for K1 end-to-end: implicit elimination of cyclic blocks (discopt#379).

``eliminate_implicit`` rebuilds a model with each nonsingular cyclic block replaced
by a ``Model.implicit`` node, and ``ImplicitElimination.recover`` reconstructs the
block variables.  The block models here are *inherent* cycles — both defining
constraints reference only the block variables, so the 2-cycle is present in every
maximum matching (robust to hash-seed-dependent matching choices); kept decision
variables appear only in the objective.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.benchmark import _local_solve
from discopt.aggregation.implicit import ImplicitElimination, eliminate_implicit

pytestmark = pytest.mark.unit


def _cyclic_model():
    # inherent 2-cycle {x, y}: both equations touch only x, y; t is a kept decision
    m = dm.Model("k1")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    t = m.continuous("t", lb=-9, ub=9)
    m.subject_to(x - (0.5 * y + 3.0) == 0.0)  # x = 0.5y + 3
    m.subject_to(y - (0.5 * x + 2.0) == 0.0)  # y = 0.5x + 2  -> x = 16/3, y = 14/3
    m.minimize((t - 1.0) ** 2 + x**2 + y**2 + t)
    return m


@pytest.mark.smoke
def test_eliminate_implicit_reduces_and_solves_same():
    _s, _i, _w, direct_obj = _local_solve(_cyclic_model())
    el = eliminate_implicit(_cyclic_model())
    assert isinstance(el, ImplicitElimination)
    assert set(el.eliminated) == {"x", "y"}
    assert [v.name for v in el.reduced_model._variables] == ["t"]
    _s, _i, _w, reduced_obj = _local_solve(el.reduced_model)
    assert float(reduced_obj) == pytest.approx(float(direct_obj), abs=1e-4)


def test_recover_solves_the_block():
    el = eliminate_implicit(_cyclic_model())
    full = el.recover({})
    # closed form: x = 0.5y + 3, y = 0.5x + 2  ->  x = 16/3, y = 14/3
    assert full["x"] == pytest.approx(16 / 3, abs=1e-4)
    assert full["y"] == pytest.approx(14 / 3, abs=1e-4)
    assert full["x"] == pytest.approx(0.5 * full["y"] + 3.0, abs=1e-6)
    assert full["y"] == pytest.approx(0.5 * full["x"] + 2.0, abs=1e-6)


def test_acyclic_model_has_no_block():
    m = dm.Model("acyclic")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-9, ub=9)
    m.subject_to(y - (x + 1.0) == 0.0)
    m.minimize(x**2 + y**2)
    with pytest.raises(ValueError, match="no eliminable"):
        eliminate_implicit(m)


def test_singular_block_is_not_eliminated():
    # x - y = 0 and x - y = 1 : an inherent but singular (and inconsistent) 2-cycle
    m = dm.Model("sing")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    m.subject_to(x - y == 0.0)
    m.subject_to(x - y == 1.0)
    m.minimize(x**2 + y**2)
    with pytest.raises(ValueError, match="no eliminable"):
        eliminate_implicit(m)


def test_nonlinear_cyclic_block_round_trips():
    # an inherent nonlinear 2-cycle still eliminates and recovers
    m = dm.Model("nl")
    x = m.continuous("x", lb=0.1, ub=9)
    y = m.continuous("y", lb=0.1, ub=9)
    t = m.continuous("t", lb=1.0, ub=5)
    m.subject_to(x - (0.3 * y + 1.0) == 0.0)
    m.subject_to(y - (0.2 * x * x + 1.0) == 0.0)  # nonlinear coupling
    m.minimize((t - 2.0) ** 2 + t)
    el = eliminate_implicit(m)
    assert set(el.eliminated) == {"x", "y"}
    full = el.recover({})
    assert full["x"] == pytest.approx(0.3 * full["y"] + 1.0, abs=1e-5)
    assert full["y"] == pytest.approx(0.2 * full["x"] ** 2 + 1.0, abs=1e-5)


@pytest.mark.smoke
def test_block_bounds_enforced():
    """C4: eliminated block variables' bounds are re-expressed on the node.

    A closed 2-cycle fixes ``x = 16/3 ≈ 5.33``.  With ``x`` bounded to ``[0, 1]``
    the model is infeasible; before the fix the bounds were dropped and the
    reduced model solved 'optimal', hiding the infeasibility.
    """
    m = dm.Model("c4")
    x = m.continuous("x", lb=0.0, ub=1.0)  # 16/3 > 1
    y = m.continuous("y", lb=-9, ub=9)
    t = m.continuous("t", lb=-9, ub=9)
    m.subject_to(x - (0.5 * y + 3.0) == 0.0)
    m.subject_to(y - (0.5 * x + 2.0) == 0.0)
    m.minimize((t - 1.0) ** 2 + t)
    el = eliminate_implicit(m)
    assert el.n_bound_inequalities == 4  # x and y each get lb+ub inequalities
    status, _i, _w, _obj = _local_solve(el.reduced_model)  # status is a str
    assert "OPTIMAL" not in status, "infeasible block bound not enforced"


def test_within_bounds_block_still_solves():
    """C4 no-regression: a closed cycle inside its bounds still solves + recovers."""
    el = eliminate_implicit(_cyclic_model())  # x,y in [-9,9]; solution 16/3, 14/3
    assert el.n_bound_inequalities > 0  # bounds are emitted (just non-binding)
    full = el.recover({})
    assert full["x"] == pytest.approx(16 / 3, abs=1e-4)
    assert full["y"] == pytest.approx(14 / 3, abs=1e-4)


@pytest.mark.smoke
def test_maximize_cyclic_block_round_trips():
    """T4: eliminate_implicit's maximize objective rebuild path."""
    m = dm.Model("maxk1")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    t = m.continuous("t", lb=-9, ub=9)
    m.subject_to(x - (0.5 * y + 3.0) == 0.0)
    m.subject_to(y - (0.5 * x + 2.0) == 0.0)
    m.maximize(-((t - 1.0) ** 2) - x**2 - y**2 - t)
    direct_s, _i, _w, direct_obj = _local_solve(_maxk1_direct())
    el = eliminate_implicit(m)
    _s, _i, _w, reduced_obj = _local_solve(el.reduced_model)
    assert float(reduced_obj) == pytest.approx(float(direct_obj), abs=1e-3)


def _maxk1_direct():
    m = dm.Model("maxk1d")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    t = m.continuous("t", lb=-9, ub=9)
    m.subject_to(x - (0.5 * y + 3.0) == 0.0)
    m.subject_to(y - (0.5 * x + 2.0) == 0.0)
    m.maximize(-((t - 1.0) ** 2) - x**2 - y**2 - t)
    return m
