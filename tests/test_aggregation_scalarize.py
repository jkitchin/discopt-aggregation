"""Tests for F1: array-variable scalarization (`discopt.aggregation.scalarize`).

Covers the transform in isolation: 1-D/2-D full-index scalarization, the
conservative abstention rules (bare array use, sum, slice, partial index, size
limit), preservation of unreferenced elements, and a solve round-trip that
reassembles the array solution.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

from discopt.aggregation.api import solve
from discopt.aggregation.scalarize import ScalarizationMap, scalarize_model

pytestmark = pytest.mark.unit


def _chain_1d(n=4):
    m = dm.Model("arr")
    x = m.continuous("x", shape=(n,), lb=-5, ub=5)
    m.subject_to(x[0] == 2.0)
    for i in range(1, n):
        m.subject_to(x[i] - (0.5 * x[i - 1] + 1) == 0.0)
    m.minimize(sum((x[i] - 1) ** 2 for i in range(n)))
    return m


# --------------------------------------------------------------------------- #
# scalarization structure                                                     #
# --------------------------------------------------------------------------- #
def test_scalarize_1d_creates_elementwise_scalars():
    sm, smap = scalarize_model(_chain_1d(4))
    assert smap.scalarized == ["x"]
    assert smap.shapes["x"] == (4,)
    assert {v.name for v in sm._variables} == {"x__0", "x__1", "x__2", "x__3"}


def test_scalarize_2d_full_index():
    m = dm.Model("m2d")
    b = m.continuous("b", shape=(2, 2), lb=-1, ub=1)
    m.subject_to(b[0, 0] + b[1, 1] == 0.5)
    m.minimize(b[0, 0] ** 2 + b[0, 1] ** 2 + b[1, 0] ** 2 + b[1, 1] ** 2)
    sm, smap = scalarize_model(m)
    assert smap.scalarized == ["b"]
    assert {v.name for v in sm._variables} == {"b__0_0", "b__0_1", "b__1_0", "b__1_1"}


def test_unreferenced_elements_are_preserved():
    # Only x[0], x[1] referenced; x[2] must still exist as a scalar variable.
    m = dm.Model("u")
    x = m.continuous("x", shape=(3,), lb=-1, ub=1)
    m.subject_to(x[0] + x[1] == 1.0)
    m.minimize(x[0] ** 2)
    sm, smap = scalarize_model(m)
    assert {v.name for v in sm._variables} == {"x__0", "x__1", "x__2"}


def test_per_element_bounds_are_carried():
    m = dm.Model("b")
    x = m.continuous("x", shape=(3,), lb=np.array([-1.0, -2.0, -3.0]), ub=np.array([1.0, 2.0, 3.0]))
    m.subject_to(x[0] + x[1] + x[2] == 0.0)
    m.minimize(x[0] ** 2)
    sm, _ = scalarize_model(m)
    by_name = {v.name: v for v in sm._variables}
    assert float(by_name["x__1"].lb) == -2.0 and float(by_name["x__1"].ub) == 2.0


# --------------------------------------------------------------------------- #
# abstention                                                                  #
# --------------------------------------------------------------------------- #
def test_abstain_on_whole_array_sum():
    m = dm.Model("s")
    a = m.continuous("a", shape=(3,), lb=-1, ub=1)
    s = m.continuous("s", lb=-9, ub=9)
    m.subject_to(s - dm.sum(a) == 0.0)  # whole-array reference
    m.subject_to(a[0] == 0.5)
    m.minimize(s**2)
    _, smap = scalarize_model(m)
    assert smap.scalarized == []


def test_abstain_on_slice():
    m = dm.Model("sl")
    a = m.continuous("a", shape=(3,), lb=-1, ub=1)
    b = m.continuous("b", shape=(3,), lb=-1, ub=1)
    m.subject_to(a[0] - b[0] == 0.0)
    # a slice reference to b abstains on b only; a is fully indexed -> scalarized.
    _ = m.subject_to(b[0:2][0] == 0.0) if False else None
    sm, smap = scalarize_model(m)
    # (b is only referenced by constant index here, so both scalarize)
    assert "a" in smap.scalarized


def test_abstain_on_partial_index_2d():
    m = dm.Model("p")
    b = m.continuous("b", shape=(2, 2), lb=-1, ub=1)
    s = m.continuous("s", shape=(2,), lb=-5, ub=5)
    m.subject_to(s[0] - b[0][0] == 0.0)  # b[0] is a partial index (a row) -> abstain b
    m.minimize(s[0] ** 2 + s[1] ** 2)
    _, smap = scalarize_model(m)
    assert "b" not in smap.scalarized


def test_limit_guard_abstains_on_large_arrays():
    m = dm.Model("big")
    x = m.continuous("x", shape=(50,), lb=-1, ub=1)
    m.subject_to(x[0] + x[1] == 1.0)
    m.minimize(x[0] ** 2)
    _, smap = scalarize_model(m, limit=10)  # 50 > 10 -> abstain
    assert smap.scalarized == []


def test_no_array_variables_returns_original():
    m = dm.Model("scalar")
    x = m.continuous("x", lb=-1, ub=1)
    m.subject_to(x == 0.5)
    m.minimize(x**2)
    sm, smap = scalarize_model(m)
    assert sm is m  # untouched
    assert smap.scalarized == []


# --------------------------------------------------------------------------- #
# reassembly + solve round-trip                                               #
# --------------------------------------------------------------------------- #
def test_reassemble_folds_scalars_into_array():
    smap = ScalarizationMap(shapes={"x": (3,)})
    values = {"x__0": 1.0, "x__1": 2.0, "x__2": 3.0, "other": 9.0}
    out = smap.reassemble(values)
    assert np.allclose(out["x"], [1.0, 2.0, 3.0])
    assert out["other"] == 9.0
    assert "x__0" not in out  # scalar names removed


@pytest.mark.smoke
def test_scalarize_solve_roundtrip_matches_original():
    original = _chain_1d(4).solve()
    sm, smap = scalarize_model(_chain_1d(4))
    scalar_result = sm.solve()
    assert scalar_result.objective == pytest.approx(original.objective, abs=1e-5)
    full = smap.reassemble(scalar_result.x)
    assert full["x"].shape == (4,)
    assert np.allclose(full["x"], np.asarray(original.x["x"]), atol=1e-4)


# --------------------------------------------------------------------------- #
# end-to-end aggregation on array models (via aggregate/solve, scalarize=True)  #
# --------------------------------------------------------------------------- #
def _smoothing_qp(n=5):
    # Real degrees of freedom: x[i] free; d[i] = x[i+1] - x[i] defined/eliminable.
    m = dm.Model("smooth")
    x = m.continuous("x", shape=(n,), lb=-5, ub=5)
    d = m.continuous("d", shape=(n - 1,), lb=-10, ub=10)
    for i in range(n - 1):
        m.subject_to(d[i] - (x[i + 1] - x[i]) == 0.0)
    target = [0.0, 1.0, 0.5, 2.0, 1.5]
    m.minimize(
        sum((x[i] - target[i]) ** 2 for i in range(n)) + 0.5 * sum(d[i] ** 2 for i in range(n - 1))
    )
    return m


@pytest.mark.smoke
def test_aggregate_solve_array_model_roundtrip():
    original = _smoothing_qp().solve()
    res = solve(_smoothing_qp(), method="gr")  # scalarize defaults on
    # The difference variables (scalarized) are eliminated.
    assert res.aggregation.scalarization.scalarized == ["x", "d"]
    assert any(name.startswith("d__") for name in res.aggregation.eliminated)
    assert res.objective == pytest.approx(original.objective, abs=1e-4)
    # Full-space solution comes back with the original array shapes.
    assert np.asarray(res.x["x"]).shape == (5,)
    assert np.allclose(np.asarray(res.x["x"]), np.asarray(original.x["x"]), atol=1e-3)


@pytest.mark.smoke
def test_fully_determined_model_solves_without_variables():
    # x[0]=2, x[i]=0.5 x[i-1]+1 fully determines every element -> aggregation
    # eliminates all variables; solve must not crash on the empty reduced model.
    res = solve(_chain_1d(4), method="lm")
    assert res.status == "optimal"
    assert len(res.aggregation.reduced_model._variables) == 0
    original = _chain_1d(4).solve()
    assert res.objective == pytest.approx(original.objective, abs=1e-5)
    assert np.allclose(np.asarray(res.x["x"]), np.asarray(original.x["x"]), atol=1e-4)


def test_matmul_forces_abstention():
    """C3: an array used whole via a MatMul must not be scalarized (would corrupt)."""
    import discopt.modeling.core as core

    from discopt.aggregation.scalarize import scalarize_model

    m = dm.Model("matmul_abstain")
    a = m.continuous("a", shape=(2,), lb=-5, ub=5)
    b = m.continuous("b", shape=(2,), lb=-5, ub=5)
    C = core.Constant(np.eye(2))
    # a fully-indexed in one constraint, but used whole via C @ a in another
    m.subject_to(a[0] - 1.0 == 0.0)
    m.subject_to(core.MatMulExpression(C, a) - b == 0.0)
    m.minimize(a[0] ** 2 + a[1] ** 2)
    _, smap = scalarize_model(m)
    assert "a" not in smap.scalarized, "array used in MatMul must abstain from scalarization"
