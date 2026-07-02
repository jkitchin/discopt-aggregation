"""End-to-end tests for :mod:`discopt.aggregation.api`.

These solve real (tiny) models with the default backend and check that solving
the *reduced* model and recovering the full solution reproduces the result of
solving the original model: identical objective, identical full-space solution,
and a recovered solution that is feasible in the original model.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

from discopt.aggregation.api import aggregate, solve
from discopt.aggregation.recovery import evaluate_expression
from discopt.aggregation.strategies import STRATEGIES

pytestmark = pytest.mark.smoke


def _as_floats(x):
    return {k: float(np.asarray(v)) for k, v in x.items()}


def _assert_feasible(model, x, tol=1e-5):
    for c in model._constraints:
        body = float(evaluate_expression(c.body, x))
        if c.sense == "==":
            assert abs(body) <= tol, f"equality violated: {body}"
        elif c.sense == "<=":
            assert body <= tol
        elif c.sense == ">=":
            assert body >= -tol


# --------------------------------------------------------------------------- #
# NLP round-trip across every method                                           #
# --------------------------------------------------------------------------- #
def _nlp():
    # min x^2 + y^2 s.t. y = x + 1  ->  x=-0.5, y=0.5, obj=0.5
    m = dm.Model("nlp")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.subject_to(y - x == 1.0)
    m.minimize(x**2 + y**2)
    return m


@pytest.mark.parametrize("method", sorted(STRATEGIES))
def test_nlp_roundtrip_matches_original(method):
    original = _nlp().solve()
    res = solve(_nlp(), method=method)

    assert res.status == "optimal"
    assert res.objective == pytest.approx(original.objective, abs=1e-4)

    full = _as_floats(res.x)
    orig_x = _as_floats(original.x)
    for name in orig_x:
        assert full[name] == pytest.approx(orig_x[name], abs=1e-3)

    _assert_feasible(_nlp(), full)


# --------------------------------------------------------------------------- #
# recursive fixed-variable chain                                               #
# --------------------------------------------------------------------------- #
def test_recursive_ld1_chain():
    # a = 3; b = 2a + 1; min b^2 + z^2 s.t. z >= 1  ->  a=3, b=7, z=1, obj=50
    def build():
        m = dm.Model("chain")
        a = m.continuous("a", lb=-10, ub=10)
        b = m.continuous("b", lb=-20, ub=20)
        z = m.continuous("z", lb=-5, ub=5)
        m.subject_to(a == 3.0)
        m.subject_to(b - 2 * a == 1.0)
        m.subject_to(z >= 1.0)
        m.minimize(b**2 + z**2)
        return m

    res = solve(build(), method="ld1")
    assert res.status == "optimal"
    assert set(res.aggregation.eliminated) == {"a", "b"}
    assert res.aggregation.n_passes == 2  # a first, then b becomes fixed
    full = _as_floats(res.x)
    assert full["a"] == pytest.approx(3.0, abs=1e-4)
    assert full["b"] == pytest.approx(7.0, abs=1e-4)
    assert full["z"] == pytest.approx(1.0, abs=1e-3)
    assert res.objective == pytest.approx(50.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# degree-2 nonlinear elimination                                               #
# --------------------------------------------------------------------------- #
def test_d2_nonlinear_elimination_roundtrip():
    # min e s.t. e = (x-2)^2, e in [0,25]  ->  x=2, e=0, obj=0
    def build():
        m = dm.Model("d2nl")
        x = m.continuous("x", lb=-5, ub=5)
        e = m.continuous("e", lb=0.0, ub=25.0)
        m.subject_to(e - (x - 2) ** 2 == 0.0)
        m.minimize(e)
        return m

    res = solve(build(), method="d2")
    assert "e" in res.aggregation.eliminated
    assert res.objective == pytest.approx(0.0, abs=1e-5)
    full = _as_floats(res.x)
    assert full["x"] == pytest.approx(2.0, abs=1e-3)
    assert full["e"] == pytest.approx(0.0, abs=1e-4)
    _assert_feasible(build(), full)


# --------------------------------------------------------------------------- #
# MINLP: integer/binary variables untouched, continuous eliminated             #
# --------------------------------------------------------------------------- #
def test_minlp_keeps_binary_and_recovers_continuous():
    # min (x-2)^2 + 3b s.t. y = x - 1, b binary  ->  b=0, x=2, y=1, obj=0
    def build():
        m = dm.Model("minlp")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        b = m.binary("b")
        m.subject_to(y - (x - 1) == 0.0)
        m.minimize((x - 2) ** 2 + 3 * b)
        return m

    res = solve(build(), method="ld2")
    # y = x - 1 can define either x or y; exactly one is eliminated.
    assert len(set(res.aggregation.eliminated) & {"x", "y"}) == 1
    assert "b" not in res.aggregation.eliminated  # binary never eliminated
    assert res.status == "optimal"
    full = _as_floats(res.x)
    assert full["x"] == pytest.approx(2.0, abs=1e-3)
    assert full["y"] == pytest.approx(1.0, abs=1e-3)
    assert full["b"] == pytest.approx(0.0, abs=1e-6)
    assert res.objective == pytest.approx(0.0, abs=1e-4)
    _assert_feasible(build(), full)


# --------------------------------------------------------------------------- #
# aggregate() semantics                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_aggregate_no_eliminations_returns_original():
    m = dm.Model("none")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.subject_to(x * y == 1.0)  # nonlinear in both -> nothing linearly eliminable
    m.minimize(x + y)
    res = aggregate(m, method="d2")
    assert res.n_eliminated == 0
    assert res.reduced_model is m  # unchanged original returned
    assert res.recovery.order == []


@pytest.mark.unit
def test_aggregate_non_recursive_single_pass():
    m = dm.Model("chain")
    a = m.continuous("a", lb=-10, ub=10)
    b = m.continuous("b", lb=-20, ub=20)
    m.subject_to(a == 3.0)
    m.subject_to(b - 2 * a == 1.0)
    m.continuous("z", lb=-5, ub=5)  # keep a free variable
    m.minimize(a + b)
    # Single pass of ld1 eliminates only the immediately-fixed 'a'.
    res = aggregate(m, method="ld1", recursive=False)
    assert res.eliminated == ["a"]
    assert res.n_passes == 1


@pytest.mark.unit
def test_unknown_method_raises():
    m = dm.Model("t")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    with pytest.raises(ValueError, match="unknown aggregation method"):
        aggregate(m, method="bogus")
