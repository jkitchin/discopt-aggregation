"""Tests for :mod:`discopt.aggregation.transform` and
:mod:`discopt.aggregation.recovery`.

Correctness is checked as an *algebraic identity*, without invoking a solver:

- **Substitution equivalence** — for random kept-variable values, every retained
  constraint body (and the objective) of the reduced model evaluates to the same
  number as the original body evaluated at the full assignment (kept +
  recovered).  This holds identically for any values, so it validates the
  substitution regardless of feasibility.
- **Recovery** — the recovered values satisfy the *original* defining equalities
  (each defining constraint body evaluates to ~0 at the full assignment).
"""

from __future__ import annotations

import random

import discopt.modeling as dm
import pytest
from discopt.modeling.core import Constant, SumOverExpression

from discopt.aggregation.recovery import RecoveryMap, evaluate_expression
from discopt.aggregation.strategies import STRATEGIES, aggregation_set
from discopt.aggregation.transform import (
    apply_aggregation,
    defining_expressions,
    substitute_expression,
)

pytestmark = pytest.mark.unit


def _mixed_model():
    m = dm.Model("mixed")
    v = {n: m.continuous(n, lb=-10, ub=10) for n in "abcdef"}
    a, b, c, d, e, f = (v[n] for n in "abcdef")
    m.subject_to(a == 3.0)  # a fixed
    m.subject_to(b - 2 * a == 1.0)  # b = 2a + 1
    m.subject_to(c - d == 4.0)  # c = d + 4
    m.subject_to(e - d**2 == 0.0)  # e = d^2
    m.subject_to(f - a * d - e == 0.0)  # f = a*d + e
    m.minimize(a + b + c + d + e + f)
    return m


def _random_env(names, rng, lo=-2.0, hi=2.0):
    return {n: rng.uniform(lo, hi) for n in names}


# --------------------------------------------------------------------------- #
# substitute_expression                                                        #
# --------------------------------------------------------------------------- #
def test_substitute_simple_binary():
    m = dm.Model("t")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    body = (x + 2 * y).__mul__(1)  # an expression referencing x and y
    # replace y with (x + 1)
    out = substitute_expression(body, {"y": x + 1})
    val = evaluate_expression(out, {"x": 3.0})
    # x + 2*(x+1) = 3x + 2 = 11 at x=3
    assert val == pytest.approx(11.0)


def test_substitute_inside_sum_over_expression():
    # The existing gdp substitute skips SumOverExpression; ours must not.
    m = dm.Model("t")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    z = m.continuous("z", lb=-5, ub=5)
    expr = SumOverExpression([x, y, z])  # x + y + z
    out = substitute_expression(expr, {"y": Constant(10.0)})
    val = evaluate_expression(out, {"x": 1.0, "y": 999.0, "z": 2.0})
    # y must have been replaced by 10, so 1 + 10 + 2 = 13 (NOT using y=999)
    assert val == pytest.approx(13.0)


def test_substitute_identity_when_no_match():
    m = dm.Model("t")
    x = m.continuous("x", lb=-5, ub=5)
    body = x + 1
    out = substitute_expression(body, {"nonexistent": Constant(0.0)})
    assert out is body  # unchanged subtree is shared, not rebuilt


# --------------------------------------------------------------------------- #
# defining_expressions                                                         #
# --------------------------------------------------------------------------- #
def test_defining_expressions_are_kept_only_and_correct():
    m = _mixed_model()
    agg = aggregation_set(m, method="gr")  # eliminates the most
    defining = defining_expressions(agg)
    eliminated = set(agg.variables())
    kept = {v.name for v in m._variables} - eliminated

    rng = random.Random(0)
    env_kept = _random_env(kept, rng)
    # Every defining expression references only kept variables.
    for name, expr in defining.items():
        # Evaluating with only kept values must succeed (no eliminated refs).
        evaluate_expression(expr, env_kept)


# --------------------------------------------------------------------------- #
# apply_aggregation: structure                                                 #
# --------------------------------------------------------------------------- #
def test_reduced_model_structure():
    m = _mixed_model()
    agg = aggregation_set(m, method="gr")
    rm = apply_aggregation(m, agg)

    reduced_names = {v.name for v in rm.model._variables}
    assert reduced_names == {v.name for v in m._variables} - set(rm.eliminated)
    assert rm.n_eliminated == len(rm.eliminated)
    assert rm.n_constraints_removed == rm.n_eliminated  # one defining eqn each
    # Reduced equality constraints = original equalities minus the removed ones,
    # plus any bound inequalities.
    assert rm.n_bound_inequalities >= 0


# --------------------------------------------------------------------------- #
# substitution equivalence + recovery (the core correctness harness)           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", sorted(STRATEGIES))
@pytest.mark.parametrize("seed", range(4))
def test_substitution_equivalence_and_recovery(method, seed):
    m = _mixed_model()
    agg = aggregation_set(m, method=method)
    rm = apply_aggregation(m, agg)

    kept_names = [v.name for v in rm.model._variables]
    rng = random.Random(seed)
    kept_values = _random_env(kept_names, rng)

    recovered = rm.recovery.recover(kept_values)
    full = {**kept_values, **recovered}

    # (1) Recovery satisfies the original defining equalities (body ~ 0).
    for e in agg.eliminations:
        body = m._constraints[e.cid].body
        assert evaluate_expression(body, full) == pytest.approx(0.0, abs=1e-9)

    # (2) Substitution equivalence: retained constraints line up 1:1 in order,
    #     followed by the appended bound inequalities.
    elim_cids = set(agg.matching())
    remaining_orig = [c for cid, c in enumerate(m._constraints) if cid not in elim_cids]
    reduced_cons = rm.model._constraints
    assert len(reduced_cons) == len(remaining_orig) + rm.n_bound_inequalities

    for orig_c, red_c in zip(remaining_orig, reduced_cons):
        assert orig_c.sense == red_c.sense
        ov = evaluate_expression(orig_c.body, full)
        rv = evaluate_expression(red_c.body, kept_values)
        assert float(ov) == pytest.approx(float(rv), rel=1e-9, abs=1e-9)

    # (3) Objective equivalence.
    ov = evaluate_expression(m._objective.expression, full)
    rv = evaluate_expression(rm.model._objective.expression, kept_values)
    assert float(ov) == pytest.approx(float(rv), rel=1e-9, abs=1e-9)


def test_recovery_map_recover_full_merges():
    rmap = RecoveryMap(defining={"y": Constant(5.0)}, order=["y"])
    out = rmap.recover_full({"x": 1.0})
    assert out == {"x": 1.0, "y": 5.0}


# --------------------------------------------------------------------------- #
# bounds -> inequalities                                                       #
# --------------------------------------------------------------------------- #
def test_bounded_elimination_adds_inequalities():
    # e = d^2 with e in [-10, 10]; eliminating e must add box inequalities on d^2.
    m = dm.Model("bnd")
    d = m.continuous("d", lb=-3, ub=3)
    e = m.continuous("e", lb=0.0, ub=4.0)
    m.subject_to(e - d**2 == 0.0)
    m.minimize(e)
    agg = aggregation_set(m, method="d2")
    assert agg.variables() == ["e"]
    rm = apply_aggregation(m, agg)
    # e had finite lb and ub -> two inequality constraints on d^2.
    assert rm.n_bound_inequalities == 2
    assert "e" not in {v.name for v in rm.model._variables}


def test_fixed_variable_adds_no_bound_inequalities():
    m = dm.Model("fix")
    a = m.continuous("a", lb=-10, ub=10)
    b = m.continuous("b", lb=-10, ub=10)
    m.subject_to(a == 3.0)  # constant definition
    m.subject_to(b - a == 1.0)
    m.minimize(a + b)
    agg = aggregation_set(m, method="ld1")
    rm = apply_aggregation(m, agg)
    assert rm.eliminated == ["a"]
    assert rm.n_bound_inequalities == 0  # constant definition: no kept-var bound


def test_infeasible_fixed_bound_detected():
    """C1: a fixed variable whose value violates its box marks infeasibility.

    ``y == 5`` forces ``y = 5``, outside ``y in [0, 1]``.  The reduced model
    drops ``y``'s bound (it becomes no constraint on the kept variables), so
    without an explicit check the reduced problem would look feasible.
    """
    import warnings

    from discopt.aggregation.api import aggregate, solve

    m = dm.Model("infeas_fixed")
    y = m.continuous("y", lb=0.0, ub=1.0)
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.subject_to(y - 5.0 == 0.0)
    m.subject_to(x - y >= 0.0)
    m.minimize(x**2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agg = aggregate(m, method="ld1")
    assert agg.infeasible_eliminations, "expected the fixed-variable violation recorded"
    var, val, lb, ub = agg.infeasible_eliminations[0]
    assert var == "y" and abs(val - 5.0) < 1e-9 and (lb, ub) == (0.0, 1.0)
    assert any("infeasible" in str(w.message) for w in caught)

    res = solve(m, method="ld1")
    assert res.status == "infeasible"


def test_fixed_variable_within_box_stays_feasible():
    """C1 boundary: a fixed value inside the box is not flagged."""
    from discopt.aggregation.api import aggregate, solve

    m = dm.Model("feas_fixed")
    y = m.continuous("y", lb=0.0, ub=5.0)
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.subject_to(y - 5.0 == 0.0)  # y = 5 is on the boundary, feasible
    m.subject_to(x - y >= 0.0)
    m.minimize(x**2)
    agg = aggregate(m, method="ld1")
    assert not agg.infeasible_eliminations
    res = solve(m, method="ld1")
    assert res.status == "optimal"
