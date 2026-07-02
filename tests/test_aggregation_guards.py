"""Tests for D2: expression-size (max_def_nodes) + conditioning caps.

Both guards are opt-in and default-off (no behaviour change).  These check that
the caps drop the intended eliminations, that the reduced-model expression size
and pivot conditioning are actually bounded, that a hand-computed chain
conditioning matches, and that a capped aggregation stays sound (round-trips).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

from discopt.aggregation.api import aggregate, solve
from discopt.aggregation.guards import (
    apply_elimination_guards,
    conditioning_estimate,
)
from discopt.aggregation.hessian_cost import model_expression_size
from discopt.aggregation.strategies import aggregation_set

pytestmark = pytest.mark.unit


def _nonlinear_chain(depth: int):
    m = dm.Model(f"chain{depth}")
    x = m.continuous("x", lb=-2, ub=2)
    s = [m.continuous(f"s{i}", lb=-50, ub=50) for i in range(depth)]

    def phi(z):
        return 0.5 * (z + 0.05 * z * z)

    m.subject_to(s[0] - phi(x) == 0.0)
    for i in range(1, depth):
        m.subject_to(s[i] - phi(s[i - 1]) == 0.0)
    m.minimize(s[depth - 1] ** 2)
    return m


def test_max_def_nodes_is_monotone():
    # tighter cap -> no more eliminations and no larger reduced DAG
    counts = []
    dags = []
    for cap in (None, 50, 15):
        r = aggregate(_nonlinear_chain(5), method="gr", max_def_nodes=cap)
        counts.append(r.n_eliminated)
        _tree, dag = model_expression_size(r.reduced_model)
        dags.append(dag)
    assert counts[0] >= counts[1] >= counts[2]
    assert dags[0] >= dags[1] >= dags[2]
    assert counts[2] < counts[0]  # the tight cap actually drops eliminations


def test_default_off_is_a_noop():
    base = aggregate(_nonlinear_chain(4), method="gr")
    same = aggregate(_nonlinear_chain(4), method="gr", max_def_nodes=None, max_condition=None)
    assert base.eliminated == same.eliminated


def test_apply_guards_noop_returns_same_object():
    agg = aggregation_set(_nonlinear_chain(3), method="gr")
    assert apply_elimination_guards(agg, None, None) is agg


def test_conditioning_estimate_matches_hand_value():
    # gr eliminates x (sorted first) via y - 0.01 x == 0 -> amplification 1/0.01 = 100
    m = dm.Model("cond")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    m.subject_to(y - 0.01 * x == 0.0)
    m.minimize(x**2 + y**2)
    agg = aggregation_set(m, method="gr")
    assert conditioning_estimate(agg) == pytest.approx(100.0, rel=1e-9)


def test_well_scaled_model_has_unit_conditioning():
    m = dm.Model("unit")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    m.subject_to(y - (x + 1.0) == 0.0)  # coeff 1 -> amplification 1
    m.minimize(x**2 + y**2)
    agg = aggregation_set(m, method="gr")
    assert conditioning_estimate(agg) == pytest.approx(1.0)


def test_max_condition_drops_ill_conditioned_elimination():
    m = dm.Model("cond2")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    m.subject_to(y - 0.01 * x == 0.0)  # eliminating x amplifies by 100
    m.minimize(x**2 + y**2)
    uncapped = aggregate(m, method="gr")
    capped = aggregate(m, method="gr", max_condition=10.0)
    assert capped.n_eliminated < uncapped.n_eliminated


def test_capped_aggregation_still_round_trips():
    # dropping eliminations stays lower-triangular; the capped solve is correct.
    def build():
        m = dm.Model("rt")
        x = m.continuous("x", lb=-5, ub=5)
        s = [m.continuous(f"s{i}", lb=-50, ub=50) for i in range(4)]
        m.subject_to(s[0] - (0.5 * x + 1.0) == 0.0)
        for i in range(1, 4):
            m.subject_to(s[i] - (0.5 * s[i - 1] + 1.0) == 0.0)
        m.minimize(sum(si**2 for si in s) + x**2)
        return m

    full = build().solve()
    capped = solve(build(), method="gr", max_def_nodes=20)
    assert capped.objective == pytest.approx(full.objective, abs=1e-5)
    # recovered eliminated variables satisfy their linear definitions
    xv = float(np.asarray(capped.x["x"]).reshape(()))
    s0 = float(np.asarray(capped.x["s0"]).reshape(()))
    assert s0 == pytest.approx(0.5 * xv + 1.0, abs=1e-6)
