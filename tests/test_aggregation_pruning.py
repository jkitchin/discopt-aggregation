"""Tests for G2: redundant-bound pruning via interval propagation.

- ``interval_evaluate`` must return a conservative *outer* enclosure — validated
  by dense sampling (every sampled value lands inside the interval).
- A box->inequality on an eliminated variable is pruned only when it is provably
  redundant; a binding bound is always kept.
- Pruning is sound: a pruned inequality holds for every point in the kept box.
"""

from __future__ import annotations

import math
import random

import discopt.modeling as dm
import pytest

from discopt.aggregation.recovery import evaluate_expression, interval_evaluate
from discopt.aggregation.strategies import aggregation_set
from discopt.aggregation.transform import apply_aggregation

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# interval_evaluate: outer-bound property vs sampling                          #
# --------------------------------------------------------------------------- #
def _assert_outer_bound(expr, boxes, n=3000, seed=0):
    rng = random.Random(seed)
    lo, hi = interval_evaluate(expr, boxes)
    for _ in range(n):
        vals = {name: rng.uniform(b[0], b[1]) for name, b in boxes.items()}
        v = float(evaluate_expression(expr, vals))
        assert lo - 1e-9 <= v <= hi + 1e-9, f"{v} not in [{lo}, {hi}]"
    return lo, hi


def _two_var_model():
    m = dm.Model("iv")
    x = m.continuous("x", lb=-2, ub=3)
    y = m.continuous("y", lb=1, ub=4)
    return m, x, y


def test_interval_linear_is_exact():
    m, x, y = _two_var_model()
    expr = 2 * x - 3 * y + 1
    boxes = {"x": (-2.0, 3.0), "y": (1.0, 4.0)}
    lo, hi = _assert_outer_bound(expr, boxes)
    # Linear: exact at the corners. min = 2*-2 - 3*4 + 1 = -15; max = 2*3 - 3*1 + 1 = 4.
    assert lo == pytest.approx(-15.0)
    assert hi == pytest.approx(4.0)


def test_interval_product_and_square_are_outer_bounds():
    m, x, y = _two_var_model()
    boxes = {"x": (-2.0, 3.0), "y": (1.0, 4.0)}
    _assert_outer_bound(x * y, boxes)
    _assert_outer_bound((x - 1) ** 2, boxes)
    _assert_outer_bound(x * x - 2 * y, boxes)


def test_interval_square_nonnegative_spanning_zero():
    m = dm.Model("sq")
    x = m.continuous("x", lb=-3, ub=2)
    lo, hi = interval_evaluate(x**2, {"x": (-3.0, 2.0)})
    assert lo == pytest.approx(0.0)  # spans zero -> min 0
    assert hi == pytest.approx(9.0)


def test_interval_division_spanning_zero_is_wide():
    m = dm.Model("div")
    x = m.continuous("x", lb=1, ub=2)
    y = m.continuous("y", lb=-1, ub=1)  # denominator spans 0
    lo, hi = interval_evaluate(x / y, {"x": (1.0, 2.0), "y": (-1.0, 1.0)})
    assert lo == -math.inf and hi == math.inf


def test_interval_division_safe_denominator():
    m = dm.Model("div2")
    x = m.continuous("x", lb=2, ub=6)
    y = m.continuous("y", lb=1, ub=2)
    _assert_outer_bound(x / y, {"x": (2.0, 6.0), "y": (1.0, 2.0)})


def test_interval_exp_monotone():
    m = dm.Model("exp")
    x = m.continuous("x", lb=-1, ub=1)
    lo, hi = interval_evaluate(dm.exp(x), {"x": (-1.0, 1.0)})
    assert lo == pytest.approx(math.exp(-1))
    assert hi == pytest.approx(math.exp(1))


def test_interval_unknown_variable_is_wide():
    m = dm.Model("u")
    x = m.continuous("x", lb=-1, ub=1)
    lo, hi = interval_evaluate(x, {})  # no box provided
    assert lo == -math.inf and hi == math.inf


# --------------------------------------------------------------------------- #
# bound pruning                                                               #
# --------------------------------------------------------------------------- #
def _prunable_model():
    # w = x1 + x2, x_i in [-1, 1] -> w ranges in [-2, 2], well inside [-10, 10].
    m = dm.Model("prune")
    x1 = m.continuous("x1", lb=-1, ub=1)
    x2 = m.continuous("x2", lb=-1, ub=1)
    w = m.continuous("w", lb=-10, ub=10)
    m.subject_to(w - (x1 + x2) == 0.0)
    m.minimize(x1**2 + x2**2 + w)
    return m


def test_redundant_bounds_are_pruned():
    m = _prunable_model()
    agg = aggregation_set(m, method="gr")
    assert agg.variables() == ["w"]  # gr eliminates w (sorts before x1, x2)

    pruned = apply_aggregation(m, agg, prune_bounds=True)
    assert pruned.n_bound_pruned == 2  # both bounds provably redundant
    assert pruned.n_bound_inequalities == 0

    kept = apply_aggregation(m, agg, prune_bounds=False)
    assert kept.n_bound_inequalities == 2
    assert kept.n_bound_pruned == 0


def test_binding_bound_is_kept():
    # w = 2*x, x in [-1, 1] -> w ranges [-2, 2]; bounds [-1.5, 1.5] genuinely bind.
    m = dm.Model("bind")
    x = m.continuous("x", lb=-1, ub=1)
    w = m.continuous("w", lb=-1.5, ub=1.5)
    m.subject_to(w - 2 * x == 0.0)
    m.minimize(x + w)
    agg = aggregation_set(m, method="gr")
    assert agg.variables() == ["w"]
    rm = apply_aggregation(m, agg, prune_bounds=True)
    assert rm.n_bound_inequalities == 2  # both bounds bind -> kept
    assert rm.n_bound_pruned == 0


def test_pruned_inequality_is_truly_redundant_over_box():
    # Soundness: the pruned bound holds for every point in the kept variables' box.
    m = _prunable_model()
    agg = aggregation_set(m, method="gr")
    rm = apply_aggregation(m, agg, prune_bounds=True)
    assert rm.n_bound_pruned == 2

    defining = rm.recovery.defining["w"]  # w = x1 + x2 (in kept vars)
    rng = random.Random(1)
    for _ in range(3000):
        vals = {"x1": rng.uniform(-1, 1), "x2": rng.uniform(-1, 1)}
        w_val = float(evaluate_expression(defining, vals))
        assert -10.0 <= w_val <= 10.0  # original bounds never violated


def test_pruning_defaults_on_but_toggleable():
    m = _prunable_model()
    agg = aggregation_set(m, method="gr")
    default = apply_aggregation(m, agg)  # prune_bounds defaults True
    assert default.n_bound_pruned == 2
