"""Tests for E2: the target-linearity-preserving filter.

``preserve_target_linearity=True`` drops an elimination whose *nonlinear*
defining expression would turn a currently-*linear* constraint nonlinear, while
leaving linear substitutions (and nonlinear substitutions into already-nonlinear
targets) untouched.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

from discopt.aggregation.api import aggregate, solve
from discopt.aggregation.strategies import _constraint_is_linear

pytestmark = pytest.mark.unit


def _nonlinear_def_linear_target():
    # y = x^2 (nonlinear def, d2-eliminable); y also in the LINEAR constraint
    # w = y + z. Substituting x^2 for y would make w = x^2 + z nonlinear.
    m = dm.Model("tl")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-1, ub=9)
    w = m.continuous("w", lb=-9, ub=9)
    z = m.continuous("z", lb=-3, ub=3)
    m.subject_to(y - x**2 == 0.0)
    m.subject_to(w - (y + z) == 0.0)
    m.minimize(w**2 + x + z)
    return m


def test_nonlinear_def_into_linear_target_is_dropped():
    m = _nonlinear_def_linear_target
    assert aggregate(m(), method="d2").eliminated == ["y"]  # default: eliminated
    guarded = aggregate(m(), method="d2", preserve_target_linearity=True)
    assert guarded.eliminated == []  # guarded: refused


def test_linear_def_into_linear_target_is_kept():
    # y = 2x + 1 (linear def): substituting into w = y + z stays linear -> kept.
    m = dm.Model("lin")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-9, ub=9)
    w = m.continuous("w", lb=-9, ub=9)
    z = m.continuous("z", lb=-3, ub=3)
    m.subject_to(y - (2 * x + 1) == 0.0)
    m.subject_to(w - (y + z) == 0.0)
    m.minimize(w**2 + x + z)
    guarded = aggregate(m, method="d2", preserve_target_linearity=True)
    # The linear 2-var constraint can define either x or y; the filter keeps the
    # elimination (a linear substitution never breaks a linear target).
    assert guarded.n_eliminated == 1
    assert set(guarded.eliminated) <= {"x", "y"}


def test_nonlinear_def_without_linear_target_is_kept():
    # y = x^2 but y only appears in a nonlinear target (w = y*z): no linear
    # constraint to protect, so the elimination is kept.
    m = dm.Model("nl")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-1, ub=9)
    w = m.continuous("w", lb=-30, ub=30)
    z = m.continuous("z", lb=-3, ub=3)
    m.subject_to(y - x**2 == 0.0)
    m.subject_to(w - y * z == 0.0)  # nonlinear target
    m.minimize(w + x + z)
    guarded = aggregate(m, method="d2", preserve_target_linearity=True)
    assert "y" in guarded.eliminated


def test_constraint_linearity_predicate():
    m = dm.Model("p")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    m.subject_to(2 * x + 3 * y == 1.0)  # linear
    m.subject_to(x * y == 1.0)  # nonlinear
    assert _constraint_is_linear(m._constraints[0]) is True
    assert _constraint_is_linear(m._constraints[1]) is False


@pytest.mark.smoke
def test_solve_with_target_linearity_still_correct():
    original = _nonlinear_def_linear_target().solve()
    res = solve(_nonlinear_def_linear_target(), method="d2", preserve_target_linearity=True)
    assert res.objective == pytest.approx(original.objective, abs=1e-4)
    full = {k: float(np.asarray(v)) for k, v in res.x.items()}
    orig = {k: float(np.asarray(v)) for k, v in original.x.items()}
    for name in orig:
        assert full[name] == pytest.approx(orig[name], abs=1e-3)
