"""Tests for E5: implicit elimination of cyclic blocks.

Validates the implicit-block primitive: Newton solve of a nonsingular cyclic
block (against a closed form and residuals), the nonsingularity gate, and the
implicit-function-theorem derivatives (against finite differences).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

from discopt.aggregation.implicit import ImplicitBlock

pytestmark = pytest.mark.unit


def _bodies(*constraints):
    return [c.body for c in constraints]


def test_solve_linear_2x2_matches_closed_form():
    # x = 0.5y + 1, y = 0.5x + 2  ->  x = 8/3, y = 10/3
    m = dm.Model("lin2")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    m.subject_to(x - (0.5 * y + 1) == 0.0)
    m.subject_to(y - (0.5 * x + 2) == 0.0)
    block = ImplicitBlock(["x", "y"], _bodies(m._constraints[0], m._constraints[1]))
    sol = block.solve({})
    assert sol is not None
    assert sol["x"] == pytest.approx(8 / 3, abs=1e-9)
    assert sol["y"] == pytest.approx(10 / 3, abs=1e-9)


def test_solve_nonlinear_cycle_residual_zero():
    # x = 0.3*y^2 + 0.1, y = 0.2*x + 0.5  (contraction near 0; unique local root)
    m = dm.Model("nl")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.subject_to(x - (0.3 * y**2 + 0.1) == 0.0)
    m.subject_to(y - (0.2 * x + 0.5) == 0.0)
    block = ImplicitBlock(["x", "y"], _bodies(m._constraints[0], m._constraints[1]))
    sol = block.solve({}, x0={"x": 0.1, "y": 0.5})
    assert sol is not None
    # residual ~ 0 at the solution
    r = block._residual(sol)
    assert np.linalg.norm(r, np.inf) < 1e-8


def test_singular_block_returns_none():
    m = dm.Model("sing")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.subject_to(x - y == 0.0)  # parallel, inconsistent rows -> singular Jacobian
    m.subject_to(x - y == 1.0)  # no solution; Newton hits the singular Jacobian
    block = ImplicitBlock(["x", "y"], _bodies(m._constraints[0], m._constraints[1]))
    assert block.solve({}) is None
    assert block.is_nonsingular({}, {"x": 0.0, "y": 0.0}) is False


def test_nonsingular_predicate():
    m = dm.Model("ns")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.subject_to(x - (0.5 * y + 1) == 0.0)
    m.subject_to(y - (0.5 * x + 2) == 0.0)
    block = ImplicitBlock(["x", "y"], _bodies(m._constraints[0], m._constraints[1]))
    assert block.is_nonsingular({}, {"x": 0.0, "y": 0.0}) is True


def test_implicit_derivative_matches_finite_difference():
    # x = 0.5y + u, y = 0.5x + 2 ; parameter u (kept). dx/du = 4/3, dy/du = 2/3.
    m = dm.Model("impl")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    u = m.continuous("u", lb=-9, ub=9)
    m.subject_to(x - (0.5 * y + u) == 0.0)
    m.subject_to(y - (0.5 * x + 2) == 0.0)
    block = ImplicitBlock(["x", "y"], _bodies(m._constraints[0], m._constraints[1]))

    u0 = 1.0
    sol = block.solve({"u": u0})
    dv = block.implicit_derivative("u", {"u": u0}, sol)
    assert dv is not None
    assert dv["x"] == pytest.approx(4 / 3, abs=1e-6)
    assert dv["y"] == pytest.approx(2 / 3, abs=1e-6)

    # finite difference cross-check
    h = 1e-6
    sp = block.solve({"u": u0 + h})
    sm = block.solve({"u": u0 - h})
    for var in ("x", "y"):
        fd = (sp[var] - sm[var]) / (2 * h)
        assert dv[var] == pytest.approx(fd, abs=1e-5)
