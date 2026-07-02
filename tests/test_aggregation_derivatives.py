"""Tests for :mod:`discopt.aggregation.derivatives` (symbolic differentiation).

Each symbolic derivative is validated against a central finite-difference of the
original expression at random points.
"""

from __future__ import annotations

import random

import discopt.modeling as dm
import pytest

from discopt.aggregation.derivatives import differentiate_expression
from discopt.aggregation.recovery import evaluate_expression

pytestmark = pytest.mark.unit


def _fd(expr, var, values, h=1e-6):
    up = dict(values)
    dn = dict(values)
    up[var] = values[var] + h
    dn[var] = values[var] - h
    return (float(evaluate_expression(expr, up)) - float(evaluate_expression(expr, dn))) / (2 * h)


def _check(expr, var, base, n=20, seed=0, tol=1e-4):
    d = differentiate_expression(expr, var)
    assert d is not None, "expected a derivative"
    rng = random.Random(seed)
    for _ in range(n):
        vals = {k: v + rng.uniform(-0.5, 0.5) for k, v in base.items()}
        got = float(evaluate_expression(d, vals))
        want = _fd(expr, var, vals)
        assert got == pytest.approx(want, abs=tol, rel=1e-4), f"{var}: {got} vs {want}"


def _xy():
    m = dm.Model("d")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    return x, y


def test_polynomial():
    x, y = _xy()
    expr = 3 * x**2 + 2 * x * y - y + 5
    _check(expr, "x", {"x": 1.0, "y": 1.0})
    _check(expr, "y", {"x": 1.0, "y": 1.0})


def test_product_and_quotient():
    x, y = _xy()
    _check(x * y, "x", {"x": 1.0, "y": 2.0})
    _check(x / (y + 3), "x", {"x": 1.0, "y": 2.0})
    _check(x / (y + 3), "y", {"x": 1.0, "y": 2.0})


def test_power_constant_exponent():
    x, _y = _xy()
    _check(x**3, "x", {"x": 1.5})
    _check((x + 1) ** 2, "x", {"x": 0.7})


def test_transcendentals():
    x, y = _xy()
    _check(dm.exp(x), "x", {"x": 0.3})
    _check(dm.sin(x) * y, "x", {"x": 0.3, "y": 2.0})
    _check(dm.sin(x) * y, "y", {"x": 0.3, "y": 2.0})
    _check(dm.sqrt(x + 3), "x", {"x": 1.0})
    _check(dm.log(x + 5), "x", {"x": 1.0})


def test_independent_variable_is_zero():
    x, y = _xy()
    d = differentiate_expression(3 * x + 5, "y")
    assert d is not None
    assert float(evaluate_expression(d, {"x": 1.0, "y": 1.0})) == 0.0


def test_variable_exponent_returns_none():
    x, y = _xy()
    assert differentiate_expression(x**y, "x") is None


def test_constant_derivative():
    from discopt.modeling.core import Constant

    d = differentiate_expression(Constant(7.0), "x")
    assert d is not None
    assert float(evaluate_expression(d, {"x": 1.0})) == 0.0
