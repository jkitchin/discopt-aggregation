"""Symbolic differentiation over the discopt expression DAG.

:func:`differentiate_expression` returns the exact partial derivative of an
expression with respect to a (scalar) variable name, as a new expression built
from the same node types.  It is the primitive the dual/multiplier recovery
(follow-up item G3) needs — the stationarity of the full Lagrangian in an
eliminated variable requires the gradients of the objective and constraints with
respect to that variable — and is independently useful for sensitivity work.

The builder folds trivial zeros and ones so the returned derivative stays
compact.  Unsupported constructs (variable exponents, reductions, indexed
references, opaque functions) yield ``None`` — callers then fall back
(e.g. to finite differences) rather than trusting a wrong derivative.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)

_ZERO = Constant(0.0)
_ONE = Constant(1.0)

# d/dx of a single-argument function f(u): the multiplier g such that
# d/dx f(u) = g(u) * du/dx.  ``None`` marks a function we do not differentiate.
_FUNC_DERIV = {
    "sin": lambda u: FunctionCall("cos", u),
    "cos": lambda u: _neg(FunctionCall("sin", u)),
    "exp": lambda u: FunctionCall("exp", u),
    "log": lambda u: _div(_ONE, u),
    "sqrt": lambda u: _div(_ONE, _mul(Constant(2.0), FunctionCall("sqrt", u))),
    "tanh": lambda u: _sub(_ONE, _mul(FunctionCall("tanh", u), FunctionCall("tanh", u))),
}


def _is_zero(e: Any) -> bool:
    return isinstance(e, Constant) and float(np.asarray(e.value).reshape(())) == 0.0


def _is_one(e: Any) -> bool:
    return isinstance(e, Constant) and float(np.asarray(e.value).reshape(())) == 1.0


def _neg(a: Expression) -> Expression:
    return a if _is_zero(a) else UnaryOp("-", a)


def _add(a: Expression, b: Expression) -> Expression:
    if _is_zero(a):
        return b
    if _is_zero(b):
        return a
    return BinaryOp("+", a, b)


def _sub(a: Expression, b: Expression) -> Expression:
    if _is_zero(b):
        return a
    if _is_zero(a):
        return _neg(b)
    return BinaryOp("-", a, b)


def _mul(a: Expression, b: Expression) -> Expression:
    if _is_zero(a) or _is_zero(b):
        return _ZERO
    if _is_one(a):
        return b
    if _is_one(b):
        return a
    return BinaryOp("*", a, b)


def _div(a: Expression, b: Expression) -> Expression:
    if _is_zero(a):
        return _ZERO
    if _is_one(b):
        return a
    return BinaryOp("/", a, b)


def _const_value(e: Any) -> float | None:
    if isinstance(e, Constant):
        return float(np.asarray(e.value).reshape(()))
    return None


def differentiate_expression(expr: Any, var: str) -> Expression | None:
    """Exact partial derivative ``d(expr)/d(var)`` as an expression, or ``None``.

    ``var`` is a scalar variable name.  Returns ``None`` when the derivative
    involves a construct this differentiator does not handle (variable exponent,
    reduction over the variable, indexed/opaque reference), signalling the caller
    to fall back rather than use an incorrect result.
    """
    if isinstance(expr, Variable):
        return _ONE if expr.name == var else _ZERO
    if isinstance(expr, (Constant, Parameter)):
        return _ZERO
    if isinstance(expr, UnaryOp):
        d = differentiate_expression(expr.operand, var)
        if d is None:
            return None
        if expr.op in ("-", "neg"):
            return _neg(d)
        if expr.op in ("+", "pos"):
            return d
        fn = _FUNC_DERIV.get(expr.op)
        if fn is None:
            return None
        return _mul(fn(expr.operand), d)
    if isinstance(expr, FunctionCall):
        if len(expr.args) != 1:
            return None
        fn = _FUNC_DERIV.get(expr.func_name)
        if fn is None:
            return None
        d = differentiate_expression(expr.args[0], var)
        if d is None:
            return None
        return _mul(fn(expr.args[0]), d)
    if isinstance(expr, BinaryOp):
        left, right = expr.left, expr.right
        dl = differentiate_expression(left, var)
        dr = differentiate_expression(right, var)
        if dl is None or dr is None:
            return None
        op = expr.op
        if op == "+":
            return _add(dl, dr)
        if op == "-":
            return _sub(dl, dr)
        if op == "*":
            return _add(_mul(dl, right), _mul(left, dr))
        if op == "/":
            # (dl*r - l*dr) / r^2
            num = _sub(_mul(dl, right), _mul(left, dr))
            return _div(num, _mul(right, right))
        if op in ("**", "^"):
            p = _const_value(right)
            if p is None or not _is_zero(dr):
                return None  # variable exponent: not handled
            # d/dx l^p = p * l^(p-1) * dl
            base_pow = _pow_const(left, p - 1.0)
            return _mul(_mul(Constant(p), base_pow), dl)
        return None
    if isinstance(expr, SumOverExpression):
        parts = [differentiate_expression(t, var) for t in expr.terms]
        if any(p is None for p in parts):
            return None
        out: Expression = _ZERO
        for p in parts:
            out = _add(out, p)  # type: ignore[arg-type]
        return out
    if isinstance(expr, SumExpression):
        return None  # array reduction: out of scope for scalar differentiation
    if isinstance(expr, IndexExpression):
        return None  # indexed reference: not a scalar-name derivative
    return None


def _pow_const(base: Expression, p: float) -> Expression:
    if p == 0.0:
        return _ONE
    if p == 1.0:
        return base
    return BinaryOp("**", base, Constant(p))


__all__ = ["differentiate_expression"]
