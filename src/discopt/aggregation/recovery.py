"""Recover eliminated variables from a reduced-space solution.

When variables are aggregated, the reduced-space model is solved for the *kept*
variables only.  Because the defining equations are strictly lower triangular
(Lemma 1), each eliminated variable is an explicit function of the kept
variables, so its value is recovered by evaluating that defining expression at
the kept-variable solution — no iterative solve.

This module provides:

- :func:`evaluate_expression` — a small numeric evaluator over the
  :mod:`discopt.modeling` expression DAG.
- :class:`RecoveryMap` — the ordered ``{eliminated_var: defining_expression}``
  produced by :func:`discopt.aggregation.transform.apply_aggregation`, with
  :meth:`RecoveryMap.recover` / :meth:`RecoveryMap.recover_full` to reconstruct
  the eliminated values from a kept-variable assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

# Unary function names (FunctionCall / UnaryOp) mapped to numpy implementations.
_FUNCS = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sign": np.sign,
    "erf": getattr(np, "erf", None),  # np.erf only in newer numpy; may be None
}


def evaluate_expression(expr: Any, values: dict[str, Any]) -> Any:
    """Numerically evaluate ``expr`` given variable ``values`` (by name).

    ``values`` maps variable names to numbers (scalars) or arrays (for
    multi-dimensional variables).  Returns a numpy scalar/array.  Raises
    :class:`KeyError` for a referenced variable missing from ``values`` and
    :class:`NotImplementedError` for an unsupported node or function.
    """
    if isinstance(expr, Variable):
        if expr.name not in values:
            raise KeyError(f"no value provided for variable {expr.name!r}")
        return np.asarray(values[expr.name], dtype=float)
    if isinstance(expr, Constant):
        return np.asarray(expr.value, dtype=float)
    if isinstance(expr, Parameter):
        return np.asarray(expr.value, dtype=float)
    if isinstance(expr, IndexExpression):
        base = evaluate_expression(expr.base, values)
        return base[expr.index]
    if isinstance(expr, UnaryOp):
        operand = evaluate_expression(expr.operand, values)
        if expr.op in ("-", "neg"):
            return -operand
        if expr.op in ("+", "pos"):
            return operand
        fn = _FUNCS.get(expr.op)
        if fn is None:
            raise NotImplementedError(f"unary op {expr.op!r} not supported in recovery")
        return fn(operand)
    if isinstance(expr, BinaryOp):
        left = evaluate_expression(expr.left, values)
        right = evaluate_expression(expr.right, values)
        op = expr.op
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op in ("**", "^"):
            return left**right
        raise NotImplementedError(f"binary op {op!r} not supported in recovery")
    if isinstance(expr, FunctionCall):
        fn = _FUNCS.get(expr.func_name)
        if fn is None:
            raise NotImplementedError(f"function {expr.func_name!r} not supported in recovery")
        args = [evaluate_expression(a, values) for a in expr.args]
        return fn(*args)
    if isinstance(expr, SumExpression):
        operand = evaluate_expression(expr.operand, values)
        axis = getattr(expr, "axis", None)
        return np.sum(operand, axis=axis)
    if isinstance(expr, SumOverExpression):
        return sum(evaluate_expression(t, values) for t in expr.terms)
    # Plain python number wrapped as an expression leaf, or unknown constant.
    if isinstance(expr, (int, float)):
        return np.asarray(expr, dtype=float)
    raise NotImplementedError(f"cannot evaluate expression node of type {type(expr).__name__}")


Interval = tuple[float, float]
_WIDE: Interval = (-float("inf"), float("inf"))


def _safe_mul(a: float, b: float) -> float:
    """Product with the interval convention ``0 * inf == 0``."""
    if a == 0.0 or b == 0.0:
        return 0.0
    return a * b


def _mul_interval(x: Interval, y: Interval) -> Interval:
    lo1, hi1 = x
    lo2, hi2 = y
    prods = [_safe_mul(lo1, lo2), _safe_mul(lo1, hi2), _safe_mul(hi1, lo2), _safe_mul(hi1, hi2)]
    return (min(prods), max(prods))


def _pow_interval(base: Interval, p: float) -> Interval:
    """Range of ``base ** p`` for a constant exponent ``p`` (conservative)."""
    lo, hi = base
    if p == 0:
        return (1.0, 1.0)
    if float(p).is_integer():
        k = int(p)
        if k > 0:
            if k % 2 == 0:  # even power: non-negative, min at 0 if the base spans it
                if lo >= 0:
                    return (lo**k, hi**k)
                if hi <= 0:
                    return (hi**k, lo**k)
                return (0.0, max(lo**k, hi**k))
            return (lo**k, hi**k)  # odd power: monotone increasing
    # Non-integer or negative exponent: only safe when the base is strictly > 0.
    if lo > 0:
        vals = [lo**p, hi**p]
        return (min(vals), max(vals))
    return _WIDE


def interval_evaluate(expr: Any, boxes: dict[str, Interval]) -> Interval:
    """Conservative outer interval enclosure of ``expr`` over variable ``boxes``.

    ``boxes`` maps each variable name to a ``(lo, hi)`` interval.  The returned
    ``(lo, hi)`` is guaranteed to *contain* the true range of ``expr`` over those
    boxes (an outer bound), so it is sound to use for proving a constraint
    redundant.  Unknown or unsupported nodes fall back to ``(-inf, +inf)``, which
    only ever forgoes an optimisation — never an unsound one.
    """
    if isinstance(expr, Variable):
        return boxes.get(expr.name, _WIDE)
    if isinstance(expr, Constant):
        v = float(np.asarray(expr.value).reshape(()))
        return (v, v)
    if isinstance(expr, Parameter):
        try:
            v = float(np.asarray(expr.value).reshape(()))
        except (TypeError, ValueError):
            return _WIDE
        return (v, v)
    if isinstance(expr, UnaryOp):
        lo, hi = interval_evaluate(expr.operand, boxes)
        if expr.op in ("-", "neg"):
            return (-hi, -lo)
        if expr.op in ("+", "pos"):
            return (lo, hi)
        if expr.op == "exp":
            return (float(np.exp(lo)), float(np.exp(hi)))
        if expr.op in ("tanh", "arctan", "atan"):
            fn = np.tanh if expr.op == "tanh" else np.arctan
            return (float(fn(lo)), float(fn(hi)))
        if expr.op in ("sin", "cos"):
            return (-1.0, 1.0)
        if expr.op == "sqrt" and lo >= 0:
            return (float(np.sqrt(lo)), float(np.sqrt(hi)))
        if expr.op == "log" and lo > 0:
            return (float(np.log(lo)), float(np.log(hi)))
        return _WIDE
    if isinstance(expr, BinaryOp):
        left = interval_evaluate(expr.left, boxes)
        right = interval_evaluate(expr.right, boxes)
        if expr.op == "+":
            return (left[0] + right[0], left[1] + right[1])
        if expr.op == "-":
            return (left[0] - right[1], left[1] - right[0])
        if expr.op == "*":
            return _mul_interval(left, right)
        if expr.op == "/":
            if right[0] <= 0 <= right[1]:  # denominator spans zero
                return _WIDE
            return _mul_interval(left, (1.0 / right[1], 1.0 / right[0]))
        if expr.op in ("**", "^"):
            if right[0] == right[1]:  # constant exponent
                return _pow_interval(left, right[0])
            return _WIDE
        return _WIDE
    if isinstance(expr, FunctionCall):
        # Reuse the unary-op enclosures for single-argument functions.
        if len(expr.args) == 1:
            return interval_evaluate(UnaryOp(expr.func_name, expr.args[0]), boxes)
        return _WIDE
    if isinstance(expr, SumOverExpression):
        lo = hi = 0.0
        for t in expr.terms:
            tl, th = interval_evaluate(t, boxes)
            lo += tl
            hi += th
        return (lo, hi)
    if isinstance(expr, (int, float)):
        return (float(expr), float(expr))
    # SumExpression over arrays, IndexExpression, and anything else: be safe.
    return _WIDE


def _to_scalar(value: Any) -> Any:
    """Collapse a size-1 array result to a python float; leave arrays as-is."""
    arr = np.asarray(value)
    if arr.ndim == 0 or arr.size == 1:
        return float(arr.reshape(()))
    return arr


@dataclass
class RecoveryMap:
    """Ordered defining expressions for the eliminated variables.

    Attributes:
        defining: ``{var: expression}`` giving each eliminated variable as a
            function of the *kept* variables (references kept variables by name).
        order: eliminated variable names in topological (inputs-first) order.
    """

    defining: dict[str, Expression] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def recover(self, kept_values: dict[str, Any]) -> dict[str, Any]:
        """Reconstruct the eliminated variable values from kept-variable values.

        Evaluates the defining expressions in elimination order.  Later
        definitions may reference earlier eliminated variables, so recovered
        values are accumulated into the evaluation environment as we go.
        """
        env = dict(kept_values)
        recovered: dict[str, Any] = {}
        for name in self.order:
            val = _to_scalar(evaluate_expression(self.defining[name], env))
            recovered[name] = val
            env[name] = val
        return recovered

    def recover_full(self, kept_values: dict[str, Any]) -> dict[str, Any]:
        """Kept values merged with the recovered eliminated values."""
        return {**kept_values, **self.recover(kept_values)}


def feasibility_violations(model: Any, point: dict[str, Any], tol: float = 1e-7) -> list[str]:
    """Return human-readable feasibility violations of ``point`` in ``model``.

    Checks every algebraic constraint body against its sense (``body == 0`` or
    ``body <= 0``; ``>=`` handled defensively) and every variable's box bounds,
    with an absolute/relative tolerance ``tol``.  A constraint referencing a
    variable absent from ``point`` is skipped.  Returns an empty list when
    ``point`` is feasible.

    Intended for *exact* points (values recovered from defining expressions, not
    a solver iterate), where residual bodies are computed in closed form — so a
    nonempty result is a genuine infeasibility, not solver noise.
    """
    from discopt.modeling.core import Constraint

    violations: list[str] = []
    for cid, c in enumerate(getattr(model, "_constraints", [])):
        if not isinstance(c, Constraint):
            continue
        try:
            body = np.asarray(evaluate_expression(c.body, point), dtype=float)
        except (KeyError, NotImplementedError):
            continue  # references an unknown variable/node — cannot judge here
        worst = float(np.max(np.abs(body))) if body.size else 0.0
        scale = max(1.0, worst)
        sense = getattr(c, "sense", "==")
        if sense == "==":
            if worst > tol * scale:
                violations.append(f"constraint {cid} (== 0) residual {worst:.3g}")
        elif sense == "<=":
            hi = float(np.max(body)) if body.size else 0.0
            if hi > tol * scale:
                violations.append(f"constraint {cid} (<= 0) exceeds by {hi:.3g}")
        elif sense == ">=":
            lo = float(np.min(body)) if body.size else 0.0
            if lo < -tol * scale:
                violations.append(f"constraint {cid} (>= 0) short by {-lo:.3g}")

    for v in getattr(model, "_variables", []):
        if v.name not in point:
            continue
        val = np.asarray(point[v.name], dtype=float)
        lb = np.asarray(v.lb, dtype=float)
        ub = np.asarray(v.ub, dtype=float)
        margin = tol * np.maximum(1.0, np.abs(np.where(np.isfinite(lb), lb, 0.0)))
        below = val < lb - margin
        margin_u = tol * np.maximum(1.0, np.abs(np.where(np.isfinite(ub), ub, 0.0)))
        above = val > ub + margin_u
        if np.any(below) or np.any(above):
            violations.append(
                f"variable {v.name!r} value {np.asarray(val).ravel()[:4]} "
                f"outside box [{np.min(lb):.3g}, {np.max(ub):.3g}]"
            )
    return violations


__all__ = [
    "evaluate_expression",
    "interval_evaluate",
    "RecoveryMap",
    "feasibility_violations",
]
