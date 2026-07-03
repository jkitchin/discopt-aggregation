"""Elimination guards: bound reduced-model expression size and conditioning (D2).

Two opt-in guards, applied after a strategy selects an :class:`AggregationSet`
and before it is applied.  Both default off (no behaviour change).

* **``max_def_nodes``** — drop any elimination whose *resolved* defining
  expression (inlined into kept-variable terms, exactly as the reduced model will
  hold it) exceeds a node-count cap.  This bounds the expression-graph / Hessian
  blow-up that aggressive substitution causes on deep nonlinear chains (plan item
  R2 / discopt#383): ``pivot_tol`` and ``max_fill`` do not — the former is a
  numerical pivot guard, the latter a *Jacobian-incidence* estimate, neither the
  substituted-expression size.

* **``max_condition``** — drop any elimination whose worst-case recovery-error
  amplification exceeds a cap.  For an eliminated ``v`` whose *resolved* defining
  expression ``D_v`` (inlined into kept-variable terms) is what recovery
  evaluates, a unit error in a kept variable ``k`` perturbs the recovered ``v`` by
  ``∂D_v/∂k``; the amplification is ``A(v) = max_k |∂D_v/∂k|`` bounded over the
  kept-variable boxes.  This composes the pivot *and* the coefficients each
  referenced variable enters with — unlike a bare ``1/|coeff|`` product, which
  misses a large multiplier such as ``v = 10^6·u`` (unit pivot, huge
  amplification).  ``pivot_tol`` only bounds a *single* pivot.

Dropping any subset of a lower-triangular elimination set leaves it lower
triangular (a later definition then references a *kept* variable instead of an
inlined one), so both guards are always sound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from discopt.modeling.core import BinaryOp, Constant, Variable

from discopt.aggregation.derivatives import differentiate_expression
from discopt.aggregation.exprwalk import collect_variable_names
from discopt.aggregation.hessian_cost import _children, expression_dag_size
from discopt.aggregation.recovery import interval_evaluate
from discopt.aggregation.strategies import AggregationSet
from discopt.aggregation.transform import substitute_expression

_INF_BOUND = 9.9e19  # below discopt's +-9.999e19 sentinel


def _var_names(expr: Any) -> set[str]:
    """Names of the :class:`Variable` leaves in ``expr`` (iterative)."""
    names: set[str] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Variable):
            names.add(node.name)
        stack.extend(_children(node))
    return names


def _model_boxes(inc: Any) -> dict[str, tuple[float, float]]:
    """Interval box ``{name: (lo, hi)}`` for the incidence model's variables."""
    boxes: dict[str, tuple[float, float]] = {}
    for v in getattr(inc.model, "_variables", []):
        lb = float(np.min(v.lb))
        ub = float(np.max(v.ub))
        boxes[v.name] = (
            -float("inf") if lb <= -_INF_BOUND else lb,
            float("inf") if ub >= _INF_BOUND else ub,
        )
    return boxes


def _raw_define(inc: Any, cid: int, var: str, coeff: float) -> Any:
    """The pre-inline defining expression ``var = -rest/coeff`` (rest = body|var=0)."""
    body = inc.constraints[cid].body
    rest = substitute_expression(body, {var: Constant(0.0)})
    return BinaryOp("*", rest, Constant(-1.0 / coeff))


def _amplification(resolved: Any, boxes: dict[str, tuple[float, float]]) -> float:
    """Worst-case recovery-error amplification ``max_k |∂resolved/∂k|`` over boxes.

    ``resolved`` is a defining expression already inlined into kept-variable
    terms, so its partials w.r.t. each kept variable are exactly the sensitivities
    recovery propagates.  A constant definition amplifies nothing (``0.0``); a
    non-differentiable node cannot be bounded and returns ``inf`` (conservatively
    dropped by ``max_condition``).
    """
    names, analyzable = collect_variable_names(resolved)
    if not analyzable:
        return float("inf")
    worst = 0.0
    for k in names:
        d = differentiate_expression(resolved, k)
        if d is None:
            return float("inf")
        lo, hi = interval_evaluate(d, boxes)
        worst = max(worst, abs(lo), abs(hi))
    return worst


@dataclass
class _Walk:
    resolved: dict[str, Any]  # kept var -> resolved (inlined) defining expr


def _step(
    walk: _Walk, inc: Any, e: Any, boxes: dict[str, tuple[float, float]]
) -> tuple[Any, float]:
    """Resolved expression and amplification for elimination ``e`` given kept state."""
    raw = _raw_define(inc, e.cid, e.var, e.coeff)
    resolved = substitute_expression(raw, walk.resolved)  # inline kept elims only
    return resolved, _amplification(resolved, boxes)


def conditioning_estimate(agg: AggregationSet) -> float:
    """Worst-case recovery-error amplification over the composed elimination chain.

    ``A(v) = max_k |∂D_v/∂k|`` bounded over the kept-variable boxes, maximised
    over eliminations, where ``D_v`` is ``v``'s resolved (inlined) defining
    expression.  ``1.0`` means unit-scaled; large values mean recovery of some
    eliminated variable is ill-conditioned (a small pivot, a large coefficient, or
    a compounding chain).  Returns ``1.0`` for an empty set.
    """
    boxes = _model_boxes(agg.incidence)
    walk = _Walk(resolved={})
    worst = 1.0
    for e in agg.eliminations:
        resolved, amplification = _step(walk, agg.incidence, e, boxes)
        walk.resolved[e.var] = resolved
        worst = max(worst, amplification)
    return worst


def apply_elimination_guards(
    agg: AggregationSet,
    max_def_nodes: int | None = None,
    max_condition: float | None = None,
) -> AggregationSet:
    """Filter ``agg``, dropping eliminations over the size / conditioning caps.

    ``max_def_nodes`` caps the resolved defining-expression node count;
    ``max_condition`` caps the cumulative pivot amplification.  Both ``None``
    (default) is a no-op returning ``agg`` unchanged.
    """
    if max_def_nodes is None and max_condition is None:
        return agg

    boxes = _model_boxes(agg.incidence)
    walk = _Walk(resolved={})
    kept = []
    dropped_nodes = 0
    dropped_cond = 0
    for e in agg.eliminations:  # topological order
        resolved, amplification = _step(walk, agg.incidence, e, boxes)
        if max_condition is not None and amplification > max_condition:
            dropped_cond += 1
            continue
        if max_def_nodes is not None and expression_dag_size(resolved) > max_def_nodes:
            dropped_nodes += 1
            continue
        walk.resolved[e.var] = resolved
        kept.append(e)

    return AggregationSet(
        eliminations=kept,
        incidence=agg.incidence,
        method=agg.method,
        n_dropped=agg.n_dropped + dropped_nodes + dropped_cond,
        stats={
            **agg.stats,
            "dropped_def_nodes": dropped_nodes,
            "dropped_condition": dropped_cond,
        },
    )


__all__ = ["apply_elimination_guards", "conditioning_estimate"]
