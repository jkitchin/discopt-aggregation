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

* **``max_condition``** — drop any elimination whose cumulative pivot
  amplification exceeds a cap.  Eliminating ``v`` (``v = -rest/coeff``) amplifies
  ``v``'s recovery by ``1/|coeff|``; if ``rest`` references an earlier-eliminated
  ``u``, the amplification compounds as ``(1/|coeff_v|) * A(u)``.  ``pivot_tol``
  only bounds a *single* pivot; this bounds the product along the composed chain.

Dropping any subset of a lower-triangular elimination set leaves it lower
triangular (a later definition then references a *kept* variable instead of an
inlined one), so both guards are always sound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from discopt.modeling.core import BinaryOp, Constant, Variable

from discopt.aggregation.hessian_cost import _children, expression_dag_size
from discopt.aggregation.strategies import AggregationSet
from discopt.aggregation.transform import substitute_expression


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


def _raw_define(inc: Any, cid: int, var: str, coeff: float) -> Any:
    """The pre-inline defining expression ``var = -rest/coeff`` (rest = body|var=0)."""
    body = inc.constraints[cid].body
    rest = substitute_expression(body, {var: Constant(0.0)})
    return BinaryOp("*", rest, Constant(-1.0 / coeff))


@dataclass
class _Walk:
    resolved: dict[str, Any]  # kept var -> resolved (inlined) defining expr
    ampl: dict[str, float]  # kept var -> cumulative pivot amplification


def _step(walk: _Walk, inc: Any, e: Any) -> tuple[Any, float]:
    """Resolved expression and amplification for elimination ``e`` given kept state."""
    raw = _raw_define(inc, e.cid, e.var, e.coeff)
    referenced = [walk.ampl[u] for u in _var_names(raw) if u in walk.ampl]
    amplification = (1.0 / abs(e.coeff)) * (max(referenced) if referenced else 1.0)
    resolved = substitute_expression(raw, walk.resolved)  # inline kept elims only
    return resolved, amplification


def conditioning_estimate(agg: AggregationSet) -> float:
    """Worst-case cumulative pivot amplification over the composed elimination chain.

    ``1.0`` means every pivot is unit-scaled; large values mean a chain of small
    pivots compounds (ill-conditioned back-substitution / recovery).  Returns
    ``1.0`` for an empty set.
    """
    walk = _Walk(resolved={}, ampl={})
    worst = 1.0
    for e in agg.eliminations:
        resolved, amplification = _step(walk, agg.incidence, e)
        walk.resolved[e.var] = resolved
        walk.ampl[e.var] = amplification
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

    walk = _Walk(resolved={}, ampl={})
    kept = []
    dropped_nodes = 0
    dropped_cond = 0
    for e in agg.eliminations:  # topological order
        resolved, amplification = _step(walk, agg.incidence, e)
        if max_condition is not None and amplification > max_condition:
            dropped_cond += 1
            continue
        if max_def_nodes is not None and expression_dag_size(resolved) > max_def_nodes:
            dropped_nodes += 1
            continue
        walk.resolved[e.var] = resolved
        walk.ampl[e.var] = amplification
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
