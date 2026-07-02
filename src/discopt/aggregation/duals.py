"""Recover full-space Lagrange multipliers from a reduced-space solution (G3b).

Solving the reduced model yields multipliers for the *surviving* constraints
(and bounds).  The eliminated defining equalities were removed, so their
multipliers are missing.  This module recovers them by the stationarity of the
full Lagrangian in each eliminated variable.

discopt normalises every constraint to ``body (sense) 0`` and reports a
multiplier ``μ_c`` such that the stationarity condition is uniformly

    ∇f(z*) + Σ_c μ_c ∇(body_c)(z*) + (bound terms) = 0

for equalities and inequalities alike (verified empirically).  For an eliminated
variable ``v_i`` with defining constraint ``cid_i`` (``∂body_{cid_i}/∂v_i =
coeff_i``), that row gives

    μ_{cid_i} = -( ∂f/∂v_i
                  + Σ_{surviving c} μ_c ∂body_c/∂v_i
                  + Σ_{later-eliminated j} μ_{cid_j} ∂body_{cid_j}/∂v_i ) / coeff_i,

which is solved by back-substitution in **reverse** elimination order (strict
lower-triangularity of the defining system makes this upper-triangular, so every
term on the right is already known).  Gradients come from
:func:`~discopt.aggregation.derivatives.differentiate_expression` evaluated at
the recovered full-space point.

Two entry points: :func:`recover_duals` for a single ``apply_aggregation``
result, and :func:`recover_duals_from_result` for a full (recursive, multi-pass)
:class:`~discopt.aggregation.api.AggregationResult` — the latter composes each
pass's constraint-origin map so the same back-substitution runs in the base
model.  Scope for both: minimise objective and no *active* variable bounds on
eliminated variables (interior solution) — validated by the full-space KKT
stationarity residual.  If any required derivative is unsupported, recovery
reports ``available=False`` rather than a wrong result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from discopt.modeling.core import Constraint, ObjectiveSense

from discopt.aggregation.derivatives import differentiate_expression
from discopt.aggregation.recovery import evaluate_expression


@dataclass
class DualRecovery:
    """Recovered full-space multipliers.

    Attributes:
        available: False if recovery could not be performed (e.g. an unsupported
            derivative, a maximise objective, or missing reduced duals).
        constraint_duals: ``{original_cid: multiplier}`` for every original
            constraint — surviving ones mapped from the reduced solve, defining
            equalities recovered by back-substitution.
        full_x: the recovered full-space primal point used.
    """

    available: bool
    constraint_duals: dict[int, float] = field(default_factory=dict)
    full_x: dict[str, Any] = field(default_factory=dict)


def _grad(expr: Any, var: str, values: dict[str, Any], cache: dict) -> float | None:
    """``d(expr)/d(var)`` evaluated at ``values``; ``None`` if not differentiable."""
    key = (id(expr), var)
    if key not in cache:
        cache[key] = differentiate_expression(expr, var)
    d = cache[key]
    if d is None:
        return None
    return float(np.asarray(evaluate_expression(d, values)).reshape(()))


def recover_duals(reduced: Any, reduced_result: Any) -> DualRecovery:
    """Recover full-space constraint multipliers for a :class:`ReducedModel`.

    Args:
        reduced: the :class:`~discopt.aggregation.transform.ReducedModel` from a
            single ``apply_aggregation``.
        reduced_result: the solve result of ``reduced.model`` (must expose
            ``x`` and ``constraint_duals``).
    """
    agg = reduced.aggregation_set
    if agg is None:
        return DualRecovery(available=False)
    model = agg.incidence.model
    obj = model._objective
    if obj is not None and obj.sense != ObjectiveSense.MINIMIZE:
        return DualRecovery(available=False)  # maximise sign handling out of scope

    rx = getattr(reduced_result, "x", None)
    rduals = getattr(reduced_result, "constraint_duals", None)
    if not isinstance(rx, dict) or not isinstance(rduals, dict):
        return DualRecovery(available=False)

    full_x = reduced.recovery.recover_full(rx)

    # Map each reduced constraint's dual back to its original constraint.  The
    # solve result keys constraint duals positionally ("c0", "c1", ...) in model
    # constraint order (constraint.name is usually None), and constraint_origin
    # is aligned with reduced.model._constraints, so index i -> "c{i}".
    surviving: dict[int, float] = {}
    for i, origin in enumerate(reduced.constraint_origin):
        if not isinstance(origin, int):
            continue  # bound->inequality; not needed without active elim. bounds
        key: Any = f"c{i}"
        if key not in rduals:  # fall back to the constraint's own name
            key = getattr(reduced.model._constraints[i], "name", None)
        surviving[origin] = float(np.asarray(rduals.get(key, 0.0)).reshape(()))

    cache: dict = {}
    obj_expr = obj.expression if obj is not None else None

    # Back-substitute the defining-equality multipliers in reverse order.
    defining: dict[int, float] = {}
    for e in reversed(agg.eliminations):
        total = 0.0
        if obj_expr is not None:
            g = _grad(obj_expr, e.var, full_x, cache)
            if g is None:
                return DualRecovery(available=False)
            total += g
        for cid, mu in surviving.items():
            if mu == 0.0:
                continue
            g = _grad(model._constraints[cid].body, e.var, full_x, cache)
            if g is None:
                return DualRecovery(available=False)
            total += mu * g
        for cid, mu in defining.items():  # later-eliminated (already solved)
            g = _grad(model._constraints[cid].body, e.var, full_x, cache)
            if g is None:
                return DualRecovery(available=False)
            total += mu * g
        defining[e.cid] = -total / e.coeff

    return DualRecovery(available=True, constraint_duals={**surviving, **defining}, full_x=full_x)


def _resolve_to_base(cid: int, level: int, origin_maps: list) -> int | None:
    """Map a constraint index in the pass-``level`` model down to the base model.

    ``origin_maps[k]`` maps a pass-``k+1`` constraint index to its pass-``k``
    origin (an int) or a bound-origin tuple.  Composing maps ``level-1 .. 0``
    descends to the base model.  Returns ``None`` if a bound-origin (a
    box->inequality, not an original constraint) is encountered.
    """
    for m in range(level - 1, -1, -1):
        entry = origin_maps[m][cid]
        if not isinstance(entry, int):
            return None
        cid = entry
    return cid


def recover_duals_from_result(result: Any, reduced_result: Any) -> DualRecovery:
    """Recover full-space multipliers from an :class:`AggregationResult` (D1).

    Unlike :func:`recover_duals` (single ``apply_aggregation``), this handles the
    **recursive** default path: it composes each pass's ``constraint_origin`` map
    to work entirely in the base (scalarized) model, then back-substitutes the
    defining-equality multipliers in reverse elimination order across *all* passes
    (the combined defining system is strictly lower triangular, so this is a valid
    upper-triangular back-substitution).

    Scope: minimise objective and interior solution (no active eliminated-variable
    bounds); returns ``available=False`` otherwise, or if any required derivative
    is unsupported.  Validated by :func:`kkt_stationarity_residual` on the base
    model.
    """
    base = getattr(result, "base_model", None)
    if base is None:
        return DualRecovery(available=False)
    obj = base._objective
    if obj is not None and obj.sense != ObjectiveSense.MINIMIZE:
        return DualRecovery(available=False)  # maximise sign handling out of scope

    rx = getattr(reduced_result, "x", None)
    rduals = getattr(reduced_result, "constraint_duals", None)
    if not isinstance(rx, dict) or not isinstance(rduals, dict):
        return DualRecovery(available=False)

    # Full-space point in the base (scalar) space — not reassembled into arrays.
    full_x = result.recovery.recover_full(rx)

    origin_maps = result.origin_maps
    n_passes = len(origin_maps)

    # Surviving constraints: final reduced index j -> base constraint id.
    surviving: dict[int, float] = {}
    for j in range(len(result.reduced_model._constraints)):
        base_cid = _resolve_to_base(j, n_passes, origin_maps)
        if base_cid is None:
            continue  # bound->inequality; not needed without active elim. bounds
        key: Any = f"c{j}"
        if key not in rduals:
            key = getattr(result.reduced_model._constraints[j], "name", None)
        surviving[base_cid] = float(np.asarray(rduals.get(key, 0.0)).reshape(()))

    cache: dict = {}
    obj_expr = obj.expression if obj is not None else None

    # Back-substitute defining multipliers in reverse elimination order, on base bodies.
    defining: dict[int, float] = {}
    for var, local_cid, pass_index in reversed(result.elim_records):
        base_cid = _resolve_to_base(local_cid, pass_index, origin_maps)
        if base_cid is None:
            return DualRecovery(available=False)
        coeff = _grad(base._constraints[base_cid].body, var, full_x, cache)
        if coeff is None or coeff == 0.0:
            return DualRecovery(available=False)
        total = 0.0
        if obj_expr is not None:
            g = _grad(obj_expr, var, full_x, cache)
            if g is None:
                return DualRecovery(available=False)
            total += g
        for cid, mu in {**surviving, **defining}.items():
            if mu == 0.0:
                continue
            g = _grad(base._constraints[cid].body, var, full_x, cache)
            if g is None:
                return DualRecovery(available=False)
            total += mu * g
        defining[base_cid] = -total / coeff

    return DualRecovery(available=True, constraint_duals={**surviving, **defining}, full_x=full_x)


def kkt_stationarity_residual(
    model: Any, full_x: dict[str, Any], constraint_duals: dict[int, float]
) -> float:
    """Max abs. stationarity residual ``∂f/∂w + Σ_c μ_c ∂body_c/∂w`` over variables.

    Assumes no active variable bounds (interior solution); a small residual then
    certifies the recovered multipliers satisfy the full-space KKT stationarity.
    Returns ``inf`` if a required derivative is unsupported.
    """
    obj = model._objective
    obj_expr = obj.expression if obj is not None else None
    cache: dict = {}
    worst = 0.0
    for v in model._variables:
        if int(getattr(v, "size", 1)) != 1:
            continue  # scalar stationarity only
        name = v.name
        res = 0.0
        if obj_expr is not None:
            g = _grad(obj_expr, name, full_x, cache)
            if g is None:
                return float("inf")
            res += g
        for cid, mu in constraint_duals.items():
            c = model._constraints[cid]
            if not isinstance(c, Constraint):
                continue
            g = _grad(c.body, name, full_x, cache)
            if g is None:
                return float("inf")
            res += mu * g
        worst = max(worst, abs(res))
    return worst


__all__ = [
    "DualRecovery",
    "recover_duals",
    "recover_duals_from_result",
    "kkt_stationarity_residual",
]
