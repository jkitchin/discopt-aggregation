"""Recover full-space Lagrange multipliers from a reduced-space solution (G3b).

Solving the reduced model yields multipliers for the *surviving* constraints
(and bounds).  The eliminated defining equalities were removed, so their
multipliers are missing.  This module recovers them by the stationarity of the
full Lagrangian in each eliminated variable.

discopt normalises every constraint to ``body (sense) 0`` and reports a
multiplier ``μ_c`` such that the stationarity condition is uniformly

    ∇f(z*) + Σ_c μ_c ∇(body_c)(z*) + (bound terms) = 0

for equalities and inequalities alike (verified empirically).  A **maximise**
objective is reported by discopt in the ``minimise(-f)`` convention — a direct
solve of ``m.maximize(-(x**2) - y**2)`` s.t. ``y - (x+1) == 0`` yields multipliers
that satisfy ``∇(-f) + Σ μ ∇body = 0`` (not ``∇f + …``; verified empirically).
Recovery therefore multiplies the objective-gradient term by ``obj_sign``
(``-1`` for maximise, ``+1`` for minimise) so recovered duals stay in discopt's
own convention and remain directly comparable to a direct solve.  For an
eliminated variable ``v_i`` with defining constraint ``cid_i``
(``∂body_{cid_i}/∂v_i = coeff_i``), that row gives

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
model.  Scope for both: minimise **or** maximise objective (the sign is
normalised, see below) and no *active* variable bounds on eliminated variables
(interior solution) — validated by the full-space KKT stationarity residual.  If
any required derivative is unsupported, recovery reports ``available=False``
rather than a wrong result.
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
            derivative, missing reduced duals, an active eliminated-variable
            bound, a singular defining system, or a failed KKT self-check).
            A maximise objective is **in** scope (its sign is normalised).
        constraint_duals: ``{original_cid: multiplier}`` for every original
            constraint — surviving ones mapped from the reduced solve, defining
            equalities recovered by solving the defining-multiplier system.
        full_x: the recovered full-space primal point used.
        reason: when ``available`` is False, a short machine-readable tag for why
            (e.g. ``"active_eliminated_bound:y"``,
            ``"unsupported_derivative"``, ``"singular_defining_system"``,
            ``"residual_check_failed"``, ``"missing_reduced_duals"``).
    """

    available: bool
    constraint_duals: dict[int, float] = field(default_factory=dict)
    full_x: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


def _grad(expr: Any, var: str, values: dict[str, Any], cache: dict) -> float | None:
    """``d(expr)/d(var)`` evaluated at ``values``; ``None`` if not differentiable."""
    key = (id(expr), var)
    if key not in cache:
        cache[key] = differentiate_expression(expr, var)
    d = cache[key]
    if d is None:
        return None
    return float(np.asarray(evaluate_expression(d, values)).reshape(()))


_INF_BOUND = 9.9e19  # below discopt's +-9.999e19 sentinel


def _objective_sign(model: Any) -> float:
    """``+1`` for a minimise objective, ``-1`` for maximise.

    discopt reports a maximise objective's multipliers in the ``minimise(-f)``
    convention (verified empirically — see the module docstring), so the
    objective-gradient term in every stationarity row is scaled by this sign.
    """
    obj = getattr(model, "_objective", None)
    if obj is not None and obj.sense != ObjectiveSense.MINIMIZE:
        return -1.0
    return 1.0


def _active_eliminated_bound(
    model: Any, full_x: dict[str, Any], var: str, tol: float = 1e-8
) -> bool:
    """Whether eliminated ``var`` sits on a finite bound at the recovered point.

    An active eliminated-variable bound contributes a γ-term to stationarity that
    this recovery does not model, so an active bound puts the case out of scope
    (the caller then reports ``available=False``).
    """
    v = next((vv for vv in model._variables if vv.name == var), None)
    if v is None or var not in full_x:
        return False
    val = float(np.asarray(full_x[var]).reshape(()))
    lb = float(np.min(v.lb))
    ub = float(np.max(v.ub))
    scale = tol * max(1.0, abs(val))
    if lb > -_INF_BOUND and abs(val - lb) <= scale:
        return True
    if ub < _INF_BOUND and abs(ub - val) <= scale:
        return True
    return False


def _solve_defining_duals(
    model: Any,
    full_x: dict[str, Any],
    elim_pairs: list[tuple[str, int]],
    surviving: dict[int, float],
    cache: dict,
) -> dict[int, float] | None:
    """Solve the defining-equality multipliers as a linear system.

    Stationarity in each eliminated variable ``v_i`` reads
    ``∂f/∂v_i + Σ_surviving μ_c ∂body_c/∂v_i + Σ_j μ_{cid_j} ∂body_{cid_j}/∂v_i = 0``.
    With one defining equality per elimination this is a square system
    ``A μ = b``, ``A[i, j] = ∂body_{cid_j}/∂v_i`` and
    ``b[i] = -(∂f/∂v_i + Σ_surviving μ_c ∂body_c/∂v_i)``, solved directly.

    A single strictly-lower-triangular pass makes ``A`` triangular (the
    back-substitution special case), but across recursive passes the combined
    defining system can be *coupled* — a pass-0 defining body may reference a
    variable eliminated in pass 1 — so no back-substitution order exists and the
    linear solve is required.  Returns ``None`` for an unsupported derivative or a
    singular / ill-conditioned system.
    """
    n = len(elim_pairs)
    if n == 0:
        return {}
    cids = [cid for _v, cid in elim_pairs]
    obj = model._objective
    obj_expr = obj.expression if obj is not None else None
    obj_sign = _objective_sign(model)
    amat = np.zeros((n, n))
    bvec = np.zeros(n)
    for i, (vi, _cid_i) in enumerate(elim_pairs):
        rhs = 0.0
        if obj_expr is not None:
            g = _grad(obj_expr, vi, full_x, cache)
            if g is None:
                return None
            rhs += obj_sign * g
        for cid, mu in surviving.items():
            if mu == 0.0:
                continue
            g = _grad(model._constraints[cid].body, vi, full_x, cache)
            if g is None:
                return None
            rhs += mu * g
        bvec[i] = -rhs
        for j, cid_j in enumerate(cids):
            g = _grad(model._constraints[cid_j].body, vi, full_x, cache)
            if g is None:
                return None
            amat[i, j] = g
    cond = np.linalg.cond(amat)
    if not np.isfinite(cond) or cond > 1e12:
        return None
    mu_vec = np.linalg.solve(amat, bvec)
    return {cids[j]: float(mu_vec[j]) for j in range(n)}


def _finalize_duals(
    model: Any,
    full_x: dict[str, Any],
    elim_pairs: list[tuple[str, int]],
    surviving: dict[int, float],
) -> DualRecovery:
    """Shared tail: active-bound guard, linear solve, KKT self-check."""
    for var, _cid in elim_pairs:
        if _active_eliminated_bound(model, full_x, var):
            return DualRecovery(
                available=False, reason=f"active_eliminated_bound:{var}", full_x=full_x
            )
    cache: dict = {}
    defining = _solve_defining_duals(model, full_x, elim_pairs, surviving, cache)
    if defining is None:
        return DualRecovery(
            available=False, reason="unsupported_derivative_or_singular", full_x=full_x
        )
    duals = {**surviving, **defining}
    # Self-check: a successful recovery must satisfy full-space KKT stationarity.
    # This turns any residual algebra gap into an honest available=False rather
    # than a wrong-but-confident answer.
    resid = kkt_stationarity_residual(model, full_x, duals)
    if not np.isfinite(resid) or resid > 1e-6:
        return DualRecovery(available=False, reason="residual_check_failed", full_x=full_x)
    return DualRecovery(available=True, constraint_duals=duals, full_x=full_x)


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
        return DualRecovery(available=False, reason="no_aggregation_set")
    model = agg.incidence.model

    rx = getattr(reduced_result, "x", None)
    rduals = getattr(reduced_result, "constraint_duals", None)
    if not isinstance(rx, dict) or not isinstance(rduals, dict):
        return DualRecovery(available=False, reason="missing_reduced_duals")

    full_x = reduced.recovery.recover_full(rx)

    # Map each reduced constraint's dual back to its original constraint.  The
    # solve result keys constraint duals positionally ("c0", "c1", ...) in model
    # constraint order (constraint.name is usually None), and constraint_origin
    # is aligned with reduced.model._constraints, so index i -> "c{i}".
    surviving: dict[int, float] = {}
    for i, origin in enumerate(reduced.constraint_origin):
        if not isinstance(origin, int):
            continue  # bound->inequality; handled by the active-bound guard
        key: Any = f"c{i}"
        if key not in rduals:  # fall back to the constraint's own name
            key = getattr(reduced.model._constraints[i], "name", None)
        surviving[origin] = float(np.asarray(rduals.get(key, 0.0)).reshape(()))

    elim_pairs = [(e.var, e.cid) for e in agg.eliminations]
    return _finalize_duals(model, full_x, elim_pairs, surviving)


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

    Scope: minimise **or** maximise objective (sign normalised) and interior
    solution (no active eliminated-variable bounds); returns ``available=False``
    otherwise, or if any required derivative is unsupported.  Validated by
    :func:`kkt_stationarity_residual` on the base model.
    """
    base = getattr(result, "base_model", None)
    if base is None:
        return DualRecovery(available=False, reason="no_base_model")

    rx = getattr(reduced_result, "x", None)
    rduals = getattr(reduced_result, "constraint_duals", None)
    if not isinstance(rx, dict) or not isinstance(rduals, dict):
        return DualRecovery(available=False, reason="missing_reduced_duals")

    # Full-space point in the base (scalar) space — not reassembled into arrays.
    full_x = result.recovery.recover_full(rx)

    origin_maps = result.origin_maps
    n_passes = len(origin_maps)

    # Surviving constraints: final reduced index j -> base constraint id.
    surviving: dict[int, float] = {}
    for j in range(len(result.reduced_model._constraints)):
        base_cid = _resolve_to_base(j, n_passes, origin_maps)
        if base_cid is None:
            continue  # bound->inequality; handled by the active-bound guard
        key: Any = f"c{j}"
        if key not in rduals:
            key = getattr(result.reduced_model._constraints[j], "name", None)
        surviving[base_cid] = float(np.asarray(rduals.get(key, 0.0)).reshape(()))

    # Resolve every elimination's defining equality to a base-model constraint id.
    # Across recursive passes these defining bodies can be mutually coupled, so
    # the multipliers are solved as one linear system, not back-substituted.
    elim_pairs: list[tuple[str, int]] = []
    for var, local_cid, pass_index in result.elim_records:
        base_cid = _resolve_to_base(local_cid, pass_index, origin_maps)
        if base_cid is None:
            return DualRecovery(available=False, reason="defining_eq_pruned")
        elim_pairs.append((var, base_cid))

    return _finalize_duals(base, full_x, elim_pairs, surviving)


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
    obj_sign = _objective_sign(model)
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
            res += obj_sign * g
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
