"""Public entry points for variable aggregation.

Two functions cover normal use:

- :func:`aggregate` — build a reduced-space model from a model using a named
  strategy, returning the reduced model plus a recovery map.
- :func:`solve` — aggregate, solve the reduced model, and recover the full-space
  solution in one call.

Recursion follows the paper's Section 3.3 procedure.  The structure-preserving
two-variable strategies (``ecd2``/``ld2``/``d2``) first aggregate fixed
variables (``ld1``) to a fixed point, then aggregate two-variable constraints to
a fixed point; ``ld1`` aggregates fixed variables to a fixed point; the
approximate-maximum strategies (``gr``/``lm``) are applied directly.  Passing
``recursive=False`` performs exactly one pass of the named strategy.

Recovery maps compose across passes purely by variable name: each defining
expression is a function of names, so evaluating them in reverse-chronological
order (later-eliminated variables first) reconstructs every eliminated variable
from the reduced-space solution.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

from discopt.aggregation.guards import apply_elimination_guards
from discopt.aggregation.incidence import build_incidence
from discopt.aggregation.recovery import (
    RecoveryMap,
    evaluate_expression,
    feasibility_violations,
)
from discopt.aggregation.scalarize import ScalarizationMap, scalarize_model
from discopt.aggregation.strategies import drop_linearity_breaking, resolve_strategy
from discopt.aggregation.transform import apply_aggregation

_STRUCTURE_PRESERVING_2VAR = ("ecd2", "ld2", "d2")
_DEGREE_K_RE = re.compile(r"^d\d+$")


def _pipeline_for(method: str) -> list[str]:
    """The ordered list of strategies applied for ``method`` (paper Section 3.3)."""
    if method == "ld1":
        return ["ld1"]
    # Structure-preserving families (2-var and the degree-k generalisation) run
    # fixed-variable elimination first, then the degree-bounded strategy.
    if method in _STRUCTURE_PRESERVING_2VAR or _DEGREE_K_RE.match(method):
        return ["ld1", method]
    return [method]  # gr, lm — approximate-maximum, applied directly


@dataclass
class AggregationResult:
    """Result of :func:`aggregate`.

    Attributes:
        reduced_model: the reduced-space model (equal to the original model when
            nothing could be eliminated).
        recovery: reconstructs eliminated variables from a reduced solution.
        eliminated: eliminated variable names across all passes (in order).
        method: the requested method.
        n_passes: number of elimination passes performed.
        original_model: the input model.
        scalarization: if array variables were scalarized first, the map used to
            reassemble them during recovery; ``None`` otherwise.
        base_model: the (scalarized) model the eliminations act on — the space in
            which dual recovery is performed.  Equals the scalarized model when
            scalarization ran, else ``original_model``.
        origin_maps: per-pass ``constraint_origin`` lists, in pass order; composing
            them maps a reduced-model constraint index back to a ``base_model``
            constraint index (dual recovery, D1).
        elim_records: ``(var, local_cid, pass_index)`` for every elimination, in
            order — ``local_cid`` indexes the model of pass ``pass_index``.
        infeasible_eliminations: ``(var, value, lb, ub)`` for every fixed-variable
            elimination whose constant value falls outside the eliminated
            variable's box, across all passes.  Nonempty ⇒ the original model is
            infeasible (:func:`solve` reports ``infeasible``).
    """

    reduced_model: Any
    recovery: RecoveryMap
    eliminated: list[str]
    method: str
    n_passes: int
    original_model: Any
    stats: dict = field(default_factory=dict)
    scalarization: ScalarizationMap | None = None
    base_model: Any = None
    origin_maps: list[list] = field(default_factory=list)
    elim_records: list[tuple[str, int, int]] = field(default_factory=list)
    infeasible_eliminations: list[tuple[str, float, float, float]] = field(default_factory=list)

    @property
    def n_eliminated(self) -> int:
        return len(self.eliminated)

    def recover_full(self, kept_values: dict[str, Any]) -> dict[str, Any]:
        """Kept-variable values merged with the recovered eliminated values.

        When array variables were scalarized, the recovered scalar elements are
        folded back into arrays so callers see the original variable shapes.
        """
        recovered = self.recovery.recover_full(kept_values)
        if self.scalarization is not None:
            recovered = self.scalarization.reassemble(recovered)
        return recovered


def aggregate(
    model: Any,
    method: str = "d2",
    recursive: bool = True,
    pivot_tol: float = 1e-6,
    max_fill: float | None = None,
    prune_bounds: bool = True,
    scalarize: bool = True,
    preserve_target_linearity: bool = False,
    tearing: str = "greedy",
    decomposable: bool = False,
    max_def_nodes: int | None = None,
    max_condition: float | None = None,
) -> AggregationResult:
    """Aggregate variables of ``model`` using ``method``.

    Args:
        model: the model to reduce.
        method: one of ``ld1``, ``ecd2``, ``ld2``, ``d2`` (default), ``gr``,
            ``lm``.
        recursive: when True (default) apply the strategy pipeline to a fixed
            point (paper Section 3.3); when False perform a single pass of the
            named strategy only.
        pivot_tol: relative pivot tolerance guarding against ill-conditioned
            substitutions (see :func:`~discopt.aggregation.incidence.linear_vars`).
            Default ``1e-6``; pass ``0.0`` for exact paper behaviour.
        max_fill: optional per-elimination fill cap that skips eliminations
            estimated to introduce more than ``max_fill`` new Jacobian
            incidences (see :meth:`IncidenceModel.fill_estimate`).  ``None``
            (default) disables the cap.
        prune_bounds: when True (default) omit box->inequality constraints on an
            eliminated variable that interval propagation proves redundant.
        scalarize: when True (default) rewrite fully-indexed array variables into
            scalars first, so aggregation reaches discretized models (a no-op on
            models with no array variables).
        preserve_target_linearity: when True, drop any elimination whose nonlinear
            defining expression would turn a currently-linear constraint nonlinear
            (Amarger et al.). Default False.
        tearing: how cyclic blocks are torn in ``gr``/``lm`` — ``"greedy"``
            (default) or ``"exact"`` (maximum acyclic subset, eliminates at least
            as many per block).
        decomposable: when True, local-search the ``lm`` maximum matching for a
            more decomposable block structure (higher Theorem-2 lower bound)
            before tearing. Default False.
        max_def_nodes: optional cap on the *resolved* defining-expression node
            count of an elimination; over-cap eliminations are dropped, bounding
            the reduced-model expression-graph / Hessian blow-up (D2). ``None``
            (default) disables the cap.
        max_condition: optional cap on the cumulative pivot amplification along the
            composed elimination chain; over-cap eliminations are dropped (D2).
            ``None`` (default) disables the cap.

    Returns:
        An :class:`AggregationResult`.  If no variable can be eliminated, the
        reduced model is the original model and the recovery map is empty.
    """
    key = method.lower()
    resolve_strategy(key)  # validate (raises ValueError for an unknown method)

    user_model = model
    scal_map: ScalarizationMap | None = None
    if scalarize:
        scalar_model, smap = scalarize_model(model)
        if smap.scalarized:  # only take the rewritten model if something changed
            model = scalar_model
            scal_map = smap

    pipeline = _pipeline_for(key) if recursive else [key]
    current = model
    all_defining: dict[str, Any] = {}
    order_stack: list[list[str]] = []
    eliminated: list[str] = []
    origin_maps: list[list] = []
    elim_records: list[tuple[str, int, int]] = []
    infeasible_eliminations: list[tuple[str, float, float, float]] = []

    for stage in pipeline:
        while True:
            inc = build_incidence(
                current,
                pivot_tol=pivot_tol,
                max_fill=max_fill,
                tearing=tearing,
                decomposable=decomposable,
            )
            agg = resolve_strategy(stage)(inc)
            if preserve_target_linearity:
                agg = drop_linearity_breaking(agg)
            if max_def_nodes is not None or max_condition is not None:
                agg = apply_elimination_guards(agg, max_def_nodes, max_condition)
            if agg.n == 0:
                break
            reduced = apply_aggregation(current, agg, prune_bounds=prune_bounds)
            # Record per-pass state for dual recovery (D1): this pass's
            # constraint-origin map, and each elimination tagged with the pass
            # index whose model-space its cid lives in.
            pass_index = len(origin_maps)
            for e in agg.eliminations:
                elim_records.append((e.var, e.cid, pass_index))
            origin_maps.append(reduced.constraint_origin)
            all_defining.update(reduced.recovery.defining)
            order_stack.append(reduced.recovery.order)
            eliminated.extend(reduced.eliminated)
            infeasible_eliminations.extend(reduced.infeasible_eliminations)
            current = reduced.model
            if not recursive:
                break

    if infeasible_eliminations:
        names = ", ".join(
            f"{v}={val:.3g}∉[{lb:.3g},{ub:.3g}]" for v, val, lb, ub in infeasible_eliminations
        )
        warnings.warn(
            f"aggregation found the model infeasible: fixed variable(s) {names}",
            UserWarning,
            stacklevel=2,
        )

    # Later-eliminated variables must be recovered first: reverse pass order.
    combined_order = [name for order in reversed(order_stack) for name in order]
    recovery = RecoveryMap(defining=all_defining, order=combined_order)
    return AggregationResult(
        reduced_model=current,
        recovery=recovery,
        eliminated=eliminated,
        method=key,
        n_passes=len(order_stack),
        original_model=user_model,
        stats={"n_eliminated": len(eliminated), "n_passes": len(order_stack)},
        scalarization=scal_map,
        base_model=model,
        origin_maps=origin_maps,
        elim_records=elim_records,
        infeasible_eliminations=infeasible_eliminations,
    )


@dataclass
class AggregatedSolveResult:
    """Result of :func:`solve`.

    Attributes:
        status: solver status string of the reduced-model solve.
        objective: objective value (identical objective to the original model).
        x: full-space solution ``{name: value}`` (kept variables from the solve,
            plus recovered eliminated variables).
        reduced_result: the raw :class:`SolveResult` from the reduced model.
        aggregation: the :class:`AggregationResult` used.
    """

    status: Any
    objective: Any
    x: dict[str, Any]
    reduced_result: Any
    aggregation: AggregationResult
    duals: Any = None  # DualRecovery when solve(..., recover_duals=True), else None


def solve(
    model: Any,
    method: str = "d2",
    recursive: bool = True,
    pivot_tol: float = 1e-6,
    max_fill: float | None = None,
    prune_bounds: bool = True,
    scalarize: bool = True,
    preserve_target_linearity: bool = False,
    tearing: str = "greedy",
    decomposable: bool = False,
    max_def_nodes: int | None = None,
    max_condition: float | None = None,
    recover_duals: bool = False,
    **solve_kwargs: Any,
) -> AggregatedSolveResult:
    """Aggregate ``model``, solve the reduced model, and recover the full solution.

    ``pivot_tol``, ``max_fill``, ``prune_bounds``, ``scalarize``,
    ``preserve_target_linearity``, ``tearing``, ``decomposable``,
    ``max_def_nodes`` and ``max_condition`` are forwarded to :func:`aggregate`;
    extra keyword arguments are forwarded to the reduced model's ``solve``.

    When ``recover_duals`` is True, the result's ``duals`` field carries a
    :class:`~discopt.aggregation.duals.DualRecovery` of the full-space Lagrange
    multipliers (recovered across all aggregation passes); it reports
    ``available=False`` for the cases outside recovery's scope (maximise
    objective, active eliminated-variable bounds, or an unsupported derivative).
    """
    result = aggregate(
        model,
        method=method,
        recursive=recursive,
        pivot_tol=pivot_tol,
        max_fill=max_fill,
        prune_bounds=prune_bounds,
        scalarize=scalarize,
        preserve_target_linearity=preserve_target_linearity,
        tearing=tearing,
        decomposable=decomposable,
        max_def_nodes=max_def_nodes,
        max_condition=max_condition,
    )
    # A fixed-variable elimination proved the model infeasible (C1): report it
    # instead of solving a reduced model that dropped the violated bound.
    if result.infeasible_eliminations:
        return AggregatedSolveResult(
            status="infeasible",
            objective=None,
            x=result.recover_full({}) if not result.reduced_model._variables else {},
            reduced_result=None,
            aggregation=result,
        )

    # Fully-determined model: aggregation eliminated every variable, so there is
    # nothing left to solve. Recover the (unique) point, then verify it satisfies
    # the residual (variable-free) constraints and the original variables' bounds
    # before claiming optimality — an over-determined system can be infeasible.
    if not result.reduced_model._variables:
        full_x = result.recover_full({})
        violations = feasibility_violations(result.reduced_model, {})
        violations += feasibility_violations(result.original_model, full_x)
        if violations:
            return AggregatedSolveResult(
                status="infeasible",
                objective=None,
                x=full_x,
                reduced_result=None,
                aggregation=result,
            )
        obj = result.original_model._objective
        objective = float(evaluate_expression(obj.expression, full_x)) if obj is not None else None
        return AggregatedSolveResult(
            status="optimal",
            objective=objective,
            x=full_x,
            reduced_result=None,
            aggregation=result,
        )

    reduced_result = result.reduced_model.solve(**solve_kwargs)

    x = reduced_result.x
    full_x = result.recover_full(x) if isinstance(x, dict) else x

    duals = None
    if recover_duals:
        from discopt.aggregation.duals import recover_duals_from_result

        duals = recover_duals_from_result(result, reduced_result)

    return AggregatedSolveResult(
        status=reduced_result.status,
        objective=reduced_result.objective,
        x=full_x,
        reduced_result=reduced_result,
        aggregation=result,
        duals=duals,
    )


__all__ = [
    "aggregate",
    "AggregationResult",
    "solve",
    "AggregatedSolveResult",
]
