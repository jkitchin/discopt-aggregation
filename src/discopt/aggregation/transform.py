"""Apply an aggregation set: build the reduced-space model and recovery map.

Given a :class:`~discopt.aggregation.strategies.AggregationSet`, this module

1. derives each eliminated variable's *defining expression* in terms of the kept
   variables (:func:`defining_expressions`), and
2. rebuilds a reduced :class:`discopt.modeling.core.Model`
   (:func:`apply_aggregation`) in which the eliminated variables and their
   defining equalities are removed, the defining expressions are substituted
   everywhere they appeared, and each eliminated *bounded* variable's box is
   re-expressed as up to two inequality constraints on its defining expression
   (paper Section 2.4).

The reduced model is built through the public modeling API with fresh variables,
so it is a self-contained model ready to solve; the returned
:class:`~discopt.aggregation.recovery.RecoveryMap` reconstructs the eliminated
variables from the reduced solution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, cast

import discopt.modeling as dm
import numpy as np
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    ObjectiveSense,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

from discopt.aggregation.exprwalk import child_exprs, collect_variable_names
from discopt.aggregation.recovery import RecoveryMap, interval_evaluate
from discopt.aggregation.strategies import AggregationSet

try:  # pragma: no cover - import shim
    from discopt.modeling.core import CustomCall as _CustomCall
except ImportError:  # pragma: no cover
    _CustomCall = None  # type: ignore[assignment,misc]
try:  # pragma: no cover - import shim
    from discopt.modeling.core import MatMulExpression as _MatMul
except ImportError:  # pragma: no cover
    _MatMul = None  # type: ignore[assignment,misc]

_INF = 9.9e19  # below discopt's +-9.999e19 sentinel; real bounds < 9.9e19 are kept


def substitute_expression(expr: Any, var_map: dict[str, Expression]) -> Any:
    """Return a copy of ``expr`` with variables replaced per ``var_map`` (by name).

    Unlike a partial walker, this recurses through *every* expression node kind
    used by the modeling layer — including :class:`SumExpression` and
    :class:`SumOverExpression` reductions — so a variable buried inside a sum is
    still substituted.  Nodes untouched by the map are returned unchanged
    (identity-preserving, so unaffected subtrees are shared, not copied).
    """
    if isinstance(expr, Variable):
        return var_map.get(expr.name, expr)
    if isinstance(expr, Constant):
        return expr
    if isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable) and expr.base.name in var_map:
            return IndexExpression(var_map[expr.base.name], expr.index)
        new_base = substitute_expression(expr.base, var_map)
        return expr if new_base is expr.base else IndexExpression(new_base, expr.index)
    if isinstance(expr, UnaryOp):
        new_operand = substitute_expression(expr.operand, var_map)
        return expr if new_operand is expr.operand else UnaryOp(expr.op, new_operand)
    if isinstance(expr, BinaryOp):
        new_left = substitute_expression(expr.left, var_map)
        new_right = substitute_expression(expr.right, var_map)
        if new_left is expr.left and new_right is expr.right:
            return expr
        return BinaryOp(expr.op, new_left, new_right)
    if isinstance(expr, FunctionCall):
        new_args = tuple(substitute_expression(a, var_map) for a in expr.args)
        if all(n is o for n, o in zip(new_args, expr.args)):
            return expr
        return FunctionCall(expr.func_name, *new_args)
    if isinstance(expr, SumExpression):
        new_operand = substitute_expression(expr.operand, var_map)
        if new_operand is expr.operand:
            return expr
        return SumExpression(new_operand, getattr(expr, "axis", None))
    if isinstance(expr, SumOverExpression):
        new_terms = [substitute_expression(t, var_map) for t in expr.terms]
        if all(n is o for n, o in zip(new_terms, expr.terms)):
            return expr
        return SumOverExpression(new_terms)
    if _CustomCall is not None and isinstance(expr, _CustomCall):
        # Opaque callable, but its arguments are ordinary expressions: rebuilding
        # the call with substituted args preserves semantics (fn(sub(a), ...)).
        new_args = tuple(substitute_expression(a, var_map) for a in expr.args)
        if all(n is o for n, o in zip(new_args, expr.args)):
            return expr
        return _CustomCall(expr.fn, *new_args, name=getattr(expr, "name", None))
    if _MatMul is not None and isinstance(expr, _MatMul):
        new_left = substitute_expression(expr.left, var_map)
        new_right = substitute_expression(expr.right, var_map)
        if new_left is expr.left and new_right is expr.right:
            return expr
        return _MatMul(new_left, new_right)
    if isinstance(expr, (Constant, Parameter, int, float)):
        return expr
    # An unrecognised node might hide a variable slated for substitution; passing
    # it through unchanged would leave a dangling reference.  Refuse loudly.
    if child_exprs(expr) is None:
        raise NotImplementedError(
            f"substitute_expression cannot rewrite node of type "
            f"{type(expr).__name__!r}; refusing rather than risk a dangling "
            f"reference to a substituted variable"
        )
    return expr


def defining_expressions(agg: AggregationSet) -> dict[str, Expression]:
    """Defining expression of each eliminated variable, in kept-variable terms.

    For an elimination ``coeff*var + rest(others) == 0`` the defining expression
    is ``var = -rest/coeff``, where ``rest`` is the constraint body with ``var``
    set to zero.  Eliminations are processed in topological order so that a later
    definition referencing an earlier-eliminated variable is inlined, leaving
    every returned expression a function of the kept variables only.
    """
    inc = agg.incidence
    resolved: dict[str, Expression] = {}
    for e in agg.eliminations:
        body = inc.constraints[e.cid].body
        rest = substitute_expression(body, {e.var: Constant(0.0)})
        define: Expression = BinaryOp("*", rest, Constant(-1.0 / e.coeff))
        define = substitute_expression(define, resolved)  # inline earlier elims
        resolved[e.var] = define
    return resolved


def _references_variable(expr: Any) -> bool:
    """Whether ``expr`` references any variable (unrecognised nodes count as yes)."""
    names, analyzable = collect_variable_names(expr)
    return (not analyzable) or len(names) > 0


def _make_constraint(body: Expression, sense: str) -> Constraint:
    # Comparison operators on an Expression build a Constraint (overloaded);
    # mypy sees the generic __eq__/__le__ return type, hence the cast.
    if sense == "==":
        return cast(Constraint, body == 0)
    if sense == "<=":
        return cast(Constraint, body <= 0)
    if sense == ">=":
        return cast(Constraint, body >= 0)
    raise ValueError(f"unknown constraint sense {sense!r}")


@dataclass
class ReducedModel:
    """Result of applying an aggregation to a model.

    Attributes:
        model: the reduced-space :class:`discopt.modeling.core.Model`.
        recovery: reconstructs eliminated variables from a reduced solution.
        eliminated: eliminated variable names, in elimination order.
        method: the aggregation method used.
        n_eliminated: number of variables eliminated.
        n_constraints_removed: number of defining equalities removed.
        n_bound_inequalities: number of box->inequality constraints introduced.
        n_bound_pruned: number of box->inequality constraints proven redundant by
            interval propagation and therefore omitted.
        constraint_origin: per reduced constraint (aligned with
            ``model._constraints``), the source: an ``int`` original-constraint
            index for a carried-over constraint, or ``("bound_lo"|"bound_hi",
            var)`` for an eliminated variable's box->inequality.  Enables mapping
            reduced-solution duals back to the original constraints.
        aggregation_set: the :class:`AggregationSet` applied (carries the source
            model, defining constraints and coefficients) — used by dual recovery.
        infeasible_eliminations: ``(var, value, lb, ub)`` for each fixed-variable
            (constant-definition) elimination whose defining value provably lies
            outside the eliminated variable's box.  A nonempty list means the
            *original* model is infeasible; :func:`~discopt.aggregation.api.solve`
            turns this into an ``infeasible`` status.
    """

    model: Any
    recovery: RecoveryMap
    eliminated: list[str]
    method: str
    n_eliminated: int
    n_constraints_removed: int
    n_bound_inequalities: int
    n_bound_pruned: int = 0
    constraint_origin: list = field(default_factory=list)
    aggregation_set: Any = None
    infeasible_eliminations: list[tuple[str, float, float, float]] = field(default_factory=list)


def _kept_variable_boxes(reduced: Any) -> dict[str, tuple[float, float]]:
    """Interval box ``{name: (lo, hi)}`` for the reduced model's variables."""
    boxes: dict[str, tuple[float, float]] = {}
    for v in reduced._variables:
        lb = float(np.min(v.lb))
        ub = float(np.max(v.ub))
        boxes[v.name] = (
            -float("inf") if lb <= -_INF else lb,
            float("inf") if ub >= _INF else ub,
        )
    return boxes


def apply_aggregation(model: Any, agg: AggregationSet, prune_bounds: bool = True) -> ReducedModel:
    """Build the reduced-space model and recovery map for ``agg`` applied to ``model``.

    When ``prune_bounds`` is true (default), a box->inequality constraint on an
    eliminated variable's defining expression is omitted when interval
    propagation over the kept variables' boxes proves it always holds.  The
    interval enclosure is an outer bound of the expression's true range, so a
    pruned inequality is provably redundant — pruning never removes a constraint
    the feasible set depends on.
    """
    eliminated_order = agg.variables()
    eliminated = set(eliminated_order)
    orig_by_name = {v.name: v for v in model._variables}

    defining = defining_expressions(agg)

    reduced = dm.Model(f"{getattr(model, 'name', 'model')}_reduced")

    # Recreate every kept variable by name (scalar or array), same type/bounds.
    newvar_map: dict[str, Expression] = {}
    for v in model._variables:
        if v.name in eliminated:
            continue
        if v.var_type == VarType.CONTINUOUS:
            nv = reduced.continuous(v.name, shape=v.shape, lb=v.lb, ub=v.ub)
        elif v.var_type == VarType.INTEGER:
            nv = reduced.integer(v.name, shape=v.shape, lb=v.lb, ub=v.ub)
        elif v.var_type == VarType.BINARY:
            # binary() takes no lb/ub (a binary is [0,1] by construction), but a
            # binary can be *tightened* to a fixed 0 or 1 by mutating its bounds
            # (b.lb = [1.0]); those bounds are part of the model and must survive
            # into the reduced model, else a fixed binary is silently re-freed.
            nv = reduced.binary(v.name, shape=v.shape)
            nv.lb = np.asarray(v.lb, dtype=float).reshape(v.shape)
            nv.ub = np.asarray(v.ub, dtype=float).reshape(v.shape)
        else:  # pragma: no cover - defensive
            raise NotImplementedError(f"unsupported variable type {v.var_type!r}")
        newvar_map[v.name] = nv

    # Defining expressions rewritten onto the reduced model's variables, then a
    # single combined substitution map: kept -> new var, eliminated -> definition.
    defining_new = {
        name: substitute_expression(expr, newvar_map) for name, expr in defining.items()
    }
    combined: dict[str, Expression] = {**newvar_map, **defining_new}

    # Carry over every constraint except the defining equalities, substituted.
    # Track, per reduced constraint, which original constraint it came from.
    constraint_origin: list = []
    elim_cids = set(agg.matching())
    for cid, c in enumerate(model._constraints):
        if cid in elim_cids:
            continue
        if not isinstance(c, Constraint):  # pragma: no cover - defensive
            raise NotImplementedError(
                f"aggregation only supports algebraic Constraint objects; found {type(c).__name__}"
            )
        new_body = substitute_expression(c.body, combined)
        reduced.subject_to(_make_constraint(new_body, c.sense))
        constraint_origin.append(cid)

    # Box of each eliminated variable -> inequalities on its defining expression,
    # skipping any that interval propagation proves redundant.
    boxes = _kept_variable_boxes(reduced) if prune_bounds else {}
    n_bound_ineq = 0
    n_bound_pruned = 0
    infeasible_eliminations: list[tuple[str, float, float, float]] = []
    for name in eliminated_order:
        d = defining_new[name]
        if not _references_variable(d):
            # A fixed variable (constant definition): its box is a feasibility
            # check on a constant.  It adds no constraint on the kept variables,
            # but if the constant lies outside the eliminated variable's box the
            # *original* model is infeasible — record it so solve() can report it
            # rather than silently returning a feasible reduced problem.
            v0 = orig_by_name[name]
            lb0 = float(np.min(v0.lb))
            ub0 = float(np.max(v0.ub))
            val = float(np.asarray(interval_evaluate(d, {})[0]))
            m_lo = 1e-9 * max(1.0, abs(lb0))
            m_hi = 1e-9 * max(1.0, abs(ub0))
            out_of_box = (lb0 > -_INF and val < lb0 - m_lo) or (ub0 < _INF and val > ub0 + m_hi)
            # IA2: a *fixed* integer/binary variable is feasible only when its
            # value is integral.  An integer candidate reaches this branch only
            # when the flag admitted it (integers are not candidates otherwise),
            # so with the flag off nothing changes — a non-integral fix stays the
            # solver's infeasibility to find.
            non_integral = (
                v0.var_type in (VarType.INTEGER, VarType.BINARY) and abs(val - round(val)) > 1e-9
            )
            if out_of_box or non_integral:
                infeasible_eliminations.append((name, val, lb0, ub0))
            continue
        v = orig_by_name[name]
        lb = float(np.min(v.lb))
        ub = float(np.max(v.ub))
        # IA3: an eliminated *integer/binary* variable only takes integer values,
        # so its box can be tightened to the integer grid before emitting/pruning:
        # ``lb' = ceil(lb)``, ``ub' = floor(ub)`` (with a 1e-9 tolerance so a bound
        # already essentially integral is not nudged off by float noise).  This
        # makes pruning strictly more effective — a wider box is still sound — and
        # emits the tighter inequality when a bound survives.  Only the *compared*
        # box tightens; the defining expression's interval enclosure stays
        # real-valued and conservative (we do not prove the expression itself is
        # integer-valued, so rounding it would be unsound).
        if v.var_type in (VarType.INTEGER, VarType.BINARY):
            if lb > -_INF:
                lb = math.ceil(lb - 1e-9)
            if ub < _INF:
                ub = math.floor(ub + 1e-9)
        d_lo, d_hi = interval_evaluate(d, boxes) if prune_bounds else (-_INF, _INF)
        if lb > -_INF:
            margin = 1e-9 * max(1.0, abs(lb))
            if d_lo >= lb + margin:  # expression provably stays above lb
                n_bound_pruned += 1
            else:
                reduced.subject_to(d >= lb)
                constraint_origin.append(("bound_lo", name))
                n_bound_ineq += 1
        if ub < _INF:
            margin = 1e-9 * max(1.0, abs(ub))
            if d_hi <= ub - margin:  # expression provably stays below ub
                n_bound_pruned += 1
            else:
                reduced.subject_to(d <= ub)
                constraint_origin.append(("bound_hi", name))
                n_bound_ineq += 1

    # Objective, substituted.
    obj = model._objective
    if obj is not None:
        new_obj = substitute_expression(obj.expression, combined)
        if obj.sense == ObjectiveSense.MINIMIZE:
            reduced.minimize(new_obj)
        else:
            reduced.maximize(new_obj)

    _int_types = {VarType.INTEGER, VarType.BINARY}
    integer_vars = frozenset(
        name
        for name in eliminated_order
        if getattr(orig_by_name.get(name), "var_type", None) in _int_types
    )
    recovery = RecoveryMap(
        defining=defining, order=list(eliminated_order), integer_vars=integer_vars
    )
    return ReducedModel(
        model=reduced,
        recovery=recovery,
        eliminated=list(eliminated_order),
        method=agg.method,
        n_eliminated=len(eliminated_order),
        n_constraints_removed=len(elim_cids),
        n_bound_inequalities=n_bound_ineq,
        n_bound_pruned=n_bound_pruned,
        constraint_origin=constraint_origin,
        aggregation_set=agg,
        infeasible_eliminations=infeasible_eliminations,
    )


__all__ = [
    "substitute_expression",
    "defining_expressions",
    "apply_aggregation",
    "ReducedModel",
]
