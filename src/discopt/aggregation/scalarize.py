"""Scalarize array variables so aggregation can reach discretized models (F1).

The incidence layer only treats *scalar* continuous variables as aggregation
candidates, so a model built with array variables (``m.continuous("x",
shape=(n,))``) — the shape of every discretized-DAE model — aggregates nothing.
:func:`scalarize_model` rewrites such arrays into per-element scalar variables
(``x`` of shape ``(3,)`` becomes ``x__0, x__1, x__2``), after which the ordinary
aggregation machinery applies.  :class:`ScalarizationMap` reassembles the arrays
from the scalar solution so callers see the original variable shapes back.

Abstention is conservative: an array variable is scalarized only when *every*
reference to it in the model is a fully-specifying constant index — ``a[i]`` for
a 1-D array, ``a[i, j]`` for a 2-D array.  Any whole-array use (a bare ``a`` in
arithmetic, ``sum(a)``, a slice ``a[1:3]``, a partial index ``b[0]`` on a 2-D
array, or a non-constant index) leaves that array untouched.  Arrays with more
than ``limit`` elements are also left alone, to avoid exploding the model.

Only continuous arrays are scalarized (the aggregation targets); integer/binary
arrays are left as arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)

from discopt.aggregation.transform import _make_constraint

_DEFAULT_LIMIT = 10_000


def _index_key(index: Any, shape: tuple[int, ...]) -> Any:
    """Return a hashable element key if ``index`` fully specifies one element.

    ``i`` for a 1-D array, ``(i, j, ...)`` for an n-D array, with every component
    a constant integer in range.  Returns ``None`` for partial indices, slices,
    negative or out-of-range indices, or non-integer components — the caller then
    abstains on that array.
    """
    ndim = len(shape)

    def _ok(i: Any, size: int) -> bool:
        return isinstance(i, (int, np.integer)) and not isinstance(i, bool) and 0 <= int(i) < size

    if ndim == 1:
        if _ok(index, shape[0]):
            return int(index)
        return None
    if isinstance(index, tuple) and len(index) == ndim:
        if all(_ok(i, shape[d]) for d, i in enumerate(index)):
            return tuple(int(i) for i in index)
    return None


def _scalar_name(array_name: str, key: Any) -> str:
    if isinstance(key, tuple):
        return array_name + "__" + "_".join(str(i) for i in key)
    return f"{array_name}__{key}"


def _iter_elements(shape: tuple[int, ...]):
    """Yield ``(key, elem_index)`` for every element of ``shape``."""
    if len(shape) == 1:
        for i in range(shape[0]):
            yield i, i
    else:
        for idx in np.ndindex(*shape):
            yield tuple(int(i) for i in idx), idx


def _candidate_arrays(model: Any) -> dict[str, tuple[int, ...]]:
    """Continuous array variables (``size > 1``) as ``{name: shape}``."""
    out: dict[str, tuple[int, ...]] = {}
    for v in model._variables:
        if v.var_type == VarType.CONTINUOUS and int(getattr(v, "size", 1)) > 1:
            out[v.name] = tuple(int(s) for s in v.shape)
    return out


def _find_abstentions(model: Any, candidates: dict[str, tuple[int, ...]]) -> set[str]:
    """Names of candidate arrays that must NOT be scalarized (non-scalar uses)."""
    abstain: set[str] = set()

    def scan(expr: Any) -> None:
        if isinstance(expr, IndexExpression):
            base = expr.base
            if isinstance(base, Variable) and base.name in candidates:
                if _index_key(expr.index, candidates[base.name]) is None:
                    abstain.add(base.name)  # slice / partial / non-constant index
                return  # the base variable is consumed by this index
            scan(base)
            return
        if isinstance(expr, Variable):
            if expr.name in candidates:
                abstain.add(expr.name)  # bare whole-array reference
            return
        if isinstance(expr, BinaryOp):
            scan(expr.left)
            scan(expr.right)
        elif isinstance(expr, UnaryOp):
            scan(expr.operand)
        elif isinstance(expr, FunctionCall):
            for a in expr.args:
                scan(a)
        elif isinstance(expr, SumExpression):
            scan(expr.operand)
        elif isinstance(expr, SumOverExpression):
            for t in expr.terms:
                scan(t)
        # Constant / Parameter / other leaf: nothing to scan.

    for c in model._constraints:
        if isinstance(c, Constraint):
            scan(c.body)
    obj = getattr(model, "_objective", None)
    if obj is not None:
        scan(obj.expression)
    return abstain


def _rewrite(expr: Any, index_map: dict, var_map: dict[str, Expression]) -> Any:
    """Replace scalarized ``a[i]`` with its scalar, and remap kept variables."""
    if isinstance(expr, IndexExpression):
        base = expr.base
        if isinstance(base, Variable) and (base.name, _key_of(expr.index)) in index_map:
            return index_map[(base.name, _key_of(expr.index))]
        new_base = _rewrite(base, index_map, var_map)
        return expr if new_base is base else IndexExpression(new_base, expr.index)
    if isinstance(expr, Variable):
        return var_map.get(expr.name, expr)
    if isinstance(expr, Constant):
        return expr
    if isinstance(expr, BinaryOp):
        return BinaryOp(
            expr.op,
            _rewrite(expr.left, index_map, var_map),
            _rewrite(expr.right, index_map, var_map),
        )
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, _rewrite(expr.operand, index_map, var_map))
    if isinstance(expr, FunctionCall):
        return FunctionCall(expr.func_name, *(_rewrite(a, index_map, var_map) for a in expr.args))
    if isinstance(expr, SumExpression):
        return SumExpression(
            _rewrite(expr.operand, index_map, var_map), getattr(expr, "axis", None)
        )
    if isinstance(expr, SumOverExpression):
        return SumOverExpression([_rewrite(t, index_map, var_map) for t in expr.terms])
    return expr


def _key_of(index: Any) -> Any:
    return tuple(int(i) for i in index) if isinstance(index, tuple) else index


def _recreate(new_model: Any, v: Any) -> Any:
    if v.var_type == VarType.CONTINUOUS:
        return new_model.continuous(v.name, shape=v.shape, lb=v.lb, ub=v.ub)
    if v.var_type == VarType.INTEGER:
        return new_model.integer(v.name, shape=v.shape, lb=v.lb, ub=v.ub)
    if v.var_type == VarType.BINARY:
        return new_model.binary(v.name, shape=v.shape)
    raise NotImplementedError(f"unsupported variable type {v.var_type!r}")  # pragma: no cover


@dataclass
class ScalarizationMap:
    """Reassembles array variables from a scalarized solution.

    Attributes:
        shapes: ``{array_name: shape}`` of the arrays that were scalarized.
    """

    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def scalarized(self) -> list[str]:
        return list(self.shapes)

    def reassemble(self, values: dict[str, Any]) -> dict[str, Any]:
        """Return ``values`` with scalar elements folded back into arrays."""
        out = dict(values)
        for name, shape in self.shapes.items():
            arr = np.empty(shape, dtype=float)
            for key, elem in _iter_elements(shape):
                sname = _scalar_name(name, key)
                arr[elem] = float(np.asarray(values[sname]).reshape(()))
                out.pop(sname, None)
            out[name] = arr
        return out


def scalarize_model(model: Any, limit: int = _DEFAULT_LIMIT) -> tuple[Any, ScalarizationMap]:
    """Rewrite fully-indexed continuous array variables into scalars.

    Returns ``(new_model, smap)``.  If no array can be scalarized, ``new_model``
    is the original ``model`` and ``smap`` is empty.
    """
    candidates = _candidate_arrays(model)
    if not candidates:
        return model, ScalarizationMap()

    abstain = _find_abstentions(model, candidates)
    to_scalarize = {
        name: shape
        for name, shape in candidates.items()
        if name not in abstain and int(np.prod(shape)) <= limit
    }
    if not to_scalarize:
        return model, ScalarizationMap()

    new = dm.Model(f"{getattr(model, 'name', 'model')}_scalar")
    var_map: dict[str, Expression] = {}
    index_map: dict[tuple[str, Any], Expression] = {}

    for v in model._variables:
        if v.name in to_scalarize:
            shape = to_scalarize[v.name]
            lb = np.broadcast_to(np.asarray(v.lb, dtype=float), shape)
            ub = np.broadcast_to(np.asarray(v.ub, dtype=float), shape)
            for key, elem in _iter_elements(shape):
                sv = new.continuous(
                    _scalar_name(v.name, key), lb=float(lb[elem]), ub=float(ub[elem])
                )
                index_map[(v.name, key)] = sv
        else:
            var_map[v.name] = _recreate(new, v)

    for c in model._constraints:
        if not isinstance(c, Constraint):  # pragma: no cover - defensive
            raise NotImplementedError(
                f"scalarization only supports algebraic constraints; found {type(c).__name__}"
            )
        new.subject_to(_make_constraint(_rewrite(c.body, index_map, var_map), c.sense))

    obj = getattr(model, "_objective", None)
    if obj is not None:
        new_obj = _rewrite(obj.expression, index_map, var_map)
        if obj.sense == ObjectiveSense.MINIMIZE:
            new.minimize(new_obj)
        else:
            new.maximize(new_obj)

    return new, ScalarizationMap(shapes=dict(to_scalarize))


__all__ = ["scalarize_model", "ScalarizationMap"]
