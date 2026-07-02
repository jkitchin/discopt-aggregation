"""Single authority for walking modeling-layer expression DAGs.

Several modules need to enumerate the sub-expressions or variable names of an
expression: incidence (occurrence counting), transform/scalarize (substitution),
and the numeric helpers (guards, hessian-cost, implicit).  Historically each
rolled its own ``isinstance`` ladder, and a node kind one walker forgot became a
silent hole — most dangerously :class:`CustomCall` and :class:`MatMulExpression`,
whose variables discopt's ``_collect_var_names`` reports as ``None`` (opaque for
*its* differentiability purposes).  For occurrence counting and substitution,
though, their children are ordinary reachable expressions.

:func:`child_exprs` is the one place that knows the node set.  It recognises
every modeling-core kind, **including** the opaque-but-walkable ``CustomCall``
(``.args``) and ``MatMulExpression`` (``.left``/``.right``), and returns
``None`` only for a genuinely unrecognised node — the signal for callers to
abstain (scalarize) or refuse loudly (incidence/transform) rather than silently
drop a hidden variable.
"""

from __future__ import annotations

from typing import Any

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    FunctionCall,
    IndexExpression,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)

# CustomCall / MatMulExpression are present in current discopt; guard the import
# so the plugin still loads against an older core that lacks them.
try:  # pragma: no cover - import shim
    from discopt.modeling.core import CustomCall
except ImportError:  # pragma: no cover
    CustomCall = None  # type: ignore[assignment,misc]
try:  # pragma: no cover - import shim
    from discopt.modeling.core import MatMulExpression
except ImportError:  # pragma: no cover
    MatMulExpression = None  # type: ignore[assignment,misc]


def child_exprs(node: Any) -> list[Any] | None:
    """Immediate expression children of ``node``.

    Returns ``[]`` for leaves (``Variable``, ``Constant``, ``Parameter``, plain
    numbers), the child list for every recognised interior node, and ``None``
    only for an unrecognised node kind — meaning the caller cannot certify which
    variables the node touches and must abstain or refuse.
    """
    if isinstance(node, (Variable, Constant, Parameter, int, float)):
        return []
    if isinstance(node, IndexExpression):
        return [node.base]
    if isinstance(node, UnaryOp):
        return [node.operand]
    if isinstance(node, BinaryOp):
        return [node.left, node.right]
    if isinstance(node, FunctionCall):
        return list(node.args)
    if isinstance(node, SumExpression):
        return [node.operand]
    if isinstance(node, SumOverExpression):
        return list(node.terms)
    if CustomCall is not None and isinstance(node, CustomCall):
        return list(node.args)
    if MatMulExpression is not None and isinstance(node, MatMulExpression):
        return [node.left, node.right]
    return None


def collect_variable_names(expr: Any) -> tuple[set[str], bool]:
    """``(names, analyzable)`` for ``expr``.

    ``names`` is the set of variable names reachable through :func:`child_exprs`.
    ``analyzable`` is ``False`` if any encountered node kind is unrecognised
    (``child_exprs`` returned ``None``), in which case ``names`` may be
    incomplete and the caller must treat the expression as opaque.
    """
    names: set[str] = set()
    analyzable = True
    stack: list[Any] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Variable):
            names.add(node.name)
            continue
        kids = child_exprs(node)
        if kids is None:
            analyzable = False
            continue
        stack.extend(kids)
    return names, analyzable


__all__ = ["child_exprs", "collect_variable_names"]
