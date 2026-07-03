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

    Shared sub-DAGs are visited once (memoised by ``id``), so this is ``O(#distinct
    nodes)`` even when the expression's *tree* expansion is exponential — the case
    aggressive substitution routinely produces.
    """
    names: set[str] = set()
    analyzable = True
    seen: set[int] = set()
    stack: list[Any] = [expr]
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        if isinstance(node, Variable):
            names.add(node.name)
            continue
        kids = child_exprs(node)
        if kids is None:
            analyzable = False
            continue
        stack.extend(kids)
    return names, analyzable


# Unary ops that preserve affine (linear) participation of their operand's vars.
_AFFINE_UNARY = frozenset({"+", "-", "neg", "pos"})


def classify_linearity(expr: Any, nonlinear_names: frozenset[str] = frozenset()) -> dict[str, bool]:
    """Map each variable in ``expr`` to whether it participates *affinely*.

    ``{name: True}`` means the variable's isolated contribution to ``expr`` is
    affine (a constant coefficient); ``{name: False}`` means it appears
    nonlinearly (under a power/transcendental, in a product or quotient with
    another variable, or inside an opaque node).  Computed by a single
    ``O(#distinct nodes)`` DAG-memoised bottom-up pass — the scalable replacement
    for calling discopt's per-name ``_isolated_affine_coeff`` (an ``O(tree)``
    walk) once per variable, which is intractable on substitution-expanded bodies.

    ``nonlinear_names`` are always reported nonlinear regardless of syntactic
    form — used to match discopt's conservative treatment of *array* variables,
    which it never classifies affine.  On fully-scalar bodies (every reduced /
    scalarized model) the pass matches ``_isolated_affine_coeff`` exactly.

    Conservative: any construct whose linearity cannot be certified marks its
    variables nonlinear, so the nonlinear set is never *under*-reported.
    """
    memo: dict[int, dict[str, bool]] = {}

    def nonlinear(d: dict[str, bool]) -> dict[str, bool]:
        return dict.fromkeys(d, False)

    def merge(a: dict[str, bool], b: dict[str, bool]) -> dict[str, bool]:
        # additive combination: a var is affine iff affine in every place it appears
        out = dict(a)
        for k, aff in b.items():
            out[k] = out.get(k, True) and aff
        return out

    def analyze(node: Any) -> dict[str, bool]:
        nid = id(node)
        cached = memo.get(nid)
        if cached is not None:
            return cached
        result = _analyze(node)
        memo[nid] = result
        return result

    def _analyze(node: Any) -> dict[str, bool]:
        if isinstance(node, Variable):
            return {node.name: node.name not in nonlinear_names}
        if isinstance(node, (Constant, Parameter, int, float)):
            return {}
        if isinstance(node, IndexExpression):
            base = node.base
            if isinstance(base, Variable):
                return {base.name: base.name not in nonlinear_names}
            return analyze(base)
        if isinstance(node, UnaryOp):
            inner = analyze(node.operand)
            return inner if node.op in _AFFINE_UNARY else nonlinear(inner)
        if isinstance(node, BinaryOp):
            left = analyze(node.left)
            right = analyze(node.right)
            op = node.op
            if op in ("+", "-"):
                return merge(left, right)
            if op == "*":
                if not left:  # constant * expr -> preserves affine
                    return right
                if not right:  # expr * constant
                    return left
                return nonlinear(merge(left, right))  # var * var -> nonlinear
            if op == "/":
                return left if not right else nonlinear(merge(left, right))
            # power (or any other binary op): variable base/operands -> nonlinear
            return nonlinear(merge(left, right))
        if isinstance(node, SumExpression):
            return analyze(node.operand)
        if isinstance(node, SumOverExpression):
            out: dict[str, bool] = {}
            for t in node.terms:
                out = merge(out, analyze(t))
            return out
        if isinstance(node, FunctionCall):
            out = {}
            for a in node.args:
                out = merge(out, analyze(a))
            return nonlinear(out)
        # CustomCall / MatMulExpression / unrecognised: all reachable vars nonlinear
        names, _ = collect_variable_names(node)
        return dict.fromkeys(names, False)

    return analyze(expr)


__all__ = ["child_exprs", "collect_variable_names", "classify_linearity"]
