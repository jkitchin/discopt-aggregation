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

import numpy as np
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


def _provably_scalar(node: Any) -> bool:
    """True only if ``node`` is certified to evaluate to a scalar (shape ``()``).

    An array-valued result can only originate from an array-shaped *leaf*
    (variable, constant, or parameter): broadcasting, indexing, reduction and
    matmul all yield a scalar when every leaf they combine is scalar.  So a node
    all of whose reachable leaves are scalar is itself scalar, whatever the
    interior ops.  Returns ``False`` — declining to certify — for any non-scalar
    leaf or any node :func:`child_exprs` cannot walk (which could hide an array
    leaf).  Soundness-conservative: a caller may rely on a ``True`` result, but a
    ``False`` means "not certified", not "definitely array".

    Used to decide whether a :class:`SumExpression` (a reduction) merely passes
    its operand through — true only when the operand is scalar, so the sum is the
    identity — versus collapsing scalar-variable coefficients by an element count
    the affine walk does not track.
    """
    seen: set[int] = set()
    stack: list[Any] = [node]
    while stack:
        n = stack.pop()
        nid = id(n)
        if nid in seen:
            continue
        seen.add(nid)
        if isinstance(n, Variable):
            if getattr(n, "shape", ()) != ():
                return False
            continue
        if isinstance(n, (Constant, Parameter)):
            try:
                if np.asarray(getattr(n, "value", None)).size != 1:
                    return False
            except (TypeError, ValueError):
                return False
            continue
        if isinstance(n, (int, float)):
            continue
        kids = child_exprs(n)
        if kids is None:
            return False
        stack.extend(kids)
    return True


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
            # A sum over an array operand multiplies each scalar variable's
            # coefficient by the (untracked) element count, so only a sum whose
            # operand is provably scalar (the sum is then the identity) preserves
            # affine coefficients; otherwise every reachable variable is nonlinear.
            inner = analyze(node.operand)
            return inner if _provably_scalar(node.operand) else nonlinear(inner)
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


def _leaf_const(node: Any) -> float | None:
    """Scalar constant value of ``node`` if it is a *leaf* constant, else ``None``.

    Mirrors discopt's ``_const_value``: only a scalar :class:`Constant` /
    :class:`Parameter` (or a bare Python number) is a pure constant.  A composite
    subtree that happens to evaluate to a constant (e.g. ``2 * 3``) is **not** —
    matching ``_isolated_affine_coeff``, whose multiply/divide rules fold only a
    literal constant operand, so ``y * (2 * 3)`` reads nonlinear both there and here.
    """
    if isinstance(node, bool):
        return None
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, (Constant, Parameter)):
        value = getattr(node, "value", None)
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
        if arr.size == 1:
            return float(arr.reshape(-1)[0])
    return None


# Internal per-node representation used by :func:`affine_coefficients`:
# ``(const, coeffs, nonlinear)`` where ``coeffs`` maps a variable participating
# *only* affinely to its summed coefficient, ``nonlinear`` is the set of variables
# participating nonlinearly, and the two name sets are kept disjoint (a name that
# ever appears nonlinearly is nonlinear).  ``None`` marks an unanalyzable node.


def affine_coefficients(
    expr: Any, nonlinear_names: frozenset[str] = frozenset()
) -> tuple[float, dict[str, float | None]] | None:
    """Affine decomposition of ``expr`` in one DAG-memoised bottom-up pass.

    Returns ``(constant_term, {name: coeff_or_None})`` where the coefficient is
    the *summed* affine coefficient when the variable participates only affinely
    (``2*y + 3*y`` → ``5.0``, matching ``_isolated_affine_coeff``'s per-name
    semantics), ``None`` when the variable participates nonlinearly, and where a
    variable absent from ``expr`` simply does not appear in the map.  The whole
    result is ``None`` only for a genuinely unanalyzable node (an unrecognised
    kind that :func:`child_exprs` cannot walk).

    This is the scalable replacement for calling discopt's ``O(tree)``
    ``_isolated_affine_coeff`` once per candidate variable: a single
    ``O(#distinct nodes)`` pass memoised by ``id(node)`` (the
    :func:`classify_linearity` pattern) covers every variable at once, so a body
    whose *tree* expansion is huge but whose shared DAG is small is analysed in
    time proportional to the DAG.

    ``nonlinear_names`` forces the listed variables nonlinear regardless of
    syntactic form — discopt's conservative treatment of *array* variables, which
    it never reports affine.  On fully-scalar bodies the per-``(body, var)``
    result matches ``_isolated_affine_coeff`` exactly (coefficient to machine
    precision when affine, ``None``-agreement otherwise).

    ``constant_term`` is exact for a fully-affine body (the body as written:
    ``3*x + 2*y - 7`` → ``(-7.0, {'x': 3.0, 'y': 2.0})``); for bodies with
    nonlinear parts only the affine coefficients are guaranteed, and the constant
    term reflects only the constants foldable through the affine skeleton.
    """
    memo: dict[int, Any] = {}

    def analyze(node: Any) -> Any:
        nid = id(node)
        if nid in memo:
            return memo[nid]
        result = _analyze(node)
        memo[nid] = result
        return result

    def all_names(rep: Any) -> set[str]:
        # Names touched by a sub-result, for the nonlinear-collapse cases.
        const, coeffs, nonlin = rep
        return set(coeffs) | nonlin

    def collapse_nonlinear(*reps: Any) -> Any:
        # Every variable reachable through the given sub-results goes nonlinear.
        if any(r is None for r in reps):
            return None
        names: set[str] = set()
        for r in reps:
            names |= all_names(r)
        return (0.0, {}, names)

    def nonlinear_or_const(node: Any, reps: list[Any]) -> Any:
        # A nonlinear construct (transcendental, power, opaque, ...) collapses its
        # variables to nonlinear — UNLESS every sub-result is variable-free, in
        # which case the whole node is a pure constant and must contribute its
        # *exact* value, not be dropped to 0.0 (e.g. sin(2) = 0.909..., which the
        # IA1 integrality check on the constant term depends on).  A variable-free
        # node that cannot be evaluated exactly yields None, declining the whole
        # extraction rather than inventing a constant.
        if any(r is None for r in reps):
            return None
        if all(not r[1] and not r[2] for r in reps):
            from discopt.aggregation.recovery import evaluate_expression

            try:
                return (float(np.asarray(evaluate_expression(node, {}))), {}, set())
            except Exception:  # noqa: BLE001 - unevaluable variable-free node: decline
                return None
        return collapse_nonlinear(*reps)

    def _analyze(node: Any) -> Any:
        if isinstance(node, Variable):
            name = node.name
            if name in nonlinear_names:
                return (0.0, {}, {name})
            return (0.0, {name: 1.0}, set())
        if isinstance(node, (Constant, Parameter, int, float)):
            k = _leaf_const(node)
            return (k if k is not None else 0.0, {}, set())
        if isinstance(node, IndexExpression):
            # Indexed occurrence is not the scalar-affine pattern: any variable
            # reached through it is nonlinear (matches _isolated_affine_coeff).
            names, analyzable = collect_variable_names(node)
            if not analyzable:
                return None
            return (0.0, {}, set(names))
        if isinstance(node, UnaryOp):
            inner = analyze(node.operand)
            if inner is None:
                return None
            if node.op in _AFFINE_UNARY:
                if node.op in ("-", "neg"):
                    const, coeffs, nonlin = inner
                    return (-const, {n: -c for n, c in coeffs.items()}, set(nonlin))
                return inner
            return nonlinear_or_const(node, [inner])  # abs / sin / exp / ...
        if isinstance(node, BinaryOp):
            left = analyze(node.left)
            right = analyze(node.right)
            if left is None or right is None:
                return None
            op = node.op
            if op in ("+", "-"):
                return _merge_add(left, right, +1.0 if op == "+" else -1.0)
            if op == "*":
                return _mul(node.left, node.right, left, right)
            if op == "/":
                return _div(node.right, left, right)
            # power or any other binary op over variable operands -> nonlinear
            return nonlinear_or_const(node, [left, right])
        if isinstance(node, SumExpression):
            # See classify_linearity: a sum only preserves affine coefficients
            # when its operand is provably scalar (identity sum).  Over an array
            # operand the true coefficient is the per-element one times the
            # element count, which this walk does not track, so decline by
            # collapsing every reachable variable to nonlinear.
            inner = analyze(node.operand)
            if inner is None:
                return None
            return inner if _provably_scalar(node.operand) else collapse_nonlinear(inner)
        if isinstance(node, SumOverExpression):
            reps = [analyze(t) for t in node.terms]
            return nonlinear_or_const(node, reps)
        if isinstance(node, FunctionCall):
            reps = [analyze(a) for a in node.args]
            return nonlinear_or_const(node, reps)
        # CustomCall / MatMulExpression / unrecognised: every reachable variable
        # nonlinear; a genuinely unwalkable node yields the outer None.
        kids = child_exprs(node)
        if kids is None:
            return None
        reps = [analyze(k) for k in kids]
        return nonlinear_or_const(node, reps)

    def _merge_add(left: Any, right: Any, sign: float) -> Any:
        lconst, lcoeffs, lnonlin = left
        rconst, rcoeffs, rnonlin = right
        const = lconst + sign * rconst
        nonlin = lnonlin | rnonlin
        coeffs: dict[str, float] = dict(lcoeffs)
        for n, c in rcoeffs.items():
            coeffs[n] = coeffs.get(n, 0.0) + sign * c
        for n in nonlin:
            coeffs.pop(n, None)
        return (const, coeffs, nonlin)

    def _scale(rep: Any, factor: float) -> Any:
        const, coeffs, nonlin = rep
        return (factor * const, {n: factor * c for n, c in coeffs.items()}, set(nonlin))

    def _mul(lnode: Any, rnode: Any, left: Any, right: Any) -> Any:
        left_has = bool(left[1] or left[2])
        right_has = bool(right[1] or right[2])
        if left_has and right_has:
            return collapse_nonlinear(left, right)  # var on both factors
        if left_has:
            k = _leaf_const(rnode)
            if k is None:
                return collapse_nonlinear(left, right)
            return _scale(left, k)
        if right_has:
            k = _leaf_const(lnode)
            if k is None:
                return collapse_nonlinear(left, right)
            return _scale(right, k)
        # Neither side carries a variable: fold the constant term.
        return (left[0] * right[0], {}, set())

    def _div(rnode: Any, left: Any, right: Any) -> Any:
        right_has = bool(right[1] or right[2])
        if right_has:
            return collapse_nonlinear(left, right)  # variable in a denominator
        k = _leaf_const(rnode)
        if k is None or k == 0.0:
            return collapse_nonlinear(left, right)
        return _scale(left, 1.0 / k)

    top = analyze(expr)
    if top is None:
        return None
    const, coeffs, nonlin = top
    out: dict[str, float | None] = dict(coeffs)
    for n in nonlin:
        out[n] = None
    return (const, out)


__all__ = [
    "child_exprs",
    "collect_variable_names",
    "classify_linearity",
    "affine_coefficients",
]
