"""Expression-graph size / CSE diagnostics for the Hessian-cost investigation (R2).

Aggressive aggregation (``gr``/``lm``) substitutes a variable's defining
expression into every constraint/objective that used it.  On a *deep nonlinear
substitution chain* (``s[i] = phi(s[i-1])`` with ``phi`` referencing its argument
more than once), the composed expression's **tree** grows exponentially with
chain depth — the paper's "Hessian evaluation becomes the bottleneck" (S2).

This module measures that growth structurally, without a solve:

* :func:`expression_tree_size` — nodes counted **with multiplicity** (the size a
  compiler that does *not* share subexpressions must process).
* :func:`expression_dag_size` — **distinct** nodes (by object identity); the size
  a common-subexpression-eliminating (CSE) compiler would process.
* :func:`sharing_ratio` — ``tree / dag``; ``> 1`` means shared structure a
  CSE-aware lowering could exploit.
* :func:`model_expression_size` — the same, summed over a model's objective and
  constraint bodies.

**Finding (R2, measured).** A single :func:`substitute_expression` call *does*
preserve sharing (the substituted subtree is one shared object).  But two things
compound on a deep chain: (1) the plugin's multi-variable substitution
composition does not keep the whole chain shared, and — decisively — (2) discopt's
expression→jaxpr lowering does **not** CSE shared nodes: a *manually* linear-DAG
chain (``dag`` growing ~6 nodes/level) still takes **exponential** time to build
its Hessian (measured 0.03 s → 2.97 s over depths 6→10).  So the dominant fix is
upstream (CSE-aware lowering in discopt — filed as a discopt issue), and the
plugin-side mitigation is to *cap the substituted-expression size* rather than
aggregate deep nonlinear chains (``max_def_nodes``, plan item D2).
"""

from __future__ import annotations

from typing import Any

from discopt.modeling.core import (
    BinaryOp,
    FunctionCall,
    IndexExpression,
    SumExpression,
    SumOverExpression,
    UnaryOp,
)

# Sentinel returned by tree-size when the (possibly exponential) count exceeds
# the cap — so callers never hang enumerating an exponential tree.
TREE_SIZE_CAP = 5_000_000


def _children(expr: Any) -> list:
    """Immediate sub-expression operands of ``expr`` (empty for leaves)."""
    if isinstance(expr, BinaryOp):
        return [expr.left, expr.right]
    if isinstance(expr, UnaryOp):
        return [expr.operand]
    if isinstance(expr, IndexExpression):
        return [expr.base]
    if isinstance(expr, SumExpression):
        return [expr.operand]
    if isinstance(expr, SumOverExpression):
        return list(expr.terms)
    if isinstance(expr, FunctionCall):
        return list(expr.args)
    return []


def expression_tree_size(expr: Any, cap: int = TREE_SIZE_CAP) -> int:
    """Number of nodes in ``expr`` counting shared subtrees once *per reference*.

    Returns ``-1`` if the count exceeds ``cap`` (an exponential tree), so this
    never hangs on a deeply-substituted expression.
    """
    stack = [expr]
    n = 0
    while stack:
        n += 1
        if n > cap:
            return -1
        stack.extend(_children(stack.pop()))
    return n


def expression_dag_size(expr: Any) -> int:
    """Number of *distinct* nodes in ``expr`` (by object identity) — the DAG size."""
    seen: set[int] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.extend(_children(node))
    return len(seen)


def expression_depth(expr: Any) -> int:
    """Maximum nesting depth of ``expr`` (iterative; safe on deep expressions).

    A flat ``sum(t0 + t1 + ... + tk)`` folds left-associatively into a depth-``k``
    chain.  The plugin's *recursive* AST walkers (scalarize/substitution) overflow
    Python's stack near depth ~1000, so this diagnostic lets callers detect a model
    that is too deep to aggregate before hitting a ``RecursionError`` — building
    such objectives as balanced sums keeps the depth logarithmic (plan item R3).
    """
    best = 0
    stack = [(expr, 1)]
    while stack:
        node, d = stack.pop()
        if d > best:
            best = d
        for c in _children(node):
            stack.append((c, d + 1))
    return best


def sharing_ratio(expr: Any, cap: int = TREE_SIZE_CAP) -> float:
    """``tree_size / dag_size`` — how much shared structure ``expr`` has.

    ``1.0`` means a pure tree (no sharing); larger means a CSE-aware compiler
    could avoid re-processing shared subtrees.  ``inf`` if the tree exceeds the
    cap while the DAG is finite (an exponentially-shared expression).
    """
    dag = expression_dag_size(expr)
    tree = expression_tree_size(expr, cap=cap)
    if dag == 0:
        return 1.0
    if tree < 0:
        return float("inf")
    return tree / dag


def model_expression_size(model: Any, cap: int = TREE_SIZE_CAP) -> tuple[int, int]:
    """``(tree_size, dag_size)`` summed over ``model``'s objective + constraint bodies.

    ``tree_size`` is ``-1`` if any body exceeds the cap.  The DAG size is measured
    per body and summed (bodies do not share nodes across each other).
    """
    exprs: list[Any] = []
    obj = getattr(model, "_objective", None)
    if obj is not None and getattr(obj, "expression", None) is not None:
        exprs.append(obj.expression)
    for c in getattr(model, "_constraints", []):
        body = getattr(c, "body", None)
        if body is not None:
            exprs.append(body)
    tree_total = 0
    dag_total = 0
    for e in exprs:
        t = expression_tree_size(e, cap=cap)
        if t < 0:
            tree_total = -1
        elif tree_total >= 0:
            tree_total += t
        dag_total += expression_dag_size(e)
    return tree_total, dag_total


__all__ = [
    "expression_tree_size",
    "expression_dag_size",
    "expression_depth",
    "sharing_ratio",
    "model_expression_size",
    "TREE_SIZE_CAP",
]
