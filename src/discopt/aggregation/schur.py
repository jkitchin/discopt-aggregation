"""Structured / Schur KKT reuse — plugin-side analysis (plan item K2).

The plugin's job for K2 is to *identify* the reducible block that pounce's
block-triangular / Schur KKT partition (pounce#180 item 2; Parker, Garcia & Bent,
arXiv:2602.17968) should Schur-complement out: the variables eliminated by
linear-matching aggregation together with their defining equality constraints form
a block-lower-triangular submatrix of the KKT system.

:func:`kkt_schur_indices` returns that block as **KKT-space indices** — the shape
`pounce.Problem.set_kkt_schur_block` consumes.  pounce's augmented KKT has
dimension ``dim = n_x + n_s + n_c + n_d`` in block order ``[x, slack, eq-dual,
ineq-dual]`` (pounce dev-note ``issue-180-item2-schur-kkt-scope``), with one slack
per inequality (``n_s = n_ineq``).  Aggregation only eliminates via **equality**
constraints, so the reducible block is:

* each eliminated variable → its position in the ``x`` block ``[0, n_x)``;
* each defining equality → the ``eq-dual`` block, index
  ``n_x + n_ineq + eq_rank`` where ``eq_rank`` is the equality's rank among the
  model's equality constraints.

Equality-only models are the special case ``n_ineq = 0`` (eq-dual block ``[n, n+m)``).

**Integration boundary.** Handing these indices to the solver — building the
``pounce.Problem`` and calling ``set_kkt_schur_block`` before the solve — belongs
in discopt's pounce bridge (``discopt.solvers.nlp_pounce``), which owns the
evaluator→Problem construction.  This plugin deliberately does *not* reach into
discopt's private solver internals to do that; it only produces the analysis for
discopt (or a caller) to feed.  Verified end-to-end externally: with the block
installed the solution is identical (pounce falls back if unsuitable) and the
factorization is ~1.2–1.4× faster on `reaction_diffusion` — the clean path is a
small ``kkt_schur_block=`` passthrough in ``solve_nlp_from_model``.
"""

from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np


@overload
def kkt_schur_indices(
    model: Any, *, method: str = ..., pivot_tol: float = ..., return_model: Literal[False] = ...
) -> list[int]: ...
@overload
def kkt_schur_indices(
    model: Any, *, method: str = ..., pivot_tol: float = ..., return_model: Literal[True]
) -> tuple[list[int], Any]: ...
def kkt_schur_indices(
    model: Any,
    *,
    method: str = "lm",
    pivot_tol: float = 1e-6,
    return_model: bool = False,
) -> list[int] | tuple[list[int], Any]:
    """KKT-space indices of the aggregation's reducible block (K2).

    Scalarizes ``model``, runs ``method`` (``lm`` by default) to find the matched
    (variable, defining-equality) pairs, and maps them to KKT-space indices in
    pounce's ``[x, slack, eq-dual, ineq-dual]`` layout.  The ``x`` block is the
    NLP evaluator's *flat* variable layout: each model variable contributes
    ``prod(shape)`` scalars in declaration order, so a surviving (abstained)
    array shifts every later variable's index.  Each eliminated variable (always
    a scalar) → its flat offset in ``[0, n_x)``; its defining equality → ``n_x +
    n_ineq + eq_rank`` in the eq-dual block.  Handles inequality constraints
    (``n_ineq > 0``); equality-only is the ``n_ineq = 0`` case.

    The indices live in the KKT space of the *analyzed* (scalarized) model, which
    is what the caller must solve.  Pass ``return_model=True`` to receive
    ``(indices, analyzed_model)`` so the two cannot be mismatched; otherwise the
    caller is responsible for solving the same (scalarized) model.  The result is
    the ``indices`` argument for ``pounce.Problem.set_kkt_schur_block`` (handed to
    the solver by discopt's pounce bridge, not by this plugin).
    """
    from discopt.aggregation.scalarize import scalarize_model
    from discopt.aggregation.strategies import aggregation_set

    scalar_model, smap = scalarize_model(model)
    analyzed = scalar_model if smap.scalarized else model

    # Flat variable layout: prod(shape) scalars per variable, declaration order —
    # matching discopt's NLPEvaluator.variable_bounds.
    var_offset: dict[str, int] = {}
    cum = 0
    for v in analyzed._variables:
        var_offset[v.name] = cum
        cum += int(getattr(v, "size", int(np.prod(getattr(v, "shape", ()) or ()))))
    n_x = cum

    # Classify constraints: equalities get a rank; inequalities each add a slack.
    eq_rank: dict[int, int] = {}
    n_ineq = 0
    rank = 0
    for cid, c in enumerate(analyzed._constraints):
        if getattr(c, "sense", "==") == "==":
            eq_rank[cid] = rank
            rank += 1
        else:
            n_ineq += 1
    eq_dual_base = n_x + n_ineq  # n_slack = n_ineq

    agg = aggregation_set(analyzed, method=method, pivot_tol=pivot_tol)

    indices: set[int] = set()
    for e in agg.eliminations:
        indices.add(var_offset[e.var])  # x block (eliminated vars are scalars)
        indices.add(eq_dual_base + eq_rank[e.cid])  # eq-dual block
    result = sorted(indices)
    return (result, analyzed) if return_model else result


__all__ = ["kkt_schur_indices"]
