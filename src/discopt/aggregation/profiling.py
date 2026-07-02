"""Build-time profiling of aggregation at scale (plan item R3).

The paper's models are large discretized DAEs; this measures how the *build*
(not solve) scales — scalarization, incidence construction, and the full
``aggregate`` — on models with thousands of indexed variables.

Findings (measured, R3):

* **The incidence / matching graph kernel scales linearly** in the number of
  variables — it is not the bottleneck.
* **Scalarization and the full aggregate scale ~quadratically**, because each
  elimination substitutes its defining expression into the (growing) objective
  and constraint bodies, and every pass re-walks them.  This is the real scaling
  limit for aggressive, whole-model aggregation.
* **The recursive AST walkers overflow** (``RecursionError``) on deeply-nested
  expressions — notably a flat ``sum(t0 + ... + tk)`` of ~1000+ terms, which
  folds into a depth-~1000 left-associative chain.  Building such objectives as
  *balanced* sums (depth ``O(log k)``) sidesteps the wall; the scaling models
  here do exactly that, and :func:`~discopt.aggregation.hessian_cost.expression_depth`
  lets callers detect an at-risk model in advance.

These point at the same two follow-ups already in the plan: a native indexed
incidence (F1) to cut the scalarization blow-up, and iterative (non-recursive)
walkers to lift the depth wall.
"""

from __future__ import annotations

import time
from typing import Any, Callable


def balanced_sum(terms: list) -> Any:
    """Sum ``terms`` as a balanced tree (depth ``O(log n)``) not a left-deep chain.

    ``sum(terms)`` / ``a + b + c + ...`` folds left-associatively into a depth-``n``
    expression that overflows the recursive AST walkers around ``n ~ 1000``; a
    balanced reduction keeps the depth logarithmic.  Requires a non-empty list.
    """
    items = list(terms)
    if not items:
        raise ValueError("balanced_sum requires at least one term")
    while len(items) > 1:
        items = [
            items[i] + items[i + 1] if i + 1 < len(items) else items[i]
            for i in range(0, len(items), 2)
        ]
    return items[0]


def indep_scaling_model(n: int) -> Any:
    """``n`` independent degree-2 intermediates ``y[i] = 2 x[i] + 1`` (+ objective).

    Each intermediate depends on a single decision, so ``d2`` eliminates all ``n``
    without chaining (no expression blow-up) — the clean regime for measuring
    build-time scaling.  The objective is a *balanced* sum so deep-expression
    recursion does not confound the measurement.
    """
    import discopt.modeling as dm

    m = dm.Model(f"indep_{n}")
    x = m.continuous("x", shape=(n,), lb=-5.0, ub=5.0)
    y = m.continuous("y", shape=(n,), lb=-20.0, ub=20.0)
    for i in range(n):
        m.subject_to(y[i] - (2.0 * x[i] + 1.0) == 0.0)
    m.minimize(balanced_sum([y[i] ** 2 + x[i] for i in range(n)]))
    return m


def profile_build(model: Any, method: str = "d2") -> dict[str, Any]:
    """Time the build stages for ``model`` under ``method``.

    Returns ``{flat_vars, scalarize_s, incidence_s, aggregate_s, n_eliminated}``.
    ``aggregate_s`` is the whole ``aggregate`` (which scalarizes internally); the
    separate ``scalarize_s`` / ``incidence_s`` break out those components.
    """
    from discopt.aggregation.api import aggregate
    from discopt.aggregation.incidence import build_incidence
    from discopt.aggregation.scalarize import scalarize_model

    flat_vars = sum(int(getattr(v, "size", 1)) for v in model._variables)

    t0 = time.perf_counter()
    scalar_model, _ = scalarize_model(model)
    scalarize_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    build_incidence(scalar_model)
    incidence_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = aggregate(model, method=method)
    aggregate_s = time.perf_counter() - t0

    return {
        "flat_vars": flat_vars,
        "scalarize_s": scalarize_s,
        "incidence_s": incidence_s,
        "aggregate_s": aggregate_s,
        "n_eliminated": result.n_eliminated,
    }


def build_scaling(
    builder: Callable[[int], Any], sizes: list[int], method: str = "d2"
) -> list[dict[str, Any]]:
    """Profile ``builder(n)`` for each ``n`` in ``sizes`` (a scaling curve)."""
    return [{"n": n, **profile_build(builder(n), method=method)} for n in sizes]


__all__ = ["balanced_sum", "indep_scaling_model", "profile_build", "build_scaling"]
