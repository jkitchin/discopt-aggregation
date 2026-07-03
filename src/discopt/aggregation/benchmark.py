"""Runtime instrumentation for the aggregation study (plan item S2).

Solves the original and reduced models with a **local** NLP solve
(``discopt.solvers.nlp_pounce.solve_nlp_from_model``) — the setting the paper
studies (local IPOPT), and the one that exposes iteration counts.  This matters:
``discopt.Model.solve()`` runs a *global* spatial branch-and-bound on nonconvex
NLPs, which is both far slower and not the paper's comparison.

Each :class:`BenchmarkRow` records the aggregation time (``t_elim``), the reduced
size and structural Hessian coupling, and the local solve's status, iteration
count, and wall time.  Together these reproduce the paper's Table 6 trends: the
structure-preserving methods keep per-iteration cost low, while the
approximate-maximum methods (``gr``/``lm``) cut variables/iterations but densify
the Hessian, so their per-iteration cost — and total solve time — can rise
sharply (the paper's "Hessian evaluation becomes the bottleneck").
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from discopt.aggregation.api import aggregate
from discopt.aggregation.metrics import model_structure
from discopt.aggregation.scalarize import scalarize_model

_DEFAULT_METHODS = ("ld1", "ecd2", "ld2", "d2", "gr", "lm")


@dataclass
class BenchmarkRow:
    """One row of a runtime benchmark (original model or one aggregation method)."""

    method: str
    n_eliminated: int
    n_variables: int  # flat scalar count of the (reduced) model
    nnz_hessian: int  # structural Lagrangian-Hessian coupling (per-iteration cost proxy)
    status: str
    iterations: int  # -1 if not solved
    solve_wall_time: float  # seconds (solver-internal); 0 if not solved
    aggregate_time: float  # seconds to compute + apply the aggregation (t_elim)
    objective: float


def _local_solve(
    model: Any, options: dict[str, Any] | None = None
) -> tuple[str, int, float, float]:
    """Local NLP solve; returns (status, iterations, wall_time, objective).

    ``options`` is forwarded to
    :func:`discopt.solvers.nlp_pounce.solve_nlp_from_model` (POUNCE/Ipopt option
    dict).  Use ``{"max_iter": N}`` to cap iterations: a run that stops at the
    limit reports a non-``OPTIMAL`` status (``ITERATION_LIMIT``) and so counts as
    *not converged* — the paper's reliability protocol.  ``options=None`` (the
    default) leaves the solve byte-identical to an uncapped call.
    """
    from discopt.solvers.nlp_pounce import solve_nlp_from_model

    r = solve_nlp_from_model(model, options=options)
    return (
        str(getattr(r, "status", "")),
        int(getattr(r, "iterations", -1)),
        float(getattr(r, "wall_time", 0.0)),
        float(getattr(r, "objective", float("nan"))),
    )


def benchmark_model(
    model: Any,
    methods: tuple[str, ...] = _DEFAULT_METHODS,
    run_solve: bool = True,
    options: dict[str, Any] | None = None,
) -> list[BenchmarkRow]:
    """Benchmark ``model`` and each aggregation method.

    The first row is the original model; each subsequent row applies one method.
    With ``run_solve=True`` (default) each model is solved locally and the row
    carries iteration count and wall time; with ``run_solve=False`` only the
    structural columns are filled (``iterations=-1``), which is the cheap way to
    characterise methods whose reduced model is expensive to solve (``gr``/``lm``
    on densifying problems).

    ``options`` is forwarded to every local solve (see :func:`_local_solve`);
    e.g. ``{"max_iter": 3000}`` caps iterations as in the paper's reliability
    protocol.  ``options=None`` (the default) leaves solves uncapped.
    """
    rows: list[BenchmarkRow] = []
    # Measure the original in its scalarized form so its structural columns are
    # on the same (per-scalar) granularity as the aggregated models, which are
    # scalarized internally — otherwise array-named originals undercount.
    scalar_model, smap = scalarize_model(model)
    base_model = scalar_model if smap.scalarized else model
    base = model_structure(base_model)
    st, it, wt, obj = (
        _local_solve(base_model, options=options) if run_solve else ("", -1, 0.0, float("nan"))
    )
    rows.append(
        BenchmarkRow("original", 0, base.n_variables, base.nnz_hessian, st, it, wt, 0.0, obj)
    )
    for method in methods:
        t0 = perf_counter()
        result = aggregate(model, method=method)
        t_elim = perf_counter() - t0
        struct = model_structure(result.reduced_model)
        st, it, wt, obj = (
            _local_solve(result.reduced_model, options=options)
            if run_solve
            else ("", -1, 0.0, float("nan"))
        )
        rows.append(
            BenchmarkRow(
                method=method,
                n_eliminated=result.n_eliminated,
                n_variables=struct.n_variables,
                nnz_hessian=struct.nnz_hessian,
                status=st,
                iterations=it,
                solve_wall_time=wt,
                aggregate_time=t_elim,
                objective=obj,
            )
        )
    return rows


def callback_breakdown(model: Any, method: str = "original", repeats: int = 20) -> dict[str, float]:
    """Per-evaluation cost breakdown across the NLP callbacks (Table 6 analogue, P3).

    The paper's Table 6 reports the *fraction of solve time* spent in function,
    Jacobian, and Hessian evaluation.  POUNCE's result object exposes no such
    breakdown (a documented gap; see the parity plan), so this measures the
    complementary, solver-independent quantity: the wall time of each evaluator
    callback (objective, gradient, Jacobian, Lagrangian Hessian) at a sample
    point, averaged over ``repeats`` calls, as fractions of their total.  It
    reproduces the paper's qualitative finding — that aggressive aggregation
    (``gr``/``lm``) shifts cost sharply toward Hessian evaluation — without
    needing solver-internal timers.  ``method="original"`` measures the input
    model; any aggregation method measures its reduced model.

    Returns ``{"func", "grad", "jac", "hess", "hess_over_func"}`` (the first four
    are fractions summing to 1; the last is the Hessian/function cost ratio).
    """
    from discopt.solvers.nlp_pounce import NLPEvaluator

    target = model if method == "original" else aggregate(model, method=method).reduced_model
    ev = NLPEvaluator(target)
    lb, ub = ev.variable_bounds
    x = np.clip(0.5 * (np.clip(lb, -100, 100) + np.clip(ub, -100, 100)), -100, 100)
    n_con = ev.n_constraints() if callable(ev.n_constraints) else ev.n_constraints
    lam = np.ones(int(n_con))

    # Warm up JIT compilation so we time steady-state evaluation, not tracing.
    ev.evaluate_objective(x)
    ev.evaluate_gradient(x)
    ev.evaluate_jacobian_values(x)
    ev.evaluate_lagrangian_hessian(x, 1.0, lam)

    def _time(fn: Any) -> float:
        t0 = perf_counter()
        for _ in range(repeats):
            fn()
        return perf_counter() - t0

    t_func = _time(lambda: ev.evaluate_objective(x))
    t_grad = _time(lambda: ev.evaluate_gradient(x))
    t_jac = _time(lambda: ev.evaluate_jacobian_values(x))
    t_hess = _time(lambda: ev.evaluate_lagrangian_hessian(x, 1.0, lam))
    total = t_func + t_grad + t_jac + t_hess or 1.0
    return {
        "func": t_func / total,
        "grad": t_grad / total,
        "jac": t_jac / total,
        "hess": t_hess / total,
        "hess_over_func": t_hess / t_func if t_func else float("inf"),
    }


__all__ = ["BenchmarkRow", "benchmark_model", "callback_breakdown"]
