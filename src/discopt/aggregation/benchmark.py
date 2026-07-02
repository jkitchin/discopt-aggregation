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


def _local_solve(model: Any) -> tuple[str, int, float, float]:
    """Local NLP solve; returns (status, iterations, wall_time, objective)."""
    from discopt.solvers.nlp_pounce import solve_nlp_from_model

    r = solve_nlp_from_model(model)
    return (
        str(getattr(r, "status", "")),
        int(getattr(r, "iterations", -1)),
        float(getattr(r, "wall_time", 0.0)),
        float(getattr(r, "objective", float("nan"))),
    )


def benchmark_model(
    model: Any, methods: tuple[str, ...] = _DEFAULT_METHODS, run_solve: bool = True
) -> list[BenchmarkRow]:
    """Benchmark ``model`` and each aggregation method.

    The first row is the original model; each subsequent row applies one method.
    With ``run_solve=True`` (default) each model is solved locally and the row
    carries iteration count and wall time; with ``run_solve=False`` only the
    structural columns are filled (``iterations=-1``), which is the cheap way to
    characterise methods whose reduced model is expensive to solve (``gr``/``lm``
    on densifying problems).
    """
    rows: list[BenchmarkRow] = []
    # Measure the original in its scalarized form so its structural columns are
    # on the same (per-scalar) granularity as the aggregated models, which are
    # scalarized internally — otherwise array-named originals undercount.
    scalar_model, smap = scalarize_model(model)
    base_model = scalar_model if smap.scalarized else model
    base = model_structure(base_model)
    st, it, wt, obj = _local_solve(base_model) if run_solve else ("", -1, 0.0, float("nan"))
    rows.append(
        BenchmarkRow("original", 0, base.n_variables, base.nnz_hessian, st, it, wt, 0.0, obj)
    )
    for method in methods:
        t0 = perf_counter()
        result = aggregate(model, method=method)
        t_elim = perf_counter() - t0
        struct = model_structure(result.reduced_model)
        st, it, wt, obj = (
            _local_solve(result.reduced_model) if run_solve else ("", -1, 0.0, float("nan"))
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


__all__ = ["BenchmarkRow", "benchmark_model"]
