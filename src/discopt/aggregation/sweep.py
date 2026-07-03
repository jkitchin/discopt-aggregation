"""Convergence-reliability sweeps for the aggregation study (plan item S3).

Runs a **local** NLP solve across a parameter grid for the original model and for
each aggregation method, counting how many instances converge — the paper's
Table 7 / Figures 9-11 analogue.  Local solves (``solve_nlp_from_model``) are the
paper's setting and expose iteration counts; grids are kept small so the sweep is
quick.

The paper's headline is that aggregation tends to *improve* reliability (more
instances converge).  Demonstrating that requires a problem tuned so the
full-space form sometimes fails within the iteration limit — the paper's authors
searched for such parameter ranges.  This harness produces the comparison for any
problem; the bundled well-behaved problems mostly converge for every method, so a
uniform 100% here is expected, not a null result.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Mapping

from discopt.aggregation.api import aggregate
from discopt.aggregation.benchmark import _local_solve


@dataclass
class CellResult:
    """Outcome of one (parameter-point, method) instance in a sweep."""

    params: dict[str, float]
    converged: bool
    iterations: int  # -1 if not converged
    wall_time: float


@dataclass
class SweepResult:
    """Convergence statistics for one method over a parameter grid.

    ``cells`` retains every per-grid-point outcome (for the paper's Figures 9-11
    convergence grids and :func:`virtual_best`); the scalar fields are aggregates
    over it.
    """

    method: str
    n_instances: int
    n_converged: int
    pct_converged: float
    avg_iterations: float  # mean over converged instances
    avg_wall_time: float  # mean over converged instances (seconds)
    cells: list[CellResult] = field(default_factory=list)


def _solve_instance(model: Any, method: str) -> tuple[bool, int, float]:
    """Solve one instance locally; returns (converged, iterations, wall_time)."""
    if method == "original":
        target = model
    else:
        target = aggregate(model, method=method).reduced_model
        if not target._variables:  # fully determined by aggregation: trivially solved
            return True, 0, 0.0
    status, iters, wall, _obj = _local_solve(target)
    return status.endswith("OPTIMAL"), iters, wall


def reliability_sweep(
    problem: Any, methods: tuple[str, ...] = ("ld1", "d2"), points: int = 3
) -> dict[str, SweepResult]:
    """Sweep ``problem`` over its parameter grid, per method (plus ``original``).

    Args:
        problem: a :class:`~discopt.aggregation.testproblems.TestProblem`.
        methods: aggregation methods to compare (kept to fast, structure-preserving
            ones by default — gr/lm reduced models can be very slow to solve).
        points: grid points per swept parameter (``points**2`` instances for a
            two-parameter problem).

    Returns a ``{method: SweepResult}`` map, keyed also by ``"original"``.
    """
    grid = problem.sweep_grid(points)
    results: dict[str, SweepResult] = {}
    for method in ("original", *methods):
        cells: list[CellResult] = []
        for params in grid:
            model = problem.build(**params)
            ok, it, wt = _solve_instance(model, method)
            cells.append(CellResult(params=dict(params), converged=ok, iterations=it, wall_time=wt))
        results[method] = _summarize(method, cells)
    return results


def _summarize(method: str, cells: list[CellResult]) -> SweepResult:
    conv = [c for c in cells if c.converged]
    n = len(cells)
    return SweepResult(
        method=method,
        n_instances=n,
        n_converged=len(conv),
        pct_converged=100.0 * len(conv) / n if n else 0.0,
        avg_iterations=mean([c.iterations for c in conv]) if conv else 0.0,
        avg_wall_time=mean([c.wall_time for c in conv]) if conv else 0.0,
        cells=cells,
    )


def virtual_best(results: Mapping[str, SweepResult]) -> SweepResult:
    """The paper's Fig-12 *virtual best*: per grid point, the best method's outcome.

    A cell converges in the virtual best if *any* method converges there; its
    iteration count is the minimum over the methods that converged.  Requires the
    input :class:`SweepResult`s to share the same grid order (they do when produced
    by one :func:`reliability_sweep`).  By construction the virtual best converges
    on at least as many instances as any single method.
    """
    methods = list(results.values())
    if not methods:
        return _summarize("virtual_best", [])
    n = methods[0].n_instances
    if any(m.n_instances != n for m in methods):
        raise ValueError("virtual_best requires all methods to share one grid")
    cells: list[CellResult] = []
    for i in range(n):
        row = [m.cells[i] for m in methods]
        conv = [c for c in row if c.converged]
        if conv:
            best = min(conv, key=lambda c: c.iterations)
            cells.append(
                CellResult(
                    params=best.params,
                    converged=True,
                    iterations=best.iterations,
                    wall_time=best.wall_time,
                )
            )
        else:
            cells.append(
                CellResult(params=row[0].params, converged=False, iterations=-1, wall_time=0.0)
            )
    return _summarize("virtual_best", cells)


def sweep_to_csv(results: Mapping[str, SweepResult], path: str) -> None:
    """Write every per-cell outcome to ``path`` (columns: method, params…, converged…)."""
    rows = [(m, c) for m, res in results.items() for c in res.cells]
    param_keys = sorted({k for _m, c in rows for k in c.params})
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", *param_keys, "converged", "iterations", "wall_time"])
        for m, c in rows:
            w.writerow(
                [
                    m,
                    *[c.params.get(k, "") for k in param_keys],
                    int(c.converged),
                    c.iterations,
                    c.wall_time,
                ]
            )


__all__ = ["SweepResult", "CellResult", "reliability_sweep", "virtual_best", "sweep_to_csv"]
