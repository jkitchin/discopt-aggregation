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

from dataclasses import dataclass
from statistics import mean
from typing import Any

from discopt.aggregation.api import aggregate
from discopt.aggregation.benchmark import _local_solve


@dataclass
class SweepResult:
    """Convergence statistics for one method over a parameter grid."""

    method: str
    n_instances: int
    n_converged: int
    pct_converged: float
    avg_iterations: float  # mean over converged instances
    avg_wall_time: float  # mean over converged instances (seconds)


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
        converged = 0
        iters: list[int] = []
        walls: list[float] = []
        for params in grid:
            model = problem.build(**params)
            ok, it, wt = _solve_instance(model, method)
            if ok:
                converged += 1
                iters.append(it)
                walls.append(wt)
        n = len(grid)
        results[method] = SweepResult(
            method=method,
            n_instances=n,
            n_converged=converged,
            pct_converged=100.0 * converged / n if n else 0.0,
            avg_iterations=mean(iters) if iters else 0.0,
            avg_wall_time=mean(walls) if walls else 0.0,
        )
    return results


__all__ = ["SweepResult", "reliability_sweep"]
