"""Tests for the S2 runtime benchmark harness.

Uses **local** solves (fast, iteration-count-aware) on tiny instances, and the
fast methods for solve-based rows; the densifying gr/lm methods are characterised
structurally (their reduced-model solve can be very slow — the paper's Hessian
bottleneck — so they are not solved in the test).
"""

from __future__ import annotations

import pytest

from discopt.aggregation.benchmark import BenchmarkRow, _local_solve, benchmark_model
from discopt.aggregation.testproblems import reaction_diffusion

pytestmark = pytest.mark.smoke


def test_benchmark_rows_have_iterations_and_times():
    rows = benchmark_model(reaction_diffusion(n=5), methods=("ld1", "d2"))
    assert [r.method for r in rows] == ["original", "ld1", "d2"]
    assert all(isinstance(r, BenchmarkRow) for r in rows)
    # Local solve exposes a positive iteration count and a wall time.
    for r in rows:
        assert r.status.endswith("OPTIMAL")
        assert r.iterations > 0
        assert r.solve_wall_time >= 0.0
    # Aggregation rows record a nonzero elimination and a measured t_elim.
    assert rows[1].n_eliminated > 0
    assert rows[1].aggregate_time >= 0.0


def test_structure_preserving_reduces_variables():
    rows = {r.method: r for r in benchmark_model(reaction_diffusion(n=6), methods=("d2",))}
    base, d2 = rows["original"], rows["d2"]
    # d2 removes variables (structural columns are scalarized-consistent).
    assert d2.n_variables < base.n_variables
    assert d2.n_eliminated > 0


@pytest.mark.unit
def test_structural_only_characterizes_aggressive_methods_cheaply():
    # run_solve=False avoids the (slow) gr/lm reduced-model solve while still
    # reporting the Hessian-coupling growth that drives their per-iteration cost.
    rows = {
        r.method: r
        for r in benchmark_model(reaction_diffusion(n=8), methods=("gr", "lm"), run_solve=False)
    }
    assert rows["gr"].iterations == -1  # not solved (reduced solve is slow)
    # gr eliminates the most variables; its per-iteration blow-up comes from the
    # substituted objective's expression graph, not a larger distinct-pair count.
    assert rows["gr"].n_eliminated > rows["original"].n_eliminated
    assert rows["gr"].n_variables < rows["original"].n_variables


@pytest.mark.unit
def test_local_solve_max_iter_cap_reports_not_converged():
    """RL1: an iteration cap turns a converging solve into a not-converged one.

    The uncapped local solve of a well-behaved instance converges (``OPTIMAL``)
    in more than three iterations; capping it at ``max_iter=3`` stops at the
    limit with a non-``OPTIMAL`` status (``ITERATION_LIMIT``) — the paper counts
    that as *not converged*.
    """
    model = reaction_diffusion(n=6)
    status, iters, _wall, _obj = _local_solve(model)
    assert status.endswith("OPTIMAL")
    assert iters > 3  # uncapped run needs more than the cap we impose below

    capped_status, capped_iters, _cw, _co = _local_solve(model, options={"max_iter": 3})
    assert not capped_status.endswith("OPTIMAL")  # stopped at the limit
    assert capped_iters == 3  # clamped to the cap


@pytest.mark.unit
def test_local_solve_options_none_is_byte_identical_default():
    """RL1: ``options=None`` reproduces the uncapped solve exactly.

    Compares the deterministic fields (status, iterations, objective); wall time
    is a measured quantity and varies run to run.
    """
    model = reaction_diffusion(n=6)
    status_a, iters_a, _wall_a, obj_a = _local_solve(model)
    status_b, iters_b, _wall_b, obj_b = _local_solve(model, options=None)
    assert (status_a, iters_a, obj_a) == (status_b, iters_b, obj_b)


@pytest.mark.unit
def test_benchmark_model_threads_options_to_every_solve():
    """RL1: ``options`` reaches the original and each aggregation row's solve."""
    rows = benchmark_model(reaction_diffusion(n=6), methods=("ld1",), options={"max_iter": 3})
    # Every solved row stopped at the cap (none reached OPTIMAL).
    assert all(not r.status.endswith("OPTIMAL") for r in rows)
    assert all(r.iterations == 3 for r in rows)
    # Default (no options) still converges — proving the cap, not a broken model.
    default_rows = benchmark_model(reaction_diffusion(n=6), methods=("ld1",))
    assert all(r.status.endswith("OPTIMAL") for r in default_rows)
