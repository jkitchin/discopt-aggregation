"""Tests for the S3 reliability-sweep harness.

Runs a small local-solve sweep and checks the convergence bookkeeping. The
bundled reaction-diffusion problem is convex, so every method converges on every
grid point (100%); the test asserts the harness structure and that aggregation
never reduces the converged count below the original.
"""

from __future__ import annotations

import pytest

from discopt.aggregation.sweep import SweepResult, reliability_sweep
from discopt.aggregation.testproblems import REACTION_DIFFUSION

pytestmark = pytest.mark.smoke


def test_reliability_sweep_structure_and_counts():
    results = reliability_sweep(REACTION_DIFFUSION, methods=("ld1", "d2"), points=3)
    assert set(results) == {"original", "ld1", "d2"}
    for method, r in results.items():
        assert isinstance(r, SweepResult)
        assert r.n_instances == 9  # 3 x 3 grid
        assert 0 <= r.n_converged <= r.n_instances
        assert 0.0 <= r.pct_converged <= 100.0

    # Convex problem: everything converges, and no method converges *fewer*
    # instances than the original (the paper's "aggregation never hurts" check).
    base = results["original"].n_converged
    assert base == 9
    for method in ("ld1", "d2"):
        assert results[method].n_converged >= base


def test_sweep_reports_iterations_for_converged():
    results = reliability_sweep(REACTION_DIFFUSION, methods=("d2",), points=3)
    # Converged instances report a positive average iteration count.
    assert results["original"].avg_iterations > 0
    assert results["d2"].avg_iterations > 0
