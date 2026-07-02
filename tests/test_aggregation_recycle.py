"""Tests for R1: the recycle-loop reliability/iteration-reduction problem.

``recycle_loop`` is a torn steady-state recycle (parallel units, tear-residual
objective) whose intermediates aggregation eliminates down to the single tear
variable ``R``.  The structural tests here are fast (no solve): the problem
builds, registers, and aggregates to one variable.  The empirical iteration
reduction (full-space ~2x the reduced iterations) is exercised by the worked
example / sweep, not asserted here, to keep the unit suite solver-free and fast.
"""

from __future__ import annotations

import pytest

from discopt.aggregation.api import aggregate
from discopt.aggregation.testproblems import PROBLEMS, RECYCLE_LOOP, recycle_loop

pytestmark = pytest.mark.unit


def test_recycle_loop_builds_with_intermediates():
    m = recycle_loop(n=4)
    # R + inp[4] + out[4] + rec  → 4 array/scalar variable objects
    names = {v.name for v in m._variables}
    assert {"R", "inp", "out", "rec"} <= names


def test_recycle_loop_registered():
    assert "recycle_loop" in PROBLEMS
    assert PROBLEMS["recycle_loop"] is RECYCLE_LOOP
    assert set(RECYCLE_LOOP.nominal) == {"feed", "alpha"}


def test_aggregation_reduces_recycle_to_single_variable():
    m = recycle_loop(n=4)
    result = aggregate(m, method="d2")
    reduced = result.reduced_model
    # the whole loop collapses to a single degree of freedom (the matcher may
    # keep R or an equivalent split variable — either way, one variable remains)
    assert len(reduced._variables) == 1
    assert result.n_eliminated > 0


@pytest.mark.parametrize("n", [3, 5])
def test_reduction_holds_across_sizes(n):
    reduced = aggregate(recycle_loop(n=n), method="d2").reduced_model
    assert len(reduced._variables) == 1


def test_sweep_grid_shape():
    grid = RECYCLE_LOOP.sweep_grid(points=3)
    assert len(grid) == 9  # 3 x 3 over (feed, alpha)
    assert all(set(p) == {"feed", "alpha"} for p in grid)
