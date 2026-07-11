"""Tests for R3: build-time profiling at scale + the depth diagnostic.

Fast, structural checks (small sizes): the profiling harness returns sane
timings, the scaling model aggregates fully and round-trips correctly, balanced
sums stay shallow (avoiding the recursive-walker depth wall), and
``expression_depth`` detects a deep left-associative sum.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.api import solve
from discopt.aggregation.hessian_cost import expression_depth
from discopt.aggregation.profiling import (
    balanced_sum,
    build_scaling,
    indep_scaling_model,
    profile_build,
)

pytestmark = pytest.mark.unit


def test_profile_build_keys_and_full_elimination():
    row = profile_build(indep_scaling_model(200), method="d2")
    assert set(row) == {
        "flat_vars",
        "scalarize_s",
        "incidence_s",
        "aggregate_s",
        "n_eliminated",
    }
    assert row["flat_vars"] == 400  # x[200] + y[200]
    assert row["n_eliminated"] == 200  # all y[i] eliminated
    assert all(row[k] >= 0.0 for k in ("scalarize_s", "incidence_s", "aggregate_s"))


def test_build_scaling_returns_row_per_size():
    rows = build_scaling(indep_scaling_model, [50, 100], method="d2")
    assert [r["n"] for r in rows] == [50, 100]
    assert rows[1]["n_eliminated"] == 100


@pytest.mark.smoke
def test_scaling_model_roundtrips():
    # aggregate + solve recovers the eliminated intermediates correctly. The
    # recovered point reassembles the scalarized arrays back to their array names.
    import numpy as np

    res = solve(indep_scaling_model(100), method="d2")
    xv = np.asarray(res.x["x"]).reshape(-1)
    yv = np.asarray(res.x["y"]).reshape(-1)
    assert xv.shape == (100,) and yv.shape == (100,)
    assert np.allclose(yv, 2.0 * xv + 1.0, atol=1e-6)


def test_balanced_sum_is_shallow():
    m = dm.Model("d")
    xs = [m.continuous(f"x{i}") for i in range(64)]
    balanced = balanced_sum([xi**2 for xi in xs])
    left_deep = xs[0] ** 2
    for xi in xs[1:]:
        left_deep = left_deep + xi**2
    # balanced depth ~ log2(64) region; left-deep depth ~ 64 (grows with terms)
    assert expression_depth(balanced) < expression_depth(left_deep)
    assert expression_depth(left_deep) > 60


def test_expression_depth_is_iterative_and_safe():
    # a deep chain that would overflow a recursive walker is handled iteratively
    m = dm.Model("deep")
    x = m.continuous("x")
    e = x
    for _ in range(5000):
        e = e + 1.0
    assert expression_depth(e) >= 5000
