"""Tests for the S1 parameterized test problems.

Checks that each problem builds, solves at nominal parameters, aggregates
(structurally and via a solve round-trip), and exposes a usable sweep grid.
"""

from __future__ import annotations

import numpy as np
import pytest

from discopt.aggregation.api import solve
from discopt.aggregation.metrics import compare_methods
from discopt.aggregation.testproblems import (
    GAS_PIPELINE,
    PROBLEMS,
    REACTION_DIFFUSION,
    UNIT_SELECTION,
    gas_pipeline,
    reaction_diffusion,
    unit_selection,
)


@pytest.mark.unit
def test_registry_and_sweep_grid():
    assert set(PROBLEMS) == {
        "reaction_diffusion",
        "unit_selection",
        "gas_pipeline",
        "recycle_loop",
        "cstr_dynamic",
        "inventory_chain",
    }
    grid = REACTION_DIFFUSION.sweep_grid(points=11)
    assert len(grid) == 11 * 11  # two swept parameters
    # grid spans the declared ranges
    ks = {round(g["k"], 6) for g in grid}
    assert min(ks) == pytest.approx(0.1) and max(ks) == pytest.approx(5.0)


@pytest.mark.unit
def test_build_applies_parameters():
    m = REACTION_DIFFUSION.build(n=5, k=2.0, c_in=0.5)
    # boundary constraint c[0] == c_in reflects the override.
    assert any("0.5" in repr(c.body) for c in m._constraints)
    assert sum(v.size for v in m._variables) == (6 + 4)  # c[0..5] + res[0..3]


@pytest.mark.unit
def test_structural_comparison_runs_on_problem():
    # Structural metrics need no solve; every method eliminates the intermediates.
    rows = {r["method"]: r for r in compare_methods(REACTION_DIFFUSION.build(n=6))}
    # Approximate-maximum reaches the multi-variable residual definitions.
    assert rows["gr"]["n_eliminated"] >= rows["d2"]["n_eliminated"]
    assert rows["lm"]["n_eliminated"] >= 1


@pytest.mark.smoke
def test_solves_at_nominal():
    m = reaction_diffusion(n=8)
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0, abs=1e-6)  # residuals driven to zero


@pytest.mark.smoke
@pytest.mark.parametrize("method", ["ld1", "d2"])
def test_aggregation_roundtrip_on_problem(method):
    # Structure-preserving methods (ld1/d2) do not densify the objective here, so
    # the reduced model solves quickly and round-trips exactly. (gr/lm eliminate
    # the multi-variable residual definitions and substitute them into sum(res^2),
    # densifying the Hessian — the reduced solve is then much slower on discopt,
    # the paper's own "aggressive aggregation can hurt solve time" finding; those
    # are exercised structurally in test_structural_comparison_runs_on_problem.)
    original = reaction_diffusion(n=8).solve()
    res = solve(reaction_diffusion(n=8), method=method)
    assert res.aggregation.n_eliminated > 0
    assert res.objective == pytest.approx(original.objective, abs=1e-5)
    # full-space profile recovered at the original array shape
    assert np.asarray(res.x["c"]).shape == (9,)


# --------------------------------------------------------------------------- #
# MINLP (unit_selection) and nonconvex NLP (gas_pipeline)                       #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_minlp_and_nonconvex_structural():
    for problem in (UNIT_SELECTION, GAS_PIPELINE):
        rows = {r["method"]: r for r in compare_methods(problem.build())}
        assert rows["d2"]["n_eliminated"] >= 1  # degree-2 defs are eliminable


@pytest.mark.smoke
def test_minlp_roundtrip_keeps_binaries():
    # unit_selection is an MIQP: the cost intermediates aggregate, binaries don't.
    original = unit_selection().solve()
    res = solve(unit_selection(), method="d2")
    assert res.aggregation.n_eliminated >= 1
    assert not any(name.startswith("b") for name in res.aggregation.eliminated)
    assert res.objective == pytest.approx(original.objective, abs=1e-4)


@pytest.mark.smoke
def test_nonconvex_pipeline_roundtrip():
    # gas_pipeline is a nonconvex NLP; d2 eliminates the squared-pressure defs.
    original = gas_pipeline().solve()
    res = solve(gas_pipeline(), method="d2")
    assert res.aggregation.n_eliminated >= 1
    assert res.objective == pytest.approx(original.objective, abs=1e-4)


@pytest.mark.smoke
def test_cstr_dynamic_dae_problem():
    """P4: the discretized-DAE control problem round-trips and aggregates well."""
    from discopt.aggregation.api import aggregate
    from discopt.aggregation.benchmark import _local_solve
    from discopt.aggregation.testproblems import CSTR_DYNAMIC, cstr_dynamic

    m = cstr_dynamic(n=8)
    nv = sum(int(getattr(v, "size", 1)) for v in m._variables)
    assert nv == 24  # c(8) + u(8) + rate(8)

    # nominal solves locally to optimality (structural fingerprint)
    status, _it, _w, obj = _local_solve(m)
    assert status.endswith("OPTIMAL")

    # d2 eliminates the algebraic rate variables — a meaningful fraction (paper's
    # DIST shows ~36%); reduced model solves to the same optimum, fast, no blow-up
    agg = aggregate(m, method="d2")
    assert agg.n_eliminated / nv > 0.10
    r_status, _i, _w, r_obj = _local_solve(agg.reduced_model)
    assert r_status.endswith("OPTIMAL")
    assert float(r_obj) == pytest.approx(float(obj), abs=1e-4)

    # registered for the sweep harness
    assert CSTR_DYNAMIC.build(n=6) is not None
