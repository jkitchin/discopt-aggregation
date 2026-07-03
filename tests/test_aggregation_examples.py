"""Tests that keep the worked examples (`discopt.aggregation.examples`) correct.

The examples double as executable documentation, so they are pinned here: the
per-method elimination counts on the showcase model, and that the solve /
metrics / safeguard examples run and produce the expected outcomes.
"""

from __future__ import annotations

import pytest

from discopt.aggregation import examples

pytestmark = pytest.mark.unit


def test_all_methods_show_increasing_aggressiveness():
    elim = examples.example_all_methods()
    counts = {m: len(v) for m, v in elim.items()}
    # Structure-preserving methods nest via the shared ld1 prefix.
    assert counts["ld1"] == 1
    assert counts["ecd2"] == 2
    assert counts["ld2"] == 3
    assert counts["d2"] == 4
    # Approximate-maximum reach the 4-variable equation too.
    assert counts["gr"] == 5
    assert counts["lm"] == 5
    # Nesting of the structure-preserving sets.
    assert set(elim["ld1"]) <= set(elim["ecd2"]) <= set(elim["ld2"]) <= set(elim["d2"])


def test_degree_k_example_runs():
    examples.example_degree_k()


def test_recursive_pipeline_example_runs():
    examples.example_recursive_pipeline()  # prints; must not raise


def test_structure_comparison_example_runs():
    examples.example_structure_comparison()


@pytest.mark.smoke
def test_exact_hessian_example_runs():
    examples.example_exact_hessian()


def test_pivot_tolerance_example_runs():
    examples.example_pivot_tolerance()


def test_fill_cap_example_runs():
    examples.example_fill_cap()


def test_target_linearity_example_runs():
    examples.example_target_linearity()


def test_symbolic_derivative_example_runs():
    examples.example_symbolic_derivative()


@pytest.mark.smoke
def test_dual_recovery_example_runs():
    out = examples.example_dual_recovery()
    assert out["available"] and out["residual"] < 1e-6  # valid recovered KKT point
    # DU2: the active-eliminated-bound case recovers its gamma-term and self-checks.
    assert out["bound_available"] and out["bound_residual"] < 1e-6
    assert out["bound_duals"]["y"] == pytest.approx(2.53553389, abs=1e-4)


def test_recursive_duals_example_runs():
    out = examples.example_recursive_duals()
    assert out["available"] and out["residual"] < 1e-6
    assert out["n_passes"] >= 2  # genuinely multi-pass


def test_structural_diagnosis_example_runs():
    examples.example_structural_diagnosis()


def test_exact_tearing_example_runs():
    out = examples.example_exact_tearing()
    assert out["n_kept"] == 3  # 3-cycle -> keep 2 + isolated d


def test_scalable_tearing_example_runs():
    examples.example_scalable_tearing()


def test_decomposable_matching_example_runs():
    examples.example_decomposable_matching()


def test_implicit_block_example_runs():
    examples.example_implicit_block()


def test_implicit_blocks_example_runs():
    examples.example_implicit_blocks()


def test_block_triangular_ordering_example_runs():
    examples.example_block_triangular_ordering()


def test_elimination_guards_example_runs():
    examples.example_elimination_guards()


def test_bound_pruning_example_runs():
    out = examples.example_bound_pruning()
    # w in [-2,2] provably inside [-10,10] -> both inequalities pruned
    assert out["n_without"] == 2 and out["n_with"] == 0 and out["n_pruned"] == 2


@pytest.mark.smoke
def test_array_scalarization_example_runs():
    examples.example_array_scalarization()  # solves; must not raise


def test_test_problem_example_runs():
    examples.example_test_problem()  # structural only; no solve


@pytest.mark.smoke
def test_runtime_example_runs():
    examples.example_runtime()  # local solves on tiny instance


@pytest.mark.smoke
def test_recycle_reliability_example_runs():
    examples.example_recycle_reliability()


def test_hessian_cost_example_runs():
    examples.example_hessian_cost()


def test_build_scaling_example_runs():
    examples.example_build_scaling()


def test_presolve_overlap_example_runs():
    examples.example_presolve_overlap()


def test_reliability_example_runs():
    out = examples.example_reliability()  # small local-solve sweep
    # convex problem: every method converges on every instance, and no method
    # converges fewer instances than another (the paper's headline direction)
    for stats in out.values():
        assert 0.0 <= stats["pct"] <= 100.0
        assert stats["converged"] <= stats["instances"]
    assert out["d2"]["converged"] >= out["ld1"]["converged"]


@pytest.mark.smoke
def test_end_to_end_solve_example_runs():
    examples.example_end_to_end_solve()  # solves; must not raise


@pytest.mark.smoke
def test_performance_example_reduces_problem_size():
    # Tiny sizes and one method to keep the test fast; assert the deterministic
    # structural reduction only (timing/speedup is machine-dependent).
    rows = examples.example_performance(sizes=(20,), methods=("lm",), reps=1)
    assert len(rows) == 1
    row = rows[0]
    assert row["n_eliminated"] > 0
    assert row["reduced_vars"] < row["orig_vars"]  # smaller reduced model


def test_integer_aggregation_example_runs():
    out = examples.example_integer_aggregation()
    assert out["off"] == 0 and out["on"] >= out["n_before"] // 2 - 1  # chain collapses
    assert out["all_integral"]
    assert abs(out["obj"] - out["direct_obj"]) < 1e-6  # reduced == direct MINLP optimum
