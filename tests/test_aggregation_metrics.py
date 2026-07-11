"""Tests for :mod:`discopt.aggregation.metrics` (paper Section 5.1 / Table 4).

Validates the structural metrics against models with known structure, and checks
that the cross-method comparison reproduces the paper's qualitative trends:
approximate-maximum methods eliminate the most but can inflate the total Jacobian
density, while structure-preserving methods never increase it.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.metrics import compare_methods, exact_hessian_nnz, model_structure

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# model_structure                                                             #
# --------------------------------------------------------------------------- #
def test_structure_counts_basic():
    m = dm.Model("s")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    z = m.continuous("z", lb=-5, ub=5)
    m.subject_to(x + y + z == 1.0)  # 3 vars, all linear
    m.subject_to(x * y <= 2.0)  # 2 vars, both nonlinear
    m.minimize(x**2 + z)  # objective nonlinear in x
    s = model_structure(m)
    assert s.n_variables == 3
    assert s.n_constraints == 2
    assert s.n_equality == 1
    assert s.nnz_jacobian == 3 + 2  # 3 in first constraint, 2 in second
    assert s.avg_nnz_per_con == pytest.approx(2.5)
    # linear participations: first constraint 3 linear, second 0 -> avg 1.5
    assert s.avg_linear_nnz_per_con == pytest.approx(1.5)


def test_hessian_pairs_bilinear():
    # x*y couples {x},{y},{x,y} -> 3 structural Hessian entries.
    m = dm.Model("h")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.subject_to(x * y == 1.0)
    m.minimize(x + y)  # linear objective contributes nothing
    assert model_structure(m).nnz_hessian == 3


def test_hessian_pairs_square():
    # x**2 couples only {x} -> 1 entry.
    m = dm.Model("h2")
    x = m.continuous("x", lb=-5, ub=5)
    m.subject_to(x**2 == 1.0)
    m.minimize(x)
    assert model_structure(m).nnz_hessian == 1


def test_hessian_zero_for_linear_model():
    m = dm.Model("lin")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.subject_to(x + y == 1.0)
    m.minimize(x + 2 * y)
    assert model_structure(m).nnz_hessian == 0


def test_hessian_array_diagonal_block_expanded_per_scalar():
    """RC9: an array variable's diagonal Hessian block is counted per scalar
    element (n(n+1)/2), so the documented invariant nnz_hessian_exact <=
    nnz_hessian holds even for array variables — it did not when the block counted
    as a single name-pair."""
    m = dm.Model("harr")
    x = m.continuous("x", shape=(4,), lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize(dm.sum(x**2) + y**4)
    s = model_structure(m, exact_hessian=True)
    # {x} diagonal 4*5/2 = 10, {y} diagonal 1, {x,y} off-diagonal 4*1 = 4 -> 15
    assert s.nnz_hessian == 15
    assert s.nnz_hessian_exact is not None
    assert s.nnz_hessian_exact <= s.nnz_hessian


def test_hessian_scalar_counts_unchanged_by_expansion():
    """RC9 no-regression: for scalar-only models the per-scalar expansion equals
    the plain name-pair count (paper Table 4 numbers are preserved)."""
    m = dm.Model("hsc")
    a = m.continuous("a", lb=-5, ub=5)
    b = m.continuous("b", lb=-5, ub=5)
    m.minimize(a**2 + a * b + b**2)  # {a}, {b}, {a, b} -> 3
    assert model_structure(m).nnz_hessian == 3


# --------------------------------------------------------------------------- #
# compare_methods                                                             #
# --------------------------------------------------------------------------- #
def _study_model():
    # y is defined linearly by a 3-input equation and then used elsewhere, so
    # eliminating it (gr/lm) substitutes a 3-variable expression into other
    # constraints -- increasing density -- while structure-preserving methods
    # (which need <=2-variable defining constraints) leave y alone.
    m = dm.Model("study")
    a = m.continuous("a", lb=-5, ub=5)
    b = m.continuous("b", lb=-5, ub=5)
    c = m.continuous("c", lb=-5, ub=5)
    y = m.continuous("y", lb=-20, ub=20)
    p = m.continuous("p", lb=-5, ub=5)
    w = m.continuous("w", lb=-50, ub=50)
    f = m.continuous("f", lb=-5, ub=5)  # a genuine fixed variable
    m.subject_to(f == 2.0)  # LD1-eliminable
    m.subject_to(y - (a + b + c) == 0.0)  # 4-var defining eqn (only gr/lm)
    m.subject_to(w - y * p == 0.0)  # y used nonlinearly here
    m.minimize(w**2 + a + b + c + p)
    return m


def test_compare_methods_row_shape():
    rows = compare_methods(_study_model())
    assert rows[0]["method"] == "original"
    assert rows[0]["n_eliminated"] == 0
    assert {r["method"] for r in rows[1:]} == {"ld1", "ecd2", "ld2", "d2", "gr", "lm"}
    for r in rows:
        assert {"n_variables", "n_constraints", "nnz_jacobian", "nnz_hessian"} <= set(r)


def test_approximate_max_eliminates_at_least_structure_preserving():
    rows = {r["method"]: r for r in compare_methods(_study_model())}
    assert rows["gr"]["n_eliminated"] >= rows["d2"]["n_eliminated"]
    assert rows["lm"]["n_eliminated"] >= rows["ld1"]["n_eliminated"]
    # gr/lm can eliminate the 4-variable-defined y; d2 cannot.
    assert rows["gr"]["n_eliminated"] > rows["ld1"]["n_eliminated"]


def test_variable_count_drops_by_number_eliminated():
    rows = {r["method"]: r for r in compare_methods(_study_model())}
    base_vars = rows["original"]["n_variables"]
    for meth, r in rows.items():
        assert r["n_variables"] == base_vars - r["n_eliminated"]


def test_structure_preserving_does_not_increase_total_jacobian_nnz():
    # Substituting a <=2-variable defining equation never adds variables to any
    # constraint, and the defining equality is removed -- so the total number of
    # Jacobian incidences cannot increase. (The *average* per constraint can rise
    # because a sparse defining constraint was dropped; that is expected.)
    rows = {r["method"]: r for r in compare_methods(_study_model())}
    base = rows["original"]["nnz_jacobian"]
    for meth in ("ld1", "ecd2", "ld2", "d2"):
        assert rows[meth]["nnz_jacobian"] <= base


def test_approximate_max_can_increase_jacobian_density():
    # Aggressive aggregation substitutes multi-variable expressions into other
    # constraints, which *can* increase the total Jacobian incidences -- the
    # density inflation the paper describes -- unlike the structure-preserving
    # methods on this model.
    rows = {r["method"]: r for r in compare_methods(_study_model())}
    assert rows["gr"]["nnz_jacobian"] > rows["d2"]["nnz_jacobian"]
    assert rows["lm"]["nnz_jacobian"] > rows["ld1"]["nnz_jacobian"]


def test_compare_methods_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown"):
        compare_methods(_study_model(), methods=["nope"])


def test_compare_methods_accepts_degree_k():
    """T2: compare_methods accepts d<k> methods, not only the six named ones."""
    rows = compare_methods(_study_model(), methods=["d2", "d3"])
    assert [r["method"] for r in rows] == ["original", "d2", "d3"]


# --------------------------------------------------------------------------- #
# exact Hessian nnz (F4)                                                      #
# --------------------------------------------------------------------------- #
def test_exact_hessian_none_unless_requested():
    m = dm.Model("h")
    x = m.continuous("x", lb=-5, ub=5)
    m.subject_to(x**2 == 1.0)
    m.minimize(x)
    assert model_structure(m).nnz_hessian_exact is None


@pytest.mark.smoke
def test_exact_hessian_bilinear_is_subset_of_structural():
    # x*y couples only the (x, y) off-diagonal exactly, but the structural
    # proxy counts {x}, {y}, {x, y} = 3.
    m = dm.Model("xy")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.subject_to(x * y == 1.0)
    m.minimize(x + y)
    s = model_structure(m, exact_hessian=True)
    assert s.nnz_hessian == 3
    assert s.nnz_hessian_exact == 1
    assert s.nnz_hessian_exact <= s.nnz_hessian


@pytest.mark.smoke
def test_exact_hessian_separable_quadratic():
    # a^2 + b^2 -> two diagonal entries, no coupling.
    m = dm.Model("sep")
    a = m.continuous("a", lb=-5, ub=5)
    b = m.continuous("b", lb=-5, ub=5)
    m.subject_to(a + b >= 1.0)
    m.minimize(a**2 + b**2)
    assert exact_hessian_nnz(m) == 2


@pytest.mark.smoke
def test_exact_hessian_never_exceeds_structural_across_methods():
    rows = compare_methods(_study_model(), exact_hessian=True)
    for r in rows:
        assert r["nnz_hessian_exact"] is not None
        assert r["nnz_hessian_exact"] <= r["nnz_hessian"]


def test_compare_methods_original_row_scalarized_granularity():
    """T3: the original row is scalarized, matching aggregated rows' granularity."""
    import discopt.modeling as dm

    from discopt.aggregation.scalarize import scalarize_model

    # a small array model (2 scalars) with an eliminable intermediate
    m = dm.Model("t3")
    c = m.continuous("c", shape=(2,), lb=-9, ub=9)
    s = m.continuous("s", lb=-9, ub=9)
    m.subject_to(s - (c[0] + c[1]) == 0.0)
    m.minimize(c[0] ** 2 + c[1] ** 2 + s**2)
    rows = compare_methods(m, methods=["d2"])
    scal, _ = scalarize_model(m)
    scal_nvars = sum(int(getattr(v, "size", 1)) for v in scal._variables)
    assert rows[0]["method"] == "original"
    assert rows[0]["n_variables"] == scal_nvars  # not the array-count of 2
