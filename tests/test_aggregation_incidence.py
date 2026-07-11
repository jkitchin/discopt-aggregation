"""Tests for :mod:`discopt.aggregation.incidence`.

Validates the bipartite incidence construction against hand-built models with
known structure: candidate selection (scalar continuous only), edge sets, and
linear vs nonlinear edge labelling with correct constant coefficients.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.incidence import (
    all_vars,
    build_incidence,
    candidate_variables,
    linear_vars,
)
from discopt.aggregation.matching import maximum_matching

pytestmark = pytest.mark.unit


def _vars(m):
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    z = m.continuous("z", lb=-10, ub=10)
    return x, y, z


# --------------------------------------------------------------------------- #
# candidate selection                                                         #
# --------------------------------------------------------------------------- #
def test_candidates_are_scalar_continuous_only():
    m = dm.Model("t")
    x, y, _z = _vars(m)
    m.integer("k", lb=0, ub=5)
    m.binary("b")
    cands = candidate_variables(m)
    assert set(cands) == {"x", "y", "z"}
    assert cands["x"] is x


def test_array_variables_excluded_from_candidates():
    m = dm.Model("t")
    m.continuous("x", lb=-1, ub=1)
    m.continuous("a", shape=(3,), lb=-1, ub=1)  # non-scalar -> excluded
    cands = candidate_variables(m)
    assert set(cands) == {"x"}


# --------------------------------------------------------------------------- #
# all_vars / linear_vars                                                       #
# --------------------------------------------------------------------------- #
def test_all_and_linear_vars_affine_constraint():
    m = dm.Model("t")
    x, y, _z = _vars(m)
    m.subject_to(x + 2 * y == 1)
    c = m._constraints[0]
    names = {"x", "y", "z"}
    assert all_vars(c, names) == {"x", "y"}
    lv = linear_vars(c, names)
    assert lv == {"x": 1.0, "y": 2.0}


def test_linear_vars_excludes_nonlinear_occurrences():
    m = dm.Model("t")
    x, y, z = _vars(m)
    m.subject_to(x * z + y == 0)  # x,z nonlinear (product); y linear
    c = m._constraints[0]
    names = {"x", "y", "z"}
    assert all_vars(c, names) == {"x", "y", "z"}
    assert linear_vars(c, names) == {"y": 1.0}


def test_linear_vars_power_is_nonlinear():
    m = dm.Model("t")
    x, y, _z = _vars(m)
    m.subject_to(y + x**2 == 0)  # y linear, x under a power -> nonlinear
    c = m._constraints[0]
    names = {"x", "y", "z"}
    assert linear_vars(c, names) == {"y": 1.0}


# --------------------------------------------------------------------------- #
# build_incidence                                                             #
# --------------------------------------------------------------------------- #
def test_build_incidence_edges_and_labels():
    m = dm.Model("t")
    x, y, z = _vars(m)
    m.subject_to(x + 2 * y == 1)  # c0: x,y linear
    m.subject_to(x * z == 1)  # c1: x,z nonlinear
    m.subject_to(y + z**2 == 0)  # c2: y linear, z nonlinear
    m.subject_to(x + y <= 5)  # inequality -> ignored
    m.minimize(x + y + z)

    inc = build_incidence(m)

    # Only equality constraints become left nodes.
    assert set(inc.constraints) == {0, 1, 2}

    assert inc.constraint_vars(0) == {"x", "y"}
    assert inc.constraint_vars(1) == {"x", "z"}
    assert inc.constraint_vars(2) == {"y", "z"}

    # Linear labelling.
    assert inc.constraint_linear_vars(0) == {"x", "y"}
    assert inc.constraint_linear_vars(1) == set()
    assert inc.constraint_linear_vars(2) == {"y"}

    assert inc.is_linear_edge(0, "x") and inc.is_linear_edge(0, "y")
    assert not inc.is_linear_edge(1, "x")
    assert inc.is_linear_edge(2, "y") and not inc.is_linear_edge(2, "z")

    # Coefficients on the linear edges.
    assert inc.linear_coeff[(0, "x")] == 1.0
    assert inc.linear_coeff[(0, "y")] == 2.0
    assert inc.linear_coeff[(2, "y")] == 1.0


def test_incidence_ignores_inequalities_only_model():
    m = dm.Model("t")
    x, y, _z = _vars(m)
    m.subject_to(x + y <= 5)
    m.subject_to(x - y >= -3)
    inc = build_incidence(m)
    assert inc.constraints == {}
    assert inc.graph.edges == set()


def test_incidence_no_candidate_variables():
    m = dm.Model("t")
    a = m.continuous("a", shape=(2,), lb=-1, ub=1)
    m.subject_to(a[0] + a[1] == 1)  # references only a non-scalar variable
    inc = build_incidence(m)
    # No scalar candidates -> the constraint contributes no edges.
    assert inc.variables == {}
    assert inc.graph.edges == set()


# --------------------------------------------------------------------------- #
# integration with the matching kernel                                        #
# --------------------------------------------------------------------------- #
def test_linear_graph_feeds_maximum_matching():
    m = dm.Model("t")
    x, y, z = _vars(m)
    m.subject_to(x + y + z == 1)  # all linear
    m.subject_to(y + z == 2)  # linear
    inc = build_incidence(m)
    match = maximum_matching(inc.linear_graph)
    # Two equalities, three linear variables -> both constraints matched.
    assert len(match) == 2
    for cid, var in match.items():
        assert var in inc.constraint_linear_vars(cid)


def test_customcall_blocks_dangling_reference():
    """C3: a variable inside an opaque CustomCall is seen and substituted, not dropped.

    ``y`` is defined by ``y == x^2+1`` and also appears inside a CustomCall in
    another constraint.  Eliminating ``y`` must substitute its definition into
    the CustomCall (no dangling reference to the deleted variable).
    """
    import discopt.modeling.core as core

    from discopt.aggregation.api import aggregate
    from discopt.aggregation.exprwalk import collect_variable_names

    m = dm.Model("customcall")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    m.subject_to(y - (x**2 + 1.0) == 0.0)
    m.subject_to(core.CustomCall(lambda a, b: a * b, y, z) - 4.0 == 0.0)
    m.minimize(x**2 + z**2)
    agg = aggregate(m, method="lm")
    for c in agg.reduced_model._constraints:
        names, analyzable = collect_variable_names(c.body)
        assert analyzable
        assert "y" not in names, "eliminated variable left dangling inside CustomCall"


def test_unrecognised_node_refuses():
    """C3: a genuinely unrecognised node makes incidence refuse rather than guess."""
    import pytest

    from discopt.aggregation.incidence import build_incidence

    class _Weird:
        # not an Expression the walker knows; child_exprs -> None
        pass

    m = dm.Model("weird")
    x = m.continuous("x")
    y = m.continuous("y")
    m.subject_to(y - x == 0.0)
    m.minimize(x**2)
    # Inject an opaque body discopt's walker cannot analyse.
    m._constraints.append(type(m._constraints[0])(_Weird(), "=="))
    with pytest.raises(NotImplementedError):
        build_incidence(m)


# --------------------------------------------------------------------------- #
# SC1: DAG-memoised affine-coefficient extractor                              #
# --------------------------------------------------------------------------- #
#
# ``linear_vars`` used to call discopt's ``_isolated_affine_coeff`` once per
# candidate variable, and that walker is O(tree): on a body whose *tree*
# expansion is huge but whose shared DAG is small (exactly what aggressive
# substitution produces), a single call took tens of seconds.  SC1 routes
# ``linear_vars`` (and ``strategies._constraint_is_linear``) through the
# DAG-memoised ``exprwalk.affine_coefficients`` — one O(#distinct nodes) pass
# that extracts every variable's coefficient at once.


def _shared_dag_body(m, depth):
    """A body whose tree expansion is ``2**depth`` but whose DAG is ``O(depth)``.

    ``e0 = x + 1`` then ``e_{k} = e_{k-1} + e_{k-1}`` reuses the *same* node on
    both sides of each ``+``, so the DAG grows by one node per doubling while the
    tree doubles.  ``x`` therefore appears ``2**depth`` times, each affinely with
    coefficient 1, so its summed affine coefficient is exactly ``2**depth``.
    """
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    e = x + 1.0
    for _ in range(depth):
        e = e + e  # shared node: tree doubles, DAG grows by one
    return x, y, y + e - 3.0


def test_linear_vars_scales_on_shared_dags():
    """SC1 repro: linear_vars on a tree>>dag body is fast and correct.

    Pre-SC1 (per-candidate ``_isolated_affine_coeff``) this walked the ~1M-node
    *tree* and took ~10 s at depth 20; post-SC1 the single DAG-memoised pass
    returns in well under a second.  We also assert the coefficients so the speed
    win cannot come from dropping the analysis.
    """
    import time

    from discopt.modeling.core import Constraint

    depth = 20
    m = dm.Model("shared")
    _x, _y, body = _shared_dag_body(m, depth)
    c = Constraint(body=body, sense="==")

    t0 = time.perf_counter()
    lv = linear_vars(c, {"x", "y"})
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"linear_vars took {elapsed:.2f}s on a shared-DAG body"
    # x occurs 2**depth times, each affine with coeff 1; y once with coeff 1.
    assert lv["x"] == pytest.approx(float(2**depth))
    assert lv["y"] == pytest.approx(1.0)


def test_build_incidence_scales_on_shared_dag_equality():
    """The full build_incidence path stays fast when an equality body is tree>>dag."""
    import time

    depth = 20
    m = dm.Model("shared2")
    _x, _y, body = _shared_dag_body(m, depth)
    m.subject_to(body == 0.0)
    m.minimize(m._variables[0] ** 2)

    t0 = time.perf_counter()
    inc = build_incidence(m)
    elapsed = time.perf_counter() - t0

    assert elapsed < 2.0, f"build_incidence took {elapsed:.2f}s on a shared-DAG body"
    # The single equality is analysed and x reads linear (its 2**depth coefficient
    # dominates, so it survives the relative-pivot filter; y's coefficient of 1 is
    # correctly filtered out as ill-conditioned relative to it).
    (cid,) = inc.constraints
    assert inc.is_linear_edge(cid, "x")


def test_affine_coefficients_constant_term_and_signs():
    """Constant-term convention (needed by IA1): body as written."""
    from discopt.aggregation.exprwalk import affine_coefficients

    m = dm.Model("ct")
    x = m.continuous("x")
    y = m.continuous("y")
    const, coeffs = affine_coefficients(3 * x + 2 * y - 7)
    assert const == pytest.approx(-7.0)
    assert coeffs["x"] == pytest.approx(3.0)
    assert coeffs["y"] == pytest.approx(2.0)


def test_sum_over_scalar_operand_stays_affine():
    """A ``sum`` whose operand is provably scalar is the identity: coefficients
    pass through unchanged (regression guard for the RC1 decline not over-firing)."""
    from discopt.aggregation.exprwalk import affine_coefficients, classify_linearity

    m = dm.Model("sc")
    y = m.continuous("y")
    const, coeffs = affine_coefficients(dm.sum(2 * y + 1))
    assert const == pytest.approx(1.0)
    assert coeffs["y"] == pytest.approx(2.0)
    assert classify_linearity(dm.sum(2 * y + 1)) == {"y": True}


def test_sum_over_array_operand_declines_scalar_coefficient():
    """RC1: a scalar variable broadcast against an array *inside* a ``sum`` has no
    single affine coefficient this walk can certify (discopt evaluates the sum as
    ``2*y + sum(x)``, its shape inference reports the operand array-shaped, and a
    naive per-element read gives yet a third value).  The walk must therefore
    decline — report ``y`` nonlinear — rather than emit the un-summed per-element
    coefficient, which previously drove an unsound elimination."""
    from discopt.aggregation.exprwalk import affine_coefficients, classify_linearity

    m = dm.Model("arr")
    y = m.continuous("y")
    x = m.continuous("x", shape=(3,), lb=1, ub=1)
    _const, coeffs = affine_coefficients(dm.sum(2 * y + x))
    assert coeffs.get("y") is None  # declined, never the pre-fix 2.0
    assert classify_linearity(dm.sum(2 * y + x))["y"] is False


def test_affine_coefficients_hand_models_match_oracle():
    """Hand bodies: affine_coefficients agrees with _isolated_affine_coeff."""
    from discopt._jax.dependent_vars import _isolated_affine_coeff

    from discopt.aggregation.exprwalk import affine_coefficients, collect_variable_names

    m = dm.Model("hand")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    bodies = [
        x + 2 * y - 3,
        x * y - z,
        x**2 + y,
        x / z,
        2 * y + 3 * y,
        y + y**3,  # y must read nonlinear (None)
        3 * x + 2 * y - 7,
        x - z + 5.0,
        (x + 1.0) * 2.0 + y,
        x * 0.0 + y,  # x has coefficient 0 -> still affine (0.0), not None
    ]
    for body in bodies:
        _, coeffs = affine_coefficients(body)
        names, _ = collect_variable_names(body)
        for nm in names:
            iso = _isolated_affine_coeff(body, nm)
            mine = coeffs.get(nm, 0.0)
            if iso is None:
                assert mine is None, f"{nm} in {body}: oracle None, got {mine}"
            else:
                assert mine is not None and mine == pytest.approx(float(iso), abs=1e-12)
    # Spot-check the nonlinear reads explicitly.
    assert affine_coefficients(y + y**3)[1]["y"] is None
    assert affine_coefficients(x * y - z)[1]["x"] is None


def test_affine_coefficients_matches_oracle_on_problem_corpus():
    """Oracle equivalence over PROBLEMS + ld1/ecd2/ld2/d2 reduced models.

    Mirrors the 0.3.0 classify_linearity validation harness: for every constraint
    body of every registered problem and each of its structure-preserving reduced
    models, affine_coefficients must agree with discopt's _isolated_affine_coeff
    per (body, variable) — coefficient to 1e-12 when affine, None-agreement
    otherwise.  Target >= 400 checks, zero mismatches.
    """
    from discopt._jax.dependent_vars import _isolated_affine_coeff
    from discopt.modeling.core import Constraint

    from discopt.aggregation.api import aggregate
    from discopt.aggregation.exprwalk import affine_coefficients, collect_variable_names
    from discopt.aggregation.testproblems import PROBLEMS

    checks = 0
    mismatches = 0

    def compare(body):
        nonlocal checks, mismatches
        extracted = affine_coefficients(body)
        names, _ = collect_variable_names(body)
        for nm in names:
            iso = _isolated_affine_coeff(body, nm)
            mine = None if extracted is None else extracted[1].get(nm, 0.0)
            checks += 1
            ok = (iso is None and mine is None) or (
                iso is not None and mine is not None and abs(float(iso) - float(mine)) < 1e-12
            )
            if not ok:
                mismatches += 1

    for problem in PROBLEMS.values():
        model = problem.build()
        for c in model._constraints:
            if isinstance(c, Constraint):
                compare(c.body)
        for method in ("ld1", "ecd2", "ld2", "d2"):
            reduced = aggregate(model, method=method).reduced_model
            for c in reduced._constraints:
                if isinstance(c, Constraint):
                    compare(c.body)

    assert checks >= 400, f"only {checks} checks (need >= 400)"
    assert mismatches == 0, f"{mismatches}/{checks} oracle mismatches"


def test_aggregate_roundtrip_correctness_after_extractor_swap():
    """End-to-end: the extractor swap preserves elimination correctness.

    A wrong coefficient would corrupt the defining substitution and break the
    round-trip; solving the reduced model must still reproduce the original
    objective.
    """
    from discopt.aggregation.api import aggregate
    from discopt.aggregation.benchmark import _local_solve
    from discopt.aggregation.testproblems import cstr_dynamic

    model = cstr_dynamic(n=6)
    reduced = aggregate(model, method="lm").reduced_model
    _, _, _, obj_original = _local_solve(model)
    _, _, _, obj_reduced = _local_solve(reduced)
    assert obj_reduced == pytest.approx(obj_original, abs=1e-4)
