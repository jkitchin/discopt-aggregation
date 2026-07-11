"""IA1 — integer candidate admission and the integrality-implication filter.

Everything here is opt-in behind ``integer_aggregation=True``.  With the flag off
(the default) integer/binary variables are never aggregation candidates, so the
default behaviour is byte-identical to the continuous-only pipeline.

The soundness rule (standard MIP presolve; Achterberg et al. 2020 §4): an integer
variable ``y`` may be eliminated via ``coeff*y + rest = 0`` only when integrality
of ``y`` is *implied* — the constraint is fully affine, the pivot ``coeff == ±1``,
every other participating variable is integer/binary, and every other coefficient
and the constant term are integral.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

from discopt.aggregation.api import aggregate, solve
from discopt.aggregation.incidence import build_incidence
from discopt.aggregation.testproblems import PROBLEMS


def _single_constraint_incidence(build, integer_aggregation):
    """Build a one-equality model with ``build(m)`` and return (inc, cid)."""
    m = dm.Model("t")
    build(m)
    inc = build_incidence(m, integer_aggregation=integer_aggregation)
    # the sole equality constraint
    (cid,) = list(inc.constraints)
    return inc, cid


# --------------------------------------------------------------------------- #
# candidate admission / edge filter                                            #
# --------------------------------------------------------------------------- #
def test_integer_pair_eliminated_only_with_flag():
    """``j = i + 3`` (both integer): eliminable with the flag, inert without."""

    def build(m):
        i = m.integer("i", lb=0, ub=10)
        j = m.integer("j", lb=0, ub=20)
        m.subject_to(j - (i + 3) == 0.0)
        m.minimize((i - 4) ** 2)

    m_off = dm.Model("off")
    build(m_off)
    off = aggregate(m_off, method="d2", integer_aggregation=False)
    assert off.n_eliminated == 0

    m_on = dm.Model("on")
    build(m_on)
    on = aggregate(m_on, method="d2", integer_aggregation=True)
    assert on.n_eliminated == 1
    assert set(on.eliminated) <= {"i", "j"}


def test_integer_pair_is_a_linear_edge_with_flag():
    """Both integer vars of ``j = i + 3`` are admitted linear edges with the flag."""

    def build(m):
        m.integer("i", lb=0, ub=10)
        m.integer("j", lb=0, ub=20)
        m.subject_to(m._variables[1] - (m._variables[0] + 3) == 0.0)

    inc, cid = _single_constraint_incidence(build, integer_aggregation=True)
    assert inc.constraint_linear_vars(cid) == {"i", "j"}
    assert inc.integer_names == frozenset({"i", "j"})


def test_flag_off_has_no_integer_candidates():
    """Flag off: integer vars are not candidates, so the constraint drops out."""

    def build(m):
        m.integer("i", lb=0, ub=10)
        m.integer("j", lb=0, ub=20)
        m.subject_to(m._variables[1] - (m._variables[0] + 3) == 0.0)

    m = dm.Model("off")
    build(m)
    inc = build_incidence(m, integer_aggregation=False)
    # no candidate variables participate -> no incidence constraints at all
    assert dict(inc.constraints) == {}
    # integer_names is still tracked regardless of the flag
    assert inc.integer_names == frozenset({"i", "j"})


def test_reject_continuous_in_integer_definition():
    """``y = x + i`` with continuous ``x``: integer ``y`` (and ``i``) not admitted."""

    def build(m):
        m.integer("y", lb=0, ub=20)
        m.continuous("x", lb=-10, ub=10)
        m.integer("i", lb=0, ub=10)
        m.subject_to(m._variables[0] - (m._variables[1] + m._variables[2]) == 0.0)

    inc, cid = _single_constraint_incidence(build, integer_aggregation=True)
    lin = inc.constraint_linear_vars(cid)
    # the continuous x participates linearly (continuous rule unchanged); the
    # integer y and i are rejected because a continuous var breaks integrality.
    assert "y" not in lin
    assert "i" not in lin
    assert "x" in lin


def test_pivot_must_be_unit_for_integer():
    """``y = 2*i``: ``y`` admissible (pivot 1); eliminating ``i`` rejected (pivot 2)."""

    def build(m):
        m.integer("y", lb=0, ub=40)
        m.integer("i", lb=0, ub=10)
        m.subject_to(m._variables[0] - 2 * m._variables[1] == 0.0)

    inc, cid = _single_constraint_incidence(build, integer_aggregation=True)
    lin = inc.constraint_linear_vars(cid)
    assert "y" in lin  # pivot on y is 1, other coeff 2 integral -> admissible
    assert "i" not in lin  # pivot on i is 2 != ±1 -> rejected


def test_reject_fractional_constant():
    """``y = i + 0.5``: non-integral constant rejects every integer candidate."""

    def build(m):
        m.integer("y", lb=0, ub=20)
        m.integer("i", lb=0, ub=10)
        m.subject_to(m._variables[0] - (m._variables[1] + 0.5) == 0.0)

    inc, cid = _single_constraint_incidence(build, integer_aggregation=True)
    assert inc.constraint_linear_vars(cid) == set()


def test_reject_fractional_coefficient():
    """``y = 0.5*i + z``: a non-integral coefficient rejects every integer candidate."""

    def build(m):
        m.integer("y", lb=0, ub=20)
        m.integer("i", lb=0, ub=10)
        m.integer("z", lb=0, ub=10)
        m.subject_to(m._variables[0] - (0.5 * m._variables[1] + m._variables[2]) == 0.0)

    inc, cid = _single_constraint_incidence(build, integer_aggregation=True)
    assert inc.constraint_linear_vars(cid) == set()


def test_binary_sum_admissible():
    """``b = i1 + i2`` (binary ``b``, integers ``i1,i2``): integrality implied."""

    def build(m):
        m.binary("b")
        m.integer("i1", lb=0, ub=5)
        m.integer("i2", lb=0, ub=5)
        m.subject_to(m._variables[0] - (m._variables[1] + m._variables[2]) == 0.0)

    inc, cid = _single_constraint_incidence(build, integer_aggregation=True)
    lin = inc.constraint_linear_vars(cid)
    assert {"b", "i1", "i2"} <= lin
    assert "b" in inc.integer_names


def test_nonlinear_body_never_admits_integer():
    """``y = x**2`` (integer ``y``): a nonlinear body fails the affine requirement.

    The d2 'nonlinear f(x)' case admits a *continuous* target, but an integer
    target requires a fully-affine body, so integer ``y`` is never admitted.
    """

    def build_int(m):
        m.integer("y", lb=0, ub=100)
        m.continuous("x", lb=-10, ub=10)
        m.subject_to(m._variables[0] - m._variables[1] ** 2 == 0.0)

    inc, cid = _single_constraint_incidence(build_int, integer_aggregation=True)
    assert "y" not in inc.constraint_linear_vars(cid)

    # sanity: the *continuous* analogue does admit the target on the same body.
    def build_cont(m):
        m.continuous("y", lb=0, ub=100)
        m.continuous("x", lb=-10, ub=10)
        m.subject_to(m._variables[0] - m._variables[1] ** 2 == 0.0)

    inc_c, cid_c = _single_constraint_incidence(build_cont, integer_aggregation=True)
    assert "y" in inc_c.constraint_linear_vars(cid_c)


def test_integer_pair_recovers_via_defining_expression():
    """Recovery reconstructs the eliminated integer from the kept one."""

    def build(m):
        i = m.integer("i", lb=0, ub=10)
        j = m.integer("j", lb=0, ub=20)
        m.subject_to(j - (i + 3) == 0.0)
        m.minimize((i - 4) ** 2)

    m = dm.Model("rt")
    build(m)
    res = aggregate(m, method="d2", integer_aggregation=True)
    (elim,) = res.eliminated
    kept = "i" if elim == "j" else "j"
    kept_val = 4 if kept == "i" else 7
    full = res.recover_full({kept: kept_val})
    assert full["i"] == 4
    assert full["j"] == 7


# --------------------------------------------------------------------------- #
# no-regression: the flag must not perturb the continuous corpus               #
# --------------------------------------------------------------------------- #
def test_flag_does_not_change_continuous_corpus():
    """On the PROBLEMS corpus the flag leaves eliminated lists untouched.

    The continuous-only problems have no integer candidates, so the flag is a
    no-op; ``unit_selection``'s binaries appear only in inequalities (never in a
    defining equality), so they stay untouched even with the flag ON — an
    adversarial check that binaries are not spuriously eliminated.
    ``inventory_chain`` is excluded: its integers appear in defining equalities,
    so the flag is *supposed* to change its result (that is IA5's whole point).
    """
    for name, problem in PROBLEMS.items():
        if name == "inventory_chain":
            continue
        m = problem.build()
        for method in ("ld1", "d2", "lm"):
            off = aggregate(m, method=method, integer_aggregation=False).eliminated
            on = aggregate(m, method=method, integer_aggregation=True).eliminated
            assert off == on, f"{name}/{method}: flag changed eliminated list"


def test_solve_accepts_integer_aggregation_kwarg():
    """T1 parity: solve() exposes the integer_aggregation knob."""
    import inspect

    from discopt.aggregation.api import solve

    assert "integer_aggregation" in inspect.signature(solve).parameters


# --------------------------------------------------------------------------- #
# IA2 — fixed integer variables and integral-infeasibility                     #
# --------------------------------------------------------------------------- #
def _fixed_integer_model(name, const):
    """``i == const`` (integer i in [0,10]) with a continuous x tied to i."""
    m = dm.Model(name)
    x = m.continuous("x", lb=-10.0, ub=10.0)
    i = m.integer("i", lb=0, ub=10)
    m.subject_to(i - float(const) == 0.0)
    m.minimize((x - i) ** 2)
    return m


def test_fixed_integer_eliminated_and_integral():
    """IA2: ``i == 3`` (integer) is eliminated, feasible, and recovers exactly 3."""
    from discopt.aggregation.api import solve

    m = _fixed_integer_model("fix_int", 3)
    agg = aggregate(m, method="ld1", integer_aggregation=True)
    assert "i" in agg.eliminated
    assert not agg.infeasible_eliminations
    res = solve(m, method="ld1", integer_aggregation=True)
    assert res.status != "infeasible"
    assert res.x["i"] == 3


def test_fixed_integer_fractional_is_infeasible():
    """IA2: ``i == 2.5`` (integer) is integral-infeasible with the flag on.

    The edge filter admits the lone fixed integer, the transform records the
    fractional value as an infeasible elimination, and ``solve`` reports it —
    rather than silently substituting a non-integral value into the model.
    """
    import warnings

    from discopt.aggregation.api import solve

    m = _fixed_integer_model("infeas_int", 2.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agg = aggregate(m, method="ld1", integer_aggregation=True)
    assert [v for v, *_ in agg.infeasible_eliminations] == ["i"]
    assert any("infeasible" in str(w.message) for w in caught)

    res = solve(m, method="ld1", integer_aggregation=True)
    assert res.status == "infeasible"


def test_fixed_integer_fractional_all_eliminated_reports_infeasible():
    """RC5: when a non-integral integer fix eliminates *every* variable, solve()
    must report infeasible, not crash.  The infeasible early-return previously
    called recover_full({}), and the recovery map asserts integrality on exactly
    this violation, so it raised AssertionError instead of returning a status."""
    from discopt.aggregation.api import solve

    m = dm.Model("all_elim_infeas")
    i = m.integer("i", lb=0, ub=10)
    m.subject_to(i == 2.5)  # fixes i non-integrally; i is the only variable
    m.minimize(i)

    res = solve(m, integer_aggregation=True)  # must not raise
    assert res.status == "infeasible"
    assert res.x == {}  # nothing feasible to recover on the infeasible branch


def test_partial_elimination_nonintegral_integer_reports_infeasible():
    """RC5 variant: a non-integral integer fix alongside a surviving variable is
    reported infeasible without crashing."""
    from discopt.aggregation.api import solve

    m = dm.Model("partial_infeas")
    i = m.integer("i", lb=0, ub=10)
    x = m.continuous("x", lb=-5.0, ub=5.0)
    m.subject_to(i == 2.5)
    m.subject_to(x >= 1.0)
    m.minimize(x**2 + i)

    res = solve(m, integer_aggregation=True)
    assert res.status == "infeasible"
    assert res.x == {}


def test_fixed_integer_out_of_bounds_is_infeasible():
    """IA2: ``i == 15`` (integral but outside [0,10]) is infeasible via the box check."""
    m = _fixed_integer_model("oob_int", 15)
    agg = aggregate(m, method="ld1", integer_aggregation=True)
    assert [v for v, *_ in agg.infeasible_eliminations] == ["i"]


def test_fixed_integer_flag_off_is_solver_job():
    """IA2: with the flag off an integer fixed to a fractional value is inert.

    ``i`` is not a candidate, so nothing is eliminated and no infeasibility is
    flagged — the non-integral fix is the solver's infeasibility to find, exactly
    as in 0.3.0.  Freezing this guarantees the new behaviour is opt-in.
    """
    for const in (3, 2.5, 15):
        m = _fixed_integer_model("off_int", const)
        agg = aggregate(m, method="ld1", integer_aggregation=False)
        assert agg.n_eliminated == 0
        # and no infeasibility is manufactured: with the flag off the integer is
        # never a candidate, so even a fractional fix is left for the solver.
        assert not agg.infeasible_eliminations


# --------------------------------------------------------------------------- #
# IA3 — integer-aware box->inequality conversion and pruning                   #
# --------------------------------------------------------------------------- #
def _sum_of_integers_model(a_lb, a_ub, b_ub=10, c_ub=10, a_type="integer"):
    """``a = b + c`` with ``a`` sorting first so ``gr`` eliminates it.

    ``a`` carries the (possibly fractional) box ``[a_lb, a_ub]``; ``b``/``c`` are
    integers in ``[0, b_ub]`` / ``[0, c_ub]``.  Returns the model plus the eliminated
    variable's defining expression is ``b + c``.
    """
    m = dm.Model("ia3")
    if a_type == "integer":
        m.integer("a", lb=a_lb, ub=a_ub)
    else:
        m.continuous("a", lb=a_lb, ub=a_ub)
    m.integer("b", lb=0, ub=b_ub)
    m.integer("c", lb=0, ub=c_ub)
    m.subject_to(m._variables[0] - (m._variables[1] + m._variables[2]) == 0.0)
    m.minimize((m._variables[1] - 1) ** 2 + (m._variables[2] - 1) ** 2 + m._variables[0])
    return m


def _apply(m, method="gr", integer_aggregation=True, prune_bounds=True):
    """Build integer-aware incidence, pick an aggregation set, apply it."""
    from discopt.aggregation.strategies import aggregation_set
    from discopt.aggregation.transform import apply_aggregation

    inc = build_incidence(m, integer_aggregation=integer_aggregation)
    agg = aggregation_set(inc, method=method)
    return agg, apply_aggregation(m, agg, prune_bounds=prune_bounds)


def _bound_constants(rm):
    """Map ``('bound_lo'|'bound_hi', name) -> numeric bound`` for a reduced model.

    Each box->inequality body is affine in the kept vars; evaluated at the kept
    origin it equals ``lb`` (for a ``lo`` bound, body ``lb - d``) or ``-ub`` (for a
    ``hi`` bound, body ``d - ub``), so we recover the compared bound directly.
    """
    from discopt.aggregation.recovery import evaluate_expression

    zeros = {v.name: 0.0 for v in rm.model._variables}
    out = {}
    for c, orig in zip(rm.model._constraints, rm.constraint_origin):
        if isinstance(orig, tuple) and orig[0] in ("bound_lo", "bound_hi"):
            val = float(evaluate_expression(c.body, zeros))
            out[orig] = val if orig[0] == "bound_lo" else -val
    return out


def test_integer_box_tightened_to_grid():
    """IA3 core repro: ``a in [-0.5, 4.5]`` integer -> constraints use ``[0, 4]``.

    Pre-fix the box->inequalities used the raw fractional bounds ``[-0.5, 4.5]``;
    the integer-aware conversion emits ``d >= 0`` and ``d <= 4`` instead.
    """
    m = _sum_of_integers_model(-0.5, 4.5)
    agg, rm = _apply(m)
    assert agg.variables() == ["a"]  # a sorts first, gr eliminates it
    consts = _bound_constants(rm)
    assert consts[("bound_lo", "a")] == 0.0  # ceil(-0.5) == 0
    assert consts[("bound_hi", "a")] == 4.0  # floor(4.5) == 4


def test_continuous_box_not_tightened():
    """Regression freeze: a *continuous* eliminated var keeps its fractional box.

    The tightening is integer-only, so the same ``[-0.5, 4.5]`` box on a continuous
    ``a`` still yields ``d >= -0.5`` / ``d <= 4.5`` — no behaviour change off the
    integer path.
    """
    m = _sum_of_integers_model(-0.5, 4.5, a_type="continuous")
    _agg, rm = _apply(m, integer_aggregation=False, prune_bounds=False)
    consts = _bound_constants(rm)
    assert consts[("bound_lo", "a")] == -0.5
    assert consts[("bound_hi", "a")] == 4.5


def test_integer_tightening_keeps_bounds_pruning_would_drop():
    """Tightening strengthens the reduced model: integer bounds bind at the grid.

    ``d = b + c`` with ``b, c in [0, 2]`` ranges over ``[0, 4]``.  With the raw
    fractional box ``[-0.5, 4.5]`` (continuous ``a``) both box->inequalities are
    provably redundant and pruned.  For an *integer* ``a`` the box tightens to the
    binding ``[0, 4]``, so both bounds now bind and are emitted (at ``0`` and ``4``)
    rather than pruned — a strictly stronger reduced feasible region.
    """
    cont = _sum_of_integers_model(-0.5, 4.5, b_ub=2, c_ub=2, a_type="continuous")
    _agg, crm = _apply(cont, integer_aggregation=False, prune_bounds=True)
    assert crm.n_bound_pruned == 2  # raw [-0.5, 4.5] is loose -> both pruned
    assert crm.n_bound_inequalities == 0

    m = _sum_of_integers_model(-0.5, 4.5, b_ub=2, c_ub=2)
    _agg, rm = _apply(m, prune_bounds=True)
    assert rm.n_bound_pruned == 0  # tightened [0, 4] binds -> nothing to prune
    assert rm.n_bound_inequalities == 2
    consts = _bound_constants(rm)
    assert consts[("bound_lo", "a")] == 0.0
    assert consts[("bound_hi", "a")] == 4.0


def test_wide_integer_box_still_prunes():
    """Pruning still fires for integers: a genuinely-redundant integer bound goes.

    ``a in [-100, 100]`` (integral bounds) with ``d = b + c in [0, 20]`` -> both
    box->inequalities are provably redundant and pruned, exactly as for a
    continuous variable.
    """
    m = _sum_of_integers_model(-100, 100)
    _agg, rm = _apply(m, prune_bounds=True)
    assert rm.n_bound_pruned == 2
    assert rm.n_bound_inequalities == 0


def test_integer_reduced_model_feasible_set_is_sound():
    """G2-style soundness: every original-feasible integer point stays feasible.

    Enumerate every integer ``(b, c)`` whose implied ``a = b + c`` lands in the
    eliminated variable's integer box, and assert the reduced model (which carries
    the *tightened* ``[0, 4]`` bounds) accepts each such kept point — the
    tightening never rejects a genuinely feasible integer assignment.
    """
    from discopt.aggregation.recovery import feasibility_violations

    m = _sum_of_integers_model(-0.5, 4.5)
    _agg, rm = _apply(m, prune_bounds=False)  # keep both bounds to test them
    checked = 0
    for b in range(0, 11):
        for c in range(0, 11):
            a = b + c
            # feasible in the original iff a is an integer inside [-0.5, 4.5]
            if -0.5 <= a <= 4.5:
                point = {"b": float(b), "c": float(c)}
                assert feasibility_violations(rm.model, point) == []
                checked += 1
    assert checked > 0
    # and a point the tightening must reject: a = 5 (b=c but b+c=5) is outside the
    # integer box, so the reduced model flags it infeasible.
    assert feasibility_violations(rm.model, {"b": 5.0, "c": 0.0}) != []


def test_binary_box_tightening_is_identity():
    """Binary eliminated variables already sit on the integer grid ([0, 1])."""
    m = dm.Model("ia3bin")
    m.binary("a")
    m.binary("b")
    m.binary("c")
    m.subject_to(m._variables[0] - (m._variables[1] + m._variables[2]) == 0.0)
    m.minimize(m._variables[1] + m._variables[2] + m._variables[0])
    agg, rm = _apply(m, prune_bounds=False)
    assert agg.variables() == ["a"]
    consts = _bound_constants(rm)
    # a in [0, 1] -> ceil(0)=0, floor(1)=1, unchanged.
    assert consts[("bound_lo", "a")] == 0.0
    assert consts[("bound_hi", "a")] == 1.0


def test_implicit_elimination_skips_integer_block_variables():
    """IA3 (implicit C4 symmetry): integer block variables are out of scope.

    A 2-cycle of *continuous* variables is eliminated by ``eliminate_implicit``;
    the byte-identical model with *integer* block variables has no eliminable
    block at all (integers are never admitted as linear candidates), so the
    integers are skipped rather than mishandled.
    """
    import pytest

    from discopt.aggregation.implicit import eliminate_implicit

    def build(kind):
        m = dm.Model("cyc")
        x = getattr(m, kind)("x", lb=-9, ub=9)
        y = getattr(m, kind)("y", lb=-9, ub=9)
        t = m.continuous("t", lb=-9, ub=9)
        m.subject_to(x - (0.5 * y + 3.0) == 0.0)
        m.subject_to(y - (0.5 * x + 2.0) == 0.0)
        m.minimize((t - 1.0) ** 2 + x**2 + y**2 + t)
        return m

    cont = eliminate_implicit(build("continuous"))
    assert set(cont.eliminated) == {"x", "y"}

    with pytest.raises(ValueError, match="no eliminable"):
        eliminate_implicit(build("integer"))


# --------------------------------------------------------------------------- #
# IA4 — integral recovery + dual scope                                        #
# --------------------------------------------------------------------------- #
def _ia4_model():
    m = dm.Model("ia4")
    i = m.integer("i", lb=0, ub=10)
    j = m.integer("j", lb=0, ub=30)
    x = m.continuous("x", lb=-5, ub=5)
    m.subject_to(j - (i + 3) == 0.0)  # j = i + 3, integrality of one implies the other
    m.minimize((i - 4.0) ** 2 + x**2 + j)
    return m


def test_integer_recovery_is_integral():
    """IA4: the recovered eliminated integer is exactly integral (rounded)."""
    agg = aggregate(_ia4_model(), method="gr", integer_aggregation=True)
    (elim,) = agg.eliminated  # one of i/j (deterministic matching picks one)
    assert elim in ("i", "j")
    assert elim in agg.recovery.integer_vars
    # feed the kept integer a value consistent with j = i + 3 and check integrality
    kept, kept_val = ("i", 4.0) if elim == "j" else ("j", 7.0)
    full = agg.recover_full({kept: kept_val, "x": 0.0})
    assert float(full[elim]).is_integer()
    assert full[elim] == (7.0 if elim == "j" else 4.0)


def test_recovery_raises_on_nonintegral_integer(monkeypatch):
    """IA4: a corrupted (non-integral) integer definition raises, never rounds a lie."""
    from discopt.modeling.core import Constant

    from discopt.aggregation.recovery import RecoveryMap

    rec = RecoveryMap(defining={"j": Constant(7.5)}, order=["j"], integer_vars=frozenset({"j"}))
    with pytest.raises(AssertionError, match="non-integral"):
        rec.recover({})


def test_duals_unavailable_when_integer_eliminated():
    """IA4: dual recovery declines (with a reason) if an integer was aggregated."""
    res = solve(_ia4_model(), method="gr", integer_aggregation=True, recover_duals=True)
    assert res.duals is not None
    assert not res.duals.available
    assert res.duals.reason == "eliminated_integer_variable"


# --------------------------------------------------------------------------- #
# IA5 — MINLP validation problem + no-regression corpus                        #
# --------------------------------------------------------------------------- #
def test_inventory_chain_roundtrips_with_integer_aggregation():
    """IA5: the integer stock chain aggregates and round-trips exactly (MINLP)."""
    from discopt.aggregation.testproblems import inventory_chain

    n = 6
    m = inventory_chain(n=n)
    direct = inventory_chain(n=n).solve()  # global solver respects integrality
    res = solve(m, method="lm", integer_aggregation=True, order="min_fill")
    assert res.status == "optimal"
    assert res.aggregation.n_eliminated >= n - 1  # the chain collapses
    assert float(res.objective) == pytest.approx(float(direct.objective), abs=1e-6)
    # every recovered eliminated integer is exactly integral
    for name in res.aggregation.eliminated:
        val = float(np.asarray(res.x[name]))
        assert val.is_integer(), f"{name} recovered non-integral {val}"


def test_inventory_chain_flag_off_eliminates_nothing():
    """IA5: with the flag off (default), the integer inventory model is inert."""
    from discopt.aggregation.testproblems import inventory_chain

    assert aggregate(inventory_chain(n=6), method="lm").n_eliminated == 0


def test_integer_flag_off_is_byte_identical_on_corpus():
    """IA5 no-regression: flag off leaves every PROBLEMS model's eliminations unchanged.

    The default (integer_aggregation=False) must reproduce 0.3.0 behavior exactly.
    Continuous problems eliminate as before; the two integer/MINLP problems
    (unit_selection, inventory_chain) eliminate only their continuous intermediates
    (or nothing), never an integer/binary.
    """
    for name, problem in PROBLEMS.items():
        m = problem.build()
        for method in ("ld1", "d2", "lm"):
            off = aggregate(m, method=method, integer_aggregation=False)
            # no integer/binary variable is ever eliminated with the flag off
            from discopt.modeling.core import VarType

            int_names = {
                v.name
                for v in m._variables
                if getattr(v, "var_type", None) in (VarType.INTEGER, VarType.BINARY)
            }
            assert not (set(off.eliminated) & int_names), (name, method, off.eliminated)


def test_binaries_never_aggregated_even_with_flag_on():
    """IA5 adversarial: unit_selection binaries stay put even with the flag ON.

    They appear only in inequalities (x <= cap*b), not in defining equalities, so
    the integrality-implication filter never admits them.
    """
    from discopt.aggregation.testproblems import unit_selection

    agg = aggregate(unit_selection(), method="lm", integer_aggregation=True)
    assert not any(name.startswith("b") for name in agg.eliminated)
