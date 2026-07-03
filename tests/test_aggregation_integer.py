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

from discopt.aggregation.api import aggregate
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
    """
    for name, problem in PROBLEMS.items():
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
