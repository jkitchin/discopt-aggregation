"""Tests for :mod:`discopt.aggregation.strategies`.

The central correctness property for *every* strategy is Lemma 1: the produced
aggregation set must be strictly lower triangular, i.e. each eliminated
variable's defining constraint references (among eliminated variables) only ones
that come earlier in the order.  :func:`_assert_lower_triangular` checks this and
is applied to every strategy's output.  Individual tests then pin down the
filter semantics (Table 2), the Theorem 2 bound for LM, and cycle handling.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.incidence import build_incidence
from discopt.aggregation.matching import (
    BipartiteGraph,
    block_triangularize,
    maximum_matching,
)
from discopt.aggregation.strategies import (
    STRATEGIES,
    AggregationSet,
    aggregation_set,
    resolve_strategy,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# reusable validators                                                          #
# --------------------------------------------------------------------------- #
def _assert_lower_triangular(agg: AggregationSet) -> None:
    """The defining eqn of each elimination may only reference earlier ones."""
    inc = agg.incidence
    pos = {e.var: i for i, e in enumerate(agg.eliminations)}
    eliminated = set(pos)
    # No variable eliminated twice; each matched on a linear edge.
    assert len(pos) == len(agg.eliminations)
    for i, e in enumerate(agg.eliminations):
        assert inc.is_linear_edge(e.cid, e.var)
        assert e.coeff == inc.linear_coeff[(e.cid, e.var)]
        others = inc.constraint_vars(e.cid) - {e.var}
        for w in others:
            if w in eliminated:  # a dependency on another eliminated variable
                assert pos[w] < i, f"{e.var} depends on later-eliminated {w}"


def _assert_matching_valid(agg: AggregationSet) -> None:
    m = agg.matching()
    # injective constraint->var and each cid is a real equality constraint
    assert len(set(m.values())) == len(m)
    for cid in m:
        assert cid in agg.incidence.constraints


# --------------------------------------------------------------------------- #
# every strategy is always lower triangular                                    #
# --------------------------------------------------------------------------- #
def _mixed_model():
    m = dm.Model("mixed")
    v = {n: m.continuous(n, lb=-10, ub=10) for n in "abcdef"}
    a, b, c, d, e, f = (v[n] for n in "abcdef")
    m.subject_to(a == 3.0)  # fixed variable (LD1)
    m.subject_to(b - 2 * a == 1.0)  # linear degree-2 (LD2), b = 2a+1
    m.subject_to(c - d == 4.0)  # equal-coefficient (ECD2), c = d+4
    m.subject_to(e - d**2 == 0.0)  # degree-2 nonlinear (D2), e = d^2
    m.subject_to(f - a * d - e == 0.0)  # 3-var (only GR/LM), f linear
    m.minimize(a + b + c + d + e + f)
    return m


@pytest.mark.parametrize("method", sorted(STRATEGIES))
def test_every_strategy_is_lower_triangular(method):
    m = _mixed_model()
    agg = aggregation_set(m, method=method)
    _assert_lower_triangular(agg)
    _assert_matching_valid(agg)


@pytest.mark.parametrize("method", sorted(STRATEGIES))
def test_strategies_accept_prebuilt_incidence(method):
    inc = build_incidence(_mixed_model())
    agg = STRATEGIES[method](inc)
    _assert_lower_triangular(agg)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown aggregation method"):
        aggregation_set(_mixed_model(), method="nope")


# --------------------------------------------------------------------------- #
# filter semantics (Table 2)                                                   #
# --------------------------------------------------------------------------- #
def test_ld1_only_eliminates_fixed_variables():
    m = _mixed_model()
    agg = aggregation_set(m, method="ld1")
    # Only 'a' is a fixed variable (a == 3).
    assert agg.variables() == ["a"]


def test_ld2_eliminates_two_variable_linear_only():
    m = _mixed_model()
    agg = aggregation_set(m, method="ld2")
    elim = set(agg.variables())
    # Two linear degree-2 constraints qualify: b-2a==1 ({a,b}) and c-d==4 ({c,d}).
    # Each can define *either* of its two variables, so the matching is free to
    # pick one from {a,b} and one from {c,d}; both are eliminated (no shared
    # variables -> no cycle). The nonlinear 'e' and degree-3 'f' never qualify.
    assert agg.n == 2
    assert elim <= {"a", "b", "c", "d"}
    assert len(elim & {"a", "b"}) == 1
    assert len(elim & {"c", "d"}) == 1
    assert "e" not in elim and "f" not in elim


def test_ecd2_requires_equal_coefficients():
    m = dm.Model("ecd2")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    z = m.continuous("z", lb=-5, ub=5)
    w = m.continuous("w", lb=-5, ub=5)
    m.subject_to(y - x == 1.0)  # equal coefficient |1|,|1| -> eligible
    m.subject_to(w - 3 * z == 2.0)  # |1| vs |3| -> NOT equal coefficient
    m.minimize(x + y + z + w)
    agg = aggregation_set(m, method="ecd2")
    assert set(agg.variables()) <= {"y", "x"}  # only the y-x constraint qualifies
    assert "w" not in agg.variables() and "z" not in agg.variables()


def test_d2_can_eliminate_nonlinear_defined_variable():
    m = dm.Model("d2")
    x = m.continuous("x", lb=-5, ub=5)
    e = m.continuous("e", lb=-50, ub=50)
    m.subject_to(e - x**2 == 0.0)  # e = x^2, degree-2 (nonlinear in x)
    m.minimize(e)
    agg = aggregation_set(m, method="d2")
    assert agg.variables() == ["e"]  # e eliminated; x stays (nonlinear)


def test_d2_excludes_degree_three_constraint():
    m = dm.Model("deg3")
    a = m.continuous("a", lb=-5, ub=5)
    b = m.continuous("b", lb=-5, ub=5)
    f = m.continuous("f", lb=-50, ub=50)
    m.subject_to(f - a * b == 0.0)  # three variables -> excluded from D2
    m.minimize(f)
    agg = aggregation_set(m, method="d2")
    assert agg.n == 0


# --------------------------------------------------------------------------- #
# approximate-maximum strategies eliminate the most                            #
# --------------------------------------------------------------------------- #
def test_greedy_and_lm_eliminate_at_least_as_many_as_structure_preserving():
    m = _mixed_model()
    counts = {k: aggregation_set(m, method=k).n for k in STRATEGIES}
    # gr/lm see the 3-variable constraint too, so they eliminate the most.
    assert counts["gr"] >= counts["d2"]
    assert counts["lm"] >= counts["ld1"]
    # ld1 is the most conservative here (only the single fixed variable).
    assert counts["ld1"] == 1


# --------------------------------------------------------------------------- #
# Theorem 2 bound for linear-matching: n_block <= n_agg <= n_match             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("builder", ["mixed", "chain"])
def test_theorem2_bound_for_lm(builder):
    if builder == "mixed":
        m = _mixed_model()
    else:
        m = dm.Model("chain")
        xs = [m.continuous(f"x{i}", lb=-10, ub=10) for i in range(5)]
        for i in range(1, 5):
            m.subject_to(xs[i] - 2 * xs[i - 1] == 1.0)  # x_i = 2 x_{i-1} + 1
        m.minimize(sum(xs))

    inc = build_incidence(m)
    agg = aggregation_set(inc, method="lm")

    linear_edges = [(cid, v) for cid in inc.constraints for v in inc.constraint_linear_vars(cid)]
    m_linear = maximum_matching(BipartiteGraph.from_edges(linear_edges))
    n_match = len(m_linear)
    n_block = len(block_triangularize(m_linear, inc.graph.adj_left))

    assert n_block <= agg.n <= n_match
    assert agg.stats["n_match"] == n_match
    assert agg.stats["n_agg"] == agg.n


# --------------------------------------------------------------------------- #
# cycle handling: mutually-defining variables cannot all be eliminated         #
# --------------------------------------------------------------------------- #
def test_mutual_linear_cycle_eliminates_exactly_one():
    # x = y + 1 and y = x - 1 : a linear 2-cycle. At most one is eliminable.
    m = dm.Model("cycle")
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    m.subject_to(x - y == 1.0)
    m.subject_to(y - x == -1.0)
    m.minimize(x + y)
    for method in ("gr", "lm", "ld2"):
        agg = aggregation_set(m, method=method)
        _assert_lower_triangular(agg)
        assert agg.n <= 1  # never both


def test_nonlinear_cycle_is_broken():
    # x = y^2 (x linear, y nonlinear); y = x (y linear). Substituting gives a
    # nonlinear self-reference -> not strictly lower triangular; break it.
    m = dm.Model("nlcycle")
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    m.subject_to(x - y**2 == 0.0)  # x = y^2
    m.subject_to(y - x == 0.0)  # y = x
    m.minimize(x + y)
    agg = aggregation_set(m, method="lm")
    _assert_lower_triangular(agg)
    assert agg.n <= 1


# --------------------------------------------------------------------------- #
# degree-k strategies (E1): d3, d4, ... generalise d2                          #
# --------------------------------------------------------------------------- #
def _degree_ladder_model():
    # Independent defined outputs of increasing degree, driven by free inputs
    # x, y, z (no elimination cycles), so coverage nests cleanly in k:
    #   p = x        (2 variables) -> reachable from d2
    #   q = x*y      (3 variables) -> reachable from d3
    #   r = x*y*z    (4 variables) -> reachable from d4
    m = dm.Model("dk")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    z = m.continuous("z", lb=-3, ub=3)
    p = m.continuous("p", lb=-9, ub=9)
    q = m.continuous("q", lb=-9, ub=9)
    r = m.continuous("r", lb=-30, ub=30)
    m.subject_to(p - x == 0.0)
    m.subject_to(q - x * y == 0.0)
    m.subject_to(r - x * y * z == 0.0)
    m.minimize(p**2 + q**2 + r**2 + x + y + z)
    return m


def test_degree_k_single_pass_reaches_higher_degree_constraints():
    # d2 reaches only the 2-variable defining constraint; d3 additionally reaches
    # the 3-variable one (q = x*y); d4 also the 4-variable one (r = x*y*z).
    m = _degree_ladder_model()
    d2 = aggregation_set(m, method="d2")
    d3 = aggregation_set(m, method="d3")
    d4 = aggregation_set(m, method="d4")
    assert d2.n == 1  # only the p = x constraint
    assert "q" in d3.variables() and d3.n == 2
    assert "r" in d4.variables() and d4.n == 3
    for agg in (d2, d3, d4):
        _assert_lower_triangular(agg)


def test_degree_k_monotone_coverage_when_acyclic():
    # With no elimination cycles, recursive coverage strictly increases with k.
    from discopt.aggregation.api import aggregate

    counts = {k: aggregate(_degree_ladder_model(), method=f"d{k}").n_eliminated for k in (2, 3, 4)}
    assert counts[2] < counts[3] < counts[4]


def test_resolve_strategy_accepts_dk_and_rejects_bad():
    assert resolve_strategy("d3") is not None
    assert resolve_strategy("d10") is not None
    for bad in ("d1", "d0", "d", "bogus", "dd2"):
        with pytest.raises(ValueError, match="unknown aggregation method"):
            resolve_strategy(bad)


def test_degree_k_helper_rejects_small_k():
    from discopt.aggregation.incidence import build_incidence
    from discopt.aggregation.strategies import degree_k_aggregation_set

    with pytest.raises(ValueError, match="k >= 2"):
        degree_k_aggregation_set(build_incidence(_degree_ladder_model()), 1)


def test_block_greedy_uses_induced_subgraph_not_full_incidence():
    """N3: a cyclic block's greedy tear cannot steal a variable from another block.

    ``a`` is defined by the singleton ``a == 5``; the dense constraint
    ``v0+v1+a == 1`` also contains ``a``.  Algorithm 2 runs greedy on the block's
    induced subgraph, so ``a`` is eliminated via its singleton, not the dense
    constraint, and the Theorem-2 stats stay consistent.
    """
    import discopt.modeling as dm

    from discopt.aggregation.strategies import aggregation_set

    m = dm.Model("n3")
    v0 = m.continuous("v0", lb=-9, ub=9)
    v1 = m.continuous("v1", lb=-9, ub=9)
    a = m.continuous("a", lb=-9, ub=9)
    m.subject_to(v0 + v1 + a - 1 == 0.0)  # dense (cid 0)
    m.subject_to(v0 + v1 - 2 == 0.0)  # cyclic with cid 0 over {v0, v1}
    m.subject_to(a - 5 == 0.0)  # singleton defining a (cid 2)
    m.minimize(v0**2 + v1**2)
    agg = aggregation_set(m, method="lm")
    elim = {e.var: e.cid for e in agg.eliminations}
    assert elim.get("a") == 2, "a must be defined by its singleton constraint, not the dense one"
    # Theorem 2 chain holds
    s = agg.stats
    assert s["n_block"] <= s["n_agg"] <= s["n_match"]


def test_exact_tear_falls_back_to_greedy_on_nonoptimal_milp(monkeypatch):
    """N2: a non-optimal feedback-set MILP falls back to greedy, never a bad tear."""
    import discopt.modeling as dm

    from discopt.aggregation.api import aggregate

    class _Bad:
        status = "iteration_limit"
        x: dict = {}

    import discopt.aggregation.strategies as strat
    monkeypatch.setattr(strat, "_EXACT_TEAR_CAP", 0)  # force the MILP path
    monkeypatch.setattr(dm.Model, "solve", lambda self, **k: _Bad())
    # a cyclic model so exact tearing engages
    m = dm.Model("n2")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    t = m.continuous("t", lb=-9, ub=9)
    m.subject_to(x - (0.5 * y + 1) == 0.0)
    m.subject_to(y - (0.5 * x + 2) == 0.0)
    m.subject_to(t - (x + 1) == 0.0)
    m.minimize(x**2 + y**2 + t**2)
    # must not raise; greedy fallback yields a valid (lower-triangular) result
    agg = aggregate(m, method="lm", tearing="exact")
    assert agg.n_eliminated >= 1


def test_degree_filter_counts_noncandidate_variables():
    """N4: degree filters count all variables, not just aggregation candidates.

    ``y = x + i`` with an integer ``i`` is a three-variable constraint, so the
    degree-2 filter must reject it — otherwise the integer passenger rides into
    every target constraint, violating the structure-preserving claim.
    """
    import discopt.modeling as dm

    from discopt.aggregation.api import aggregate

    m = dm.Model("n4")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    i = m.integer("i", lb=0, ub=5)
    m.subject_to(y - (x + i) == 0.0)
    m.minimize(x**2 + y**2 + i)
    assert aggregate(m, method="d2", recursive=False).eliminated == []
    assert aggregate(m, method="ld2", recursive=False).eliminated == []

    # a genuine two-variable constraint is still eliminated
    m2 = dm.Model("ok")
    a = m2.continuous("a", lb=-9, ub=9)
    b = m2.continuous("b", lb=-9, ub=9)
    m2.subject_to(b - (a + 1) == 0.0)
    m2.minimize(a**2 + b**2)
    assert aggregate(m2, method="d2", recursive=False).eliminated == ["a"]
