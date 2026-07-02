"""Tests for E3: exact tearing of cyclic blocks (maximum acyclic subset).

The exact tear keeps the largest subset of a cyclic block whose induced
dependency subgraph is acyclic (a maximum induced DAG / complement of a minimum
feedback vertex set).  The core algorithm is unit-tested against known digraphs;
integration tests confirm it eliminates at least as many variables as the greedy
tear and stays strictly lower triangular.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.api import aggregate
from discopt.aggregation.strategies import _induced_acyclic, _max_acyclic_subset

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# core algorithm                                                              #
# --------------------------------------------------------------------------- #
def test_max_acyclic_dag_keeps_everything():
    deps = {"a": set(), "b": {"a"}, "c": {"b"}}  # a -> b -> c
    assert _max_acyclic_subset(list(deps), deps) == {"a", "b", "c"}


def test_max_acyclic_two_cycle_keeps_one():
    deps = {"x": {"y"}, "y": {"x"}}
    assert len(_max_acyclic_subset(list(deps), deps)) == 1


def test_max_acyclic_three_cycle_plus_isolated():
    # 3-cycle a->b->c->a (drop one) plus an isolated node d (always kept).
    deps = {"a": {"c"}, "b": {"a"}, "c": {"b"}, "d": set()}
    keep = _max_acyclic_subset(list(deps), deps)
    assert len(keep) == 3  # two of the cycle + the isolated node
    assert "d" in keep  # the exact tear never needlessly drops the free node


def test_induced_acyclic_predicate():
    deps = {"a": {"b"}, "b": {"a"}, "c": set()}
    assert _induced_acyclic({"a", "c"}, deps) is True  # no cycle among {a, c}
    assert _induced_acyclic({"a", "b"}, deps) is False  # a<->b cycle


# --------------------------------------------------------------------------- #
# integration: exact tear >= greedy tear, and stays lower triangular          #
# --------------------------------------------------------------------------- #
def _cycle_model():
    # A 3-constraint linear cycle x=y, y=z, z=x.
    m = dm.Model("cyc3")
    x = m.continuous("x", lb=-9, ub=9)
    y = m.continuous("y", lb=-9, ub=9)
    z = m.continuous("z", lb=-9, ub=9)
    m.subject_to(x - y == 0.0)
    m.subject_to(y - z == 0.0)
    m.subject_to(z - x == 0.0)
    m.minimize(x**2 + (y - 1) ** 2 + z**2)
    return m


def _assert_lower_triangular(agg):
    inc = agg.incidence
    pos = {e.var: i for i, e in enumerate(agg.eliminations)}
    for i, e in enumerate(agg.eliminations):
        for w in inc.constraint_vars(e.cid) - {e.var}:
            if w in pos:
                assert pos[w] < i


@pytest.mark.parametrize("method", ["gr", "lm"])
def test_exact_tear_at_least_greedy_and_valid(method):
    greedy = aggregate(_cycle_model(), method=method, tearing="greedy")
    exact = aggregate(_cycle_model(), method=method, tearing="exact")
    assert exact.n_eliminated >= greedy.n_eliminated
    # the exact-tear aggregation set is still strictly lower triangular
    from discopt.aggregation.incidence import build_incidence
    from discopt.aggregation.strategies import aggregation_set

    inc = build_incidence(_cycle_model(), tearing="exact")
    _assert_lower_triangular(aggregation_set(inc, method=method))


def test_default_tearing_is_greedy():
    # Default must be unchanged (greedy) so existing behaviour is preserved.
    default = aggregate(_cycle_model(), method="lm")
    greedy = aggregate(_cycle_model(), method="lm", tearing="greedy")
    assert default.n_eliminated == greedy.n_eliminated
