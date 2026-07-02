"""Tests for E4: decomposable-matching local search.

The local search improves the block decomposability (``n_block``, Theorem 2's
lower bound) of the linear-matching maximum matching by cardinality-preserving
swaps.  Whether it *strictly* improves is model-dependent — on small models the
Hopcroft-Karp matching is often already optimal, and inherent cycles cannot be
matched away (the paper used metaheuristics on large models).  These tests pin
the guaranteed invariants: cardinality is preserved, ``n_block`` never decreases,
the result stays a valid maximum matching, and aggregation eliminates at least as
many variables with the option on.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.api import aggregate
from discopt.aggregation.incidence import build_incidence
from discopt.aggregation.matching import BipartiteGraph, maximum_matching
from discopt.aggregation.strategies import _decomposable_matching, _n_block

pytestmark = pytest.mark.unit


def _cyclic_model():
    m = dm.Model("d")
    v = [m.continuous(f"x{i}", lb=-9, ub=9) for i in range(4)]
    m.subject_to(v[0] - v[1] == 0.0)
    m.subject_to(v[1] - (v[0] + v[2]) == 0.0)
    m.subject_to(v[2] - v[3] == 0.0)
    m.subject_to(v[3] - v[2] == 0.0)
    m.minimize(sum(x**2 for x in v))
    return m


def _chain_model(n=6):
    m = dm.Model("chain")
    xs = [m.continuous(f"x{i}", lb=-10, ub=10) for i in range(n)]
    for i in range(1, n):
        m.subject_to(xs[i] - (2 * xs[i - 1] + 1) == 0.0)
    m.minimize(sum(x**2 for x in xs))
    return m


@pytest.mark.parametrize("builder", [_cyclic_model, _chain_model])
def test_decomposable_matching_invariants(builder):
    inc = build_incidence(builder())
    edges = [(cid, v) for cid in inc.constraints for v in inc.constraint_linear_vars(cid)]
    m0 = maximum_matching(BipartiteGraph.from_edges(edges))
    md = _decomposable_matching(inc, m0)
    # cardinality preserved and still a valid (injective) matching
    assert len(md) == len(m0)
    assert len(set(md.values())) == len(md)
    for cid, var in md.items():
        assert var in inc.constraint_linear_vars(cid)
    # n_block never decreases
    assert _n_block(md, inc) >= _n_block(m0, inc)


@pytest.mark.parametrize("builder", [_cyclic_model, _chain_model])
def test_aggregate_decomposable_at_least_default(builder):
    default = aggregate(builder(), method="lm", decomposable=False)
    decomp = aggregate(builder(), method="lm", decomposable=True)
    assert decomp.n_eliminated >= default.n_eliminated
    # decomposable result is still strictly lower triangular
    inc = build_incidence(builder(), decomposable=True)
    from discopt.aggregation.strategies import aggregation_set

    agg = aggregation_set(inc, method="lm")
    pos = {e.var: i for i, e in enumerate(agg.eliminations)}
    for i, e in enumerate(agg.eliminations):
        for w in inc.constraint_vars(e.cid) - {e.var}:
            if w in pos:
                assert pos[w] < i


def test_default_is_not_decomposable():
    # Default off: unchanged behaviour.
    off = aggregate(_cyclic_model(), method="lm")
    explicit = aggregate(_cyclic_model(), method="lm", decomposable=False)
    assert off.n_eliminated == explicit.n_eliminated
