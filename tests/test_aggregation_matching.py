"""Tests for the aggregation graph kernel (:mod:`discopt.aggregation.matching`).

Correctness is validated against independent oracles wherever possible:

- Maximum matching cardinality vs :func:`scipy.sparse.csgraph.maximum_bipartite_matching`
  and vs a tiny, independently-written Kuhn augmenting-path oracle.
- Strongly-connected components vs
  :func:`scipy.sparse.csgraph.connected_components` (``connection="strong"``).
- Block triangularization against the structural invariants from the paper
  (strictly-lower-triangular -> all singletons; mutual cycle -> one 2-block;
  the dependency-order invariant).
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, maximum_bipartite_matching

from discopt.aggregation.matching import (
    BipartiteGraph,
    block_triangularize,
    maximum_matching,
    strongly_connected_components,
    topological_order,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Independent reference oracles                                               #
# --------------------------------------------------------------------------- #
def _kuhn_max_matching_size(n_left: int, adj: dict[int, set[int]]) -> int:
    """Independent Kuhn (Hungarian-style) augmenting-path oracle -> match size."""
    match_right: dict[int, int] = {}

    def try_kuhn(u: int, seen: set[int]) -> bool:
        for v in adj.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            if v not in match_right or try_kuhn(match_right[v], seen):
                match_right[v] = u
                return True
        return False

    size = 0
    for u in range(n_left):
        if try_kuhn(u, set()):
            size += 1
    return size


def _scipy_match_size(n_left: int, n_right: int, adj: dict[int, set[int]]) -> int:
    rows, cols = [], []
    for u, vs in adj.items():
        for v in vs:
            rows.append(u)
            cols.append(v)
    data = np.ones(len(rows), dtype=np.int8)
    mat = csr_matrix((data, (rows, cols)), shape=(n_left, n_right))
    m = maximum_bipartite_matching(mat, perm_type="column")
    return int((m >= 0).sum())


def _random_bipartite(rng: random.Random, n_left: int, n_right: int, p: float):
    adj: dict[int, set[int]] = {u: set() for u in range(n_left)}
    edges = []
    for u in range(n_left):
        for v in range(n_right):
            if rng.random() < p:
                adj[u].add(v)
                edges.append((("c", u), ("v", v)))
    return adj, edges


# --------------------------------------------------------------------------- #
# BipartiteGraph basics                                                       #
# --------------------------------------------------------------------------- #
def test_from_edges_builds_symmetric_adjacency():
    g = BipartiteGraph.from_edges([("c1", "x"), ("c1", "y"), ("c2", "x")])
    assert g.neighbors_of_left("c1") == {"x", "y"}
    assert g.neighbors_of_right("x") == {"c1", "c2"}
    assert g.edges == {("c1", "x"), ("c1", "y"), ("c2", "x")}


def test_isolated_nodes_are_retained():
    g = BipartiteGraph.from_edges([("c1", "x")], left=["c1", "c2"], right=["x", "y"])
    assert "c2" in g.left and "y" in g.right
    assert g.neighbors_of_left("c2") == set()


# --------------------------------------------------------------------------- #
# maximum_matching                                                            #
# --------------------------------------------------------------------------- #
def test_matching_is_valid_and_maximum_fig2():
    # Figure 2 of the paper: eq1 touches {x, y, z}; eq2 touches {x, y}.
    g = BipartiteGraph.from_edges(
        [("eq1", "x"), ("eq1", "y"), ("eq1", "z"), ("eq2", "x"), ("eq2", "y")]
    )
    m = maximum_matching(g)
    # Two constraints, both matchable -> perfect on the constraint side.
    assert len(m) == 2
    _assert_valid_matching(g, m)


def test_matching_empty_graph():
    assert maximum_matching(BipartiteGraph()) == {}


def test_matching_no_edges():
    g = BipartiteGraph.from_edges([], left=["c1", "c2"], right=["x"])
    assert maximum_matching(g) == {}


@pytest.mark.parametrize("seed", range(25))
def test_matching_size_matches_oracles(seed):
    rng = random.Random(seed)
    n_left = rng.randint(1, 9)
    n_right = rng.randint(1, 9)
    p = rng.choice([0.2, 0.4, 0.6, 0.8])
    adj, edges = _random_bipartite(rng, n_left, n_right, p)

    g = BipartiteGraph.from_edges(edges)
    m = maximum_matching(g)
    _assert_valid_matching(g, m)

    kuhn = _kuhn_max_matching_size(n_left, adj)
    scipy_size = _scipy_match_size(n_left, n_right, adj)
    assert len(m) == kuhn == scipy_size


def _assert_valid_matching(g: BipartiteGraph, m: dict) -> None:
    # Every matched pair is a real edge.
    for u, v in m.items():
        assert v in g.neighbors_of_left(u)
    # Injective on the right (no right node matched twice).
    assert len(set(m.values())) == len(m)


# --------------------------------------------------------------------------- #
# strongly_connected_components                                               #
# --------------------------------------------------------------------------- #
def _scipy_scc_labels(n: int, succ: dict[int, set[int]]):
    rows, cols = [], []
    for u, ws in succ.items():
        for w in ws:
            rows.append(u)
            cols.append(w)
    data = np.ones(len(rows), dtype=np.int8)
    mat = csr_matrix((data, (rows, cols)), shape=(n, n))
    ncomp, labels = connected_components(mat, directed=True, connection="strong")
    return ncomp, labels


def _partition_key(components: list[list[int]]):
    return frozenset(frozenset(c) for c in components)


@pytest.mark.parametrize("seed", range(30))
def test_scc_matches_scipy(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 12)
    succ = {u: set() for u in range(n)}
    for u in range(n):
        for w in range(n):
            if u != w and rng.random() < 0.25:
                succ[u].add(w)

    comps = strongly_connected_components(range(n), lambda x: succ[x])
    # Same partition of nodes as scipy's strong-connection labelling.
    ncomp, labels = _scipy_scc_labels(n, succ)
    scipy_comps: dict[int, list[int]] = {}
    for node, lab in enumerate(labels):
        scipy_comps.setdefault(int(lab), []).append(node)
    assert _partition_key(comps) == _partition_key(list(scipy_comps.values()))
    assert len(comps) == ncomp


def test_scc_single_cycle_is_one_component():
    # 0 -> 1 -> 2 -> 0
    succ = {0: {1}, 1: {2}, 2: {0}}
    comps = strongly_connected_components(range(3), lambda x: succ[x])
    assert len(comps) == 1
    assert set(comps[0]) == {0, 1, 2}


def test_scc_dag_is_all_singletons():
    succ = {0: {1, 2}, 1: {2}, 2: set()}
    comps = strongly_connected_components(range(3), lambda x: succ[x])
    assert sorted(sorted(c) for c in comps) == [[0], [1], [2]]


# --------------------------------------------------------------------------- #
# topological_order                                                           #
# --------------------------------------------------------------------------- #
def test_topological_order_respects_edges():
    succ = {0: {1, 2}, 1: {3}, 2: {3}, 3: set()}
    order = topological_order(range(4), lambda x: succ[x])
    pos = {n: i for i, n in enumerate(order)}
    for u, ws in succ.items():
        for w in ws:
            assert pos[u] < pos[w]


def test_topological_order_rejects_cycle():
    succ = {0: {1}, 1: {0}}
    with pytest.raises(ValueError, match="cycle"):
        topological_order(range(2), lambda x: succ[x])


# --------------------------------------------------------------------------- #
# block_triangularize                                                         #
# --------------------------------------------------------------------------- #
def test_block_triangularize_strictly_lower_triangular_all_singletons():
    # Defining equations already strictly lower triangular:
    #   c1 defines x1 (only x1)
    #   c2 defines x2 (x2 depends on x1)
    #   c3 defines x3 (x3 depends on x1, x2)
    adj_left = {
        "c1": {"x1"},
        "c2": {"x1", "x2"},
        "c3": {"x1", "x2", "x3"},
    }
    matching = {"c1": "x1", "c2": "x2", "c3": "x3"}
    blocks = block_triangularize(matching, adj_left)
    assert all(len(b) == 1 for b in blocks)
    # Inputs before outputs: x1's block precedes x2's precedes x3's.
    order = [edge[1] for block in blocks for edge in block]
    assert order.index("x1") < order.index("x2") < order.index("x3")


def test_block_triangularize_mutual_dependency_is_one_block():
    # c1 defines x but also references y; c2 defines y but references x.
    # x and y are mutually dependent -> a single irreducible 2x2 block.
    adj_left = {"c1": {"x", "y"}, "c2": {"x", "y"}}
    matching = {"c1": "x", "c2": "y"}
    blocks = block_triangularize(matching, adj_left)
    assert len(blocks) == 1
    assert len(blocks[0]) == 2
    assert {edge[1] for edge in blocks[0]} == {"x", "y"}


def test_block_triangularize_mixed_blocks_and_order_invariant():
    # x1 independent; {x2,x3} a mutual cycle; x4 depends on x3.
    adj_left = {
        "c1": {"x1"},
        "c2": {"x2", "x3"},
        "c3": {"x2", "x3"},
        "c4": {"x3", "x4"},
    }
    matching = {"c1": "x1", "c2": "x2", "c3": "x3", "c4": "x4"}
    blocks = block_triangularize(matching, adj_left)

    sizes = sorted(len(b) for b in blocks)
    assert sizes == [1, 1, 2]  # x1, x4 singletons; {x2,x3} a 2-block

    # Dependency-order invariant: for every projection edge a2 -> a, the block
    # containing a2 is not after the block containing a.
    block_of = {}
    for bi, block in enumerate(blocks):
        for _c, v in block:
            block_of[v] = bi
    dep_edges = [("x3", "x4")]  # x4 depends on x3
    for a2, a in dep_edges:
        assert block_of[a2] <= block_of[a]


def test_block_triangularize_empty_matching():
    assert block_triangularize({}, {}) == []
