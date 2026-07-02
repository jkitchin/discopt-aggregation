"""Pure-graph kernel for variable aggregation.

This module implements the graph algorithms the aggregation strategies rely on,
independent of any discopt / JAX / Rust machinery (stdlib only):

- :func:`maximum_matching` — Hopcroft-Karp maximum-cardinality matching on a
  bipartite graph.
- :func:`strongly_connected_components` — Tarjan's algorithm (iterative).
- :func:`topological_order` — Kahn topological sort of a DAG.
- :func:`block_triangularize` — permute a matched bipartite graph into
  irreducible block lower-triangular form (paper Algorithms 7-9, Appendix A).

Nodes may be any hashable objects.  A bipartite graph carries two disjoint node
sets we call ``left`` and ``right``; in the aggregation setting ``left`` nodes
are equality constraints and ``right`` nodes are variables, and an edge means
"this variable participates in this constraint".

References
----------
Naik, Biegler, Bent, Parker, *Variable aggregation for nonlinear optimization
problems*, arXiv:2502.13869 — Section 2.3 (bipartite graph), Appendix A (block
triangularization); Duff & Reid (1978); Pothen & Fan (1990); Tarjan (1972).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable, Iterator

_INF = float("inf")


@dataclass
class BipartiteGraph:
    """A bipartite graph with adjacency in both directions.

    ``left`` and ``right`` are the two (disjoint) node sets.  ``edges`` is the
    set of ``(left_node, right_node)`` pairs.  ``adj_left`` / ``adj_right`` are
    the derived adjacency maps; build a graph with :meth:`from_edges` to have
    them populated consistently.
    """

    left: set = field(default_factory=set)
    right: set = field(default_factory=set)
    adj_left: dict[Hashable, set] = field(default_factory=dict)  # left -> set of right
    adj_right: dict[Hashable, set] = field(default_factory=dict)  # right -> set of left

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[tuple[Hashable, Hashable]],
        left: Iterable[Hashable] | None = None,
        right: Iterable[Hashable] | None = None,
    ) -> "BipartiteGraph":
        """Build a graph from ``(left, right)`` edges.

        ``left`` / ``right`` may list isolated nodes that appear in no edge; any
        endpoint of an edge is added automatically.
        """
        g = cls()
        if left:
            for u in left:
                g.left.add(u)
                g.adj_left.setdefault(u, set())
        if right:
            for v in right:
                g.right.add(v)
                g.adj_right.setdefault(v, set())
        for u, v in edges:
            g.add_edge(u, v)
        return g

    def add_edge(self, u: Hashable, v: Hashable) -> None:
        """Add edge between left node ``u`` and right node ``v``."""
        self.left.add(u)
        self.right.add(v)
        self.adj_left.setdefault(u, set()).add(v)
        self.adj_right.setdefault(v, set()).add(u)

    @property
    def edges(self) -> set:
        return {(u, v) for u, vs in self.adj_left.items() for v in vs}

    def neighbors_of_left(self, u: Hashable) -> set:
        return self.adj_left.get(u, set())

    def neighbors_of_right(self, v: Hashable) -> set:
        return self.adj_right.get(v, set())

    def induced_by_edges(self, edges: Iterable[tuple[Hashable, Hashable]]) -> "BipartiteGraph":
        """Subgraph induced by an edge subset (paper Table 1 ``induced_subgraph``)."""
        return BipartiteGraph.from_edges(edges)


def maximum_matching(graph: BipartiteGraph) -> dict:
    """Maximum-cardinality matching via Hopcroft-Karp.

    Returns a ``dict`` mapping each matched left node to its matched right node.
    Runs in ``O((n_v + n_e) * sqrt(n_v))`` (Hopcroft & Karp, 1973).  The DFS is
    iterative (explicit stack) so the algorithm scales to the tens-of-thousands
    of nodes in the paper's larger models without hitting Python's recursion
    limit.

    The returned matching is *a* maximum matching; it is not unique.
    """
    adj = graph.adj_left
    left_nodes = list(graph.left)
    pair_left: dict[Hashable, Hashable | None] = {u: None for u in left_nodes}
    pair_right: dict[Hashable, Hashable | None] = {v: None for v in graph.right}
    dist: dict[Hashable, float] = {}

    def bfs() -> bool:
        """Layer the graph; return True if an augmenting path exists."""
        queue: deque = deque()
        for u in left_nodes:
            if pair_left[u] is None:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = _INF
        found = False
        while queue:
            u = queue.popleft()
            for v in adj.get(u, ()):  # noqa: PLC0206 - iterating neighbours, not dict
                w = pair_right[v]
                if w is None:
                    found = True
                elif dist[w] == _INF:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return found

    def augment(start) -> bool:
        """Iterative DFS following BFS layers; augment if a free right reached."""
        stack: list[Hashable] = [start]
        iters: list[Iterator[Hashable]] = [iter(adj.get(start, ()))]
        took: list[Hashable | None] = [None]  # right node taken to descend per frame
        while stack:
            cur = stack[-1]
            descended = False
            for v in iters[-1]:
                w = pair_right[v]
                if w is None:
                    # Augmenting path found: match every frame to its taken v.
                    took[-1] = v
                    for i, u in enumerate(stack):
                        pair_left[u] = took[i]
                        pair_right[took[i]] = u
                    return True
                if dist.get(w, _INF) == dist[cur] + 1:
                    took[-1] = v
                    stack.append(w)
                    iters.append(iter(adj.get(w, ())))
                    took.append(None)
                    descended = True
                    break
            if not descended:
                dist[cur] = _INF  # dead end this phase
                stack.pop()
                iters.pop()
                took.pop()
        return False

    while bfs():
        for u in left_nodes:
            if pair_left[u] is None:
                augment(u)

    return {u: v for u, v in pair_left.items() if v is not None}


def strongly_connected_components(
    nodes: Iterable[Hashable], successors: Callable[[Hashable], Iterable[Hashable]]
) -> list[list]:
    """Strongly-connected components via Tarjan's algorithm (iterative).

    ``successors(v)`` returns the out-neighbours of ``v``.  Returns a list of
    components (each a list of nodes).  Tarjan emits components in *reverse*
    topological order of the condensation; :func:`block_triangularize` does not
    rely on that and re-derives the order explicitly.
    """
    index: dict = {}
    low: dict = {}
    on_stack: set = set()
    tarjan_stack: list = []
    result: list[list] = []
    counter = 0

    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(successors(root)))]
        while work:
            v, it = work[-1]
            if v not in index:
                index[v] = low[v] = counter
                counter += 1
                tarjan_stack.append(v)
                on_stack.add(v)
            recursed = False
            for w in it:
                if w not in index:
                    work.append((w, iter(successors(w))))
                    recursed = True
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            if recursed:
                continue
            if low[v] == index[v]:
                comp: list = []
                while True:
                    w = tarjan_stack.pop()
                    on_stack.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                result.append(comp)
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])

    return result


def topological_order(
    nodes: Iterable[Hashable], successors: Callable[[Hashable], Iterable[Hashable]]
) -> list:
    """Kahn topological sort of a DAG.

    ``successors(v)`` gives out-neighbours.  Raises :class:`ValueError` if the
    graph has a cycle (callers pass the acyclic condensation).
    """
    nodes = list(nodes)
    node_set = set(nodes)
    indeg = {u: 0 for u in nodes}
    succ_map: dict = {}
    for u in nodes:
        outs = [w for w in successors(u) if w in node_set]
        succ_map[u] = outs
        for w in outs:
            indeg[w] += 1
    queue = deque(u for u in nodes if indeg[u] == 0)
    order: list = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for w in succ_map[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                queue.append(w)
    if len(order) != len(nodes):
        raise ValueError("topological_order: graph contains a cycle")
    return order


def block_triangularize(matching: dict, adj_left: dict) -> list[list[tuple]]:
    """Permute a matched bipartite graph into block lower-triangular form.

    Implements the paper's ``block_triangularize`` (Algorithm 7): project the
    matched bipartite graph onto the matched *right* nodes (variables), find the
    strongly-connected components of that projection, and return the SCCs as an
    ordered partition of the matching in topological (lower-triangular) order.

    Args:
        matching: a ``dict`` mapping left nodes (constraints) to right nodes
            (variables) — as returned by :func:`maximum_matching`.  Only the
            matched edges are triangularized.
        adj_left: adjacency of the *full* bipartite graph, mapping each left
            node to the set of right nodes it touches (``graph.adj_left``).  The
            projection uses every variable in a constraint, not just the matched
            one, to build the dependency edges.

    Returns:
        An ordered list of blocks.  Each block is a list of matched
        ``(left, right)`` edges.  A block of length 1 is an irreducible
        1x1 (triangular) block; a block of length > 1 is an irreducible cycle of
        mutually-dependent defining equations that cannot be explicitly ordered.
        Blocks earlier in the list are lower (their variables are inputs to the
        later blocks' defining equations).

    The projection edge convention follows Algorithm 8: for matched variable
    ``a`` with defining constraint ``b = M^{-1}(a)``, every other variable
    ``a2`` appearing in ``b`` yields a directed edge ``a2 -> a`` ("a depends on
    a2"), so a topological order places inputs before outputs.
    """
    # var -> its defining (matched) constraint
    right_to_left = {v: u for u, v in matching.items()}
    matched_vars = list(right_to_left)
    matched_set = set(matched_vars)

    # Projection: directed dependency graph over matched variables.
    succ: dict = {a: set() for a in matched_vars}
    for a in matched_vars:
        b = right_to_left[a]
        for a2 in adj_left.get(b, ()):  # every variable in the defining constraint
            if a2 in matched_set and a2 != a:
                succ[a2].add(a)  # a2 -> a : a depends on a2

    sccs = strongly_connected_components(matched_vars, lambda n: succ[n])

    # Condensation: map each var to its component id, build a DAG over components.
    comp_id: dict = {}
    for cid, comp in enumerate(sccs):
        for node in comp:
            comp_id[node] = cid
    comp_succ: dict = {cid: set() for cid in range(len(sccs))}
    for a in matched_vars:
        for a2 in succ[a]:  # a -> a2
            ca, ca2 = comp_id[a], comp_id[a2]
            if ca != ca2:
                comp_succ[ca].add(ca2)

    ordered_cids = topological_order(range(len(sccs)), lambda c: comp_succ[c])

    blocks: list[list[tuple]] = []
    for cid in ordered_cids:
        block = [(right_to_left[a], a) for a in sccs[cid]]
        blocks.append(block)
    return blocks


@dataclass
class DulmageMendelsohn:
    """Coarse Dulmage-Mendelsohn decomposition of a bipartite graph.

    Given a maximum matching, the left/right nodes partition into three blocks
    (Dulmage & Mendelsohn 1958; Pothen & Fan 1990):

    * ``underdetermined`` — reachable from *unmatched right* nodes by alternating
      paths; has strictly more right nodes than left (excess right / degrees of
      freedom).
    * ``overdetermined`` — reachable from *unmatched left* nodes; has strictly
      more left nodes than right (excess left / structurally singular rows).
    * ``square`` — the remaining matched nodes; structurally well-determined.

    Each block is stored as ``(left_nodes, right_nodes)`` (sorted lists).  With
    left = constraints and right = variables, ``overdetermined`` left nodes are
    excess (over-constrained / singular) constraints and ``underdetermined`` right
    nodes are free variables.
    """

    underdetermined: tuple[list, list]
    square: tuple[list, list]
    overdetermined: tuple[list, list]

    @property
    def is_square(self) -> bool:
        """True iff the whole system is structurally well-determined (no excess)."""
        return (
            not self.underdetermined[0]
            and not self.underdetermined[1]
            and (not self.overdetermined[0] and not self.overdetermined[1])
        )


def dulmage_mendelsohn(graph: BipartiteGraph, matching: dict | None = None) -> DulmageMendelsohn:
    """Coarse Dulmage-Mendelsohn decomposition of ``graph``.

    ``matching`` is a maximum matching ``{left: right}`` (computed if omitted).
    Alternating-reachability from the unmatched nodes on each side gives the
    over- and under-determined blocks; the matched remainder is the square block.
    Pure-graph (no discopt / solve dependency).
    """
    if matching is None:
        matching = maximum_matching(graph)
    match_l2r = dict(matching)
    match_r2l = {right: left for left, right in matching.items()}
    matched_l = set(match_l2r)
    matched_r = set(match_r2l)

    # Overdetermined: alternating reachability from unmatched LEFT nodes.
    over_l: set = set(graph.left) - matched_l
    over_r: set = set()
    stack = list(over_l)
    while stack:
        left_node = stack.pop()
        for r in graph.neighbors_of_left(left_node):
            if r in over_r:
                continue
            over_r.add(r)
            ml = match_r2l.get(r)  # follow the matching edge back to a left node
            if ml is not None and ml not in over_l:
                over_l.add(ml)
                stack.append(ml)

    # Underdetermined: alternating reachability from unmatched RIGHT nodes.
    under_r: set = set(graph.right) - matched_r
    under_l: set = set()
    stack = list(under_r)
    while stack:
        right_node = stack.pop()
        for left_node in graph.neighbors_of_right(right_node):
            if left_node in under_l:
                continue
            under_l.add(left_node)
            mr = match_l2r.get(left_node)  # follow the matching edge to a right node
            if mr is not None and mr not in under_r:
                under_r.add(mr)
                stack.append(mr)

    square_l = matched_l - over_l - under_l
    square_r = matched_r - over_r - under_r

    return DulmageMendelsohn(
        underdetermined=(sorted(under_l), sorted(under_r)),
        square=(sorted(square_l), sorted(square_r)),
        overdetermined=(sorted(over_l), sorted(over_r)),
    )


__all__ = [
    "BipartiteGraph",
    "maximum_matching",
    "strongly_connected_components",
    "topological_order",
    "block_triangularize",
    "DulmageMendelsohn",
    "dulmage_mendelsohn",
]
