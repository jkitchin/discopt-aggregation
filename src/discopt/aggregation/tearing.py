"""Scalable exact tearing via lazy-cycle MILP (plan item S2).

The exact tear of a cyclic block keeps the largest subset of variables whose
induced defining subgraph is acyclic — equivalently the complement of a *minimum
feedback vertex set* (FVS).  The brute-force version (``strategies._exact_tear``)
enumerates subsets and is exact only to ~13 variables.

This module solves the same problem for larger blocks by **lazy cycle
enumeration + a covering MILP**, in the spirit of Baharev, Schichl, Neumaier &
Achterberg (ACM JEA 2021) for the closely-related minimum feedback *arc* set:

1. Solve ``min Σ r_v`` s.t. ``Σ_{v∈C} r_v ≥ 1`` for the cycles ``C`` found so far
   (``r_v = 1`` means *remove* ``v``), using discopt's own MILP solver.
2. Find a cycle in the subgraph on the kept variables; if none, the kept set is a
   maximum acyclic subset (done).  Otherwise add that cycle's covering constraint
   and re-solve.

Only finitely many simple cycles exist, so this terminates with the exact optimum.
It is used only under ``tearing="exact"`` for blocks past the brute-force cap; any
failure falls back to the greedy tear (callers never degrade).
"""

from __future__ import annotations

from typing import Any


def find_cycle(nodes: list[str], succ: dict[str, list[str]]) -> list[str] | None:
    """A simple directed cycle in ``(nodes, succ)``, or ``None`` if acyclic.

    ``succ[v]`` are the out-neighbours of ``v``.  Iterative DFS with white/grey/
    black colouring; on a back edge to a grey node the cycle is reconstructed from
    the current DFS path.
    """
    white, grey, black = 0, 1, 2
    color = {v: white for v in nodes}
    for root in nodes:
        if color[root] != white:
            continue
        color[root] = grey
        stack: list[tuple[str, Any]] = [(root, iter(succ.get(root, ())))]
        path = [root]
        while stack:
            node, it = stack[-1]
            advanced = False
            for w in it:
                if color.get(w, black) == white:
                    color[w] = grey
                    stack.append((w, iter(succ.get(w, ()))))
                    path.append(w)
                    advanced = True
                    break
                if color.get(w, black) == grey:  # back edge -> cycle
                    return path[path.index(w) :]
            if not advanced:
                color[node] = black
                stack.pop()
                path.pop()
    return None


def _solve_min_cover(nodes: list[str], cycles: list[list[str]]) -> set[str]:
    """Minimum set of ``nodes`` hitting every cycle in ``cycles`` (discopt MILP)."""
    import discopt.modeling as dm

    m = dm.Model("fvs")
    r = {v: m.binary(f"r{i}") for i, v in enumerate(nodes)}
    for cyc in cycles:
        m.subject_to(sum(r[v] for v in cyc) >= 1)
    m.minimize(sum(r.values()))
    result = m.solve()
    x = result.x
    return {v for v in nodes if round(float(x[r[v].name])) == 1}


def max_acyclic_subset_milp(
    nodes: list[str], deps: dict[str, set[str]], max_rounds: int = 10_000
) -> set[str]:
    """Maximum subset of ``nodes`` with an acyclic induced subgraph (exact, MILP).

    ``deps[v]`` are the nodes ``v`` depends on (edge ``d -> v``).  Returns the kept
    set (complement of a minimum feedback vertex set).
    """
    node_set = set(nodes)
    succ: dict[str, list[str]] = {v: [] for v in nodes}
    for v in nodes:
        for d in deps[v]:
            if d in node_set:
                succ[d].append(v)  # edge d -> v

    cycles: list[list[str]] = []
    for _ in range(max_rounds):
        removed = _solve_min_cover(nodes, cycles)
        kept = [v for v in nodes if v not in removed]
        kept_set = set(kept)
        ksucc = {v: [w for w in succ[v] if w in kept_set] for v in kept}
        cyc = find_cycle(kept, ksucc)
        if cyc is None:
            return kept_set
        cycles.append(cyc)
    raise RuntimeError("max_acyclic_subset_milp did not converge")  # pragma: no cover


__all__ = ["find_cycle", "max_acyclic_subset_milp"]
