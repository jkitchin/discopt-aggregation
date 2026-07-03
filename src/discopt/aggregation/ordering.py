"""Block-triangular ordering emitter (E6 / K2, plugin side).

The linear-matching aggregation (``lm``) selects, as a byproduct, a large
nonsingular submatrix of the equality Jacobian whose block-lower-triangular form
is exactly a **Schur pivot block** for the KKT linear system a nonlinear solver
factorizes at every iteration (Parker, Garcia & Bent, *Exploiting block
triangular submatrices in KKT systems*, arXiv:2602.17968; see also the source
paper's closing suggestion, Naik et al. arXiv:2502.13869).

This module turns that structure into a **permutation** of the candidate
variables, block-lower-triangular part first, in the exact 0-based ``list[int]``
shape that the proposed upstream hooks will accept:

* feral ``OrderingMethod::External(Vec<usize>)`` — ``jkitchin/feral#107``
* pounce ordering / Schur passthrough — ``jkitchin/pounce#180``

Until those land the ordering cannot be *consumed* by the solver, so this is the
plugin-side half of K2: computed and tested now, a one-line handoff later.  The
permutation is over the *candidate* (scalar continuous) variable index space
``list(inc.variables)``; wiring maps those names to the solver's KKT variable
indices.

Cyclic (size > 1) diagonal blocks are kept in the ordering — they are the
irreducible blocks that would need implicit elimination (E5 / ``ImplicitBlock``),
and they still belong to the block-lower-triangular submatrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from discopt.aggregation.incidence import build_incidence
from discopt.aggregation.matching import (
    BipartiteGraph,
    block_triangularize,
    maximum_matching,
)


@dataclass
class BlockTriangularOrdering:
    """A block-lower-triangular permutation of a model's candidate variables.

    Attributes:
        variables: candidate (scalar continuous) variable names in their
            *original* index order — the index space the permutation acts on.
        permutation: 0-based permutation of ``range(len(variables))``.
            ``permutation[k]`` is the original index of the variable placed at
            position ``k``.  With ``matched_first`` (the default) the leading
            ``n_matched`` positions are the block-lower-triangular submatrix.
        blocks: the matched-variable diagonal blocks in inputs-first
            (lower-triangular) order; each block is a list of variable names.
            A length-1 block is an explicitly-triangular 1x1; a longer block is
            an irreducible cycle.
        n_matched: size of the block-lower-triangular submatrix (total matched
            variables).
        n_singleton_blocks: number of explicitly-triangular 1x1 diagonal blocks.
        n_cyclic_blocks: number of irreducible (size > 1) diagonal blocks.
        matched_first: whether the triangular submatrix is placed first (True)
            or last (False) in the permutation.
        method: the aggregation strategy used to build the matching.
    """

    variables: list[str]
    permutation: list[int]
    blocks: list[list[str]]
    n_matched: int
    n_singleton_blocks: int
    n_cyclic_blocks: int
    matched_first: bool
    method: str

    @property
    def permuted_names(self) -> list[str]:
        """Candidate variable names in permuted order."""
        return [self.variables[i] for i in self.permutation]

    def is_valid_permutation(self) -> bool:
        """Whether ``permutation`` is a bijection of ``range(len(variables))``."""
        return sorted(self.permutation) == list(range(len(self.variables)))


def block_triangular_ordering(
    model: Any,
    *,
    method: str = "lm",
    pivot_tol: float = 1e-6,
    matched_first: bool = True,
) -> BlockTriangularOrdering:
    """Emit the block-lower-triangular ordering of ``model``'s candidate variables.

    Builds the ``method`` matching (``lm`` by default), block-triangularizes it
    over the full incidence (keeping cyclic diagonal blocks), and returns the
    variables permuted so the block-lower-triangular submatrix is contiguous.

    Args:
        model: a discopt model.
        method: ``"lm"`` (linear-matching, the intended Schur source) or
            ``"gr"``; any strategy whose matching spans a nonsingular submatrix.
        pivot_tol: relative pivot tolerance for the linear incidence.
        matched_first: place the triangular submatrix first (default) or last.

    Returns:
        A :class:`BlockTriangularOrdering`.  Its ``permutation`` is the 0-based
        array to hand to feral#107 / pounce#180 once those land.
    """
    if method not in ("lm", "gr"):
        raise ValueError(f"method must be 'lm' or 'gr', got {method!r}")
    inc = build_incidence(model, pivot_tol=pivot_tol)

    # The matching whose block-triangular structure is emitted.  "lm" uses the
    # maximum matching on the linear incidence (pre-drop, so cyclic blocks survive
    # as diagonal blocks); "gr" uses the greedy matching (Algorithm 1), which is
    # already lower-triangular by construction.
    cids = sorted(inc.constraints)
    if method == "gr":
        from discopt.aggregation.strategies import _greedy

        matching = _greedy(inc, cids)
    else:
        linear_edges = [(cid, v) for cid in cids for v in inc.constraint_linear_vars(cid)]
        linear_graph = BipartiteGraph.from_edges(linear_edges)
        matching = maximum_matching(linear_graph)

    blocks = block_triangularize(matching, inc.graph.adj_left)
    block_names = [[var for _cid, var in block] for block in blocks]
    ordered_matched = [var for block in block_names for var in block]

    var_names = list(inc.variables)
    index_of = {name: i for i, name in enumerate(var_names)}
    matched_set = set(ordered_matched)
    rest = [n for n in var_names if n not in matched_set]

    order_names = ordered_matched + rest if matched_first else rest + ordered_matched
    permutation = [index_of[n] for n in order_names]

    return BlockTriangularOrdering(
        variables=var_names,
        permutation=permutation,
        blocks=block_names,
        n_matched=len(ordered_matched),
        n_singleton_blocks=sum(1 for b in block_names if len(b) == 1),
        n_cyclic_blocks=sum(1 for b in block_names if len(b) > 1),
        matched_first=matched_first,
        method=method,
    )


__all__ = ["BlockTriangularOrdering", "block_triangular_ordering"]
