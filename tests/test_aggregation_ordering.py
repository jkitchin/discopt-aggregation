"""Tests for E6 / K2: the block-triangular ordering emitter.

The emitter turns the linear-matching nonsingular submatrix into a
block-lower-triangular permutation of the candidate variables — the structural
input feral#107 / pounce#180 will consume as a KKT Schur pivot ordering.  Until
those land the ordering is only *emitted*, so these tests pin the properties the
handoff relies on: the permutation is a valid bijection, the matched submatrix is
contiguous and inputs-first (strict lower-triangularity, Lemma 1), cyclic blocks
survive as diagonal blocks, and the result is deterministic.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest

from discopt.aggregation.ordering import (
    BlockTriangularOrdering,
    block_triangular_ordering,
)

pytestmark = pytest.mark.unit


def _chain_model(n=6):
    """x0 free; x_i defined by x_{i-1} — a pure lower-triangular chain."""
    m = dm.Model("chain")
    xs = [m.continuous(f"x{i}", lb=-10, ub=10) for i in range(n)]
    for i in range(1, n):
        m.subject_to(xs[i] - (2 * xs[i - 1] + 1) == 0.0)
    m.minimize(sum(x**2 for x in xs))
    return m


def _cyclic_model():
    """Contains a 2-cycle (x2 == x3, x3 == x2) → one size-2 diagonal block."""
    m = dm.Model("cyc")
    v = [m.continuous(f"x{i}", lb=-9, ub=9) for i in range(4)]
    m.subject_to(v[0] - v[1] == 0.0)
    m.subject_to(v[1] - (v[0] + v[2]) == 0.0)
    m.subject_to(v[2] - v[3] == 0.0)
    m.subject_to(v[3] - v[2] == 0.0)
    m.minimize(sum(x**2 for x in v))
    return m


@pytest.mark.parametrize("builder", [_chain_model, _cyclic_model])
def test_permutation_is_valid_bijection(builder):
    ordering = block_triangular_ordering(builder())
    assert isinstance(ordering, BlockTriangularOrdering)
    assert ordering.is_valid_permutation()
    assert sorted(ordering.permutation) == list(range(len(ordering.variables)))
    # permuted_names is a reordering of the same name set
    assert sorted(ordering.permuted_names) == sorted(ordering.variables)


def test_chain_prefix_is_block_lower_triangular():
    ordering = block_triangular_ordering(_chain_model(6))
    # 5 of 6 vars are matched; one endpoint is free. Which endpoint is free
    # depends on the maximum matching chosen (hash-seed dependent), so assert the
    # structure, not specific names.
    assert ordering.n_matched == 5
    assert ordering.n_cyclic_blocks == 0
    assert ordering.n_singleton_blocks == 5
    prefix = ordering.permuted_names[: ordering.n_matched]
    assert len(prefix) == 5 and len(set(prefix)) == 5
    assert len(ordering.permuted_names[ordering.n_matched :]) == 1  # one free var
    # blocks (inputs-first, block-lower-triangular) flatten to exactly the prefix
    assert [v for b in ordering.blocks for v in b] == prefix


def test_cyclic_block_survives_as_diagonal_block():
    ordering = block_triangular_ordering(_cyclic_model())
    assert ordering.n_cyclic_blocks >= 1
    # the 2-cycle {x2, x3} appears as one contiguous size-2 block
    cyclic = [b for b in ordering.blocks if len(b) > 1]
    assert any(set(b) == {"x2", "x3"} for b in cyclic)
    # every matched variable lands in the contiguous triangular prefix
    prefix = set(ordering.permuted_names[: ordering.n_matched])
    assert prefix == {v for b in ordering.blocks for v in b}


def test_matched_last_places_submatrix_at_end():
    front = block_triangular_ordering(_chain_model(5), matched_first=True)
    back = block_triangular_ordering(_chain_model(5), matched_first=False)
    assert back.is_valid_permutation()
    assert back.n_matched == front.n_matched
    # same blocks, opposite placement
    assert back.blocks == front.blocks
    assert back.permuted_names[-back.n_matched :] == front.permuted_names[: front.n_matched]
    # the single free var swaps from the front to the back (identity is seed-dependent)
    assert back.permuted_names[: -back.n_matched] == front.permuted_names[front.n_matched :]
    assert len(back.permuted_names[: -back.n_matched]) == 1


def test_ordering_is_deterministic():
    a = block_triangular_ordering(_cyclic_model())
    b = block_triangular_ordering(_cyclic_model())
    assert a.permutation == b.permutation
    assert a.blocks == b.blocks


def test_counts_are_consistent():
    ordering = block_triangular_ordering(_chain_model(7))
    assert ordering.n_matched == sum(len(b) for b in ordering.blocks)
    assert ordering.n_singleton_blocks + ordering.n_cyclic_blocks == len(ordering.blocks)
    assert ordering.n_matched + 1 == len(ordering.variables)  # only x0 unmatched
